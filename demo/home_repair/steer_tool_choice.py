#!/usr/bin/env python3
"""Causal check for the home-repair demo: edit SAE features, re-read the tool choice.

The demo's observational results (features, tool-choice probabilities, contrasts,
theme evidence) come from the modified-vLLM backend.  vLLM cannot intervene
mid-forward, so this script runs the *intervention* with HuggingFace
transformers inside the same container:

  1. build the three base decision prompts exactly as the demo does,
  2. read the baseline tool distribution at the ``I'll use the`` position,
  3. edit selected SAE features in the residual stream entering the SAE layer
     (delta patch, plus an encode-edit-decode "replace" arm and a
     reconstruction-only control), and re-read the distribution,
  4. record HF-vs-vLLM parity (residual cosine, baseline distribution) so the
     backend drift is visible next to the result.

Feature selection and clamp targets are taken from the vLLM evaluation report
when it is provided, so the causal check tests exactly the features the page
shows.

Usage (inside 575lab/kiji-inspector:dev):
    python demo/home_repair/steer_tool_choice.py \
        --model-name /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp \
        --vllm-report demo/home_repair/output/prompt_alignment/vllm_native_evaluation.json \
        --vllm-activations demo/home_repair/output/prompt_alignment/vllm_six_layer_demo_activations.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare_sae_backends as backend_compare
import home_repair_demo as demo
import numpy as np
import torch

from kiji_inspector.core.sae_core import JumpReLUSAE

_HAZARD_THEMES = ("safe_vs_hazardous", "diy_vs_professional")


# ---------------------------------------------------------------------------
# Pure helpers (unit tested)
# ---------------------------------------------------------------------------


def _sae_scale(sae) -> float:
    scale = getattr(sae, "rms_scale", None)
    return float(scale) if scale else 1.0


def make_feature_edit_hook(
    sae,
    edits: dict[int, float | None],
    mode: str = "delta",
    decision_token_only: bool = True,
    record: dict | None = None,
    position: int = -1,
):
    """Forward pre-hook that edits SAE features of the residual entering a layer.

    ``edits`` maps feature index -> target activation (``None`` = 0, i.e.
    ablate).  Two modes:

    * ``"delta"`` (primary): ``x += rms_scale * (target - a_i) * W_dec[i]`` on
      the real residual, where ``a_i`` is the feature's current activation.
      Because ``denormalize_output`` is affine, this is exactly the change the
      SAE decoder attributes to feature ``i`` — no reconstruction error is
      injected, so a no-op edit leaves the residual untouched.
    * ``"replace"``: encode -> edit -> decode -> denormalize and swap the
      residual for the reconstruction (the classic ablation arm).  With
      ``edits={}`` this is the reconstruction-only control.

    Only one token is modified when ``decision_token_only``: the last one by
    default, or the one at negative index ``position`` (e.g. ``-2`` when a
    tool's second token has been appended after the decision token).
    ``record`` (optional dict) receives the pre-edit activations of the edited
    features so callers can report what the SAE actually saw under HF.
    """
    if position >= 0:
        raise ValueError("position must be a negative index from the end of the sequence")
    if mode not in ("delta", "replace"):
        raise ValueError(f"Unknown mode {mode!r}")
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    scale = _sae_scale(sae)

    def _edit(hidden_slice: torch.Tensor) -> torch.Tensor:
        flat = hidden_slice.reshape(-1, hidden_slice.shape[-1]).to(
            device=sae_device, dtype=sae_dtype
        )
        with torch.no_grad():
            features = sae.encode(sae.normalize_input(flat))
            if record is not None:
                record["pre_edit"] = {
                    int(idx): float(features[-1, idx].float().item()) for idx in edits
                }
            if mode == "delta":
                modified = flat.clone().float()
                for idx, target in edits.items():
                    current = features[:, idx].float()
                    desired = torch.zeros_like(current) if target is None else current * 0 + target
                    delta = (desired - current).unsqueeze(1)  # (rows, 1)
                    modified = modified + scale * delta * sae.W_dec[idx].float().unsqueeze(0)
            else:
                for idx, target in edits.items():
                    features[:, idx] = 0.0 if target is None else target
                modified = sae.denormalize_output(sae.decode(features)).float()
        return modified.reshape(hidden_slice.shape)

    def hook(module, args, kwargs):
        if args:
            hidden = args[0]
            rest_args = args[1:]
            hidden_from_kwargs = False
        else:
            hidden = kwargs["hidden_states"]
            rest_args = None
            hidden_from_kwargs = True
        orig_device, orig_dtype = hidden.device, hidden.dtype
        if decision_token_only:
            end = position + 1 if position != -1 else hidden.shape[1]
            edited = _edit(hidden[:, position:end, :]).to(device=orig_device, dtype=orig_dtype)
            hidden = torch.cat([hidden[:, :position, :], edited, hidden[:, end:, :]], dim=1)
        else:
            hidden = _edit(hidden).to(device=orig_device, dtype=orig_dtype)
        if hidden_from_kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden
            return args, new_kwargs
        return (hidden,) + tuple(rest_args), kwargs

    return hook


def select_steering_features(
    active: list[tuple[int, float]],
    contrastive_map: dict[int, list[dict]],
    themes: tuple[str, ...] | list[str],
    side: str = "contrast",
    k: int = 3,
    labels: dict | None = None,
) -> list[dict]:
    """Active features on the requested side of the given themes, by act*|d|."""
    best: dict[int, dict] = {}
    for index, activation in active:
        for entry in contrastive_map.get(int(index), []):
            if entry.get("theme") not in themes or entry.get("direction") != side:
                continue
            weight = float(activation) * abs(float(entry.get("cohens_d", 0.0)))
            candidate = {
                "index": int(index),
                "label": demo._label_for(labels, int(index)),
                "activation": round(float(activation), 4),
                "cohensD": round(abs(float(entry.get("cohens_d", 0.0))), 4),
                "theme": entry.get("theme"),
                "weight": round(weight, 4),
            }
            if index not in best or weight > best[int(index)]["weight"]:
                best[int(index)] = candidate
    return sorted(best.values(), key=lambda row: -row["weight"])[:k]


def distribution_from_logits(logits_row: torch.Tensor, tool_to_token: dict[str, int]) -> dict:
    """Tool-choice readout from one row of next-token logits (full vocab)."""
    log_probs = torch.log_softmax(logits_row.float(), dim=-1)
    logprobs = {tid: float(log_probs[tid].item()) for tid in tool_to_token.values()}
    sampled = int(torch.argmax(logits_row).item())
    return demo.decision_from_logprobs(logprobs, tool_to_token, sampled_id=sampled)


def distribution_deltas(baseline: dict, intervened: dict) -> dict[str, float]:
    return {
        tool: round(intervened.get(tool, 0.0) - baseline.get(tool, 0.0), 4) for tool in baseline
    }


def hazard_experiments(
    active_by_problem: dict[str, list[tuple[int, float]]],
    contrastive_map: dict[int, list[dict]],
    labels: dict | None,
    k: int = 3,
    reference_activations: dict[str, dict[int, float]] | None = None,
) -> list[dict]:
    """Optional theme-based checks (hazard/professional-side features).

    1. water heater: ablate its top hazard/professional-side features.
    2. disposal: clamp the same features to the water-heater activations.

    ``active_by_problem`` should describe what the *intervened* backend
    represents (HF); ``reference_activations`` (e.g. the vLLM run) is only
    attached for reporting.
    """
    water = active_by_problem.get("water_heater_noise", [])
    hazard = select_steering_features(water, contrastive_map, _HAZARD_THEMES, "contrast", k, labels)
    if reference_activations:
        reference = reference_activations.get("water_heater_noise", {})
        for row in hazard:
            row["vllmActivation"] = reference.get(row["index"])
    experiments = []
    if hazard:
        experiments.append(
            {
                "id": "water_heater_ablate_hazard",
                "problem": "water_heater_noise",
                "mode": "ablate",
                "description": (
                    "Zero the water heater's strongest hazard/professional-side features "
                    "at the decision token."
                ),
                "features": [{**row, "target": 0.0} for row in hazard],
            }
        )
        experiments.append(
            {
                "id": "disposal_clamp_hazard",
                "problem": "disposal_stuck",
                "mode": "clamp",
                "description": (
                    "Clamp the same hazard/professional-side features on the disposal "
                    "prompt to the values they take on the water heater."
                ),
                "features": [{**row, "target": row["activation"]} for row in hazard],
            }
        )
    return experiments


def discriminating_features(
    base_active: list[tuple[int, float]],
    variant_active: list[tuple[int, float]],
    labels: dict | None,
    k: int = 5,
) -> list[dict]:
    """Features that are stronger on the base decision than on its contrast.

    Ranked by the activation gap (base - variant); these are the candidates
    for "what makes the model pick the base tool rather than the variant's".
    """
    base = {int(i): float(a) for i, a in base_active}
    variant = {int(i): float(a) for i, a in variant_active}
    rows = []
    for index, activation in base.items():
        gap = activation - variant.get(index, 0.0)
        if gap <= 0:
            continue
        rows.append(
            {
                "index": index,
                "label": demo._label_for(labels, index),
                "activation": round(activation, 4),
                "variantActivation": round(variant.get(index, 0.0), 4),
                "gap": round(gap, 4),
            }
        )
    rows.sort(key=lambda row: -row["gap"])
    return rows[:k]


def contrast_experiments(
    active_by_problem: dict[str, list[tuple[int, float]]],
    contrast_active_by_problem: dict[str, list[tuple[int, float]]],
    labels: dict | None,
    k: int = 5,
    reference_activations: dict[str, dict[int, float]] | None = None,
    problems: list[dict] | None = None,
) -> list[dict]:
    """Default causal checks, derived from each problem's contrast prompt.

    For every problem whose base and contrast decisions differ in wording:
      * ablate, on the base prompt, the top-k features that discriminate the
        base decision from the contrast decision;
      * clamp, on the contrast prompt, the same features to their base values.
    """
    experiments = []
    for problem in problems or demo._PROBLEMS:
        pid = problem["id"]
        base = active_by_problem.get(pid)
        variant = contrast_active_by_problem.get(pid)
        if not base or not variant:
            continue
        rows = discriminating_features(base, variant, labels, k)
        if reference_activations:
            reference = reference_activations.get(pid, {})
            for row in rows:
                row["vllmActivation"] = reference.get(row["index"])
        if not rows:
            continue
        base_tool = demo.tool_display(problem["initial_decision"]["tool"])
        changed = problem.get("contrast", {}).get("changed", "the contrast wording")
        experiments.append(
            {
                "id": f"{pid}_ablate_discriminating",
                "problem": pid,
                "step": f"{pid}_InitialDecision",
                "mode": "ablate",
                "description": (
                    f"On the base request, zero the {len(rows)} features that are strongest "
                    f"here relative to the contrast ({changed}). If they carry the "
                    f"decision, {base_tool} should lose probability."
                ),
                "features": [{**row, "target": 0.0} for row in rows],
            }
        )
        experiments.append(
            {
                "id": f"{pid}_clamp_discriminating",
                "problem": pid,
                "step": f"{pid}_Contrast",
                "mode": "clamp",
                "description": (
                    f"On the contrast request ({changed}), clamp the same features to "
                    f"their base-request values. If they carry the decision, "
                    f"{base_tool} should gain probability."
                ),
                "features": [{**row, "target": row["activation"]} for row in rows],
            }
        )
    return experiments


def matched_random_sets(
    active: list[tuple[int, float]],
    exclude: set[int],
    target_mass: float,
    target_size: int,
    draws: int,
    seed: int,
) -> list[list[int]]:
    """Random sets of active features, matched to a family's size and mass.

    Each draw adds randomly chosen active features (outside ``exclude``) until
    both the family's count and its summed activation are reached, so ablating
    a draw removes at least as much as ablating the family.  Deterministic for
    a given ``seed``.
    """
    pool = [(int(i), float(a)) for i, a in active if int(i) not in exclude and a > 0]
    if not pool:
        return []
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(draws):
        order = rng.permutation(len(pool))
        chosen, mass = [], 0.0
        for position in order:
            index, activation = pool[position]
            chosen.append(index)
            mass += activation
            if len(chosen) >= target_size and mass >= target_mass:
                break
        sets.append(sorted(chosen))
    return sets


def attribution_plan(
    rows: list[dict], hf_active: list[tuple[int, float]], draws: int = 3, seed: int = 0
) -> dict:
    """What to ablate for one problem's snapshot rows, plus matched controls."""
    hf_map = {int(i): float(a) for i, a in hf_active}
    families = {int(row["index"]): demo.row_family(row) for row in rows}
    exclude = {index for family in families.values() for index in family}
    plan_rows = []
    controls = []
    for position, row in enumerate(rows):
        family = families[int(row["index"])]
        present = [index for index in family if hf_map.get(index, 0.0) > 0]
        mass = sum(hf_map.get(index, 0.0) for index in family)
        plan_rows.append(
            {
                "index": int(row["index"]),
                "label": row.get("label"),
                "family": family,
                "familySize": len(family),
                "activeUnderHf": len(present),
                "hfActivation": round(max((hf_map.get(i, 0.0) for i in family), default=0.0), 4),
                "hfMass": round(mass, 4),
                "inactiveUnderHf": not present,
            }
        )
        for draw, chosen in enumerate(
            matched_random_sets(
                hf_active, exclude, mass, max(1, len(present)), draws, seed + 97 * position
            )
        ):
            controls.append(
                {
                    "matchesRow": int(row["index"]),
                    "draw": draw,
                    "features": chosen,
                    "size": len(chosen),
                    "hfMass": round(sum(hf_map.get(i, 0.0) for i in chosen), 4),
                }
            )
    return {"rows": plan_rows, "allFeatures": sorted(exclude), "controls": controls}


def summarize_attribution(
    plan: dict,
    baseline: dict,
    row_readings: list[dict],
    all_reading: dict | None,
    control_readings: list[dict],
    target_tool: str,
) -> dict:
    """Combine the readings of one problem's attribution run into the UI block."""
    base_dist = baseline["distribution"]
    base_target = float(base_dist.get(target_tool, 0.0))

    def _delta(reading: dict) -> float:
        return round(float(reading["distribution"].get(target_tool, 0.0)) - base_target, 4)

    rows = []
    for plan_row, reading in zip(plan["rows"], row_readings, strict=True):
        rows.append(
            {
                **plan_row,
                "intervened": reading["distribution"],
                "deltaTarget": _delta(reading),
                "deltas": distribution_deltas(base_dist, reading["distribution"]),
                "argmaxChanged": reading.get("display") != baseline.get("display"),
            }
        )
    controls = []
    for plan_control, reading in zip(plan["controls"], control_readings, strict=True):
        controls.append({**plan_control, "deltaTarget": _delta(reading)})
    control_threshold = max((abs(c["deltaTarget"]) for c in controls), default=0.0)
    result = {
        "targetTool": target_tool,
        "baseline": base_dist,
        "hfChoice": baseline.get("display"),
        "rows": rows,
        "controls": controls,
        "controlThreshold": round(control_threshold, 4),
        "controlMeanAbsDelta": round(
            sum(abs(c["deltaTarget"]) for c in controls) / len(controls) if controls else 0.0, 4
        ),
    }
    if all_reading is not None:
        result["allRows"] = {
            "features": plan["allFeatures"],
            "size": len(plan["allFeatures"]),
            "intervened": all_reading["distribution"],
            "deltaTarget": _delta(all_reading),
            "argmaxChanged": all_reading.get("display") != baseline.get("display"),
        }
    return result


def injection_plan(
    rows: list[dict],
    base_active: list[tuple[int, float]],
    open_active: list[tuple[int, float]],
    draws: int = 3,
    seed: int = 0,
) -> dict:
    """What to inject into the open (no-ask) prompt, plus matched controls.

    Each snapshot row's family is clamped to its activations on the base
    (explicit-ask) prompt; ``allRows`` clamps every family together; ``allBase``
    clamps *every* feature active on the base prompt to its base value (the
    upper bound of what the SAE feature space can put back).  Controls are
    random sets of base-active features outside the families, matched to each
    row's count and activation mass, clamped to their base values.
    """
    base_map = {int(i): float(a) for i, a in base_active if a > 0}
    open_map = {int(i): float(a) for i, a in open_active if a > 0}
    families = {int(row["index"]): demo.row_family(row) for row in rows}
    exclude = {index for family in families.values() for index in family}
    plan_rows = []
    controls = []
    for position, row in enumerate(rows):
        family = families[int(row["index"])]
        targets = {index: base_map[index] for index in family if index in base_map}
        mass = sum(targets.values())
        plan_rows.append(
            {
                "index": int(row["index"]),
                "label": row.get("label"),
                "family": family,
                "familySize": len(family),
                "targets": {str(k): round(v, 4) for k, v in targets.items()},
                "baseMass": round(mass, 4),
                "openMass": round(sum(open_map.get(index, 0.0) for index in family), 4),
                "absentOnBase": not targets,
            }
        )
        for draw, chosen in enumerate(
            matched_random_sets(
                list(base_map.items()),
                exclude,
                mass,
                max(1, len(targets)),
                draws,
                seed + 97 * position,
            )
        ):
            controls.append(
                {
                    "matchesRow": int(row["index"]),
                    "draw": draw,
                    "targets": {str(index): round(base_map[index], 4) for index in chosen},
                    "size": len(chosen),
                    "baseMass": round(sum(base_map[index] for index in chosen), 4),
                }
            )
    all_rows = {}
    for plan_row in plan_rows:
        all_rows.update({int(k): float(v) for k, v in plan_row["targets"].items()})
    return {
        "rows": plan_rows,
        "allRowsTargets": {str(k): round(v, 4) for k, v in sorted(all_rows.items())},
        "allBaseTargets": {str(k): round(v, 4) for k, v in sorted(base_map.items())},
        "controls": controls,
    }


def summarize_injection(
    plan: dict,
    baseline: dict,
    row_readings: list[dict],
    all_rows_reading: dict | None,
    all_base_reading: dict | None,
    control_readings: list[dict],
    target_tool: str,
) -> dict:
    """Combine the readings of one problem's injection run (open prompt)."""
    base_dist = baseline["distribution"]
    base_target = float(base_dist.get(target_tool, 0.0))

    def _delta(reading: dict) -> float:
        return round(float(reading["distribution"].get(target_tool, 0.0)) - base_target, 4)

    def _arm(reading: dict, extra: dict) -> dict:
        return {
            **extra,
            "intervened": reading["distribution"],
            "deltaTarget": _delta(reading),
            "deltas": distribution_deltas(base_dist, reading["distribution"]),
            "argmaxChanged": reading.get("display") != baseline.get("display"),
            "choice": reading.get("display"),
        }

    rows = [
        _arm(reading, plan_row)
        for plan_row, reading in zip(plan["rows"], row_readings, strict=True)
    ]
    controls = [
        {**plan_control, "deltaTarget": _delta(reading), "choice": reading.get("display")}
        for plan_control, reading in zip(plan["controls"], control_readings, strict=True)
    ]
    control_threshold = max((abs(c["deltaTarget"]) for c in controls), default=0.0)
    result = {
        "targetTool": target_tool,
        "baseline": base_dist,
        "hfChoice": baseline.get("display"),
        "rows": rows,
        "controls": controls,
        "controlThreshold": round(control_threshold, 4),
        "controlMeanAbsDelta": round(
            sum(abs(c["deltaTarget"]) for c in controls) / len(controls) if controls else 0.0, 4
        ),
    }
    if all_rows_reading is not None:
        result["allRows"] = _arm(all_rows_reading, {"size": len(plan["allRowsTargets"])})
    if all_base_reading is not None:
        result["allBase"] = _arm(all_base_reading, {"size": len(plan["allBaseTargets"])})
    return result


# ---------------------------------------------------------------------------
# Runner (needs the model)
# ---------------------------------------------------------------------------


def _load_report(path: str | None, layer: int) -> tuple[dict | None, dict | None]:
    if not path:
        return None, None
    report = json.loads(Path(path).read_text())
    layer_result = next(
        (result for result in report.get("layers", []) if result.get("layer") == layer), None
    )
    return report, layer_result


def _active_from_report(report: dict, layer_result: dict) -> dict[str, list[tuple[int, float]]]:
    active_by_problem = {}
    for prompt, rows in zip(report["prompts"], layer_result["active_features"], strict=True):
        if prompt.get("kind", "base") != "base":
            continue
        active_by_problem[prompt["problem"]] = [
            (int(row["index"]), float(row["activation"])) for row in rows
        ]
    return active_by_problem


def _active_by_step_from_report(
    report: dict, layer_result: dict
) -> dict[str, list[tuple[int, float]]]:
    return {
        prompt["step"]: [(int(row["index"]), float(row["activation"])) for row in rows]
        for prompt, rows in zip(report["prompts"], layer_result["active_features"], strict=True)
    }


def _labels_from_report(layer_result: dict) -> dict[str, str]:
    labels = {}
    for rows in layer_result.get("active_features", []):
        for row in rows:
            labels[str(row["index"])] = row.get("label", "")
    return labels


def _run_forward(engine, prompt: str) -> torch.Tensor:
    inputs = engine.tokenizer(prompt, return_tensors="pt").to(engine._input_device)
    with torch.inference_mode():
        try:
            outputs = engine.model(**inputs, logits_to_keep=1)
        except TypeError:
            outputs = engine.model(**inputs)
    return outputs.logits[0, -1, :].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-local-dir", default="output")
    parser.add_argument("--layer", type=int, default=demo._SAE_LAYER)
    parser.add_argument("--threshold-offset", type=float, default=demo._HF_THRESHOLD_OFFSET)
    parser.add_argument("--vllm-report", default=None)
    parser.add_argument("--vllm-activations", default=None)
    parser.add_argument("--results-dir", default="demo/home_repair/output/steering")
    parser.add_argument("--experiments", default=None, help="JSON file overriding experiments")
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument(
        "--hazard-experiments",
        action="store_true",
        help="Also run the theme-based hazard ablation/clamp checks.",
    )
    parser.add_argument(
        "--amplify",
        type=float,
        default=3.0,
        help="Extra clamp arm at this multiple of the target activation (1 disables).",
    )
    parser.add_argument(
        "--no-attribution",
        action="store_true",
        help="Skip the per-row ablation of the page's snapshot features.",
    )
    parser.add_argument(
        "--no-injection",
        action="store_true",
        help="Skip clamping the snapshot families into the open (no-ask) prompt.",
    )
    parser.add_argument("--control-draws", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    layer = args.layer
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    report, layer_result = _load_report(args.vllm_report, layer)
    contrastive_map = demo.filter_contrastive_map(
        demo._load_contrastive_feature_map(args.sae_local_dir, layer)
    )
    checkpoint = Path(args.sae_local_dir) / f"layer_{layer}" / "sae_checkpoints" / "sae_final.pt"
    sae_device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(str(checkpoint), device=sae_device)
    sae.eval()
    with torch.no_grad():
        sae.threshold.add_(args.threshold_offset)
    labels = _labels_from_report(layer_result) if layer_result else None
    _, feature_descs = demo._load_sae_local(args.sae_local_dir, layer, device="cpu")
    if feature_descs:
        labels = {**(labels or {}), **dict(feature_descs)}

    engine = demo.HFEngine(
        model_name=args.model_name,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=1,
        allow_thinking=False,
    )
    prompts = [
        item
        for item in demo.decision_prompts(include_contrasts=True, include_probes=True)
        if item["kind"] in ("base", "contrast", "open")
    ]
    for item in prompts:
        engine.record_tool_decision(item["step"], item["request"])
    prompt_text = dict(engine.prompt_log)
    tool_to_token = demo.tool_first_token_ids(engine.tokenizer, demo._DECISION_TOOLS)

    # HF residuals at the SAE layer: parity + fallback feature source.
    activation_log = engine.extract_all_prompts([layer])
    hf_residuals = {step: acts[f"residual_{layer}"] for step, acts in activation_log}
    vllm_residuals = None
    if args.vllm_activations:
        with np.load(args.vllm_activations) as saved:
            rows = saved[f"residual_{layer}"].astype(np.float32)
        if report and len(report.get("prompts", [])) == len(rows):
            vllm_residuals = {prompt["step"]: rows[i] for i, prompt in enumerate(report["prompts"])}
        else:
            vllm_residuals = {
                item["step"]: rows[i] for i, item in enumerate(prompts) if i < len(rows)
            }

    parity: dict[str, dict] = {}
    for item in prompts:
        entry: dict = {}
        if vllm_residuals and item["step"] in vllm_residuals:
            entry.update(
                {
                    key: value
                    for key, value in backend_compare._vector_parity(
                        hf_residuals[item["step"]][None, :], vllm_residuals[item["step"]][None, :]
                    ).items()
                    if not key.startswith("per_")
                }
            )
        if entry:
            entry["cosine"] = entry.get("mean_cosine_similarity")
            entry["relativeL2"] = entry.get("mean_relative_l2_error")
        parity[item["step"]] = entry

    # Features are selected from what the HF forward pass actually represents
    # (the backend that will be intervened on); the vLLM activations of the
    # same features are attached for reference.  Selecting from vLLM alone can
    # pick features that are silent under HF, which makes an ablation vacuous.
    hf_active_by_problem: dict[str, list[tuple[int, float]]] = {}
    hf_contrast_by_problem: dict[str, list[tuple[int, float]]] = {}
    hf_open_by_problem: dict[str, list[tuple[int, float]]] = {}
    dtype = next(sae.parameters()).dtype
    for item in prompts:
        vec = torch.from_numpy(hf_residuals[item["step"]]).to(device=sae_device, dtype=dtype)
        with torch.no_grad():
            feats = sae.encode(sae.normalize_input(vec.unsqueeze(0)))[0].float().cpu().numpy()
        idx = np.flatnonzero(feats > 0)
        active = [(int(i), float(feats[i])) for i in idx]
        if item["kind"] == "contrast":
            hf_contrast_by_problem[item["problem"]] = active
        elif item["kind"] == "open":
            hf_open_by_problem[item["problem"]] = active
        else:
            hf_active_by_problem[item["problem"]] = active
    reference_activations = None
    if layer_result and layer_result.get("active_features"):
        reference_activations = {
            pid: dict(active) for pid, active in _active_from_report(report, layer_result).items()
        }
    feature_source = "hf_activations" + (" + vllm_reference" if reference_activations else "")

    if args.experiments:
        experiments = json.loads(Path(args.experiments).read_text())
    else:
        experiments = contrast_experiments(
            hf_active_by_problem,
            hf_contrast_by_problem,
            labels,
            args.n_features,
            reference_activations,
        )
        if args.hazard_experiments:
            experiments.extend(
                hazard_experiments(
                    hf_active_by_problem, contrastive_map, labels, 3, reference_activations
                )
            )
    if not experiments:
        print("No steering experiments could be derived.")

    layers = engine._get_model_layers()
    target_layer = layers[layer]
    step_by_problem = {item["problem"]: item["step"] for item in prompts if item["kind"] == "base"}

    def _read(prompt: str, hook=None) -> dict:
        handle = target_layer.register_forward_pre_hook(hook, with_kwargs=True) if hook else None
        try:
            return distribution_from_logits(_run_forward(engine, prompt), tool_to_token)
        finally:
            if handle is not None:
                handle.remove()

    baselines: dict[str, dict] = {}
    for item in prompts:
        baselines[item["step"]] = _read(prompt_text[item["step"]])
        vllm_choice = None
        for decision in (report or {}).get("decisions") or []:
            if decision.get("step") == item["step"]:
                vllm_choice = decision
        parity[item["step"]]["baselineDistributionHf"] = baselines[item["step"]]["distribution"]
        if vllm_choice:
            parity[item["step"]]["baselineDistributionVllm"] = vllm_choice.get("distribution")
        print(
            f"  Baseline {item['step']}: {baselines[item['step']]['display']} "
            f"(p={baselines[item['step']]['prob']:.2f})"
        )

    results = []
    for experiment in experiments:
        step = experiment.get("step") or step_by_problem[experiment["problem"]]
        prompt = prompt_text[step]
        edits = {
            int(row["index"]): (None if experiment["mode"] == "ablate" else float(row["target"]))
            for row in experiment["features"]
        }
        record: dict = {}
        arms = {"baseline": baselines[step]}
        arms["reconstructionOnly"] = _read(prompt, make_feature_edit_hook(sae, {}, mode="replace"))
        arms["delta"] = _read(
            prompt, make_feature_edit_hook(sae, edits, mode="delta", record=record)
        )
        arms["replace"] = _read(prompt, make_feature_edit_hook(sae, edits, mode="replace"))
        if experiment["mode"] == "clamp" and args.amplify > 1:
            amplified = {idx: (None if v is None else v * args.amplify) for idx, v in edits.items()}
            arms["amplified"] = _read(prompt, make_feature_edit_hook(sae, amplified, mode="delta"))
        base_dist = arms["baseline"]["distribution"]
        result = {
            **experiment,
            "step": step,
            "features": [
                {**row, "hfActivation": record.get("pre_edit", {}).get(int(row["index"]))}
                for row in experiment["features"]
            ],
            "arms": arms,
            "baseline": base_dist,
            "intervened": arms["delta"]["distribution"],
            "reconstructionOnly": arms["reconstructionOnly"]["distribution"],
            "replaceMode": arms["replace"]["distribution"],
            "deltas": distribution_deltas(base_dist, arms["delta"]["distribution"]),
            "replaceDeltas": distribution_deltas(base_dist, arms["replace"]["distribution"]),
            **(
                {
                    "amplified": arms["amplified"]["distribution"],
                    "amplifiedDeltas": distribution_deltas(
                        base_dist, arms["amplified"]["distribution"]
                    ),
                    "amplifyFactor": args.amplify,
                }
                if "amplified" in arms
                else {}
            ),
            "backend": "hf",
            "parity": parity.get(step, {}),
        }
        results.append(result)
        print(f"\n  Experiment {experiment['id']} ({experiment['mode']}, {step}):")
        for arm, reading in arms.items():
            print(f"    {arm:>18}: {reading['distribution']}")

    # Per-row attribution: ablate each snapshot row's feature family on the
    # base prompt (vLLM-selected rows, exactly as the page shows them) and
    # compare with mass-matched random ablations of other active features.
    attribution: dict[str, dict] = {}
    if not args.no_attribution and layer_result and layer_result.get("active_features"):
        active_by_step = _active_by_step_from_report(report, layer_result)
        # Labels from the report only, so dedupe/merge reproduces the page's rows.
        rows_by_problem, _, _ = demo.snapshot_feature_rows(
            active_by_step, _labels_from_report(layer_result)
        )
        for item in prompts:
            if item["kind"] != "base":
                continue
            pid, step = item["problem"], item["step"]
            rows = rows_by_problem.get(pid) or []
            if not rows:
                continue
            plan = attribution_plan(
                rows, hf_active_by_problem.get(pid, []), draws=args.control_draws, seed=0
            )
            target_tool = demo.tool_display(item["toolId"])
            for decision in report.get("decisions") or []:
                if decision.get("step") == step and decision.get("display"):
                    target_tool = decision["display"]
            prompt = prompt_text[step]
            row_readings = [
                _read(prompt, make_feature_edit_hook(sae, dict.fromkeys(r["family"]), "delta"))
                for r in plan["rows"]
            ]
            all_reading = _read(
                prompt, make_feature_edit_hook(sae, dict.fromkeys(plan["allFeatures"]), "delta")
            )
            control_readings = [
                _read(prompt, make_feature_edit_hook(sae, dict.fromkeys(c["features"]), "delta"))
                for c in plan["controls"]
            ]
            attribution[pid] = {
                "step": step,
                **summarize_attribution(
                    plan, baselines[step], row_readings, all_reading, control_readings, target_tool
                ),
                "backend": "hf",
                "parity": parity.get(step, {}),
            }
            print(f"\n  Attribution {pid} (target {target_tool}):")
            for row in attribution[pid]["rows"]:
                print(
                    f"    {row['index']:>6} {row['deltaTarget']:+.3f} "
                    f"(hf act {row['hfActivation']:.2f}, {row['familySize']} feats) {row['label']}"
                )
            print(f"    all rows: {attribution[pid]['allRows']['deltaTarget']:+.3f}")
            print(
                f"    control max |delta| {attribution[pid]['controlThreshold']:.3f} "
                f"(mean {attribution[pid]['controlMeanAbsDelta']:.3f})"
            )

    # Injection: put the explicit-ask request's snapshot families back into the
    # open (no-ask) prompt and see whether the HF decision moves toward the
    # tool the ask produced.  Targets are the HF activations on the base
    # prompt (the backend being intervened on); matched random controls clamp
    # other base-active features to their base values.
    injection: dict[str, dict] = {}
    if not args.no_injection and layer_result and layer_result.get("active_features"):
        active_by_step = _active_by_step_from_report(report, layer_result)
        rows_by_problem, _, _ = demo.snapshot_feature_rows(
            active_by_step, _labels_from_report(layer_result)
        )
        open_steps = {item["problem"]: item["step"] for item in prompts if item["kind"] == "open"}
        for item in prompts:
            if item["kind"] != "base" or item["problem"] not in open_steps:
                continue
            pid, base_step = item["problem"], item["step"]
            open_step = open_steps[pid]
            rows = rows_by_problem.get(pid) or []
            if not rows:
                continue
            plan = injection_plan(
                rows,
                hf_active_by_problem.get(pid, []),
                hf_open_by_problem.get(pid, []),
                draws=args.control_draws,
                seed=0,
            )
            target_tool = demo.tool_display(item["toolId"])
            for decision in report.get("decisions") or []:
                if decision.get("step") == base_step and decision.get("display"):
                    target_tool = decision["display"]
            prompt = prompt_text[open_step]

            def _clamp(targets: dict, prompt: str = prompt) -> dict:
                edits = {int(k): float(v) for k, v in targets.items()}
                return _read(prompt, make_feature_edit_hook(sae, edits, "delta"))

            row_readings = [_clamp(r["targets"]) for r in plan["rows"]]
            all_rows_reading = _clamp(plan["allRowsTargets"])
            all_base_reading = _clamp(plan["allBaseTargets"])
            control_readings = [_clamp(c["targets"]) for c in plan["controls"]]
            injection[pid] = {
                "step": open_step,
                "baseStep": base_step,
                **summarize_injection(
                    plan,
                    baselines[open_step],
                    row_readings,
                    all_rows_reading,
                    all_base_reading,
                    control_readings,
                    target_tool,
                ),
                "baseHfChoice": baselines[base_step].get("display"),
                "backend": "hf",
                "parity": parity.get(open_step, {}),
            }
            print(f"\n  Injection {pid} into {open_step} (target {target_tool}):")
            for row in injection[pid]["rows"]:
                print(
                    f"    {row['index']:>6} {row['deltaTarget']:+.3f} -> {row['choice']} "
                    f"({len(row['targets'])} feats, mass {row['baseMass']:.1f}) {row['label']}"
                )
            for key in ("allRows", "allBase"):
                arm = injection[pid][key]
                print(
                    f"    {key}: {arm['deltaTarget']:+.3f} -> {arm['choice']} ({arm['size']} feats)"
                )
            print(
                f"    control max |delta| {injection[pid]['controlThreshold']:.3f} "
                f"(mean {injection[pid]['controlMeanAbsDelta']:.3f})"
            )

    engine.cleanup()

    output = {
        "backend": "hf",
        "hfFastPath": engine.fast_path,
        "model": args.model_name,
        "layer": layer,
        "thresholdOffset": args.threshold_offset,
        "saeCheckpoint": str(checkpoint),
        "featureSource": feature_source,
        "vllmReport": args.vllm_report,
        "parity": parity,
        "experiments": results,
        "attribution": attribution,
        "injection": injection,
    }
    out_path = results_dir / "steering_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
