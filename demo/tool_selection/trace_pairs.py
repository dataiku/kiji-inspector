#!/usr/bin/env python3
"""Where the cue lives, how much of it is needed, and what the model then writes.

Three HuggingFace-backend checks (fused Mamba kernels, same container) that go
beyond the decision-token attribution of ``attribute_pairs.py``:

* **cue map** — the SAE activation of each side's cue families at *every*
  token of the prompt (request span and assistant turn), and the tool
  distribution after switching the families off at the decision token only,
  at the request tokens only, everywhere but the decision token, and
  everywhere — at the page layer and at a comparison layer (default 27);
  mass-matched random ablations give the band for the decision-token and
  everywhere conditions;
* **dose-response** — the other side's cue families clamped into this request
  at 0×, ½×, 1×, 1½×, 2×, 3× of their activation on the other side (all
  families, and the single family with the largest 1× effect), with matched
  random sets at the same scales;
* **generation** — greedy continuations of the request with the other side's
  cue families clamped at every position on every step, next to the
  unsteered continuation and a matched-random-set continuation.

Usage (inside 575lab/kiji-inspector:dev, ``pip install "kernels>=0.15.2,<0.16"``):
    python demo/tool_selection/trace_pairs.py --model-name /models/... --layer 43
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tool_selection_demo as demo
import torch

sys.path.insert(0, str(demo._DEMO_DIR.parent / "home_repair"))
import steer_tool_choice as steer  # noqa: E402

from kiji_inspector.core.sae_core import JumpReLUSAE  # noqa: E402

CONDITIONS = ("decision", "request", "allButDecision", "all")
DEFAULT_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PROFILE_FAMILIES = 6


# --------------------------------------------------------------------------- pure helpers


def request_span(offsets: list[tuple[int, int]], prompt: str, request: str) -> list[int]:
    """Token positions whose character span overlaps the request text."""
    start = prompt.index(request)
    end = start + len(request)
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


def condition_positions(condition: str, n_tokens: int, request_positions: list[int]) -> list[int]:
    """Absolute token positions edited under each ablation condition."""
    decision = n_tokens - 1
    if condition == "decision":
        return [decision]
    if condition == "request":
        return [p for p in request_positions if p != decision]
    if condition == "allButDecision":
        return list(range(decision))
    if condition == "all":
        return list(range(n_tokens))
    raise ValueError(f"Unknown condition {condition!r}")


def family_profile(
    features: np.ndarray, families: list[dict], top: int = PROFILE_FAMILIES
) -> list[dict]:
    """Per-token activation (summed over the family) of the first ``top`` families.

    ``features`` is the (tokens, d_sae) SAE activation matrix of one prompt;
    ``families`` are page rows (``index``, ``label``, ``merged``).
    """
    out = []
    for row in families[:top]:
        idxs = [int(row["index"])] + [int(i) for i in row.get("merged") or []]
        per_token = features[:, idxs].sum(axis=1)
        out.append(
            {
                "index": int(row["index"]),
                "label": row["label"],
                "familySize": len(idxs),
                "perToken": [round(float(v), 2) for v in per_token],
            }
        )
    return out


def scaled_targets(targets: dict, scale: float) -> dict[int, float]:
    return {int(k): float(v) * float(scale) for k, v in targets.items()}


def dose_summary(
    baseline: dict,
    scales: list[float],
    all_readings: list[dict],
    best_row: dict | None,
    best_readings: list[dict] | None,
    control_readings: list[list[dict]],
    target_tool: str,
) -> dict:
    """Curves of p(target) against clamp scale, with the random band per scale."""
    base_p = float(baseline["distribution"].get(target_tool, 0.0))

    def _curve(readings: list[dict]) -> list[dict]:
        return [
            {
                "scale": s,
                "p": round(float(r["distribution"].get(target_tool, 0.0)), 4),
                "choice": r["display"],
            }
            for s, r in zip(scales, readings, strict=True)
        ]

    bands = []
    for i, _ in enumerate(scales):
        deltas = [
            abs(float(c[i]["distribution"].get(target_tool, 0.0)) - base_p)
            for c in control_readings
            if len(c) > i
        ]
        bands.append(round(max(deltas), 4) if deltas else None)
    out = {
        "targetTool": target_tool,
        "baselineP": round(base_p, 4),
        "baselineChoice": baseline["display"],
        "scales": list(scales),
        "allRows": _curve(all_readings),
        "controlBand": bands,
    }
    if best_row is not None and best_readings is not None:
        out["bestRow"] = {
            "index": int(best_row["index"]),
            "label": best_row["label"],
            "familySize": best_row.get("familySize"),
            "curve": _curve(best_readings),
        }
    return out


# --------------------------------------------------------------------------- HF plumbing


def make_position_hook(sae, edits: dict[int, float | None], positions: list[int] | None):
    """Delta-patch ``edits`` (index -> target, ``None`` = ablate) at ``positions``.

    ``positions`` are absolute token indices; ``None`` edits every position of
    whatever the layer receives — during generation that is the whole prompt
    on the prefill step and the one new token on each decode step.
    """
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    scale = steer._sae_scale(sae)

    def hook(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        n_tokens = hidden.shape[1]
        pos = list(range(n_tokens)) if positions is None else [p for p in positions if p < n_tokens]
        if not pos:
            return None
        flat = hidden[0, pos, :].to(device=sae_device, dtype=sae_dtype)
        with torch.no_grad():
            feats = sae.encode(sae.normalize_input(flat)).float()
            modified = flat.clone().float()
            for idx, target in edits.items():
                current = feats[:, idx]
                desired = torch.zeros_like(current) if target is None else current * 0 + target
                modified = modified + scale * (desired - current).unsqueeze(1) * sae.W_dec[
                    idx
                ].float().unsqueeze(0)
        new_hidden = hidden.clone()
        new_hidden[0, pos, :] = modified.to(device=hidden.device, dtype=hidden.dtype)
        if args:
            return (new_hidden,) + tuple(args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = new_hidden
        return args, new_kwargs

    return hook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-local-dir", default="output")
    parser.add_argument("--layer", type=int, default=43, help="page layer")
    parser.add_argument("--compare-layers", default="27")
    parser.add_argument("--threshold-offset", type=float, default=demo.hr._HF_THRESHOLD_OFFSET)
    parser.add_argument(
        "--report", default=str(demo._DEMO_DIR / "output" / "capture" / "evaluation.json")
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    parser.add_argument("--gen-tokens", type=int, default=48)
    parser.add_argument("--control-draws", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    layer = args.layer
    compare_layers = [int(x) for x in args.compare_layers.split(",") if x.strip()]
    all_layers = [layer] + [x for x in compare_layers if x != layer]
    scales = [float(x) for x in args.scales.split(",")]
    results_dir = Path(args.results_dir or demo._DEMO_DIR / "output" / f"trace_layer{layer}")
    results_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.report).read_text())
    demo.check_report_prompts(report)
    decisions = {d["step"]: d for d in report.get("decisions") or []}

    saes = {}
    for lyr in all_layers:
        checkpoint = Path(args.sae_local_dir) / f"layer_{lyr}" / "sae_checkpoints" / "sae_final.pt"
        sae = JumpReLUSAE.from_pretrained(str(checkpoint), device="cuda")
        sae.eval()
        with torch.no_grad():
            sae.threshold.add_(args.threshold_offset)
        saes[lyr] = sae
    # Page rows per layer (same families the page shows for that layer).
    pair_rows = {
        lyr: {p["id"]: p for p in demo.build_ui_data(report, None, layer=lyr)["pairs"]}
        for lyr in all_layers
    }

    demo.use_scenario_in_home_repair_module()
    engine = demo.hr.HFEngine(
        model_name=args.model_name,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=args.gen_tokens,
        allow_thinking=False,
    )
    if engine.fast_path is False:
        print("  WARNING: naive Mamba path — these results will drift from vLLM")
    prompts = demo.decision_prompts()
    for item in prompts:
        engine.record_tool_decision(item["step"], item["request"])
    prompt_text = dict(engine.prompt_log)
    tree = demo.tool_token_tree(engine.tokenizer)
    tok = engine.tokenizer
    model_layers = engine._get_model_layers()

    def _forward(prompt: str, hooks: list[tuple[int, object]]) -> torch.Tensor:
        handles = [
            model_layers[lyr].register_forward_pre_hook(h, with_kwargs=True) for lyr, h in hooks
        ]
        try:
            return steer._run_forward(engine, prompt)
        finally:
            for h in handles:
                h.remove()

    def _read(prompt: str, hooks: list[tuple[int, object]] | None = None) -> dict:
        hooks = hooks or []
        logp = torch.log_softmax(_forward(prompt, hooks).float(), dim=-1)
        first_lp = {int(t): float(logp[t]) for t in tree["first"]}
        second_lp: dict[int, dict[int, float]] = {}
        for token in tree["shared"]:
            if math.exp(first_lp[token]) < 0.005:
                continue
            piece = tok.decode([token])
            logp2 = torch.log_softmax(_forward(prompt + piece, hooks).float(), dim=-1)
            second_lp[token] = {
                int(t2): float(logp2[t2]) for t2 in demo.second_token_ids(tree, token)
            }
        return demo.distribution_from_tree(first_lp, second_lp, tree)

    def _residuals(prompt: str) -> dict[int, torch.Tensor]:
        store: dict[int, torch.Tensor] = {}

        def _capture(lyr: int):
            def hook(module, a, kw):
                store[lyr] = (a[0] if a else kw["hidden_states"])[0].detach()

            return hook

        _forward(prompt, [(lyr, _capture(lyr)) for lyr in all_layers])
        return store

    def _features(x: torch.Tensor, lyr: int) -> np.ndarray:
        sae = saes[lyr]
        with torch.no_grad():
            f = sae.encode(sae.normalize_input(x.to(dtype=next(sae.parameters()).dtype)))
        return f.float().cpu().numpy()

    def _generate(prompt: str, hooks: list[tuple[int, object]]) -> str:
        handles = [
            model_layers[lyr].register_forward_pre_hook(h, with_kwargs=True) for lyr, h in hooks
        ]
        try:
            inputs = tok(prompt, return_tensors="pt").to(engine._input_device)
            with torch.inference_mode():
                out = engine.model.generate(
                    **inputs, max_new_tokens=args.gen_tokens, do_sample=False
                )
            return tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()

    baselines = {item["step"]: _read(prompt_text[item["step"]]) for item in prompts}
    positions_out: dict[str, dict] = {}
    dose_out: dict[str, dict] = {}
    gen_out: dict[str, dict] = {}
    # Decision-token feature vectors per layer, for mass-matched controls.
    decision_feats: dict[str, dict[int, np.ndarray]] = {}

    for pair in demo.PAIRS:
        pid = pair["id"]
        steps = {side: f"{pid}_{side.upper()}" for side in demo.SIDES}
        feats_by_step: dict[str, dict[int, np.ndarray]] = {}
        for side in demo.SIDES:
            step = steps[side]
            prompt = prompt_text[step]
            enc = tok(prompt, return_offsets_mapping=True)
            ids = enc["input_ids"]
            n_tokens = len(ids)
            req_pos = request_span(enc["offset_mapping"], prompt, pair[side]["request"])
            res = _residuals(prompt)
            feats_by_step[step] = {lyr: _features(res[lyr], lyr) for lyr in all_layers}
            decision_feats[step] = {lyr: feats_by_step[step][lyr][-1] for lyr in all_layers}
            target_tool = (decisions.get(step) or {}).get("display") or baselines[step]["display"]
            entry = {
                "step": step,
                "tokens": [tok.decode([t]) for t in ids],
                "requestSpan": [req_pos[0], req_pos[-1]] if req_pos else None,
                "targetTool": target_tool,
                "hfChoice": baselines[step]["display"],
                "baseline": baselines[step]["distribution"],
                "layers": {},
            }
            print(f"\n=== {step}: HF {baselines[step]['display']} {baselines[step]['prob']:.2f}")
            for lyr in all_layers:
                rows = pair_rows[lyr][pid][side]["features"]
                feats = feats_by_step[step][lyr]
                family_feats = [
                    int(i) for r in rows for i in [r["index"]] + list(r.get("merged") or [])
                ]
                block: dict = {"numFamilies": len(rows), "numFeatures": len(family_feats)}
                if lyr == layer:
                    block["profile"] = family_profile(feats, rows)
                if not family_feats:
                    entry["layers"][str(lyr)] = block
                    continue
                edits = dict.fromkeys(family_feats)
                active = [(int(i), float(v)) for i, v in enumerate(feats[-1]) if v > 0]
                plan = steer.attribution_plan(rows, active, draws=args.control_draws, seed=0)
                ablate: dict = {}
                for cond in CONDITIONS:
                    pos = condition_positions(cond, n_tokens, req_pos)
                    reading = _read(prompt, [(lyr, make_position_hook(saes[lyr], edits, pos))])
                    item = {
                        "positions": len(pos),
                        "p": round(float(reading["distribution"].get(target_tool, 0.0)), 4),
                        "choice": reading["display"],
                        "distribution": reading["distribution"],
                    }
                    if cond in ("decision", "all"):
                        deltas = []
                        for c in plan["controls"]:
                            r = _read(
                                prompt,
                                [
                                    (
                                        lyr,
                                        make_position_hook(
                                            saes[lyr], dict.fromkeys(c["features"]), pos
                                        ),
                                    )
                                ],
                            )
                            deltas.append(
                                abs(
                                    float(r["distribution"].get(target_tool, 0.0))
                                    - float(baselines[step]["distribution"].get(target_tool, 0))
                                )
                            )
                        item["controlBand"] = round(max(deltas), 4) if deltas else None
                    ablate[cond] = item
                    print(
                        f"  L{lyr} off {len(family_feats):>2} feats @{cond:<15} "
                        f"p({target_tool})={item['p']:.2f} -> {item['choice']}"
                        + (f"  band {item['controlBand']:.3f}" if "controlBand" in item else "")
                    )
                block["ablate"] = ablate
                entry["layers"][str(lyr)] = block
            positions_out[step] = entry

        # Dose-response and generation: each side's families into the other request.
        dose_out[pid] = {}
        gen_out[pid] = {}
        for side in demo.SIDES:
            other = "b" if side == "a" else "a"
            step, other_step = steps[side], steps[other]
            rows = pair_rows[layer][pid][side]["features"]
            if not rows:
                continue
            target_tool = (decisions.get(step) or {}).get("display") or baselines[step]["display"]
            src_active = [
                (int(i), float(v)) for i, v in enumerate(decision_feats[step][layer]) if v > 0
            ]
            dst_active = [
                (int(i), float(v)) for i, v in enumerate(decision_feats[other_step][layer]) if v > 0
            ]
            plan = steer.injection_plan(
                rows, src_active, dst_active, draws=args.control_draws, seed=0
            )
            other_prompt = prompt_text[other_step]
            n_other = len(tok(other_prompt)["input_ids"])
            sae = saes[layer]

            def _clamp(
                targets: dict,
                scale: float,
                prompt: str = other_prompt,
                n: int = n_other,
                sae_l=sae,
            ):
                return _read(
                    prompt,
                    [(layer, make_position_hook(sae_l, scaled_targets(targets, scale), [n - 1]))],
                )

            one_x = [_clamp(r["targets"], 1.0) for r in plan["rows"]]
            best_i = None
            if one_x:
                best_i = max(
                    range(len(one_x)),
                    key=lambda i: float(one_x[i]["distribution"].get(target_tool, 0.0)),
                )
            all_curve = [_clamp(plan["allRowsTargets"], s) for s in scales]
            best_curve = (
                [_clamp(plan["rows"][best_i]["targets"], s) for s in scales]
                if best_i is not None
                else None
            )
            control_curves = [[_clamp(c["targets"], s) for s in scales] for c in plan["controls"]]
            dose = dose_summary(
                baselines[other_step],
                scales,
                all_curve,
                plan["rows"][best_i] if best_i is not None else None,
                best_curve,
                control_curves,
                target_tool,
            )
            dose["fromSide"], dose["intoSide"], dose["intoStep"] = side, other, other_step
            dose["numFeatures"] = len(plan["allRowsTargets"])
            dose_out[pid][f"{side}_into_{other}"] = dose
            print(
                f"  Dose {side}->{other} p({target_tool}) on {other_step}: "
                + " ".join(f"{c['scale']}x={c['p']:.2f}" for c in dose["allRows"])
            )

            # Generation: unsteered, steered (all positions, every step), matched random set.
            full_targets = {int(k): float(v) for k, v in plan["allRowsTargets"].items()}
            baseline_text = _generate(other_prompt, [])
            steered_text = _generate(
                other_prompt, [(layer, make_position_hook(sae, full_targets, None))]
            )
            control_text = None
            if plan["controls"]:
                ctrl_targets = {int(k): float(v) for k, v in plan["controls"][0]["targets"].items()}
                control_text = _generate(
                    other_prompt, [(layer, make_position_hook(sae, ctrl_targets, None))]
                )
            gen_out[pid][f"{side}_into_{other}"] = {
                "fromSide": side,
                "intoSide": other,
                "intoStep": other_step,
                "targetTool": target_tool,
                "numFeatures": len(full_targets),
                "baseline": baseline_text,
                "steered": steered_text,
                "control": control_text,
                "controlSize": len(plan["controls"][0]["targets"]) if plan["controls"] else 0,
            }
            print(f"  GEN {other_step} baseline: {baseline_text!r}")
            print(f"  GEN {other_step} steered : {steered_text!r}")
            print(f"  GEN {other_step} control : {control_text!r}")

    engine.cleanup()
    output = {
        "backend": "hf",
        "hfFastPath": engine.fast_path,
        "model": args.model_name,
        "layer": layer,
        "compareLayers": compare_layers,
        "thresholdOffset": args.threshold_offset,
        "scales": scales,
        "genTokens": args.gen_tokens,
        "controlDraws": args.control_draws,
        "report": args.report,
        "baselines": {k: v["distribution"] for k, v in baselines.items()},
        "positions": positions_out,
        "dose": dose_out,
        "generations": gen_out,
    }
    out_path = results_dir / "trace_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
