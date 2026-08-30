#!/usr/bin/env python3
"""Causal check for the tool-selection pairs (HuggingFace backend, same container).

For each side of each pair:

* **ablation** — switch off each of that side's cue families (the features
  stronger on this side than on the other, as shown on the page) at the
  decision token and re-read the tool distribution; compare with
  mass-matched random ablations of other active features -- matched to the
  single family, and, for the all-families arm, a second set of draws matched
  to the whole cue set's count and mass; also all families together;
* **cross-patch** — clamp this side's cue families into the *other* side's
  prompt at their activations here, and see whether the other side's decision
  moves toward this side's tool.  Three control families: matched to one cue
  family's donor mass, to the whole cue set's donor mass, and to the change
  the clamp actually makes on the recipient (donor mass overstates that
  whenever a feature is already active there).

Readout is the same token-prefix tree as the vLLM capture (``file_read`` /
``file_write`` share ``" file"``: a second forward with that token appended
reads the conditional, with the edit applied at the decision token, position
``-2``).  vLLM/HF parity (residual cosine, baseline distributions) is recorded.

Usage (inside 575lab/kiji-inspector:dev):
    python demo/tool_selection/attribute_pairs.py --model-name /models/... \
        --report demo/tool_selection/output/capture/evaluation.json \
        [--activations demo/tool_selection/output/capture/activations.npz]
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
import compare_sae_backends as backend_compare  # noqa: E402
import steer_tool_choice as steer  # noqa: E402

from kiji_inspector.core.sae_core import JumpReLUSAE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-local-dir", default="output")
    parser.add_argument("--layer", type=int, default=demo.SAE_LAYER)
    parser.add_argument("--threshold-offset", type=float, default=demo.hr._HF_THRESHOLD_OFFSET)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario to run (default: tool_selection); reads demo/<scenario>/.",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--activations", default=None)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Default: demo/tool_selection/output/steering_layer<L>",
    )
    parser.add_argument("--control-draws", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--active-from-sae",
        action="store_true",
        help=(
            "Recompute each prompt's active features by encoding the --activations "
            "residuals with the loaded SAE, instead of taking them from the report. "
            "Required when --sae-local-dir is a different dictionary than the one the "
            "report was captured with (feature indices are dictionary-specific); "
            "labels then fall back to bare feature indices."
        ),
    )
    args = parser.parse_args()
    if args.active_from_sae and not args.activations:
        raise SystemExit("--active-from-sae requires --activations")
    if args.scenario:
        demo.configure(args.scenario)

    layer = args.layer
    results_dir = Path(args.results_dir or demo.DEMO_DIR / "output" / f"steering_layer{layer}")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or demo.DEMO_DIR / "output" / "capture" / "evaluation.json"
    report = json.loads(Path(report_path).read_text())
    demo.check_report_prompts(report)
    labels = demo._labels_from_report(report, layer)
    active_by_step = demo._active_by_step(report, layer)
    decisions = {d["step"]: d for d in report.get("decisions") or []}

    checkpoint = Path(args.sae_local_dir) / f"layer_{layer}" / "sae_checkpoints" / "sae_final.pt"
    sae_device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(str(checkpoint), device=sae_device)
    sae.eval()
    with torch.no_grad():
        sae.threshold.add_(args.threshold_offset)

    demo.use_scenario_in_home_repair_module()
    engine = demo.hr.HFEngine(
        model_name=args.model_name,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=1,
        allow_thinking=False,
    )
    prompts = demo.decision_prompts()
    for item in prompts:
        engine.record_tool_decision(item["step"], item["request"])
    prompt_text = dict(engine.prompt_log)
    tree = demo.tool_token_tree(engine.tokenizer)
    shared_tokens = list(tree["shared"])

    # HF residuals at the SAE layer: parity + the feature values to intervene on.
    activation_log = engine.extract_all_prompts([layer])
    hf_residuals = {step: acts[f"residual_{layer}"] for step, acts in activation_log}
    vllm_residuals = None
    if args.activations:
        with np.load(args.activations) as saved:
            rows = saved[f"residual_{layer}"].astype(np.float32)
        vllm_residuals = {p["step"]: rows[i] for i, p in enumerate(report["prompts"])}
    parity: dict[str, dict] = {}
    hf_active: dict[str, list[tuple[int, float]]] = {}
    dtype = next(sae.parameters()).dtype
    if args.active_from_sae:
        # The report's stored features belong to the dictionary used at capture
        # time; re-derive them from the canonical vLLM residuals with THIS SAE.
        # The report convention encodes vLLM residuals with offset-free
        # thresholds (the offset is an HF-side correction), so lift it here.
        labels = {}
        active_by_step = {}
        with torch.no_grad():
            sae.threshold.sub_(args.threshold_offset)
        try:
            for step, vec in vllm_residuals.items():
                tensor = torch.from_numpy(vec).to(device=sae_device, dtype=dtype)
                with torch.no_grad():
                    feats = sae.encode(sae.normalize_input(tensor.unsqueeze(0)))[0]
                feats = feats.float().cpu().numpy()
                idx = np.flatnonzero(feats > 0)
                active_by_step[step] = [(int(i), float(feats[i])) for i in idx]
        finally:
            with torch.no_grad():
                sae.threshold.add_(args.threshold_offset)
    for item in prompts:
        step = item["step"]
        entry: dict = {}
        if vllm_residuals and step in vllm_residuals:
            entry = {
                k: v
                for k, v in backend_compare._vector_parity(
                    hf_residuals[step][None, :], vllm_residuals[step][None, :]
                ).items()
                if not k.startswith("per_")
            }
            entry["cosine"] = entry.get("mean_cosine_similarity")
            entry["relativeL2"] = entry.get("mean_relative_l2_error")
        parity[step] = entry
        vec = torch.from_numpy(hf_residuals[step]).to(device=sae_device, dtype=dtype)
        with torch.no_grad():
            feats = sae.encode(sae.normalize_input(vec.unsqueeze(0)))[0].float().cpu().numpy()
        idx = np.flatnonzero(feats > 0)
        hf_active[step] = [(int(i), float(feats[i])) for i in idx]

    layers = engine._get_model_layers()
    target_layer = layers[layer]

    def _forward(prompt: str, hook=None) -> torch.Tensor:
        handle = target_layer.register_forward_pre_hook(hook, with_kwargs=True) if hook else None
        try:
            return steer._run_forward(engine, prompt)
        finally:
            if handle is not None:
                handle.remove()

    def _read(prompt: str, edits: dict | None = None, mode: str = "delta") -> dict:
        """Tool distribution; ``edits`` applied at the decision token in every forward."""

        def _hook(position: int):
            if edits is None:
                return None
            return steer.make_feature_edit_hook(sae, edits, mode, position=position)

        logits = _forward(prompt, _hook(-1)).float()
        logp = torch.log_softmax(logits, dim=-1)
        first_lp = {int(t): float(logp[t]) for t in tree["first"]}
        second_lp: dict[int, dict[int, float]] = {}
        for token in shared_tokens:
            # Second forward only where the shared prefix carries real mass;
            # otherwise the (negligible) mass is split evenly.
            if math.exp(first_lp[token]) < 0.005:
                continue
            piece = engine.tokenizer.decode([token])
            logits2 = _forward(prompt + piece, _hook(-2)).float()
            logp2 = torch.log_softmax(logits2, dim=-1)
            second_lp[token] = {
                int(t2): float(logp2[t2]) for t2 in demo.second_token_ids(tree, token)
            }
        return demo.distribution_from_tree(first_lp, second_lp, tree)

    baselines = {item["step"]: _read(prompt_text[item["step"]]) for item in prompts}
    for item in prompts:
        step = item["step"]
        parity[step]["baselineDistributionHf"] = baselines[step]["distribution"]
        if step in decisions:
            parity[step]["baselineDistributionVllm"] = decisions[step].get("distribution")
        print(
            f"  Baseline {step:<24} HF {baselines[step]['display']:<16} "
            f"p={baselines[step]['prob']:.2f} | vLLM "
            f"{decisions.get(step, {}).get('display')} p={decisions.get(step, {}).get('prob')}"
        )

    attribution: dict[str, dict] = {}
    cross_patch: dict[str, dict] = {}
    for pair in demo.PAIRS:
        pid = pair["id"]
        steps = {side: f"{pid}_{side.upper()}" for side in demo.SIDES}
        analysis = demo.pair_feature_analysis(
            active_by_step[steps["a"]], active_by_step[steps["b"]], labels
        )
        attribution[pid] = {}
        cross_patch[pid] = {}
        for side in demo.SIDES:
            other = "b" if side == "a" else "a"
            rows = analysis[f"{side}Features"]
            if not rows:
                continue
            step, other_step = steps[side], steps[other]
            target_tool = (decisions.get(step) or {}).get("display") or baselines[step]["display"]
            other_tool = demo.other_tool_for(
                target_tool,
                (decisions.get(other_step) or baselines[other_step]).get("distribution", {}),
                (decisions.get(step) or baselines[step]).get("distribution", {}),
            )
            prompt = prompt_text[step]
            plan = steer.attribution_plan(
                rows,
                hf_active[step],
                draws=args.control_draws,
                seed=0,
                other_active=hf_active.get(other_step),
            )
            row_readings = [_read(prompt, dict.fromkeys(r["family"])) for r in plan["rows"]]
            all_reading = _read(prompt, dict.fromkeys(plan["allFeatures"]))
            control_readings = [
                _read(prompt, dict.fromkeys(c["features"])) for c in plan["controls"]
            ]
            set_control_readings = [
                _read(prompt, dict.fromkeys(c["features"])) for c in plan["setControls"]
            ]
            contrast_control_readings = [
                _read(prompt, dict.fromkeys(c["features"]))
                for c in plan.get("contrastControls") or []
            ]
            summary = steer.summarize_attribution(
                plan,
                baselines[step],
                row_readings,
                all_reading,
                control_readings,
                target_tool,
                set_control_readings,
                contrast_control_readings,
            )
            base_other = float(baselines[step]["distribution"].get(other_tool, 0.0))
            for row, reading in zip(summary["rows"], row_readings, strict=True):
                row["deltaOther"] = round(
                    float(reading["distribution"].get(other_tool, 0.0)) - base_other, 4
                )
            summary["allRows"]["deltaOther"] = round(
                float(all_reading["distribution"].get(other_tool, 0.0)) - base_other, 4
            )
            summary["otherTool"] = other_tool
            attribution[pid][side] = {
                "step": step,
                **summary,
                "backend": "hf",
                "parity": parity.get(step, {}),
            }
            print(f"\n  Ablation {step} (target {target_tool}, other {other_tool}):")
            for row in summary["rows"]:
                print(
                    f"    {row['index']:>6} {row['deltaTarget']:+.3f} / other {row['deltaOther']:+.3f}"
                    f" (hf act {row['hfActivation']:.2f}, {row['familySize']} feats) {row['label']}"
                )
            print(
                f"    all rows: {summary['allRows']['deltaTarget']:+.3f}; per-family control max "
                f"|delta| {summary['controlThreshold']:.3f}; set-matched control max |delta| "
                f"{summary.get('setControlThreshold', float('nan')):.3f} "
                f"(cue mass {summary.get('setMass', 0.0):.1f} over "
                f"{summary.get('setActiveSize', 0)} features)"
            )

            # Cross-patch: this side's families into the other side's prompt.
            plan_in = steer.injection_plan(
                rows, hf_active[step], hf_active[other_step], draws=args.control_draws, seed=0
            )
            other_prompt = prompt_text[other_step]

            def _clamp(targets: dict, other_prompt: str = other_prompt) -> dict:
                return _read(other_prompt, {int(k): float(v) for k, v in targets.items()})

            inj_rows = [_clamp(r["targets"]) for r in plan_in["rows"]]
            inj_all = _clamp(plan_in["allRowsTargets"])
            inj_base = _clamp(plan_in["allBaseTargets"])
            inj_controls = [_clamp(c["targets"]) for c in plan_in["controls"]]
            inj_set_controls = [_clamp(c["targets"]) for c in plan_in["setControls"]]
            inj_delta_controls = [_clamp(c["targets"]) for c in plan_in["deltaControls"]]
            inj = steer.summarize_injection(
                plan_in,
                baselines[other_step],
                inj_rows,
                inj_all,
                inj_base,
                inj_controls,
                target_tool,
                inj_set_controls,
                inj_delta_controls,
            )
            inj["fromSide"], inj["intoSide"], inj["intoStep"] = side, other, other_step
            inj["intoBaselineChoice"] = baselines[other_step]["display"]
            cross_patch[pid][f"{side}_into_{other}"] = inj
            print(f"  Cross-patch {side}->{other} (target {target_tool} on {other_step}):")
            for row in inj["rows"]:
                print(
                    f"    {row['index']:>6} {row['deltaTarget']:+.3f} -> {row['choice']} {row['label']}"
                )
            print(
                f"    all families {inj['allRows']['deltaTarget']:+.3f} -> {inj['allRows']['choice']}; "
                f"all {inj['allBase']['size']} active {inj['allBase']['deltaTarget']:+.3f} -> "
                f"{inj['allBase']['choice']}; per-family control max |delta| "
                f"{inj['controlThreshold']:.3f}; set-matched control max |delta| "
                f"{inj.get('setControlThreshold', float('nan')):.3f}; "
                f"delta-matched control max |delta| "
                f"{inj.get('deltaControlThreshold', float('nan')):.3f} "
                f"(clamp moves {inj.get('setDeltaMass', 0.0):.1f} of "
                f"{inj.get('setMass', 0.0):.1f} donor mass)"
            )

    engine.cleanup()
    output = {
        "backend": "hf",
        "hfFastPath": engine.fast_path,
        "model": args.model_name,
        "layer": layer,
        "thresholdOffset": args.threshold_offset,
        "saeCheckpoint": str(checkpoint),
        "report": args.report,
        "parity": parity,
        "attribution": attribution,
        "crossPatch": cross_patch,
    }
    out_path = results_dir / "steering_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
