#!/usr/bin/env python3
"""How much causal signal is available at the decision token at all.

``attribute_pairs.py`` reports what interventions on SAE features do.  On their
own those counts have no denominator: a reader cannot tell whether flipping 29
of 128 cross-patch directions means the dictionary recovers most of the
available causal signal or a third of it.  This script supplies the
denominator, with no dictionary in the path:

* **ceiling** — replace the recipient's residual at the decision token with the
  donor's, i.e. ordinary activation patching in the model's own basis.  Whatever
  causal signal that token carries, this moves all of it.
* **difference-in-means** — add the mean (side A - side B) residual of the
  *other* pairs of the same contrast type, scaled to the norm of this pair's own
  donor-minus-recipient difference.  A direction that needed no per-pair
  activations and no dictionary.  Undefined where a contrast type contributes
  one pair, which is every pair of the demonstration sets.
* **random directions** — Gaussian directions at the same norm, so a large
  perturbation is not mistaken for a targeted one.

The donor-minus-recipient norm is larger than the change a cue-set clamp makes,
so those three arms bound rather than match it.  With ``--battery`` the
difference-in-means and random arms are additionally run at the norm of the cue
clamp's own residual change, reconstructed from the battery's stored targets, so
the comparison is size-matched and not merely "a bigger intervention wins".

Usage (inside 575lab/kiji-inspector:dev):
    python demo/tool_selection/ceiling_pairs.py --model-name /models/... \
        --scenario supply_chain_expanded --layer 43 \
        --results-dir demo/steering/supply_chain_expanded/output/ceiling_layer43
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import tool_selection_demo as demo
import torch

sys.path.insert(0, str(demo._DEMO_DIR.parent / "home_repair"))
import steer_tool_choice as steer  # noqa: E402

from kiji_inspector.core.sae_core import JumpReLUSAE  # noqa: E402


def _argmax_tool(reading: dict) -> str | None:
    return reading.get("display")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--layer", type=int, default=demo.SAE_LAYER)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--draws", type=int, default=3, help="random-direction controls per side")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--battery",
        default=None,
        help="steering_results.json of the same scenario/layer; enables the "
        "norm-matched arms by reconstructing the cue clamp's residual change.",
    )
    parser.add_argument("--sae-local-dir", default="output")
    parser.add_argument("--threshold-offset", type=float, default=demo.hr._HF_THRESHOLD_OFFSET)
    args = parser.parse_args()

    if args.scenario:
        demo.configure(args.scenario)
    layer = args.layer

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

    activation_log = engine.extract_all_prompts([layer])
    residuals = {step: acts[f"residual_{layer}"] for step, acts in activation_log}

    battery = json.loads(Path(args.battery).read_text()) if args.battery else None
    sae = None
    if battery:
        checkpoint = Path(args.sae_local_dir) / f"layer_{layer}" / "sae_checkpoints" / "sae_final.pt"
        sae = JumpReLUSAE.from_pretrained(
            str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu"
        )
        sae.eval()
        with torch.no_grad():
            sae.threshold.add_(args.threshold_offset)

    def clamp_delta_norm(pid: str, direction: str, recipient_step: str) -> float | None:
        """Norm of the residual change the cue-set clamp actually makes.

        The delta hook moves the residual by ``scale * (target - current) *
        W_dec[i]`` per clamped feature, so the same sum reconstructs it here
        from the battery's stored per-row targets and the recipient's own
        activations.  Returns ``None`` where the battery has no such direction.
        """
        record = ((battery or {}).get("crossPatch") or {}).get(pid, {}).get(direction)
        if not record:
            return None
        targets: dict[int, float] = {}
        for row in record.get("rows") or []:
            for index, value in (row.get("targets") or {}).items():
                targets[int(index)] = float(value)
        if not targets:
            return None
        device = next(sae.parameters()).device
        dtype = next(sae.parameters()).dtype
        vec = torch.from_numpy(residuals[recipient_step]).to(device=device, dtype=dtype)
        with torch.no_grad():
            current = sae.encode(sae.normalize_input(vec.unsqueeze(0)))[0].float()
        scale = steer._sae_scale(sae)
        delta = torch.zeros(vec.shape[-1], device=device, dtype=torch.float32)
        for index, target in targets.items():
            delta += scale * (target - float(current[index])) * sae.W_dec[index].float()
        return float(delta.norm())

    layers = engine._get_model_layers()
    target_layer = layers[layer]

    def _forward(prompt: str, hook=None) -> torch.Tensor:
        handle = target_layer.register_forward_pre_hook(hook, with_kwargs=True) if hook else None
        try:
            return steer._run_forward(engine, prompt)
        finally:
            if handle is not None:
                handle.remove()

    def _read(prompt: str, vector=None) -> dict:
        """Tool distribution, with ``vector`` patched in at the decision token."""

        def _hook(position: int):
            if vector is None:
                return None
            return steer.make_residual_patch_hook(vector, position=position)

        logits = _forward(prompt, _hook(-1)).float()
        logp = torch.log_softmax(logits, dim=-1)
        first_lp = {int(t): float(logp[t]) for t in tree["first"]}
        second_lp: dict[int, dict[int, float]] = {}
        for token in shared_tokens:
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

    # Difference-in-means needs several pairs of the same contrast type, so it
    # is defined on the sampled sets and not on the one-pair-per-type demos.
    by_title: dict[str, list[str]] = {}
    for pair in demo.PAIRS:
        by_title.setdefault(pair.get("title") or pair["id"], []).append(pair["id"])

    def direction_rng(pid: str, direction: str) -> np.random.Generator:
        """One independent stream per direction.

        A single shared stream would make each direction's control draws depend
        on how many arms ran before it, so enabling ``--battery`` would silently
        change the *unmatched* controls too.  Seeding from a stable hash of the
        direction keeps every arm reproducible on its own.
        """
        tag = zlib.crc32(f"{pid}:{direction}".encode()) & 0xFFFFFFFF
        return np.random.default_rng([args.seed, tag])

    out: dict[str, dict] = {}
    for pair in demo.PAIRS:
        pid = pair["id"]
        title = pair.get("title") or pid
        siblings = [q for q in by_title[title] if q != pid]
        out[pid] = {}
        for direction, donor_side, recipient_side in (("a_into_b", "a", "b"), ("b_into_a", "b", "a")):
            donor_step = f"{pid}_{donor_side.upper()}"
            recipient_step = f"{pid}_{recipient_side.upper()}"
            if donor_step not in residuals or recipient_step not in residuals:
                continue
            donor = torch.from_numpy(residuals[donor_step])
            recipient = torch.from_numpy(residuals[recipient_step])
            prompt = prompt_text[recipient_step]
            base = baselines[recipient_step]
            target_tool = baselines[donor_step]["display"]
            delta = donor - recipient
            norm = float(delta.norm())

            ceiling = _read(prompt, donor)

            dim_reading = None
            dim_pairs = 0
            for sibling in siblings:
                a, b = f"{sibling}_A", f"{sibling}_B"
                if a in residuals and b in residuals:
                    dim_pairs += 1
            if dim_pairs:
                acc = torch.zeros_like(donor, dtype=torch.float32)
                for sibling in siblings:
                    a, b = f"{sibling}_A", f"{sibling}_B"
                    if a not in residuals or b not in residuals:
                        continue
                    diff = torch.from_numpy(residuals[a]).float() - torch.from_numpy(
                        residuals[b]
                    ).float()
                    acc += diff if donor_side == "a" else -diff
                acc /= dim_pairs
                acc_norm = float(acc.norm())
                if acc_norm > 0:
                    scaled = acc * (norm / acc_norm)
                    dim_reading = _read(prompt, (recipient.float() + scaled).to(donor.dtype))

            rng = direction_rng(pid, direction)
            controls = []
            for draw in range(args.draws):
                noise = torch.from_numpy(
                    rng.standard_normal(donor.shape[-1]).astype(np.float32)
                )
                noise = noise * (norm / float(noise.norm()))
                reading = _read(prompt, (recipient.float() + noise).to(donor.dtype))
                controls.append(
                    {
                        "draw": draw,
                        "choice": _argmax_tool(reading),
                        "pTarget": round(float(reading["distribution"].get(target_tool, 0.0)), 4),
                    }
                )

            matched_norm = (
                clamp_delta_norm(pid, direction, recipient_step) if battery else None
            )
            dim_matched = None
            random_matched: list[dict] = []
            if matched_norm and dim_pairs and matched_norm > 0:
                unit = acc / float(acc.norm())
                reading = _read(
                    prompt, (recipient.float() + unit * matched_norm).to(donor.dtype)
                )
                dim_matched = {
                    "norm": round(matched_norm, 4),
                    "choice": _argmax_tool(reading),
                    "pTarget": round(float(reading["distribution"].get(target_tool, 0.0)), 4),
                    "flipped": _argmax_tool(reading) == target_tool
                    and base["display"] != target_tool,
                }
                matched_rng = direction_rng(pid, direction + ":matched")
                for draw in range(args.draws):
                    noise = torch.from_numpy(
                        matched_rng.standard_normal(donor.shape[-1]).astype(np.float32)
                    )
                    noise = noise * (matched_norm / float(noise.norm()))
                    r2 = _read(prompt, (recipient.float() + noise).to(donor.dtype))
                    random_matched.append(
                        {"draw": draw, "choice": _argmax_tool(r2)}
                    )

            base_p = float(base["distribution"].get(target_tool, 0.0))
            record = {
                "donorStep": donor_step,
                "recipientStep": recipient_step,
                "targetTool": target_tool,
                "baselineChoice": base["display"],
                "baselineTarget": round(base_p, 4),
                "differenceNorm": round(norm, 4),
                "recipientNorm": round(float(recipient.norm()), 4),
                "ceiling": {
                    "choice": _argmax_tool(ceiling),
                    "pTarget": round(float(ceiling["distribution"].get(target_tool, 0.0)), 4),
                    "deltaTarget": round(
                        float(ceiling["distribution"].get(target_tool, 0.0)) - base_p, 4
                    ),
                    "flipped": _argmax_tool(ceiling) == target_tool
                    and base["display"] != target_tool,
                },
                "differenceInMeans": None
                if dim_reading is None
                else {
                    "siblingPairs": dim_pairs,
                    "choice": _argmax_tool(dim_reading),
                    "pTarget": round(float(dim_reading["distribution"].get(target_tool, 0.0)), 4),
                    "deltaTarget": round(
                        float(dim_reading["distribution"].get(target_tool, 0.0)) - base_p, 4
                    ),
                    "flipped": _argmax_tool(dim_reading) == target_tool
                    and base["display"] != target_tool,
                },
                "clampDeltaNorm": None if matched_norm is None else round(matched_norm, 4),
                "differenceInMeansMatched": dim_matched,
                "randomMatched": random_matched,
                "randomMatchedFlips": sum(
                    1
                    for c in random_matched
                    if c["choice"] == target_tool and base["display"] != target_tool
                ),
                "randomControls": controls,
                "randomFlips": sum(
                    1
                    for c in controls
                    if c["choice"] == target_tool and base["display"] != target_tool
                ),
            }
            out[pid][direction] = record
            dim_txt = (
                "n/a"
                if record["differenceInMeans"] is None
                else f"{record['differenceInMeans']['deltaTarget']:+.3f}"
                f"{' FLIPS' if record['differenceInMeans']['flipped'] else ''}"
            )
            print(
                f"  {pid[:34]:34s} {direction:9s} target {target_tool:<20s} "
                f"ceiling {record['ceiling']['deltaTarget']:+.3f}"
                f"{' FLIPS' if record['ceiling']['flipped'] else '     '} | "
                f"dim {dim_txt} | random flips {record['randomFlips']}/{args.draws}"
            )

    ceiling_flips = sum(1 for p in out.values() for d in p.values() if d["ceiling"]["flipped"])
    dim_flips = sum(
        1
        for p in out.values()
        for d in p.values()
        if d["differenceInMeans"] and d["differenceInMeans"]["flipped"]
    )
    dim_n = sum(1 for p in out.values() for d in p.values() if d["differenceInMeans"])
    total = sum(len(p) for p in out.values())
    random_flips = sum(d["randomFlips"] for p in out.values() for d in p.values())
    dimm = [d["differenceInMeansMatched"] for p in out.values() for d in p.values()
            if d.get("differenceInMeansMatched")]
    dimm_flips = sum(1 for d in dimm if d["flipped"])
    randm_flips = sum(d.get("randomMatchedFlips", 0) for p in out.values() for d in p.values())
    randm_draws = sum(len(d.get("randomMatched") or []) for p in out.values() for d in p.values())
    print(
        f"\nceiling (full residual patch): {ceiling_flips}/{total} directions flip; "
        f"difference-in-means {dim_flips}/{dim_n}; "
        f"random directions at the same norm {random_flips}/{total * args.draws}"
    )
    if dimm:
        print(
            f"at the cue clamp's own norm: difference-in-means {dimm_flips}/{len(dimm)}; "
            f"random {randm_flips}/{randm_draws}"
        )

    results = {
        "model": args.model_name,
        "layer": layer,
        "scenario": args.scenario,
        "draws": args.draws,
        "seed": args.seed,
        "directions": out,
        "summary": {
            "ceilingFlips": ceiling_flips,
            "directions": total,
            "dimFlips": dim_flips,
            "dimDirections": dim_n,
            "randomFlips": random_flips,
            "randomDraws": total * args.draws,
            "dimMatchedFlips": dimm_flips,
            "dimMatchedDirections": len(dimm),
            "randomMatchedFlips": randm_flips,
            "randomMatchedDraws": randm_draws,
        },
    }
    results_dir = Path(args.results_dir or (demo.DEMO_DIR / "output" / f"ceiling_layer{layer}"))
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "ceiling_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
