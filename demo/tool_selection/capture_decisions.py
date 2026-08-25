#!/usr/bin/env python3
"""Capture the tool-selection pairs with the modified vLLM backend (canonical).

For every request of every pair: the layer-27 decision-position residual, the
exact tool distribution at ``I'll use the`` (first-token log-probabilities,
plus the second token for ``file_read`` / ``file_write`` which share their
first token), and the layer-27 SAE's active features with labels.

Usage (inside 575lab/kiji-inspector:dev):
    python demo/tool_selection/capture_decisions.py --model-name /models/... \
        [--sae-local-dir output] [--results-dir demo/tool_selection/output/capture]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import tool_selection_demo as demo
import torch

from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
)


def read_tool_choices(
    extractor, prompts: list[str], tree: dict, min_shared_mass: float = 0.0
) -> dict:
    """Exact tool distribution per prompt (one request per ``generate`` call).

    The fork only populates per-token logprobs for the first request of a
    batch, hence one call per prompt.  Shared first tokens get a second call
    with that token appended to read the conditional next-token logprobs;
    when ``min_shared_mass`` > 0 the second call is skipped for prompts where
    the shared prefix carries less than that share of the tool-token mass
    (``distribution_from_tree`` then splits the negligible mass evenly).
    """
    from vllm import SamplingParams

    storage_dir = getattr(extractor, "_storage_dir", None) or "."
    first_ids = sorted(tree["first"])
    shared = list(tree["shared"])
    second_ids = {token: demo.second_token_ids(tree, token) for token in shared}

    def _params(tag: str, token_ids: list[int], max_tokens: int):
        path = os.path.join(storage_dir, f"readout_{tag}.safetensors")
        extra = {"kv_transfer_params": {"hidden_states_path": path, "include_output_tokens": False}}
        return SamplingParams(
            max_tokens=max_tokens, temperature=0.0, logprob_token_ids=token_ids, extra_args=extra
        ), path

    def _cleanup(path: str) -> None:
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1 import (
                example_hidden_states_connector as connector,
            )

            connector.cleanup_hidden_states(path)
        except Exception:
            if os.path.exists(path):
                os.remove(path)

    logprobs_mode = None
    try:
        logprobs_mode = str(extractor.llm.llm_engine.model_config.logprobs_mode)
    except Exception:
        pass

    decisions = []
    for index, prompt in enumerate(prompts):
        params, path = _params(f"{index}_first", first_ids, 6)
        out = extractor.llm.generate([prompt], [params], use_tqdm=False)[0].outputs[0]
        _cleanup(path)
        first_lp = {int(t): float(o.logprob) for t, o in (out.logprobs or [{}])[0].items()}
        second_lp: dict[int, dict[int, float]] = {}
        for token in shared:
            if min_shared_mass > 0.0:
                total = sum(math.exp(lp) for lp in first_lp.values())
                share = (
                    math.exp(first_lp[token]) / total if token in first_lp and total > 0 else 0.0
                )
                if share < min_shared_mass:
                    continue
            piece = extractor.tokenizer.decode([token])
            params, path = _params(f"{index}_second_{token}", second_ids[token], 1)
            out2 = extractor.llm.generate([prompt + piece], [params], use_tqdm=False)[0].outputs[0]
            _cleanup(path)
            second_lp[token] = {
                int(t): float(o.logprob) for t, o in (out2.logprobs or [{}])[0].items()
            }
        decision = demo.distribution_from_tree(first_lp, second_lp, tree)
        decision["completion"] = out.text
        decision["topTokens"] = [
            {
                "tokenId": int(t),
                "logprob": round(float(o.logprob), 4),
                "text": getattr(o, "decoded_token", None),
            }
            for t, o in sorted((out.logprobs or [{}])[0].items(), key=lambda kv: -kv[1].logprob)[:8]
        ]
        decisions.append(decision)
    return {"decisions": decisions, "logprobs_mode": logprobs_mode}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-local-dir", default="output")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(demo.hr._TRAINED_LAYERS),
        help="SAE layers to capture and encode (all trained layers by default).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario to run (default: tool_selection). Its pairs.json/probes.json and "
        "output/ are read from demo/<scenario>/.",
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument(
        "--no-probes", action="store_true", help="capture only the pair sides, not probes.json"
    )
    args = parser.parse_args()
    # Rebind before any default path is resolved: the scenario decides where
    # pairs/probes are read and where results are written.
    if args.scenario:
        demo.configure(args.scenario)

    results_dir = Path(args.results_dir or demo.DEMO_DIR / "output" / "capture")
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = demo.decision_prompts(include_probes=not args.no_probes)
    requests = [item["request"] for item in metadata]

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=args.model_name,
            layers=list(args.layers),
            token_positions="decision",
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=2048,
            max_num_seqs=max(16, args.batch_size),
        )
    )
    try:
        prompts = demo.build_prompts(extractor.tokenizer, args.model_name, requests)
        tree = demo.tool_token_tree(extractor.tokenizer)
        activations = extractor.extract_batch(prompts, batch_size=args.batch_size)
        readout = read_tool_choices(extractor, prompts, tree)
    finally:
        extractor.cleanup()

    by_layer = {
        layer: np.stack([item[f"residual_{layer}"] for item in activations]).astype(np.float32)
        for layer in args.layers
    }
    np.savez_compressed(
        results_dir / "activations.npz", **{f"residual_{k}": v for k, v in by_layer.items()}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_blocks = []
    for layer in args.layers:
        sae, feature_descs = demo.hr._load_sae_local(args.sae_local_dir, layer, device=device)
        if sae is None:
            raise SystemExit(f"SAE checkpoint for layer {layer} not found.")
        labels = {str(k): v for k, v in (feature_descs or {}).items()}
        dtype = next(sae.parameters()).dtype
        with torch.no_grad():
            x = torch.from_numpy(by_layer[layer]).to(device=device, dtype=dtype)
            normalized = sae.normalize_input(x)
            encoded = sae.encode(normalized)
            feats = encoded.float().cpu().numpy()
            recon = sae.decode(encoded).float()
            err = (normalized.float() - recon).pow(2).sum(-1)
            ev = (1.0 - err / normalized.float().pow(2).sum(-1)).cpu().numpy()
        active_features = []
        for row in feats:
            idx = np.flatnonzero(row > 0)
            idx = idx[np.argsort(-row[idx])]
            active_features.append(
                [
                    {
                        "index": int(i),
                        "activation": round(float(row[i]), 4),
                        "label": labels.get(str(i), ""),
                    }
                    for i in idx
                ]
            )
        layer_blocks.append(
            {
                "layer": layer,
                "d_sae": int(feats.shape[1]),
                "active_features": active_features,
                "l0": [int(len(rows)) for rows in active_features],
                "explained_variance": [round(float(v), 4) for v in ev],
            }
        )
        del sae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    report = {
        "model": args.model_name,
        "backend": "vllm",
        "layer": demo.SAE_LAYER if demo.SAE_LAYER in args.layers else args.layers[0],
        "layers": layer_blocks,
        "threshold_offset": 0.0,
        "logprobs_mode": readout.get("logprobs_mode"),
        "scenario": str(demo._SCENARIO_PATH.relative_to(demo._REPO_ROOT)),
        "prompts": metadata,
        "decisions": [
            {**decision, "step": item["step"]}
            for item, decision in zip(metadata, readout["decisions"], strict=True)
        ],
    }
    (results_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    primary = demo.layer_block(report, report["layer"])
    for item, decision, rows in zip(
        metadata, report["decisions"], primary["active_features"], strict=True
    ):
        print(
            f"  {item['step']:<24} {decision['display']:<16} p={decision['prob']:.2f} "
            f"cov={decision['coverage']:.2f} L0={len(rows)}  | {item['request'][:70]}"
        )
    for block in layer_blocks:
        print(
            f"  layer {block['layer']}: mean L0 {np.mean(block['l0']):.0f}, "
            f"mean EV {np.mean(block['explained_variance']):.3f}"
        )
    print(f"\nWrote {results_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
