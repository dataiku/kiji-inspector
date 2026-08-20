#!/usr/bin/env python3
"""Replay exact stored training prompts through the current vLLM extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare_sae_backends as backend_compare
import evaluate_sae_layers as layer_eval
import numpy as np

from kiji_inspector.data.contrastive_dataset import ContrastiveDataset
from kiji_inspector.data.scenario import load_scenarios_meta
from kiji_inspector.extraction.extractor import build_agent_prompt
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
    recommended_chat_template_kwargs,
)


def _select_pair_indices(total_pairs: int, num_pairs: int) -> list[int]:
    if num_pairs < 1 or num_pairs > total_pairs:
        raise ValueError("num_pairs must be between 1 and total_pairs.")
    if num_pairs == 1:
        return [0]
    return [round(index * (total_pairs - 1) / (num_pairs - 1)) for index in range(num_pairs)]


def _load_replay_prompts(
    pairs_dir: Path,
    activations_dir: Path,
    pair_indices: list[int],
    tokenizer,
    model_name: str,
) -> tuple[list[dict], list[str]]:
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    scenarios = load_scenarios_meta(pairs_dir)
    stored_user_requests = json.loads((activations_dir / "prompts.json").read_text())
    template_kwargs = recommended_chat_template_kwargs(model_name, tokenizer)

    # The extraction run can use a reordered/subselected pair list, so its pair
    # position is not necessarily the row position in the current Parquet files.
    # prompts.json is the authoritative activation-row ordering. Resolve only the
    # scenario/tool metadata from the exact adjacent prompt pair.
    candidates_by_prompts: dict[tuple[str, str], list] = {}
    for pair in dataset.pairs:
        key = (pair.anchor_prompt, pair.contrast_prompt)
        candidates_by_prompts.setdefault(key, []).append(pair)

    metadata = []
    formatted_prompts = []
    for pair_index in pair_indices:
        anchor_request = stored_user_requests[2 * pair_index]
        contrast_request = stored_user_requests[2 * pair_index + 1]
        candidates = candidates_by_prompts.get((anchor_request, contrast_request), [])
        if not candidates:
            raise ValueError(
                f"Stored activation pair {pair_index} has no exact match in {pairs_dir}."
            )
        scenario_names = {pair.scenario_name for pair in candidates}
        if len(scenario_names) != 1:
            raise ValueError(
                f"Stored activation pair {pair_index} maps to multiple scenarios: "
                f"{sorted(scenario_names)}."
            )
        scenario_name = next(iter(scenario_names))
        scenario = scenarios[scenario_name]
        contrast_types = sorted({pair.contrast_type for pair in candidates})
        source_pair_ids = [pair.pair_id for pair in candidates]
        sides = (
            (
                "anchor",
                anchor_request,
                sorted({pair.anchor_tool for pair in candidates}),
                2 * pair_index,
            ),
            (
                "contrast",
                contrast_request,
                sorted({pair.contrast_tool for pair in candidates}),
                2 * pair_index + 1,
            ),
        )
        for side, user_request, expected_tools, row_index in sides:
            formatted_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=user_request,
                    tokenizer=tokenizer,
                    chat_template_kwargs=template_kwargs,
                    close_think_block=bool(template_kwargs),
                )
            )
            metadata.append(
                {
                    "step": f"row_{row_index}_{side}",
                    "problem": f"activation_pair_{pair_index}",
                    "tool": side,
                    "row_index": row_index,
                    "pair_index": pair_index,
                    "source_pair_ids": source_pair_ids,
                    "source_candidate_count": len(candidates),
                    "side": side,
                    "scenario_name": scenario_name,
                    "contrast_types": contrast_types,
                    "expected_tool_candidates": expected_tools,
                    "user_request": user_request,
                }
            )
    return metadata, formatted_prompts


def _pair_delta_parity(stored: np.ndarray, fresh: np.ndarray) -> dict:
    if stored.shape != fresh.shape or stored.shape[0] % 2:
        raise ValueError("Stored and fresh arrays must have equal, even row counts.")
    pairs = []
    for row in range(0, stored.shape[0], 2):
        stored_delta = stored[row] - stored[row + 1]
        fresh_delta = fresh[row] - fresh[row + 1]
        stored_norm = float(np.linalg.norm(stored_delta))
        pairs.append(
            {
                "cosine_similarity": 1.0 - layer_eval._cosine_distance(stored_delta, fresh_delta),
                "relative_l2_error": (
                    float(np.linalg.norm(fresh_delta - stored_delta) / stored_norm)
                    if stored_norm
                    else 0.0
                ),
            }
        )
    return {
        "mean_cosine_similarity": float(np.mean([pair["cosine_similarity"] for pair in pairs])),
        "min_cosine_similarity": float(np.min([pair["cosine_similarity"] for pair in pairs])),
        "mean_relative_l2_error": float(np.mean([pair["relative_l2_error"] for pair in pairs])),
        "per_pair": pairs,
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# Stored-training activation replay",
        "",
        f"- Exact rows replayed: {len(report['prompts'])}",
        f"- Complete pairs: {len(report['pair_indices'])}",
        f"- Activation-pair indices: {', '.join(map(str, report['pair_indices']))}",
        "- SAE thresholds: native checkpoint values",
        "",
        "| Layer | Raw cosine | Min cosine | Raw rel. L2 | Pair-delta cosine | Feature RBO | Top-k Jaccard | Stored L0 | Fresh L0 | Stored MSE | Fresh MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["layers"]:
        raw = item["raw_vector_parity"]
        delta = item["pair_delta_parity"]
        rank = item["feature_rank_parity"]
        stored = item["stored_evaluation"]
        fresh = item["fresh_evaluation"]
        lines.append(
            f"| {item['layer']} | {raw['mean_cosine_similarity']:.6f} | "
            f"{raw['min_cosine_similarity']:.6f} | "
            f"{raw['mean_relative_l2_error']:.3f} | "
            f"{delta['mean_cosine_similarity']:.6f} | "
            f"{rank['mean_rbo']:.3f} | {rank['mean_top_k_jaccard']:.3f} | "
            f"{stored['l0']['mean']:.2f} | {fresh['l0']['mean']:.2f} | "
            f"{stored['reconstruction']['normalized_mse']:.5f} | "
            f"{fresh['reconstruction']['normalized_mse']:.5f} |"
        )
    lines.extend(
        [
            "",
            "Raw-vector and pair-delta parity compare the original stored vLLM rows with the current modified-vLLM extraction of the exact same serialized agent requests.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pairs-dir", default="output/pairs")
    parser.add_argument("--training-output-dir", default="output")
    parser.add_argument("--results-dir", default="demo/home_repair/output/training_replay")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 27, 34, 43])
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    training_output_dir = Path(args.training_output_dir)
    reference_dir = training_output_dir / f"layer_{args.layers[0]}" / "activations"
    reference_metadata = json.loads((reference_dir / "metadata.json").read_text())
    total_pairs = int(reference_metadata["total_pairs"])
    pair_indices = _select_pair_indices(total_pairs, args.num_pairs)

    # Resolve and serialize every exact row before paying the model startup cost.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    metadata, formatted_prompts = _load_replay_prompts(
        pairs_dir=Path(args.pairs_dir),
        activations_dir=reference_dir,
        pair_indices=pair_indices,
        tokenizer=tokenizer,
        model_name=args.model_name,
    )

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=args.model_name,
            layers=args.layers,
            token_positions="decision",
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=2048,
            max_num_seqs=max(16, args.batch_size),
        )
    )
    try:
        extracted = extractor.extract_batch(formatted_prompts, batch_size=args.batch_size)
    finally:
        extractor.cleanup()

    row_indices = [item["row_index"] for item in metadata]
    fresh_by_layer = {
        layer: np.stack([item[f"residual_{layer}"] for item in extracted]).astype(np.float32)
        for layer in args.layers
    }
    stored_by_layer = {
        layer: np.asarray(
            np.load(
                training_output_dir / f"layer_{layer}" / "activations" / "shard_000000.npy",
                mmap_mode="r",
            )[row_indices],
            dtype=np.float32,
        )
        for layer in args.layers
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results_dir / "fresh_training_replay_activations.npz",
        **{f"residual_{layer}": values for layer, values in fresh_by_layer.items()},
    )

    layer_reports = []
    for layer in args.layers:
        print(f"\nComparing stored and fresh layer {layer}...")
        stored_eval = layer_eval._evaluate_layer(
            layer,
            stored_by_layer[layer],
            metadata,
            training_output_dir,
            threshold_offset=0.0,
            top_k=args.top_k,
            device=args.device,
        )
        fresh_eval = layer_eval._evaluate_layer(
            layer,
            fresh_by_layer[layer],
            metadata,
            training_output_dir,
            threshold_offset=0.0,
            top_k=args.top_k,
            device=args.device,
        )
        layer_reports.append(
            {
                "layer": layer,
                "raw_vector_parity": backend_compare._vector_parity(
                    stored_by_layer[layer], fresh_by_layer[layer]
                ),
                "pair_delta_parity": _pair_delta_parity(
                    stored_by_layer[layer], fresh_by_layer[layer]
                ),
                "feature_rank_parity": backend_compare._feature_rank_parity(
                    stored_eval, fresh_eval
                ),
                "stored_evaluation": stored_eval,
                "fresh_evaluation": fresh_eval,
            }
        )

    report = {
        "model": args.model_name,
        "pair_indices": pair_indices,
        "row_indices": row_indices,
        "prompts": metadata,
        "layers": layer_reports,
    }
    (results_dir / "training_replay_parity.json").write_text(json.dumps(report, indent=2))
    (results_dir / "training_replay_parity.md").write_text(_render_markdown(report))
    print(f"\nWrote replay reports to {results_dir}")


if __name__ == "__main__":
    main()
