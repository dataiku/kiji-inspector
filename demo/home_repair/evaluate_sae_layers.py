#!/usr/bin/env python3
"""Compare all trained SAEs on the three home-repair tool-choice prompts.

The subject model is loaded once and all requested residual streams are captured
in the same forward pass per prompt.  Each SAE is then evaluated with identical
activations and the configured in-memory threshold calibration.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import home_repair_demo as demo
import numpy as np
import torch

from kiji_inspector.core.sae_core import JumpReLUSAE

# Appliance keywords per scenario live in home_repair_demo so the UI builder
# and this evaluator agree on what counts as a scenario-specific label.
_SCENARIO_PATTERNS = demo._SCENARIO_PATTERNS


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0 if np.array_equal(left, right) else 1.0
    return float(1.0 - np.dot(left, right) / denominator)


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _truncated_rbo(left: list[int], right: list[int], p: float = 0.9) -> float:
    """Finite rank-biased overlap, including the residual agreement at depth k."""
    depth = min(len(left), len(right))
    if depth == 0:
        return 1.0 if not left and not right else 0.0
    left_seen: set[int] = set()
    right_seen: set[int] = set()
    weighted = 0.0
    agreement = 0.0
    for index in range(depth):
        left_seen.add(left[index])
        right_seen.add(right[index])
        agreement = len(left_seen & right_seen) / (index + 1)
        weighted += (1.0 - p) * (p**index) * agreement
    return weighted + (p**depth) * agreement


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _prompt_metadata(include_contrasts: bool = False, include_probes: bool = False) -> list[dict]:
    """Decision prompts in the demo's canonical order (base first)."""
    return demo.decision_prompts(include_contrasts=include_contrasts, include_probes=include_probes)


def _base_rows(metadata: list[dict]) -> list[int]:
    """Row indices of the base prompts (scenario metrics ignore contrast variants)."""
    return [index for index, item in enumerate(metadata) if item.get("kind", "base") == "base"]


def _load_contrastive_map(output_dir: Path, layer: int) -> dict[int, list[dict]]:
    return demo.filter_contrastive_map(demo._load_contrastive_feature_map(str(output_dir), layer))


def _capture_activations(
    model_name: str, layers: list[int]
) -> tuple[list[dict], dict[int, np.ndarray]]:
    engine = demo.HFEngine(
        model_name=model_name,
        device="auto",
        dtype="bfloat16",
        max_new_tokens=1,
        allow_thinking=False,
    )
    metadata = _prompt_metadata()
    try:
        for item in metadata:
            engine.record_tool_decision(item["step"], item["request"])
        activation_log = engine.extract_all_prompts(layers)
    finally:
        engine.cleanup()

    observed_steps = [step for step, _ in activation_log]
    expected_steps = [item["step"] for item in metadata]
    if observed_steps != expected_steps:
        raise RuntimeError("Captured prompt order does not match evaluation metadata.")

    by_layer = {
        layer: np.stack(
            [activations[f"residual_{layer}"] for _, activations in activation_log]
        ).astype(np.float32)
        for layer in layers
    }
    return metadata, by_layer


def _load_feature_labels(output_dir: Path, layer: int) -> dict[str, dict]:
    path = output_dir / f"layer_{layer}" / "activations" / "feature_descriptions.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _classify_label(label: str) -> set[str]:
    return demo.classify_scenario_label(label)


def _top_rankings(features: np.ndarray, top_k: int) -> list[list[int]]:
    rankings = []
    for row in features:
        active = np.flatnonzero(row > 0)
        order = active[np.argsort(-row[active])]
        rankings.append(order[:top_k].astype(int).tolist())
    return rankings


def _scenario_label_metrics(
    features: np.ndarray,
    rankings: list[list[int]],
    metadata: list[dict],
    labels: dict[str, dict],
) -> dict:
    correct_mass = 0.0
    contaminating_mass = 0.0
    labeled_mass = 0.0
    prompts_with_correct = 0
    prompt_details = []

    for row, ranking, item in zip(features, rankings, metadata, strict=True):
        prompt_correct = 0.0
        prompt_contaminating = 0.0
        prompt_labeled = 0.0
        correct_labels = []
        contaminating_labels = []
        for index in ranking:
            description = labels.get(str(index))
            if not description:
                continue
            activation = float(row[index])
            prompt_labeled += activation
            matches = _classify_label(description.get("label", ""))
            if item["problem"] in matches:
                prompt_correct += activation
                correct_labels.append(description.get("label", ""))
            other_matches = matches - {item["problem"]}
            if other_matches:
                prompt_contaminating += activation
                contaminating_labels.append(description.get("label", ""))

        correct_mass += prompt_correct
        contaminating_mass += prompt_contaminating
        labeled_mass += prompt_labeled
        prompts_with_correct += int(prompt_correct > 0)
        prompt_details.append(
            {
                "step": item["step"],
                "correct_mass_share": prompt_correct / prompt_labeled if prompt_labeled else 0.0,
                "contamination_mass_share": (
                    prompt_contaminating / prompt_labeled if prompt_labeled else 0.0
                ),
                "correct_labels": correct_labels,
                "contaminating_labels": contaminating_labels,
            }
        )

    scenario_mass = correct_mass + contaminating_mass
    return {
        "correct_mass_share": correct_mass / labeled_mass if labeled_mass else 0.0,
        "contamination_mass_share": (contaminating_mass / labeled_mass if labeled_mass else 0.0),
        "scenario_purity": correct_mass / scenario_mass if scenario_mass else 0.0,
        "prompts_with_correct_feature": prompts_with_correct,
        "num_prompts": len(metadata),
        "details": prompt_details,
    }


def _matched_separation(
    features: np.ndarray,
    rankings: list[list[int]],
    metadata: list[dict],
) -> dict:
    pairs = []
    for left, right in combinations(range(len(metadata)), 2):
        if metadata[left]["problem"] == metadata[right]["problem"]:
            continue
        pairs.append(
            {
                "left": metadata[left]["step"],
                "right": metadata[right]["step"],
                "cosine_distance": _cosine_distance(features[left], features[right]),
                "top_k_jaccard": _jaccard(rankings[left], rankings[right]),
            }
        )
    return {
        "definition": "All pairs of distinct repair problems.",
        "num_pairs": len(pairs),
        "mean_cosine_distance": _mean([pair["cosine_distance"] for pair in pairs]),
        "mean_top_k_jaccard": _mean([pair["top_k_jaccard"] for pair in pairs]),
        "pairs": pairs,
    }


def _ranking_stability(rankings: list[list[int]], metadata: list[dict]) -> dict:
    pairs = []
    for left, right in combinations(range(len(metadata)), 2):
        if metadata[left]["problem"] != metadata[right]["problem"]:
            continue
        if not metadata[left].get("tool") or not metadata[right].get("tool"):
            continue  # the open (no-ask) prompt names no tool
        if metadata[left]["tool"] == metadata[right]["tool"]:
            continue
        pairs.append(
            {
                "left": metadata[left]["step"],
                "right": metadata[right]["step"],
                "rbo": _truncated_rbo(rankings[left], rankings[right]),
                "top_k_jaccard": _jaccard(rankings[left], rankings[right]),
            }
        )
    return {
        "definition": (
            "Feature-rank consistency across different requested tools for the same problem."
        ),
        "num_pairs": len(pairs),
        "mean_rbo": _mean([pair["rbo"] for pair in pairs]),
        "mean_top_k_jaccard": _mean([pair["top_k_jaccard"] for pair in pairs]),
        "pairs": pairs,
    }


def _evaluate_layer(
    layer: int,
    raw_activations: np.ndarray,
    metadata: list[dict],
    output_dir: Path,
    threshold_offset: float,
    top_k: int,
    device: str,
    contrastive_map: dict[int, list[dict]] | None = None,
) -> dict:
    checkpoint = output_dir / f"layer_{layer}" / "sae_checkpoints" / "sae_final.pt"
    sae = JumpReLUSAE.from_pretrained(str(checkpoint), device=device)
    sae.eval()
    native_threshold_median = float(sae.threshold.float().median().item())
    with torch.no_grad():
        sae.threshold.add_(threshold_offset)

    dtype = next(sae.parameters()).dtype
    raw = torch.from_numpy(raw_activations).to(device=device, dtype=dtype)
    with torch.no_grad():
        normalized = sae.normalize_input(raw)
        features = sae.encode(normalized)
        reconstruction = sae.decode(features)
        raw_reconstruction = sae.denormalize_output(reconstruction)

    normalized_float = normalized.float()
    reconstruction_float = reconstruction.float()
    raw_float = raw.float()
    raw_reconstruction_float = raw_reconstruction.float()
    residual = normalized_float - reconstruction_float
    # Aggregate explained variance over all prompt/dimension entries. Centering
    # each dimension over only 12 prompts would make the denominator mostly a
    # between-prompt quantity and badly understate reconstruction quality.
    centered = normalized_float - normalized_float.mean()
    total_variance = float(centered.square().sum().item())
    explained_variance = (
        1.0 - float(residual.square().sum().item()) / total_variance if total_variance > 0 else 0.0
    )
    cosine = torch.nn.functional.cosine_similarity(normalized_float, reconstruction_float, dim=1)

    features_np = features.float().cpu().numpy()
    rankings = _top_rankings(features_np, top_k)
    labels = _load_feature_labels(output_dir, layer)
    active_counts = (features_np > 0).sum(axis=1)

    # Scenario-specificity and separation are published over the base prompts
    # only, so adding contrast variants does not move those numbers.
    base_rows = _base_rows(metadata)
    base_features = features_np[base_rows]
    base_rankings = [rankings[index] for index in base_rows]
    base_metadata = [metadata[index] for index in base_rows]

    def _label(index: int) -> str:
        return labels.get(str(index), {}).get("label", "unlabeled")

    active_features = []
    theme_evidence = []
    for row in features_np:
        active = np.flatnonzero(row > 0)
        order = active[np.argsort(-row[active])]
        pairs = [(int(index), float(row[index])) for index in order]
        active_features.append(
            [{"index": index, "activation": act, "label": _label(index)} for index, act in pairs]
        )
        theme_evidence.append(
            demo.contrastive_theme_evidence(pairs, contrastive_map, labels)
            if contrastive_map
            else None
        )

    result = {
        "layer": layer,
        "d_sae": int(features_np.shape[1]),
        "threshold_offset": threshold_offset,
        "native_threshold_median": native_threshold_median,
        "calibrated_threshold_median": native_threshold_median + threshold_offset,
        "l0": {
            "mean": float(active_counts.mean()),
            "std": float(active_counts.std()),
            "min": int(active_counts.min()),
            "max": int(active_counts.max()),
            "per_prompt": active_counts.astype(int).tolist(),
        },
        "reconstruction": {
            "normalized_mse": float(residual.square().mean().item()),
            "raw_mse": float((raw_float - raw_reconstruction_float).square().mean().item()),
            "explained_variance": explained_variance,
            "mean_cosine_similarity": float(cosine.mean().item()),
        },
        "scenario_specificity": _scenario_label_metrics(
            base_features, base_rankings, base_metadata, labels
        ),
        "matched_separation": _matched_separation(base_features, base_rankings, base_metadata),
        "top_k": top_k,
        "active_features": active_features,
        "theme_evidence": theme_evidence,
        "top_features": [
            [
                {
                    "index": index,
                    "activation": float(features_np[row_index, index]),
                    "label": _label(index),
                }
                for index in ranking
            ]
            for row_index, ranking in enumerate(rankings)
        ],
    }

    del sae, raw, normalized, features, reconstruction, raw_reconstruction
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _render_markdown(report: dict) -> str:
    lines = [
        "# Six-layer home-repair SAE evaluation",
        "",
        f"- Prompts: {len(report['prompts'])} (one initial tool decision per problem)",
        f"- Top-k: {report['top_k']}",
        f"- Threshold offset: {report['threshold_offset']:+.8f}",
        "",
        "## Sparsity and reconstruction",
        "",
        "| Layer | L0 mean | L0 range | Norm. MSE | Explained var. | Recon. cosine |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["layers"]:
        lines.append(
            f"| {result['layer']} | {result['l0']['mean']:.2f} | "
            f"{result['l0']['min']}–{result['l0']['max']} | "
            f"{result['reconstruction']['normalized_mse']:.5f} | "
            f"{result['reconstruction']['explained_variance']:.3f} | "
            f"{result['reconstruction']['mean_cosine_similarity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Specificity and cross-problem separation",
            "",
            "| Layer | Correct mass | Contam. mass | Scenario purity | Correct coverage | Problem cosine dist. | Problem top-k Jaccard |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in report["layers"]:
        specificity = result["scenario_specificity"]
        lines.append(
            f"| {result['layer']} | "
            f"{specificity['correct_mass_share']:.1%} | "
            f"{specificity['contamination_mass_share']:.1%} | "
            f"{specificity['scenario_purity']:.1%} | "
            f"{specificity['prompts_with_correct_feature']}/{specificity['num_prompts']} | "
            f"{result['matched_separation']['mean_cosine_distance']:.3f} | "
            f"{result['matched_separation']['mean_top_k_jaccard']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Correct/contaminating mass:** activation-weighted shares among labeled top-k features, using explicit appliance keywords.",
            "- **Scenario purity:** correct / (correct + contaminating) scenario-specific activation mass.",
            "- **Problem cosine distance:** feature-vector distance between each pair of repair problems; higher means more separation.",
            "- **Reconstruction:** measured in the normalized space used during SAE training.",
            "",
            "Scenario specificity is a label-dependent diagnostic, not a causal metric. Generic labels do not count for either side, and mislabeled or polysemantic features can affect both correct and contamination scores.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--results-dir", default="demo/home_repair/output")
    parser.add_argument("--layers", type=int, nargs="+", default=demo._TRAINED_LAYERS)
    parser.add_argument("--threshold-offset", type=float, default=demo._HF_THRESHOLD_OFFSET)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--activations-file",
        help="Reuse a previously captured six-layer .npz instead of loading the model.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = _prompt_metadata()
    if args.activations_file:
        activation_path = Path(args.activations_file)
        print(f"Reusing activations from {activation_path}")
        with np.load(activation_path) as saved:
            activations = {
                layer: saved[f"residual_{layer}"].astype(np.float32) for layer in args.layers
            }
    else:
        metadata, activations = _capture_activations(args.model_name, args.layers)
        np.savez_compressed(
            results_dir / "six_layer_demo_activations.npz",
            **{f"residual_{layer}": values for layer, values in activations.items()},
        )

    for layer, values in activations.items():
        if values.ndim != 2 or values.shape[0] != len(metadata):
            raise ValueError(
                f"Layer {layer} activations have shape {values.shape}; "
                f"expected ({len(metadata)}, d_model)."
            )

    layer_results = []
    for layer in args.layers:
        print(f"\nEvaluating layer {layer}...")
        layer_results.append(
            _evaluate_layer(
                layer=layer,
                raw_activations=activations[layer],
                metadata=metadata,
                output_dir=output_dir,
                threshold_offset=args.threshold_offset,
                top_k=args.top_k,
                device=args.device,
                contrastive_map=_load_contrastive_map(output_dir, layer),
            )
        )

    report = {
        "model": args.model_name,
        "layers_evaluated": args.layers,
        "threshold_offset": args.threshold_offset,
        "top_k": args.top_k,
        "prompts": metadata,
        "layers": layer_results,
    }
    json_path = results_dir / "six_layer_sae_evaluation.json"
    markdown_path = results_dir / "six_layer_sae_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2))
    markdown_path.write_text(_render_markdown(report))
    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
