"""
Step 3: Contrastive feature identification.

Identify which SAE features are decision-relevant using contrastive pairs.
For each contrastive pair, encode both the anchor and contrast prompts
through the SAE and compare which features activate differently.  Features
that consistently differ across many pairs of the same contrast type are
the decision-relevant features for that contrast.

Activations are loaded from the numpy shards produced by Step 1 (activation
extraction), so the subject model does not need to be loaded again.  Each
layer's SAE is applied independently to produce per-layer reports.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _analyze_layer(
    layer_key: str,
    sae_checkpoint: str,
    pairs_by_type: dict[str, list],
    all_acts_by_type: dict[str, list[dict[str, np.ndarray]]],
    top_k: int,
    min_effect_size: float,
    min_activation: float,
    output_dir: Path,
) -> Path:
    """Analyze contrastive features for a single layer using pre-extracted activations.

    Args:
        layer_key: e.g. "residual_20".
        sae_checkpoint: Path to this layer's trained SAE.
        pairs_by_type: {contrast_type: [pairs]}.
        all_acts_by_type: {contrast_type: [act_dicts]} — each dict has all layer keys.
        top_k: Number of top features per contrast type.
        min_effect_size: Minimum Cohen's d.
        min_activation: Minimum mean activation.
        output_dir: Directory to write contrastive_features.json.

    Returns:
        Path to the report file.
    """
    import torch

    from kiji_inspector.core.sae_core import JumpReLUSAE

    sae_path = Path(sae_checkpoint)
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(str(sae_path), device=device)
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype
    print(f"    Loaded SAE: d_model={sae.d_model}, d_sae={sae.d_sae}")

    results: dict[str, dict] = {}

    for ct_value, ct_pairs in pairs_by_type.items():
        ct_acts = all_acts_by_type[ct_value]
        n = len(ct_pairs)

        # Split into anchor (first n) and contrast (last n)
        anchor_acts = [ct_acts[i][layer_key] for i in range(n)]
        contrast_acts = [ct_acts[n + i][layer_key] for i in range(n)]

        anchor_vecs = torch.from_numpy(np.stack(anchor_acts)).to(device=device, dtype=sae_dtype)
        contrast_vecs = torch.from_numpy(np.stack(contrast_acts)).to(device=device, dtype=sae_dtype)

        # Normalize by the same RMS scale used during SAE training
        if sae.rms_scale is not None and sae.rms_scale > 0:
            anchor_vecs = anchor_vecs / sae.rms_scale
            contrast_vecs = contrast_vecs / sae.rms_scale

        with torch.no_grad():
            anchor_features = sae.encode(anchor_vecs)
            contrast_features = sae.encode(contrast_vecs)

        feature_diffs = (anchor_features - contrast_features).abs().mean(dim=0)
        anchor_mean = anchor_features.mean(dim=0)
        contrast_mean = contrast_features.mean(dim=0)
        anchor_var = anchor_features.var(dim=0)
        contrast_var = contrast_features.var(dim=0)
        n_a = anchor_features.shape[0]
        n_c = contrast_features.shape[0]

        # Cohen's d effect size
        pooled_std = torch.sqrt(
            ((n_a - 1) * anchor_var + (n_c - 1) * contrast_var) / max(n_a + n_c - 2, 1)
        )
        cohens_d = (anchor_mean - contrast_mean).abs() / (pooled_std + 1e-8)

        # Diagnostic: print stats for first contrast type to help debug 0-feature issues
        if len(results) == 0:
            max_d = cohens_d.max().item()
            max_act = max(anchor_mean.abs().max().item(), contrast_mean.abs().max().item())
            mean_act = (anchor_mean.abs().mean().item() + contrast_mean.abs().mean().item()) / 2
            n_above_effect = int((cohens_d >= min_effect_size).sum().item())
            n_above_act = int(
                ((anchor_mean.abs() > min_activation) | (contrast_mean.abs() > min_activation))
                .sum()
                .item()
            )
            print(f"    [debug] max Cohen's d: {max_d:.4f} (threshold: {min_effect_size})")
            print(
                f"    [debug] max activation: {max_act:.6f}, mean: {mean_act:.6f} (threshold: {min_activation})"
            )
            print(f"    [debug] features above effect size: {n_above_effect}/{cohens_d.shape[0]}")
            print(f"    [debug] features above min activation: {n_above_act}/{cohens_d.shape[0]}")

        # Filter: effect size >= threshold AND at least one side has meaningful activation
        effect_mask = cohens_d >= min_effect_size
        activation_mask = (anchor_mean.abs() > min_activation) | (
            contrast_mean.abs() > min_activation
        )
        valid_mask = effect_mask & activation_mask

        # Apply mask before top-K
        masked_diffs = feature_diffs.clone()
        masked_diffs[~valid_mask] = -1.0

        k_actual = min(top_k, int(valid_mask.sum().item()), feature_diffs.shape[0])
        if k_actual > 0:
            topk_vals, topk_indices = masked_diffs.topk(k_actual)
            keep = topk_vals > 0
            topk_vals = topk_vals[keep]
            topk_indices = topk_indices[keep]
        else:
            topk_vals = torch.tensor([])
            topk_indices = torch.tensor([], dtype=torch.long)

        feature_list = []
        for rank, (val, idx) in enumerate(
            zip(topk_vals.tolist(), topk_indices.tolist(), strict=True)
        ):
            idx = int(idx)
            feature_list.append(
                {
                    "rank": rank,
                    "feature_index": idx,
                    "mean_abs_diff": round(val, 6),
                    "anchor_mean_activation": round(anchor_mean[idx].item(), 6),
                    "contrast_mean_activation": round(contrast_mean[idx].item(), 6),
                    "cohens_d": round(cohens_d[idx].item(), 6),
                    "anchor_std": round(anchor_var[idx].sqrt().item(), 6),
                    "contrast_std": round(contrast_var[idx].sqrt().item(), 6),
                }
            )

        n_filtered_effect = int((~effect_mask).sum().item())
        n_filtered_activation = int((effect_mask & ~activation_mask).sum().item())

        results[ct_value] = {
            "num_pairs": n,
            "top_features": feature_list,
            "num_filtered_by_effect_size": n_filtered_effect,
            "num_filtered_by_min_activation": n_filtered_activation,
        }

        del anchor_vecs, contrast_vecs, anchor_features, contrast_features

    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute utilization summary
    all_feature_indices: list[int] = []
    for _ct_value, ct_info in results.items():
        for f in ct_info["top_features"]:
            all_feature_indices.append(f["feature_index"])

    unique_features = set(all_feature_indices)
    total_slots = len(all_feature_indices)
    feature_counts = Counter(all_feature_indices)
    multi_contrast = sum(1 for c in feature_counts.values() if c > 1)

    results["_summary"] = {
        "total_feature_slots": total_slots,
        "unique_features": len(unique_features),
        "dedup_ratio": round(len(unique_features) / max(total_slots, 1), 4),
        "features_in_multiple_contrasts": multi_contrast,
        "max_contrast_overlap": max(feature_counts.values()) if feature_counts else 0,
        "top_k_per_type": top_k,
        "min_effect_size": min_effect_size,
        "min_activation": min_activation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "contrastive_features.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"    Report: {report_path}")
    print(f"    Contrast types: {len(results) - 1}, unique features: {len(unique_features)}")
    print(f"    Dedup ratio: {results['_summary']['dedup_ratio']:.2%}")

    for ct_value, info in results.items():
        if ct_value.startswith("_"):
            continue
        n_feats = len(info["top_features"])
        top3 = info["top_features"][:3]
        top3_str = ", ".join(f"#{f['feature_index']}(d={f['cohens_d']:.2f})" for f in top3)
        print(f"    {ct_value}: {info['num_pairs']} pairs, {n_feats} features, top: {top3_str}")

    return report_path


def _load_shards_as_act_dicts(
    layer_keys: list[str],
    layer_dirs: dict[str, str],
    total_prompts: int,
) -> list[dict[str, np.ndarray]]:
    """Load shard files back into a list of per-prompt activation dicts."""
    # Load all shards per layer into a single array
    layer_arrays: dict[str, np.ndarray] = {}
    for lk in layer_keys:
        shards = []
        for shard_path in sorted(Path(layer_dirs[lk]).glob("shard_*.npy")):
            shards.append(np.load(shard_path))
        layer_arrays[lk] = np.concatenate(shards, axis=0)

    # Build per-prompt dicts
    result: list[dict[str, np.ndarray]] = []
    for i in range(total_prompts):
        result.append({lk: layer_arrays[lk][i] for lk in layer_keys})
    return result


def _load_and_regroup_activations(
    pairs: list,
    layer_keys: list[str],
    base_output_dir: Path,
) -> tuple[dict[str, list], dict[str, list[dict[str, np.ndarray]]]]:
    """Load Step 1 shards and regroup into the format ``_analyze_layer`` expects.

    Step 1 saves activations interleaved per pair:
        [anchor_0, contrast_0, anchor_1, contrast_1, ...]

    ``_analyze_layer`` expects activations grouped by contrast type with all
    anchors first then all contrasts:
        {ct: [anchor_0, anchor_3, ..., contrast_0, contrast_3, ...]}

    This function bridges the two orderings.

    Returns:
        (pairs_by_type, all_acts_by_type) ready for ``_analyze_layer``.
    """
    # Build layer_dirs from the base output directory
    layer_dirs: dict[str, str] = {}
    for lk in layer_keys:
        layer_num = lk.split("_", 1)[1]
        layer_dir = base_output_dir / f"layer_{layer_num}" / "activations"
        if not layer_dir.exists():
            raise FileNotFoundError(
                f"Step 1 activations not found for {lk}. "
                f"Expected at: {layer_dir}\n"
                "Run step 1 first to extract activations."
            )
        layer_dirs[lk] = str(layer_dir)

    # Load metadata to validate pair count
    first_layer_dir = Path(layer_dirs[layer_keys[0]])
    metadata_path = first_layer_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    total_tokens = metadata["total_tokens"]
    total_pairs_in_shards = metadata.get("total_pairs", total_tokens // 2)
    if total_pairs_in_shards != len(pairs):
        raise ValueError(
            f"Pair count mismatch: shards have {total_pairs_in_shards} pairs "
            f"but {len(pairs)} pairs were loaded. "
            "Ensure the same pairs directory is used for steps 1 and 3."
        )

    # Validate prompt ordering against prompts.json
    prompts_path = first_layer_dir / "prompts.json"
    if prompts_path.exists():
        with open(prompts_path) as f:
            saved_prompts = json.load(f)
        for i, pair in enumerate(pairs):
            if saved_prompts[2 * i] != pair.anchor_prompt:
                raise ValueError(
                    f"Prompt ordering mismatch at pair {i} (anchor). "
                    "The pairs may have been regenerated since step 1."
                )
            if saved_prompts[2 * i + 1] != pair.contrast_prompt:
                raise ValueError(
                    f"Prompt ordering mismatch at pair {i} (contrast). "
                    "The pairs may have been regenerated since step 1."
                )

    # Load all shards into a flat list of per-prompt activation dicts
    all_acts = _load_shards_as_act_dicts(layer_keys, layer_dirs, total_tokens)

    # Group pairs by contrast type
    pairs_by_type: dict[str, list] = defaultdict(list)
    pair_indices_by_type: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        pairs_by_type[pair.contrast_type].append(pair)
        pair_indices_by_type[pair.contrast_type].append(i)

    # Regroup activations: for each contrast type, all anchors first then all contrasts
    # (matching _analyze_layer's expectation: first n = anchors, next n = contrasts)
    all_acts_by_type: dict[str, list[dict[str, np.ndarray]]] = {}
    for ct_value, ct_pair_indices in pair_indices_by_type.items():
        ct_acts: list[dict[str, np.ndarray]] = []
        # Anchors first
        for pair_idx in ct_pair_indices:
            ct_acts.append(all_acts[2 * pair_idx])
        # Then contrasts
        for pair_idx in ct_pair_indices:
            ct_acts.append(all_acts[2 * pair_idx + 1])
        all_acts_by_type[ct_value] = ct_acts

    return pairs_by_type, all_acts_by_type


def identify_contrastive_features(
    pairs: list,
    sae_checkpoints: dict[str, str],
    layers: list[int],
    top_k: int,
    base_output_dir: str,
    min_effect_size: float = 0.3,
    min_activation: float = 0.01,
) -> dict[str, Path]:
    """Identify contrastive features for multiple layers.

    Loads pre-extracted activations from Step 1 numpy shards, then loops
    over layers — loading each SAE and computing contrastive features
    independently.

    Args:
        pairs: Contrastive pairs.
        sae_checkpoints: {layer_key: checkpoint_path} for each layer.
        layers: Transformer layer indices.
        top_k: Top features per contrast type.
        base_output_dir: Base output directory. Per-layer reports go to
            ``base_output_dir/layer_N/activations/contrastive_features.json``.
        min_effect_size: Minimum Cohen's d.
        min_activation: Minimum mean activation.

    Returns:
        Dict mapping layer_key to report path.
    """
    layer_keys = [f"residual_{layer}" for layer in layers]

    pairs_by_type, all_acts_by_type = _load_and_regroup_activations(
        pairs=pairs,
        layer_keys=layer_keys,
        base_output_dir=Path(base_output_dir),
    )

    report_paths: dict[str, Path] = {}

    for layer, layer_key in zip(layers, layer_keys, strict=True):
        checkpoint = sae_checkpoints[layer_key]
        output_dir = Path(base_output_dir) / f"layer_{layer}" / "activations"

        print(f"\n  --- Layer {layer} ---")
        print(f"    SAE checkpoint: {checkpoint}")

        report_path = _analyze_layer(
            layer_key=layer_key,
            sae_checkpoint=checkpoint,
            pairs_by_type=pairs_by_type,
            all_acts_by_type=all_acts_by_type,
            top_k=top_k,
            min_effect_size=min_effect_size,
            min_activation=min_activation,
            output_dir=output_dir,
        )
        report_paths[layer_key] = report_path

    return report_paths
