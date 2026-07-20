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

import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _analyze_layer(
    layer_key: str,
    sae_checkpoint: str,
    pairs_by_type: dict[str, list],
    acts_by_type: dict[str, tuple[np.ndarray, np.ndarray]],
    top_k: int,
    min_effect_size: float,
    min_activation: float,
    output_dir: Path,
) -> Path:
    """Analyze contrastive features for a single layer using pre-extracted activations.

    Args:
        layer_key: e.g. "residual_20" (used for logging).
        sae_checkpoint: Path to this layer's trained SAE.
        pairs_by_type: {contrast_type: [pairs]}.
        acts_by_type: {contrast_type: (anchor_arr, contrast_arr)} pre-stacked
            per-contrast-type activations for this layer only.
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
        anchor_arr, contrast_arr = acts_by_type[ct_value]
        n = len(ct_pairs)

        anchor_vecs = torch.from_numpy(anchor_arr).to(device=device, dtype=sae_dtype)
        contrast_vecs = torch.from_numpy(contrast_arr).to(device=device, dtype=sae_dtype)

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


def _open_layer_shards(
    layer_dir: Path,
) -> tuple[list[np.memmap], np.ndarray]:
    """Open all shard files in a layer dir as memmaps (no RAM materialization).

    Returns:
        (memmaps, cum_offsets) — ``cum_offsets[i]`` is the global row index of
        the first row in shard ``i``, and ``cum_offsets[-1]`` is the total row
        count across all shards.
    """
    shard_paths = sorted(layer_dir.glob("shard_*.npy"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npy files in {layer_dir}")
    memmaps = [np.load(p, mmap_mode="r") for p in shard_paths]
    cum = np.zeros(len(memmaps) + 1, dtype=np.int64)
    for i, m in enumerate(memmaps):
        cum[i + 1] = cum[i] + m.shape[0]
    return memmaps, cum


def _gather_rows(
    memmaps: list[np.memmap],
    cum_offsets: np.ndarray,
    global_indices: np.ndarray,
) -> np.ndarray:
    """Gather rows at ``global_indices`` from a list of virtually-concatenated memmaps.

    Only the requested rows are paged in from disk; the full array is never
    materialized in RAM.
    """
    d_model = memmaps[0].shape[1]
    total_rows = int(cum_offsets[-1])
    if len(global_indices) and (
        int(global_indices.min()) < 0 or int(global_indices.max()) >= total_rows
    ):
        raise IndexError(
            f"Activation row index out of range: requested "
            f"[{int(global_indices.min())}, {int(global_indices.max())}] but shards "
            f"hold {total_rows} rows. A shard file is likely missing or truncated, "
            "or this layer's shards don't match the layer the indices were built from."
        )
    out = np.empty((len(global_indices), d_model), dtype=memmaps[0].dtype)
    for shard_idx, m in enumerate(memmaps):
        lo, hi = int(cum_offsets[shard_idx]), int(cum_offsets[shard_idx + 1])
        mask = (global_indices >= lo) & (global_indices < hi)
        if mask.any():
            local = global_indices[mask] - lo
            out[mask] = m[local]
    return out


def _validate_step1_metadata(pairs: list, first_layer_dir: Path) -> None:
    """Validate Step 1 metadata + prompt ordering matches the loaded pairs."""
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


def _build_pair_indices(
    pairs: list,
) -> tuple[dict[str, list], dict[str, np.ndarray]]:
    """Group pair objects and their integer indices by contrast type."""
    pairs_by_type: dict[str, list] = defaultdict(list)
    pair_indices_by_type: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        pairs_by_type[pair.contrast_type].append(pair)
        pair_indices_by_type[pair.contrast_type].append(i)
    indices_arrays = {
        ct: np.asarray(idxs, dtype=np.int64) for ct, idxs in pair_indices_by_type.items()
    }
    return pairs_by_type, indices_arrays


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
    base_dir = Path(base_output_dir)
    layer_keys = [f"residual_{layer}" for layer in layers]

    first_layer_dir = base_dir / f"layer_{layers[0]}" / "activations"
    if not first_layer_dir.exists():
        raise FileNotFoundError(
            f"Step 1 activations not found for layer {layers[0]}. "
            f"Expected at: {first_layer_dir}\nRun step 1 first."
        )
    _validate_step1_metadata(pairs, first_layer_dir)

    pairs_by_type, pair_indices_by_type = _build_pair_indices(pairs)

    report_paths: dict[str, Path] = {}

    for layer, layer_key in zip(layers, layer_keys, strict=True):
        checkpoint = sae_checkpoints[layer_key]
        layer_act_dir = base_dir / f"layer_{layer}" / "activations"
        if not layer_act_dir.exists():
            raise FileNotFoundError(
                f"Step 1 activations not found for layer {layer}. "
                f"Expected at: {layer_act_dir}\nRun step 1 first."
            )

        print(f"\n  --- Layer {layer} ---")
        print(f"    SAE checkpoint: {checkpoint}")

        memmaps, cum_offsets = _open_layer_shards(layer_act_dir)
        print(
            f"    Memmapped {len(memmaps)} shards "
            f"({int(cum_offsets[-1])} rows × {memmaps[0].shape[1]} dims, "
            f"dtype={memmaps[0].dtype})"
        )

        acts_by_type: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ct_value, ct_pair_indices in pair_indices_by_type.items():
            anchor_global = 2 * ct_pair_indices
            contrast_global = 2 * ct_pair_indices + 1
            anchor_arr = _gather_rows(memmaps, cum_offsets, anchor_global)
            contrast_arr = _gather_rows(memmaps, cum_offsets, contrast_global)
            acts_by_type[ct_value] = (anchor_arr, contrast_arr)

        report_path = _analyze_layer(
            layer_key=layer_key,
            sae_checkpoint=checkpoint,
            pairs_by_type=pairs_by_type,
            acts_by_type=acts_by_type,
            top_k=top_k,
            min_effect_size=min_effect_size,
            min_activation=min_activation,
            output_dir=layer_act_dir,
        )
        report_paths[layer_key] = report_path

        del acts_by_type, memmaps, cum_offsets
        gc.collect()

    return report_paths
