"""
Step 4: Contrastive feature identification.

Identify which SAE features are decision-relevant using contrastive pairs.
For each contrastive pair, encode both the anchor and contrast prompts
through the SAE and compare which features activate differently.  Features
that consistently differ across many pairs of the same contrast type are
the decision-relevant features for that contrast.

When multiple layers are specified, the subject model is loaded once and
activations for all layers are extracted in a single pass.  Each layer's
SAE is then applied independently to produce per-layer reports.
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

    from sae.model import JumpReLUSAE

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


def identify_contrastive_features(
    pairs: list,
    sae_checkpoints: dict[str, str],
    subject_model: str,
    layers: list[int],
    batch_size: int,
    top_k: int,
    base_output_dir: str,
    min_effect_size: float = 0.3,
    min_activation: float = 0.01,
    scenarios_meta: dict | None = None,
    backend: str = "vllm",
    dp_size: int = 1,
) -> dict[str, Path]:
    """Identify contrastive features for multiple layers.

    Loads the subject model once, extracts activations for all layers in a
    single pass, then loops over layers — loading each SAE and computing
    contrastive features independently.

    Args:
        pairs: Contrastive pairs.
        sae_checkpoints: {layer_key: checkpoint_path} for each layer.
        subject_model: HuggingFace model ID.
        layers: Transformer layer indices.
        batch_size: GPU batch size for extraction.
        top_k: Top features per contrast type.
        base_output_dir: Base output directory. Per-layer reports go to
            ``base_output_dir/layer_N/activations/contrastive_features.json``.
        min_effect_size: Minimum Cohen's d.
        min_activation: Minimum mean activation.
        scenarios_meta: {scenario_name: ScenarioConfig}.
        backend: ``"vllm"`` or ``"hf"``.
        dp_size: Number of data-parallel GPUs for extraction.

    Returns:
        Dict mapping layer_key to report path.
    """
    import tempfile

    from tqdm import tqdm

    from data.scenario import default_scenario
    from extraction.extractor import build_agent_prompt

    _scenarios_meta = scenarios_meta or {}
    _default_scenario = default_scenario()

    layer_keys = [f"residual_{layer}" for layer in layers]

    # Build tokenizer (lightweight — no model loaded in main process for DP)
    if dp_size > 1 and backend == "vllm":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(subject_model, trust_remote_code=True)
    else:
        from extraction import create_extractor

        extractor = create_extractor(
            backend=backend,
            model_name=subject_model,
            layers=layers,
            token_positions="decision",
        )
        tokenizer = extractor.tokenizer

    # Group pairs by contrast type
    pairs_by_type: dict[str, list] = defaultdict(list)
    for pair in pairs:
        pairs_by_type[pair.contrast_type].append(pair)

    # Build all prompts, tracking which contrast type each belongs to
    all_prompts: list[str] = []
    prompt_ct_labels: list[str] = []  # contrast type for each prompt
    ct_prompt_counts: dict[str, int] = {}  # total prompts per contrast type

    for ct_value, ct_pairs in pairs_by_type.items():
        ct_start = len(all_prompts)
        # Anchors first, then contrasts — _analyze_layer expects this order
        for pair in ct_pairs:
            scenario = _scenarios_meta.get(pair.scenario_name, _default_scenario)
            all_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.anchor_prompt,
                    tokenizer=tokenizer,
                )
            )
        for pair in ct_pairs:
            scenario = _scenarios_meta.get(pair.scenario_name, _default_scenario)
            all_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.contrast_prompt,
                    tokenizer=tokenizer,
                )
            )
        ct_prompt_counts[ct_value] = len(all_prompts) - ct_start
        prompt_ct_labels.extend([ct_value] * ct_prompt_counts[ct_value])

    total_prompts = len(all_prompts)

    # Extract activations
    if dp_size > 1 and backend == "vllm":
        from extraction.vllm_activation_extractor import (
            run_dp_extraction_to_shards,
        )

        # Write to temp dir, then load back
        with tempfile.TemporaryDirectory(prefix="kiji_step3_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            layer_dirs: dict[str, str] = {}
            for lk in layer_keys:
                layer_num = lk.split("_", 1)[1]
                ldir = tmp_path / f"layer_{layer_num}"
                ldir.mkdir(parents=True)
                layer_dirs[lk] = str(ldir)

            config_kwargs = {
                "model_name": subject_model,
                "layers": layers,
                "token_positions": "decision",
                "gpu_memory_utilization": 0.90,
                "max_model_len": 8192,
                "trust_remote_code": True,
            }
            run_dp_extraction_to_shards(
                prompts=all_prompts,
                dp_size=dp_size,
                config_kwargs=config_kwargs,
                batch_size=batch_size,
                layer_keys=layer_keys,
                layer_dirs=layer_dirs,
                shard_size=500_000,
            )

            # Load shards back into per-prompt activation dicts
            all_acts = _load_shards_as_act_dicts(layer_keys, layer_dirs, total_prompts)
    else:
        # Single-GPU extraction
        all_acts: list[dict[str, np.ndarray]] = []
        pbar = tqdm(total=total_prompts, desc="Extracting pair activations", unit="prompt")
        for bi in range(0, total_prompts, batch_size):
            batch_prompts = all_prompts[bi : bi + batch_size]
            all_acts.extend(extractor.extract_batch(batch_prompts, batch_size=len(batch_prompts)))
            pbar.update(len(batch_prompts))
        pbar.close()
        extractor.cleanup()

    # Redistribute activations into per-contrast-type groups
    all_acts_by_type: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    for i, ct_label in enumerate(prompt_ct_labels):
        all_acts_by_type[ct_label].append(all_acts[i])

    # Now loop over layers — each gets its own SAE and report
    report_paths: dict[str, Path] = {}
    layer_keys = [f"residual_{layer}" for layer in layers]

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
