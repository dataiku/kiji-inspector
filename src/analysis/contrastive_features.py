"""
Step 4: Contrastive feature identification.

Identify which SAE features are decision-relevant using contrastive pairs.
For each contrastive pair, encode both the anchor and contrast prompts
through the SAE and compare which features activate differently.  Features
that consistently differ across many pairs of the same contrast type are
the decision-relevant features for that contrast.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def identify_contrastive_features(
    pairs: list,
    sae_checkpoint: str,
    nemotron_model: str,
    layers: list[int],
    layer_key: str,
    batch_size: int,
    top_k: int,
    output_dir: str,
    min_effect_size: float = 0.3,
    min_activation: float = 0.01,
    scenarios_meta: dict | None = None,
) -> Path:
    """Identify which SAE features are decision-relevant using contrastive pairs.

    For each contrastive pair, encode both the anchor and contrast prompts
    through the SAE and compare which features activate differently.  Features
    that consistently differ across many pairs of the same contrast type are
    the decision-relevant features for that contrast.
    """
    import torch

    from data.scenario import default_scenario
    from extraction.activation_extractor import ActivationConfig, ActivationExtractor
    from extraction.extractor import build_agent_prompt
    from sae.model import JumpReLUSAE

    # Load the trained SAE
    sae_path = Path(sae_checkpoint)
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(str(sae_path), device=device)
    sae.eval()
    print(f"  Loaded SAE: d_model={sae.d_model}, d_sae={sae.d_sae}")

    # Load Nemotron for live extraction of contrastive pairs
    config = ActivationConfig(
        model_name=nemotron_model,
        layers=layers,
        dtype=torch.bfloat16,
    )
    extractor = ActivationExtractor(config=config)

    # Build scenario lookup for per-pair prompt construction
    _scenarios_meta = scenarios_meta or {}
    _default_scenario = default_scenario()

    # Group pairs by contrast type
    pairs_by_type: dict[str, list] = defaultdict(list)
    for pair in pairs:
        pairs_by_type[pair.contrast_type].append(pair)

    results: dict[str, dict] = {}
    sae_dtype = next(sae.parameters()).dtype
    from tqdm import tqdm

    pbar = tqdm(total=len(pairs), desc="Extracting pair activations", unit="pair")

    for ct_value, ct_pairs in pairs_by_type.items():
        # Build all prompts for this contrast type
        anchor_prompts = []
        contrast_prompts = []
        for pair in ct_pairs:
            scenario = _scenarios_meta.get(pair.scenario_name, _default_scenario)
            anchor_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.anchor_prompt,
                    model_type="nemotron",
                )
            )
            contrast_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.contrast_prompt,
                    model_type="nemotron",
                )
            )

        # Extract in batches, accumulating results
        all_prompts = anchor_prompts + contrast_prompts
        all_acts: list[dict[str, np.ndarray]] = []
        for bi in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[bi : bi + batch_size]
            all_acts.extend(extractor.extract_batch(batch_prompts, batch_size=len(batch_prompts)))

        pbar.update(len(ct_pairs))

        n = len(ct_pairs)
        anchor_acts = [all_acts[i][layer_key] for i in range(n)]
        contrast_acts = [all_acts[n + i][layer_key] for i in range(n)]

        anchor_vecs = torch.from_numpy(np.stack(anchor_acts)).to(device=device, dtype=sae_dtype)
        contrast_vecs = torch.from_numpy(np.stack(contrast_acts)).to(device=device, dtype=sae_dtype)

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
            # Keep only features that passed the mask
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

    pbar.close()

    extractor.cleanup()

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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "contrastive_features.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nContrastive feature report saved to {report_path}")
    print(f"  Contrast types analyzed: {len(results) - 1}")  # exclude _summary
    print(f"  Max top-{top_k} features per type (filtered by Cohen's d >= {min_effect_size})")
    print(f"  Total feature slots: {total_slots}")
    print(f"  Unique features: {len(unique_features)}")
    print(f"  Features in multiple contrasts: {multi_contrast}")
    print(f"  Dedup ratio: {results['_summary']['dedup_ratio']:.2%}")

    for ct_value, info in results.items():
        if ct_value.startswith("_"):
            continue
        n_feats = len(info["top_features"])
        top3 = info["top_features"][:3]
        top3_str = ", ".join(f"#{f['feature_index']}(d={f['cohens_d']:.2f})" for f in top3)
        print(f"  {ct_value}: {info['num_pairs']} pairs, {n_feats} features, top: {top3_str}")

    return report_path
