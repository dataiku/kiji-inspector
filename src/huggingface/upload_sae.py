#!/usr/bin/env python3
"""Upload a trained SAE checkpoint with feature descriptions to Hugging Face.

Uploads the SAE checkpoint, feature_descriptions.json, and
contrastive_features.json to a private Hugging Face model repo.

Usage:
    python -m huggingface.upload_sae <repo_id> [--output-dir output/activations]

Example:
    python -m huggingface.upload_sae myorg/contrastive-sae
    python -m huggingface.upload_sae myorg/contrastive-sae --output-dir output/activations
    python -m huggingface.upload_sae myorg/contrastive-sae --checkpoint output/sae_checkpoints/sae_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi


def _build_model_card(
    repo_id: str,
    checkpoint_path: str,
    feature_descriptions: dict | None = None,
    contrastive_features: dict | None = None,
) -> str:
    """Generate a model card (README.md) from the SAE checkpoint and feature metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    d_model = config.get("d_model", "unknown")
    d_sae = config.get("d_sae", "unknown")
    dtype = config.get("dtype", "unknown")
    bandwidth = config.get("bandwidth", "unknown")

    state = checkpoint.get("model_state_dict", {})
    num_params = sum(v.numel() for v in state.values()) if state else "unknown"
    if isinstance(num_params, int):
        num_params_str = f"{num_params:,}"
    else:
        num_params_str = str(num_params)

    config_table = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(config.items()))

    # Feature summary
    features_section = ""
    if feature_descriptions:
        n_features = len(feature_descriptions)
        confidence_counts: dict[str, int] = {}
        for desc in feature_descriptions.values():
            conf = desc.get("confidence", "unknown")
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        confidence_table = "\n".join(
            f"| {conf} | {count} |" for conf, count in sorted(confidence_counts.items())
        )

        # Show top features by confidence
        high_conf = [
            (idx, desc)
            for idx, desc in feature_descriptions.items()
            if desc.get("confidence") == "high"
        ]
        example_features = ""
        if high_conf:
            rows = []
            for idx, desc in high_conf[:10]:
                label = desc.get("label", "unlabeled")
                description = desc.get("description", "")
                if len(description) > 80:
                    description = description[:77] + "..."
                rows.append(f"| {idx} | {label} | {description} |")
            example_features = f"""
### High-confidence features (top 10)

| Feature | Label | Description |
|---------|-------|-------------|
{chr(10).join(rows)}
"""

        features_section = f"""
## Interpreted features

- **Total features described**: {n_features}

| Confidence | Count |
|------------|-------|
{confidence_table}
{example_features}"""

    # Contrastive features summary
    contrastive_section = ""
    if contrastive_features:
        contrast_types = [k for k in contrastive_features if not k.startswith("_")]
        meta = contrastive_features.get("_meta", {})
        top_k = meta.get("top_k", "unknown")

        ct_rows = []
        for ct in sorted(contrast_types):
            info = contrastive_features[ct]
            n_pairs = info.get("num_pairs", "?")
            n_feats = len(info.get("top_features", []))
            ct_rows.append(f"| {ct} | {n_pairs:,} | {n_feats} |")

        contrastive_section = f"""
## Contrastive features

Top-K features per contrast type: {top_k}

| Contrast type | Pairs | Features |
|---------------|-------|----------|
{chr(10).join(ct_rows)}
"""

    return f"""---
license: apache-2.0
tags:
  - sparse-autoencoder
  - sae
  - mechanistic-interpretability
  - tool-selection
---

# {repo_id.split("/")[-1]}

JumpReLU Sparse Autoencoder (SAE) trained on contrastive activation data for mechanistic interpretability of tool-selection decisions.

## Architecture

```
input x (d_model={d_model})
    |
    +-> W_enc @ (x - b_dec) + b_enc -> JumpReLU(-, theta) -> features (d_sae={d_sae})
    |
    +-> W_dec @ features + b_dec -> reconstruction (d_model={d_model})
```

- **Type**: JumpReLU SAE with learnable per-feature thresholds
- **Parameters**: {num_params_str}
- **dtype**: {dtype}
- **d_model**: {d_model}
- **d_sae**: {d_sae}
- **bandwidth**: {bandwidth}

## Config

| Key | Value |
|-----|-------|
{config_table}
{features_section}{contrastive_section}
## Usage

```python
from sae.model import JumpReLUSAE

sae = JumpReLUSAE.from_pretrained("sae_final.pt", device="cuda")
features = sae.encode(activations)
reconstruction = sae.decode(features)
```

## Files

| File | Description |
|------|-------------|
| `sae_final.pt` | SAE checkpoint (weights + config) |
| `feature_descriptions.json` | Per-feature labels, descriptions, confidence, and examples |
| `contrastive_features.json` | Per-contrast-type ranked features with effect sizes |

Generated by [kiji-inspector](https://github.com/dataiku/kiji-inspector).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload a trained SAE with feature descriptions to Hugging Face.",
    )
    p.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face model repo ID (e.g. myorg/contrastive-sae).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAE checkpoint (default: auto-detect in output-dir).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/activations",
        help="Pipeline output directory containing feature JSONs (default: output/activations).",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch/revision to push to (default: main).",
    )
    p.add_argument(
        "--commit-message",
        type=str,
        default="Upload SAE checkpoint and feature descriptions",
        help="Commit message for the push.",
    )
    return p.parse_args()


def _find_checkpoint(output_dir: str, explicit: str | None) -> Path:
    """Resolve the SAE checkpoint path."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        print(f"ERROR: checkpoint not found at {p}")
        sys.exit(1)

    candidates = [
        Path(output_dir).parent / "sae_checkpoints" / "sae_final.pt",
        Path(output_dir) / "sae_final.pt",
        Path("output/sae_checkpoints/sae_final.pt"),
    ]
    for p in candidates:
        if p.is_file():
            return p

    print("ERROR: SAE checkpoint not found. Searched:")
    for p in candidates:
        print(f"  {p}")
    print("Use --checkpoint to specify the path explicitly.")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    checkpoint_path = _find_checkpoint(args.output_dir, args.checkpoint)

    # Load feature descriptions (optional)
    feature_descriptions = None
    desc_path = output_dir / "feature_descriptions.json"
    if desc_path.is_file():
        with open(desc_path) as f:
            feature_descriptions = json.load(f)
        print(f"  Loaded {len(feature_descriptions)} feature descriptions from {desc_path}")
    else:
        print(f"  Warning: {desc_path} not found, skipping feature descriptions")

    # Load contrastive features (optional)
    contrastive_features = None
    cf_path = output_dir / "contrastive_features.json"
    if cf_path.is_file():
        with open(cf_path) as f:
            contrastive_features = json.load(f)
        contrast_types = [k for k in contrastive_features if not k.startswith("_")]
        print(f"  Loaded {len(contrast_types)} contrast types from {cf_path}")
    else:
        print(f"  Warning: {cf_path} not found, skipping contrastive features")

    api = HfApi()

    print(f"\nCreating/verifying repo {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    print("Generating model card...")
    card = _build_model_card(
        args.repo_id,
        str(checkpoint_path),
        feature_descriptions=feature_descriptions,
        contrastive_features=contrastive_features,
    )

    # Upload all files
    files_to_upload = [
        (str(checkpoint_path), checkpoint_path.name, "SAE checkpoint"),
    ]
    if desc_path.is_file():
        files_to_upload.append(
            (str(desc_path), "feature_descriptions.json", "feature descriptions")
        )
    if cf_path.is_file():
        files_to_upload.append((str(cf_path), "contrastive_features.json", "contrastive features"))

    print(f"\nUploading to https://huggingface.co/{args.repo_id}...")
    for local_path, repo_path, label in files_to_upload:
        print(f"  Uploading {label}: {local_path} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            commit_message=f"{args.commit_message}: {label}",
        )

    print("  Uploading model card: README.md")
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Update model card",
    )

    print("\nDone. Uploaded:")
    for _, repo_path, label in files_to_upload:
        print(f"  {repo_path} ({label})")
    print("  README.md (model card)")


if __name__ == "__main__":
    main()
