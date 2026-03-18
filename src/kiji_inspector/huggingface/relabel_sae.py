#!/usr/bin/env python3
"""Download an SAE repo from HF, re-run feature labeling (step 5c), and re-upload.

Since activation shards (.npy) are not stored on HF, this script reconstructs
the feature_examples from the existing feature_descriptions.json, then re-runs
only 5c (LLM labeling) + 5d (report generation) for each layer.

Usage:
    python -m kiji_inspector.huggingface.relabel_sae <repo_id> \
        --judging-model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

Example:
    python -m kiji_inspector.huggingface.relabel_sae \
        575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --judging-model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
        --tp-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from kiji_inspector.analysis.feature_interpreter import (
    generate_explanation_report,
    label_features_via_llm,
)
from kiji_inspector.huggingface.upload_sae import _build_model_card, _summarise_layer


def _reconstruct_feature_examples(
    feature_descriptions_path: Path,
) -> dict[int, dict]:
    """Reconstruct feature_examples from an existing feature_descriptions.json.

    The activation shards (.npy) are not uploaded to HF, so we cannot re-run
    steps 5a/5b.  Instead we reconstruct the feature_examples dict from the
    previously saved descriptions which already contain top/bottom example
    prompts and activation statistics.

    Activation values per example are not stored in feature_descriptions.json,
    so we synthesise decreasing placeholder values (the LLM labeling prompt
    shows them for context but the actual magnitudes are secondary to the
    prompt text).
    """
    with open(feature_descriptions_path) as f:
        descriptions = json.load(f)

    feature_examples: dict[int, dict] = {}
    for feat_idx_str, info in descriptions.items():
        feat_idx = int(feat_idx_str)
        max_act = info.get("max_activation", 1.0)
        mean_act = info.get("mean_activation", 0.1)

        # Reconstruct top examples with synthetic decreasing activations
        top_prompts = info.get("top_examples", [])
        top = []
        for i, prompt in enumerate(top_prompts):
            # Linearly interpolate from max_activation down to mean_activation
            frac = i / max(len(top_prompts) - 1, 1)
            act = max_act - frac * (max_act - mean_act)
            top.append({"prompt": prompt, "activation": round(act, 6)})

        # Reconstruct bottom examples with near-zero activations
        bottom_prompts = info.get("bottom_examples", [])
        bottom = [{"prompt": prompt, "activation": 0.0} for prompt in bottom_prompts]

        feature_examples[feat_idx] = {
            "top": top,
            "bottom": bottom,
            "mean_activation": mean_act,
            "max_activation": max_act,
            "frac_nonzero": info.get("frac_nonzero", 0.0),
        }

    return feature_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download SAE repo from HF, re-run feature labeling, and re-upload.",
    )
    p.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face model repo ID (e.g. 575-lab/kiji-inspector-model).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to download repo into. Defaults to /tmp/relabel_sae/<repo_name>.",
    )
    p.add_argument(
        "--judging-model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="HuggingFace model for feature labeling via vLLM (default: Qwen3-VL-235B).",
    )
    p.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM (default: 4).",
    )
    p.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size — number of model copies on separate GPUs (default: 1).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max sequence length for vLLM (default: 16384).",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch/revision to download from and push to (default: main).",
    )
    p.add_argument(
        "--commit-message",
        type=str,
        default="Re-label feature descriptions with improved parsing",
        help="Commit message for the HF upload.",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip the upload step (useful for local testing).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_name = args.repo_id.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"/tmp/relabel_sae/{repo_name}")

    # ---- Step 1: Download repo from HF ----
    print(f"Downloading {args.repo_id} to {output_dir}...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(output_dir),
    )
    print(f"  Download complete: {output_dir}")

    # ---- Step 2: Discover layers ----
    layer_dirs = sorted(output_dir.glob("layer_*"))
    if not layer_dirs:
        print(f"ERROR: no layer_* directories found under {output_dir}")
        sys.exit(1)
    print(f"\nFound {len(layer_dirs)} layers: {', '.join(d.name for d in layer_dirs)}")

    # ---- Step 3: Re-run labeling for each layer ----
    for layer_dir in layer_dirs:
        activations_dir = layer_dir / "activations"

        # Check for existing feature descriptions to reconstruct examples from
        desc_path = activations_dir / "feature_descriptions.json"
        if not desc_path.exists():
            print(f"\n  Skipping {layer_dir.name}: no feature_descriptions.json found")
            continue

        # Check for contrastive features (needed for report generation)
        contrastive_path = activations_dir / "contrastive_features.json"
        if not contrastive_path.exists():
            print(f"\n  Skipping {layer_dir.name}: no contrastive_features.json found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Re-labeling {layer_dir.name}")
        print(f"{'=' * 60}")

        # Reconstruct feature_examples from existing descriptions
        # (activation shards are not stored on HF)
        print(f"  Reconstructing feature examples from {desc_path.name}...")
        feature_examples = _reconstruct_feature_examples(desc_path)
        print(f"  {len(feature_examples)} features to re-label")

        # 5c: Label features via LLM
        print("\n  [5c] Labeling features via LLM...")
        feature_labels = label_features_via_llm(
            feature_examples=feature_examples,
            judging_model=args.judging_model,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            max_model_len=args.max_model_len,
            output_dir=str(activations_dir),
        )
        print(f"    {len(feature_labels)} features labeled")

        # 5d: Generate report
        print("\n  [5d] Generating report...")
        report_path = generate_explanation_report(
            contrastive_features_path=str(contrastive_path),
            feature_examples=feature_examples,
            feature_labels=feature_labels,
            output_dir=str(activations_dir),
        )
        print(f"    Report saved: {report_path}")

    # ---- Step 4: Re-upload to HF ----
    if args.skip_upload:
        print(f"\n  Skipping upload (--skip-upload). Results in: {output_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Uploading to {args.repo_id}")
    print(f"{'=' * 60}")

    layer_names = [d.name for d in layer_dirs]
    layer_summaries = {}
    for layer_dir in layer_dirs:
        layer_summaries[layer_dir.name] = _summarise_layer(layer_dir)

    card = _build_model_card(args.repo_id, layer_names, layer_summaries)
    card_path = output_dir / "README.md"
    card_path.write_text(card)

    api = HfApi()
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=["layer_*/**", "README.md"],
        ignore_patterns=["**/*.npy", "**/step_*.pt"],
        delete_patterns=["*"],
    )

    print(f"\nDone. Updated {len(layer_dirs)} layers at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
