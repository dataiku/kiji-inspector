#!/usr/bin/env python3
"""Download contrastive pairs from a Hugging Face dataset to local parquet shards.

Usage:
    python src/huggingface/download_pairs.py <repo_id> [--pairs-dir output/pairs]

Example:
    python src/huggingface/download_pairs.py myorg/contrastive-pairs
    python src/huggingface/download_pairs.py myorg/contrastive-pairs --pairs-dir output/pairs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from data.contrastive_dataset import ContrastiveDataset, ContrastivePair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download contrastive pairs from a Hugging Face dataset.",
    )
    p.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face dataset repo ID (e.g. myorg/contrastive-pairs).",
    )
    p.add_argument(
        "--pairs-dir",
        type=str,
        default="output/pairs",
        help="Local directory to save parquet shards (default: output/pairs).",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch/revision to download from (default: main).",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=50_000,
        help="Rows per parquet shard (default: 50000).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Downloading dataset from https://huggingface.co/datasets/{args.repo_id}...")
    ds = load_dataset(args.repo_id, revision=args.revision, split="train")
    print(f"  Downloaded {len(ds)} rows")

    pairs = [
        ContrastivePair(
            pair_id=row["pair_id"],
            anchor_prompt=row["anchor_prompt"],
            anchor_tool=row["anchor_tool"],
            contrast_prompt=row["contrast_prompt"],
            contrast_tool=row["contrast_tool"],
            shared_intent=row["shared_intent"],
            semantic_similarity=row["semantic_similarity"],
            contrast_type=row["contrast_type"],
            distinguishing_signal=row["distinguishing_signal"],
            scenario_name=row.get("scenario_name", ""),
        )
        for row in ds
    ]

    cd = ContrastiveDataset(pairs=pairs)

    pairs_path = Path(args.pairs_dir)
    print(f"\nSaving {len(pairs)} pairs to {pairs_path}...")
    written = cd.to_parquet(pairs_path, shard_size=args.shard_size)
    print(f"  Wrote {len(written)} shard(s)")
    print("Done.")


if __name__ == "__main__":
    main()
