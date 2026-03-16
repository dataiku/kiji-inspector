#!/usr/bin/env python3
"""
Print comprehensive statistics for a directory of contrastive pair shards.

Usage:
    uv run python -m data.pair_stats output/pairs
    uv run python -m data.pair_stats output/pairs --show-examples 3
    uv run python -m data.pair_stats output/pairs --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from kiji_inspector.data.contrastive_dataset import ContrastiveDataset


def compute_stats(dataset: ContrastiveDataset) -> dict:
    """Compute comprehensive statistics from a ContrastiveDataset."""
    pairs = dataset.pairs
    n = len(pairs)
    if n == 0:
        return {"total_pairs": 0}

    # -- Scenario distribution --
    scenario_counts = Counter(p.scenario_name for p in pairs)

    # -- Contrast type distribution (global and per-scenario) --
    contrast_counts = Counter(p.contrast_type for p in pairs)
    contrast_by_scenario: dict[str, dict[str, int]] = {}
    for p in pairs:
        contrast_by_scenario.setdefault(p.scenario_name, Counter())
        contrast_by_scenario[p.scenario_name][p.contrast_type] += 1

    # -- Tool distribution --
    anchor_tool_counts = Counter(p.anchor_tool for p in pairs)
    contrast_tool_counts = Counter(p.contrast_tool for p in pairs)
    all_tool_counts = anchor_tool_counts + contrast_tool_counts
    tool_pair_counts = Counter((p.anchor_tool, p.contrast_tool) for p in pairs)

    # -- Prompt length statistics --
    anchor_lengths = [len(p.anchor_prompt) for p in pairs]
    contrast_lengths = [len(p.contrast_prompt) for p in pairs]
    all_lengths = anchor_lengths + contrast_lengths

    def _length_stats(lengths: list[int]) -> dict:
        lengths_sorted = sorted(lengths)
        n_l = len(lengths_sorted)
        return {
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
            "mean": round(sum(lengths_sorted) / n_l, 1),
            "median": lengths_sorted[n_l // 2],
            "p10": lengths_sorted[int(n_l * 0.1)],
            "p90": lengths_sorted[int(n_l * 0.9)],
        }

    # -- Semantic similarity statistics --
    similarities = [p.semantic_similarity for p in pairs]
    sim_mean = sum(similarities) / n
    sim_sorted = sorted(similarities)

    # -- Unique prompts --
    all_prompts = set()
    for p in pairs:
        all_prompts.add(p.anchor_prompt)
        all_prompts.add(p.contrast_prompt)

    return {
        "total_pairs": n,
        "total_prompts": n * 2,
        "unique_prompts": len(all_prompts),
        "duplicate_prompts": n * 2 - len(all_prompts),
        "scenarios": {
            "count": len(scenario_counts),
            "distribution": dict(scenario_counts.most_common()),
        },
        "contrast_types": {
            "count": len(contrast_counts),
            "distribution": dict(contrast_counts.most_common()),
            "by_scenario": {
                sc: dict(sorted(cts.items())) for sc, cts in sorted(contrast_by_scenario.items())
            },
        },
        "tools": {
            "unique_tools": len(all_tool_counts),
            "as_anchor": dict(anchor_tool_counts.most_common()),
            "as_contrast": dict(contrast_tool_counts.most_common()),
            "combined": dict(all_tool_counts.most_common()),
            "top_tool_pairs": [
                {"anchor": a, "contrast": c, "count": cnt}
                for (a, c), cnt in tool_pair_counts.most_common(20)
            ],
        },
        "prompt_lengths": {
            "anchor": _length_stats(anchor_lengths),
            "contrast": _length_stats(contrast_lengths),
            "all": _length_stats(all_lengths),
        },
        "semantic_similarity": {
            "mean": round(sim_mean, 4),
            "min": round(sim_sorted[0], 4),
            "max": round(sim_sorted[-1], 4),
            "median": round(sim_sorted[n // 2], 4),
        },
    }


def print_stats(stats: dict, pairs: list | None = None, show_examples: int = 0) -> None:
    """Pretty-print statistics to stdout."""
    n = stats["total_pairs"]
    if n == 0:
        print("No pairs found.")
        return

    print(f"\n{'=' * 70}")
    print("  Contrastive Pair Statistics")
    print(f"{'=' * 70}")

    print(f"\n  Total pairs       : {n:,}")
    print(f"  Total prompts     : {stats['total_prompts']:,}")
    print(f"  Unique prompts    : {stats['unique_prompts']:,}")
    if stats["duplicate_prompts"] > 0:
        print(f"  Duplicate prompts : {stats['duplicate_prompts']:,}")

    # -- Scenarios --
    sc = stats["scenarios"]
    print(f"\n  Scenarios ({sc['count']}):")
    for name, count in sc["distribution"].items():
        pct = count / n * 100
        print(f"    {name:30s} {count:>7,}  ({pct:5.1f}%)")

    # -- Contrast types --
    ct = stats["contrast_types"]
    print(f"\n  Contrast types ({ct['count']}):")
    for name, count in ct["distribution"].items():
        pct = count / n * 100
        print(f"    {name:40s} {count:>7,}  ({pct:5.1f}%)")

    # -- Contrast types by scenario --
    if len(ct["by_scenario"]) > 1:
        print("\n  Contrast types by scenario:")
        for sc_name, cts in ct["by_scenario"].items():
            sc_total = sum(cts.values())
            print(f"    {sc_name} ({sc_total:,} pairs):")
            for ct_name, ct_count in sorted(cts.items(), key=lambda x: -x[1]):
                print(f"      {ct_name:38s} {ct_count:>7,}")

    # -- Tools --
    tools = stats["tools"]
    print(f"\n  Tools ({tools['unique_tools']} unique):")
    for name, count in tools["combined"].items():
        print(f"    {name:30s} {count:>7,}")

    print("\n  Top tool pairs (anchor -> contrast):")
    for tp in tools["top_tool_pairs"][:10]:
        print(f"    {tp['anchor']:25s} -> {tp['contrast']:25s} {tp['count']:>5,}")

    # -- Prompt lengths --
    pl = stats["prompt_lengths"]["all"]
    print("\n  Prompt lengths (chars):")
    print(
        f"    Min: {pl['min']:,}  P10: {pl['p10']:,}  Median: {pl['median']:,}  "
        f"Mean: {pl['mean']:,}  P90: {pl['p90']:,}  Max: {pl['max']:,}"
    )

    # -- Semantic similarity --
    ss = stats["semantic_similarity"]
    print("\n  Semantic similarity:")
    print(
        f"    Min: {ss['min']:.4f}  Median: {ss['median']:.4f}  "
        f"Mean: {ss['mean']:.4f}  Max: {ss['max']:.4f}"
    )

    # -- Example pairs --
    if show_examples > 0 and pairs:
        import random

        samples = random.sample(pairs, min(show_examples, len(pairs)))
        print(f"\n  Example pairs ({len(samples)}):")
        for i, p in enumerate(samples, 1):
            print(f"\n  [{i}] {p.contrast_type} ({p.scenario_name})")
            print(f"      Anchor   ({p.anchor_tool}):   {p.anchor_prompt[:120]}")
            print(f"      Contrast ({p.contrast_tool}): {p.contrast_prompt[:120]}")
            print(f"      Intent: {p.shared_intent[:120]}")
            print(f"      Signal: {p.distinguishing_signal[:120]}")

    print(f"\n{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print statistics for contrastive pair shards.",
    )
    parser.add_argument(
        "pairs_dir",
        type=str,
        help="Directory containing shard_*.parquet files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        metavar="N",
        help="Show N random example pairs (default: 0).",
    )
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        print(f"Error: directory not found: {pairs_dir}", file=sys.stderr)
        sys.exit(1)

    shards = sorted(pairs_dir.glob("shard_*.parquet"))
    if not shards:
        print(f"Error: no shard_*.parquet files in {pairs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(shards)} shard(s) from {pairs_dir}...")
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    stats = compute_stats(dataset)

    # Add shard info
    stats["shards"] = {
        "count": len(shards),
        "files": [s.name for s in shards],
    }

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_stats(stats, pairs=dataset.pairs, show_examples=args.show_examples)


if __name__ == "__main__":
    main()
