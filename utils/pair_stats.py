#!/usr/bin/env python3
"""Print statistics about generated contrastive pairs.

Usage:
    uv run python utils/pair_stats.py                    # default: output/pairs
    uv run python utils/pair_stats.py path/to/pairs_dir
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contrastive_dataset import ContrastiveDataset


def print_stats(pairs_dir: str) -> None:
    pairs_path = Path(pairs_dir)
    shards = sorted(pairs_path.glob("shard_*.parquet"))
    if not shards:
        print(f"No shard_*.parquet files found in {pairs_dir}")
        sys.exit(1)

    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    pairs = dataset.pairs

    # --- Overall ---
    print("=" * 60)
    print("  Contrastive Pair Statistics")
    print(f"  Source: {pairs_path.resolve()}")
    print(f"  Shards: {len(shards)}")
    print("=" * 60)
    print(f"\nTotal pairs: {len(pairs)}")

    # --- Per scenario ---
    scenario_counts = Counter(p.scenario_name for p in pairs)
    print(f"\nScenarios ({len(scenario_counts)}):")
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"  {scenario or '(unnamed)':<30s} {count:>6,}")

    # --- Per contrast type ---
    ct_counts = Counter(p.contrast_type for p in pairs)
    print(f"\nContrast types ({len(ct_counts)}):")
    for ct, count in sorted(ct_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:<40s} {count:>6,}")

    # --- Per contrast type per scenario ---
    scenario_ct = Counter((p.scenario_name, p.contrast_type) for p in pairs)
    print("\nContrast types by scenario:")
    for scenario in sorted(scenario_counts):
        label = scenario or "(unnamed)"
        cts = {ct: c for (s, ct), c in scenario_ct.items() if s == scenario}
        print(f"  {label} ({sum(cts.values()):,} pairs, {len(cts)} types):")
        for ct, count in sorted(cts.items(), key=lambda x: -x[1]):
            print(f"    {ct:<38s} {count:>6,}")

    # --- Tool distribution ---
    anchor_tools = Counter(p.anchor_tool for p in pairs)
    contrast_tools = Counter(p.contrast_tool for p in pairs)
    all_tools = anchor_tools + contrast_tools
    print(f"\nTools ({len(all_tools)} unique, counting anchor + contrast appearances):")
    for tool, count in sorted(all_tools.items(), key=lambda x: -x[1]):
        a = anchor_tools.get(tool, 0)
        c = contrast_tools.get(tool, 0)
        print(f"  {tool:<30s} {count:>6,}  (anchor={a:,}, contrast={c:,})")

    # --- Tool pair distribution ---
    tool_pairs = Counter(
        (min(p.anchor_tool, p.contrast_tool), max(p.anchor_tool, p.contrast_tool)) for p in pairs
    )
    print(f"\nTool pairs ({len(tool_pairs)} unique combinations):")
    for (t1, t2), count in sorted(tool_pairs.items(), key=lambda x: -x[1])[:20]:
        print(f"  {t1} <-> {t2:<25s} {count:>6,}")
    if len(tool_pairs) > 20:
        print(f"  ... and {len(tool_pairs) - 20} more")

    # --- Semantic similarity ---
    sims = [p.semantic_similarity for p in pairs]
    print("\nSemantic similarity:")
    print(f"  min={min(sims):.3f}  max={max(sims):.3f}  mean={sum(sims) / len(sims):.3f}")
    buckets = Counter(int(s * 10) for s in sims)  # bucket by 0.0-0.1, 0.1-0.2, ...
    print("  Distribution:")
    for bucket in range(11):
        count = buckets.get(bucket, 0)
        bar = "#" * (count * 40 // max(max(buckets.values()), 1))
        lo = bucket / 10
        hi = lo + 0.1
        print(f"    [{lo:.1f}-{hi:.1f}) {count:>6,}  {bar}")

    # --- Prompt lengths ---
    anchor_lens = [len(p.anchor_prompt) for p in pairs]
    contrast_lens = [len(p.contrast_prompt) for p in pairs]
    all_lens = anchor_lens + contrast_lens
    print("\nPrompt lengths (chars):")
    print(
        f"  Anchor:   min={min(anchor_lens):,}  max={max(anchor_lens):,}  mean={sum(anchor_lens) // len(anchor_lens):,}"
    )
    print(
        f"  Contrast: min={min(contrast_lens):,}  max={max(contrast_lens):,}  mean={sum(contrast_lens) // len(contrast_lens):,}"
    )
    print(
        f"  Overall:  min={min(all_lens):,}  max={max(all_lens):,}  mean={sum(all_lens) // len(all_lens):,}"
    )

    # --- Duplicate check ---
    anchor_set = {p.anchor_prompt for p in pairs}
    contrast_set = {p.contrast_prompt for p in pairs}
    all_prompts = [p.anchor_prompt for p in pairs] + [p.contrast_prompt for p in pairs]
    unique_prompts = set(all_prompts)
    dup_count = len(all_prompts) - len(unique_prompts)
    cross_overlap = anchor_set & contrast_set
    print("\nDuplicates:")
    print(f"  Unique prompts: {len(unique_prompts):,} / {len(all_prompts):,}")
    print(f"  Duplicate prompts: {dup_count:,}")
    print(f"  Anchor/contrast overlap: {len(cross_overlap):,} prompts appear in both roles")

    id_counts = Counter(p.pair_id for p in pairs)
    dup_ids = {pid: c for pid, c in id_counts.items() if c > 1}
    if dup_ids:
        print(f"  Duplicate pair_ids: {len(dup_ids):,}")
    else:
        print("  Duplicate pair_ids: 0")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stats about generated contrastive pairs.")
    parser.add_argument(
        "pairs_dir",
        nargs="?",
        default="output/pairs",
        help="Directory containing shard_*.parquet files (default: output/pairs).",
    )
    args = parser.parse_args()
    print_stats(args.pairs_dir)


if __name__ == "__main__":
    main()
