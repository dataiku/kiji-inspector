"""This script loads n pairs from output/pairs/shard_*.parquet and displays the pairs."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contrastive_dataset import ContrastiveDataset


def display_pair(pair, index: int) -> None:
    """Display a single contrastive pair."""
    print(f"\n{'=' * 80}")
    print(f"Pair {index}: {pair.pair_id}")
    print(f"{'=' * 80}")
    print(f"Scenario: {pair.scenario_name}")
    print(f"Contrast Type: {pair.contrast_type}")
    print(f"Shared Intent: {pair.shared_intent}")
    print(f"Distinguishing Signal: {pair.distinguishing_signal}")
    print(f"Semantic Similarity: {pair.semantic_similarity:.2f}")
    print()
    print(f"ANCHOR:")
    print(f"  Tool: {pair.anchor_tool}")
    print(f"  Prompt: {pair.anchor_prompt}")
    print()
    print(f"CONTRAST:")
    print(f"  Tool: {pair.contrast_tool}")
    print(f"  Prompt: {pair.contrast_prompt}")


def main():
    parser = argparse.ArgumentParser(description="Display contrastive pairs from parquet files.")
    parser.add_argument(
        "-n", "--num-pairs", type=int, default=5, help="Number of pairs to display (default: 5)"
    )
    parser.add_argument(
        "-d",
        "--pairs-dir",
        type=str,
        default="output/pairs",
        help="Directory containing shard_*.parquet files (default: output/pairs)",
    )
    parser.add_argument(
        "-c",
        "--contrast-type",
        type=str,
        default=None,
        help="Filter by contrast type (e.g., 'read_vs_write')",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        default=None,
        help="Filter by scenario name (e.g., 'tool_selection')",
    )
    parser.add_argument(
        "--list-types", action="store_true", help="List all available contrast types and exit"
    )
    parser.add_argument(
        "--list-scenarios", action="store_true", help="List all available scenarios and exit"
    )
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics and exit")

    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        print(f"Error: Directory not found: {pairs_dir}")
        sys.exit(1)

    print(f"Loading pairs from {pairs_dir}...")
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    print(f"Loaded {len(dataset.pairs)} pairs.")

    if args.stats:
        print(f"\n{'=' * 80}")
        print("Dataset Statistics")
        print(f"{'=' * 80}")
        print(f"Total pairs: {len(dataset.pairs)}")

        scenarios = {}
        contrast_types = {}
        tool_pairs = {}

        for pair in dataset.pairs:
            scenarios[pair.scenario_name] = scenarios.get(pair.scenario_name, 0) + 1
            contrast_types[pair.contrast_type] = contrast_types.get(pair.contrast_type, 0) + 1
            tp = (pair.anchor_tool, pair.contrast_tool)
            tool_pairs[tp] = tool_pairs.get(tp, 0) + 1

        print(f"\nScenarios ({len(scenarios)}):")
        for name, count in sorted(scenarios.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")

        print(f"\nContrast Types ({len(contrast_types)}):")
        for name, count in sorted(contrast_types.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")

        print(f"\nTop 10 Tool Pairs:")
        for (t1, t2), count in sorted(tool_pairs.items(), key=lambda x: -x[1])[:10]:
            print(f"  {t1} vs {t2}: {count}")

        sys.exit(0)

    if args.list_types:
        contrast_types = sorted({p.contrast_type for p in dataset.pairs})
        print(f"\nAvailable contrast types ({len(contrast_types)}):")
        for ct in contrast_types:
            count = sum(1 for p in dataset.pairs if p.contrast_type == ct)
            print(f"  {ct}: {count} pairs")
        sys.exit(0)

    if args.list_scenarios:
        scenarios = sorted({p.scenario_name for p in dataset.pairs})
        print(f"\nAvailable scenarios ({len(scenarios)}):")
        for s in scenarios:
            count = sum(1 for p in dataset.pairs if p.scenario_name == s)
            print(f"  {s}: {count} pairs")
        sys.exit(0)

    # Filter pairs
    pairs = dataset.pairs

    if args.contrast_type:
        pairs = [p for p in pairs if p.contrast_type == args.contrast_type]
        print(f"Filtered to {len(pairs)} pairs with contrast type '{args.contrast_type}'")

    if args.scenario:
        pairs = [p for p in pairs if p.scenario_name == args.scenario]
        print(f"Filtered to {len(pairs)} pairs with scenario '{args.scenario}'")

    if not pairs:
        print("No pairs found matching the filters.")
        sys.exit(1)

    # Display pairs
    num_to_show = min(args.num_pairs, len(pairs))
    print(f"\nDisplaying {num_to_show} pairs:")

    for i, pair in enumerate(pairs[:num_to_show]):
        display_pair(pair, i + 1)

    if len(pairs) > num_to_show:
        print(f"\n... and {len(pairs) - num_to_show} more pairs.")


if __name__ == "__main__":
    main()
