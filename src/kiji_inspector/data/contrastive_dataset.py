from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContrastivePair:
    """A pair of similar requests with different optimal tools.

    This is a text-only dataclass.  Activations are extracted separately
    and saved as numpy shards for SAE training.  Contrastive pairs are
    used *post-hoc* to identify which SAE features are decision-relevant.
    """

    pair_id: str

    # The two requests
    anchor_prompt: str
    anchor_tool: str
    contrast_prompt: str
    contrast_tool: str

    # What makes them similar
    shared_intent: str  # e.g., "find information about API rate limits"
    semantic_similarity: float  # 0-1 score

    # What distinguishes them
    contrast_type: str  # e.g. "internal_vs_external", "growth_vs_value"
    distinguishing_signal: str  # Human description of the key difference

    # Which scenario generated this pair
    scenario_name: str = ""  # e.g. "tool_selection", "investment"


@dataclass
class ContrastiveDataset:
    """Dataset organised around contrastive pairs (text only)."""

    pairs: list[ContrastivePair]

    # Metadata
    contrast_type_distribution: dict[str, int] = field(default_factory=dict)
    tool_pair_distribution: dict[tuple[str, str], int] = field(default_factory=dict)

    def get_by_contrast_type(self, contrast_type: str) -> list[ContrastivePair]:
        """Filter pairs by contrast type."""
        return [p for p in self.pairs if p.contrast_type == contrast_type]

    def balance_by_contrast_type(self, max_per_type: int = 500) -> ContrastiveDataset:
        """Create a balanced subset with equal representation of contrast types."""
        # Collect all unique contrast types from the actual pairs
        all_types = sorted({p.contrast_type for p in self.pairs})
        balanced_pairs = []
        for ct in all_types:
            type_pairs = self.get_by_contrast_type(ct)
            balanced_pairs.extend(type_pairs[:max_per_type])
        return ContrastiveDataset(pairs=balanced_pairs)

    # ── Parquet I/O ──────────────────────────────────────────────

    def to_parquet(
        self,
        output_dir: str | Path,
        shard_size: int = 50_000,
    ) -> list[Path]:
        """Save the dataset as sharded parquet files (text only).

        Args:
            output_dir: Directory to write shard files into.
            shard_size: Max rows per shard (default 50k).

        Returns:
            List of written file paths.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = self._pairs_to_rows()
        if not rows:
            return []

        written: list[Path] = []
        for shard_idx, start in enumerate(range(0, len(rows), shard_size)):
            shard_rows = rows[start : start + shard_size]
            table = pa.Table.from_pydict(self._rows_to_columnar(shard_rows))
            path = output_dir / f"shard_{shard_idx:05d}.parquet"
            pq.write_table(table, path)
            written.append(path)

        return written

    @classmethod
    def from_parquet(cls, input_dir: str | Path) -> ContrastiveDataset:
        """Load a dataset from sharded parquet files.

        Args:
            input_dir: Directory containing ``shard_*.parquet`` files.

        Returns:
            Reconstructed ContrastiveDataset.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        input_dir = Path(input_dir)
        shard_files = sorted(input_dir.glob("shard_*.parquet"))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.parquet files in {input_dir}")

        tables = [pq.read_table(f) for f in shard_files]
        table = pa.concat_tables(tables)
        return cls._table_to_dataset(table)

    # ── Private helpers ──────────────────────────────────────────

    def _pairs_to_rows(self) -> list[dict]:
        rows = []
        for p in self.pairs:
            rows.append(
                {
                    "pair_id": p.pair_id,
                    "anchor_prompt": p.anchor_prompt,
                    "anchor_tool": p.anchor_tool,
                    "contrast_prompt": p.contrast_prompt,
                    "contrast_tool": p.contrast_tool,
                    "shared_intent": p.shared_intent,
                    "semantic_similarity": p.semantic_similarity,
                    "contrast_type": p.contrast_type,
                    "distinguishing_signal": p.distinguishing_signal,
                    "scenario_name": p.scenario_name,
                }
            )
        return rows

    @staticmethod
    def _rows_to_columnar(rows: list[dict]) -> dict[str, list]:
        columns: dict[str, list] = {k: [] for k in rows[0]}
        for row in rows:
            for k, v in row.items():
                columns[k].append(v)
        return columns

    @classmethod
    def _table_to_dataset(cls, table) -> ContrastiveDataset:
        col = table.to_pydict()
        n = len(col["pair_id"])

        # Backward compat: old parquet files may not have scenario_name
        has_scenario = "scenario_name" in col

        pairs = []
        for i in range(n):
            pairs.append(
                ContrastivePair(
                    pair_id=col["pair_id"][i],
                    anchor_prompt=col["anchor_prompt"][i],
                    anchor_tool=col["anchor_tool"][i],
                    contrast_prompt=col["contrast_prompt"][i],
                    contrast_tool=col["contrast_tool"][i],
                    shared_intent=col["shared_intent"][i],
                    semantic_similarity=col["semantic_similarity"][i],
                    contrast_type=col["contrast_type"][i],
                    distinguishing_signal=col["distinguishing_signal"][i],
                    scenario_name=col["scenario_name"][i] if has_scenario else "tool_selection",
                )
            )

        return cls(pairs=pairs)
