#!/usr/bin/env python3
"""Turn the population readout into a pair-level census (CPU only).

Joins ``population_sweep.py``'s per-prompt readout back onto the pairs
parquet and reports, over every unique tool_selection pair:

* the **flip census** — how many pairs actually flip the model's first tool
  (flip = min(p_A, p_B) when the two sides' top tools differ), by contrast
  type and by flip strength;
* **agreement with the generator's intended tool** overall and by confidence
  bin (a reliability curve against the pair generator's intent — intent is
  not ground truth, and the report says so);
* the **seen / unseen stratum** — pairs whose prompts never entered SAE
  training behave the same or differently?

Usage:
    uv run python demo/spec_sheet/population_report.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]


def flip_of(decision_a: dict | None, decision_b: dict | None) -> float:
    """min(p_A, p_B) when both sides resolved and their top tools differ."""
    if not decision_a or not decision_b:
        return 0.0
    tool_a, tool_b = decision_a.get("toolId"), decision_b.get("toolId")
    if not tool_a or not tool_b or tool_a == tool_b:
        return 0.0
    return min(float(decision_a.get("prob") or 0.0), float(decision_b.get("prob") or 0.0))


def census(
    pair_rows: list[dict], decision_of: dict[str, dict], sae_prompts: set[str]
) -> tuple[list[dict], dict]:
    """Pair records + summary for every unique tool_selection (anchor, contrast)."""
    records: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in pair_rows:
        if row["scenario_name"] != "tool_selection":
            continue
        key = (row["anchor_prompt"], row["contrast_prompt"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        decision_a = decision_of.get(row["anchor_prompt"])
        decision_b = decision_of.get(row["contrast_prompt"])
        if decision_a is None or decision_b is None:
            continue
        record = {
            "anchor": row["anchor_prompt"],
            "contrast": row["contrast_prompt"],
            "contrastType": row["contrast_type"],
            "toolA": decision_a.get("toolId"),
            "probA": round(float(decision_a.get("prob") or 0.0), 4),
            "toolB": decision_b.get("toolId"),
            "probB": round(float(decision_b.get("prob") or 0.0), 4),
            "intendedA": row["anchor_tool"],
            "intendedB": row["contrast_tool"],
            "flip": round(flip_of(decision_a, decision_b), 4),
            "seenBySae": row["anchor_prompt"] in sae_prompts
            and row["contrast_prompt"] in sae_prompts,
        }
        record["agreeA"] = record["toolA"] == record["intendedA"]
        record["agreeB"] = record["toolB"] == record["intendedB"]
        records.append(record)

    def _stratum(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "pairs": n,
            "flipping": sum(1 for r in rows if r["flip"] > 0),
            "flipAtLeast03": sum(1 for r in rows if r["flip"] >= 0.3),
            "flipAtLeast06": sum(1 for r in rows if r["flip"] >= 0.6),
            "agreement": round(sum(r["agreeA"] + r["agreeB"] for r in rows) / (2 * n), 4)
            if n
            else None,
        }

    by_type: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["contrastType"]].append(record)
    for contrast_type, rows in sorted(grouped.items()):
        by_type[contrast_type] = _stratum(rows)

    bins: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        for prob, agree in (
            (record["probA"], record["agreeA"]),
            (record["probB"], record["agreeB"]),
        ):
            label = f"{min(int(prob * 10), 9) / 10:.1f}"
            bins[label][0] += int(agree)
            bins[label][1] += 1
    reliability = {
        label: {"agreement": round(hit / total, 4), "n": total}
        for label, (hit, total) in sorted(bins.items())
        if total
    }

    tool_counts = Counter(r["toolA"] for r in records) + Counter(r["toolB"] for r in records)
    summary = {
        "overall": _stratum(records),
        "byContrastType": by_type,
        "seen": _stratum([r for r in records if r["seenBySae"]]),
        "unseenBySae": _stratum([r for r in records if not r["seenBySae"]]),
        "reliabilityVsIntent": reliability,
        "modelToolCounts": dict(tool_counts.most_common()),
        "caveat": "agreement is measured against the pair generator's intended tool, not ground truth",
    }
    return records, summary


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", default=str(_REPO_ROOT / "output" / "pairs"))
    parser.add_argument("--population-dir", default=str(_DEMO_DIR / "output" / "population"))
    parser.add_argument(
        "--sae-prompts",
        default=str(_REPO_ROOT / "output" / "layer_43" / "activations" / "prompts.json"),
    )
    args = parser.parse_args()

    population = Path(args.population_dir)
    decision_of: dict[str, dict] = {}
    with (population / "readout.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            decision_of[row["request"]] = row

    parquet_files = sorted(Path(args.pairs_dir).glob("shard_*.parquet"))
    pair_rows = pd.concat([pd.read_parquet(p) for p in parquet_files]).to_dict("records")
    sae_prompts = set(json.loads(Path(args.sae_prompts).read_text()))

    records, summary = census(pair_rows, decision_of, sae_prompts)
    (population / "population_pairs.json").write_text(json.dumps(records) + "\n")
    (population / "population_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Wrote {population}/population_summary.json ({len(records)} pairs)")


if __name__ == "__main__":
    main()
