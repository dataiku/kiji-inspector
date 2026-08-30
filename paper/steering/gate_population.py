#!/usr/bin/env python3
"""Per-contrast-type sizes of the gate-passing population the expanded samples were drawn from.

``extract_results.py`` needs these to report a design-weighted flip rate, and
they cannot be recovered from ``pairs.json``: the emitted sample records its
own size and the population *total*, but not how that total is distributed over
contrast types.  Recovering the distribution means re-running the gate over the
sweep corpora (300 MB each), which is far too slow for the report builder, so it
is done once here and cached.

The gate flags below are the ones each expanded README documents; the resulting
totals must match the ``sample.population`` recorded in the corresponding
``pairs.json`` (1,094 and 1,616), which this script asserts.

    uv run python paper/steering/gate_population.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "demo" / "steering" / "sweep" / "output" / "sweep_candidates"
OUT = Path(__file__).resolve().parent / "results" / "gate_population.json"

sys.path.insert(0, str(ROOT / "demo" / "steering" / "sweep"))
sys.path.insert(0, str(ROOT / "demo" / "tool_selection"))

# scenario -> the gate its expanded README documents.  supply chain and customer
# support differ in --min-jaccard: the customer-support contrast types are
# written as longer sentences and none reaches J = 0.7.
GATES = {
    "supply_chain": {"min_jaccard": 0.7, "expected": 1094},
    "customer_support": {"min_jaccard": 0.55, "expected": 1616},
}
MIN_FLIP = 0.6
MIN_COVERAGE = 0.5
MAX_SIDE_PROB = 0.8


def slim_sweep(path: Path) -> dict[str, dict]:
    """``request -> {toolId, prob, coverage}``, streamed a line at a time.

    ``rank_flips.load_sweep`` keeps the whole row; at 300 MB per corpus that is
    gigabytes of dict for three fields.  A later re-sweep of a request wins, as
    there.
    """
    out: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a killed run
            out[row["request"]] = {
                "toolId": row["toolId"],
                "prob": row["prob"],
                "coverage": row["coverage"],
            }
    return out


def gate_passing(scenario: str, min_jaccard: float) -> list[dict]:
    from rank_flips import score

    meta = json.loads((SWEEP / scenario / "meta.json").read_text())
    sweep = slim_sweep(SWEEP / scenario / "sweep.jsonl")
    scored = [s for s in (score(m, sweep, MIN_COVERAGE) for m in meta) if s]
    usable = [s for s in scored if not s["unscorable"] and not s["namesATool"]]
    strong = [s for s in usable if s["flip"] > 0 and s["flip"] >= MIN_FLIP]
    strong = [s for s in strong if min(s["probA"], s["probB"]) < MAX_SIDE_PROB]
    strong = [s for s in strong if s["jaccard"] >= min_jaccard]
    return strong


def main() -> None:
    out = {
        "rule": {
            "minFlip": MIN_FLIP,
            "minCoverage": MIN_COVERAGE,
            "maxSideProb": MAX_SIDE_PROB,
            "excludeToolNamed": True,
            "minJaccard": {k: v["min_jaccard"] for k, v in GATES.items()},
        },
        "scenarios": {},
    }
    for scenario, cfg in GATES.items():
        strong = gate_passing(scenario, cfg["min_jaccard"])
        by_theme = Counter(s["theme"] for s in strong)
        if len(strong) != cfg["expected"]:
            raise SystemExit(
                f"{scenario}: gate produced {len(strong)} pairs, but pairs.json records "
                f"{cfg['expected']} — the flags here no longer match the ones the sample "
                f"was drawn under, and the weights would be wrong."
            )
        out["scenarios"][scenario] = {
            "population": len(strong),
            "byTheme": dict(by_theme.most_common()),
        }
        print(f"{scenario}: {len(strong):,} gate-passing")
        for theme, count in by_theme.most_common():
            print(f"  {theme:<40} {count:>6,}  ({100 * count / len(strong):5.1f}%)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
