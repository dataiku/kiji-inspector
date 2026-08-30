#!/usr/bin/env python3
"""Stage 3: find the steering examples in a sweep and rank them.

``select_pairs.py`` answers a different question -- it picks *one page's worth*
of pairs (one per theme, at most two per tool combination, n=8) for the demo.
At dataset scale the question is instead "which of these pairs flip, and which
flips are worth intervening on", so this script keeps every flipping pair,
scores it the same way, and reports the population.

Scoring follows ``select_pairs.py`` exactly so results stay comparable:

    flip  = min(p_A, p_B) when the two sides' top tools differ, else 0
    score = flip * jaccard(content words of A, content words of B)

Pairs whose two sides land in the same merged readout bucket (``file_read`` and
``file_write`` share a first token) cannot be scored and are counted separately
rather than reported as non-flips.

``--emit-pairs`` writes a ``pairs.json`` in the exact shape
``tool_selection_demo.py`` consumes, so a selection made here can be fed
straight into the capture/attribute/trace stages that produce a steering run.

Usage:
    uv run python demo/steering/sweep/rank_flips.py \\
        --meta   demo/steering/sweep/output/sweep_candidates/tool_selection/meta.json \\
        --sweep  demo/steering/sweep/output/sweep_candidates/tool_selection/sweep.jsonl \\
        --top 50 --emit-pairs demo/tool_selection/output/steering_pairs.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DEMO_DIR))

from select_pairs import (  # noqa: E402
    content_words,
    cue_words_for,
    names_a_tool,
    _title,
)


def load_sweep(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a killed run
            rows[row["request"]] = row  # a later re-sweep of a request wins
    return rows


def score(meta_row: dict, sweep: dict[str, dict], min_coverage: float) -> dict | None:
    a = sweep.get(meta_row["anchor"])
    b = sweep.get(meta_row["contrast"])
    if not a or not b:
        return None
    tool_a, tool_b = a["toolId"], b["toolId"]
    unscorable = None
    if not tool_a or not tool_b:
        unscorable = "no_tool"
    elif a["coverage"] < min_coverage or b["coverage"] < min_coverage:
        unscorable = "low_coverage"
    elif tool_a == tool_b:
        unscorable = "same_bucket" if "|" in tool_a else None

    flip = 0.0
    if unscorable is None and tool_a != tool_b:
        flip = min(float(a["prob"]), float(b["prob"]))
    words_a = set(content_words(meta_row["anchor"]))
    words_b = set(content_words(meta_row["contrast"]))
    jaccard = len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0.0
    return {
        "theme": meta_row["theme"],
        "anchor": meta_row["anchor"],
        "contrast": meta_row["contrast"],
        "signal": meta_row.get("signal", ""),
        "toolA": tool_a,
        "toolB": tool_b,
        "probA": round(float(a["prob"]), 4),
        "probB": round(float(b["prob"]), 4),
        "coverageA": round(float(a["coverage"]), 4),
        "coverageB": round(float(b["coverage"]), 4),
        "datasetToolA": meta_row.get("anchorTool"),
        "datasetToolB": meta_row.get("contrastTool"),
        "flip": round(flip, 4),
        "jaccard": round(jaccard, 4),
        "score": round(flip * jaccard, 4),
        "unscorable": unscorable,
        "namesATool": names_a_tool(meta_row["anchor"]) or names_a_tool(meta_row["contrast"]),
    }


def to_pair_records(rows: list[dict]) -> list[dict]:
    """``pairs.json`` records, the shape ``tool_selection_demo.py`` consumes."""
    records = []
    seen_themes = Counter(row["theme"] for row in rows)
    used: Counter = Counter()
    for row in rows:
        cues_a = cue_words_for(row["anchor"], row["contrast"], row["signal"])
        cues_b = cue_words_for(row["contrast"], row["anchor"], row["signal"])
        # Bare theme when it identifies the pair (the one-per-theme demo case);
        # numbered only where a theme really does contribute several pairs.
        theme = row["theme"]
        if seen_themes[theme] > 1:
            pair_id = f"{theme}_{used[theme]}"
            used[theme] += 1
        else:
            pair_id = theme
        records.append(
            {
                "id": pair_id,
                "title": _title(row["theme"]),
                "cue": f"{' / '.join(cues_a) or '...'} vs {' / '.join(cues_b) or '...'}",
                "signal": row["signal"],
                "score": row["score"],
                "flip": row["flip"],
                "jaccard": row["jaccard"],
                "a": {
                    "request": row["anchor"],
                    "cueWords": cues_a,
                    "sweep": {"tool": row["toolA"], "prob": row["probA"]},
                },
                "b": {
                    "request": row["contrast"],
                    "cueWords": cues_b,
                    "sweep": {"tool": row["toolB"], "prob": row["probB"]},
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--meta", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--min-flip", type=float, default=0.6)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Drop readouts where the tool tokens hold less than this much probability mass "
        "(top-k logprobs can miss a tool entirely).",
    )
    parser.add_argument(
        "--max-side-prob",
        type=float,
        default=None,
        help="Keep only pairs where the less confident side is below this probability. "
        "A flip is not the same as a usable intervention target: the repo measured that "
        "saturated pairs (both sides ~1.00) are overdetermined -- switching all cue features "
        "off moves them <=1pp -- while the load-bearing pairs sat at p=0.61-0.77. "
        "'--max-side-prob 0.8' selects that regime.",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=None,
        help="Drop pairs whose two sides share less than this fraction of content words. "
        "A demo built on 'one cue, one flip' needs a real minimal pair; some contrast types "
        "are written as structurally different sentences and top out well below any usable "
        "floor (supply_chain's just_in_time_vs_safety_stock peaks at J=0.36 over ~10k pairs), "
        "in which case the theme should be dropped rather than misrepresented.",
    )
    parser.add_argument(
        "--select-demo",
        action="store_true",
        help="Apply select_pairs.py's page constraints to --emit-pairs: at most one pair per "
        "theme and --per-tools per unordered tool combination, highest score first. Use this "
        "to turn a sweep into a demo's pairs.json instead of a raw top-N list.",
    )
    parser.add_argument("--per-tools", type=int, default=2)
    parser.add_argument("--top", type=int, default=25, help="Rows to print")
    parser.add_argument("--exclude-tool-named", action="store_true",
                        help="Drop pairs naming a tool outright, as select_pairs.py does.")
    parser.add_argument("--output", default=None, help="Write all scored rows as JSON")
    parser.add_argument("--emit-pairs", default=None, help="Write a pairs.json of the top rows")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="With --emit-pairs: emit a uniform random sample of the gate-passing pairs "
        "instead of the top-scored ones. Top-scored selection is right for a demo page; "
        "for estimating a flip *rate* over the gated population it is biased upward, "
        "so evaluation sets should use this.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample")
    parser.add_argument(
        "--theme-cap",
        type=int,
        default=None,
        help="With --sample: at most this many pairs per theme. The gated population is "
        "dominated by whichever contrast type the template generator wrote most minimal "
        "variants of, and within-theme pairs are near-duplicates, so an uncapped sample "
        "spends most of its budget on correlated copies of one theme.",
    )
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    sweep = load_sweep(Path(args.sweep))
    scored = [s for s in (score(m, sweep, args.min_coverage) for m in meta) if s]
    missing = len(meta) - len(scored)

    reasons = Counter(s["unscorable"] for s in scored if s["unscorable"])
    usable = [s for s in scored if not s["unscorable"]]
    if args.exclude_tool_named:
        usable = [s for s in usable if not s["namesATool"]]
    flipping = [s for s in usable if s["flip"] > 0]
    strong = sorted(
        (s for s in flipping if s["flip"] >= args.min_flip), key=lambda s: -s["score"]
    )
    saturated = sum(1 for s in strong if min(s["probA"], s["probB"]) >= 0.95)
    if args.max_side_prob is not None:
        strong = [s for s in strong if min(s["probA"], s["probB"]) < args.max_side_prob]
    if args.min_jaccard is not None:
        dropped_themes = {s["theme"] for s in strong}
        strong = [s for s in strong if s["jaccard"] >= args.min_jaccard]
        dropped_themes -= {s["theme"] for s in strong}
        if dropped_themes:
            print(
                f"  themes with no pair at J>={args.min_jaccard}: {sorted(dropped_themes)} "
                f"(their contrast type is not a one-cue edit)"
            )

    print(f"{len(meta):,} pairs in meta, {missing:,} not found in sweep")
    print(f"{len(usable):,} scorable; unscorable: {dict(reasons) or 'none'}")
    rate = 100 * len(flipping) / len(usable) if usable else 0
    print(f"{len(flipping):,} flip ({rate:.1f}%), {len(strong):,} at flip >= {args.min_flip}", end="")
    print(f" and a side < {args.max_side_prob}" if args.max_side_prob is not None else "")
    if args.max_side_prob is None and saturated:
        print(
            f"  note: {saturated:,} of these are saturated (both sides p>=0.95). The repo "
            f"measured such pairs as descriptive, not load-bearing -- see --max-side-prob."
        )

    by_theme = Counter(s["theme"] for s in strong)
    by_tools = Counter(tuple(sorted((s["toolA"], s["toolB"]))) for s in strong)
    print("\nsteering examples per theme:")
    for theme, count in by_theme.most_common():
        pool = sum(1 for s in usable if s["theme"] == theme)
        print(f"  {theme:<36} {count:>6,} of {pool:>6,}")
    print("\ntop tool combinations:")
    for (tool_a, tool_b), count in by_tools.most_common(10):
        print(f"  {tool_a:<22} <-> {tool_b:<22} {count:>6,}")

    print(f"\ntop {min(args.top, len(strong))} by score (flip x lexical overlap):")
    for row in strong[: args.top]:
        print(
            f"\n  {row['score']:.2f} (flip {row['flip']:.2f}, J {row['jaccard']:.2f}) "
            f"{row['theme']}  {row['toolA']} -> {row['toolB']}"
        )
        print(f"       A: {row['anchor']}")
        print(f"       B: {row['contrast']}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "meta": args.meta,
                    "sweep": args.sweep,
                    "rule": {
                        "flip": "min(p_A, p_B) if top tools differ else 0",
                        "score": "flip * jaccard(content words A, content words B)",
                        "minFlip": args.min_flip,
                        "minCoverage": args.min_coverage,
                    },
                    "counts": {
                        "pairs": len(meta),
                        "scorable": len(usable),
                        "flipping": len(flipping),
                        "strong": len(strong),
                        "unscorable": dict(reasons),
                    },
                    "rows": strong,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.output}")

    if args.emit_pairs:
        emit = strong
        if args.sample is not None:
            if args.select_demo:
                raise SystemExit("--sample and --select-demo are mutually exclusive")
            pool = list(strong)
            random.Random(args.seed).shuffle(pool)
            if args.theme_cap is not None:
                emit, taken = [], Counter()
                for row in pool:
                    if taken[row["theme"]] >= args.theme_cap:
                        continue
                    emit.append(row)
                    taken[row["theme"]] += 1
                    if len(emit) >= args.sample:
                        break
            else:
                emit = pool[: args.sample]
            themes = Counter(row["theme"] for row in emit)
            print(f"\nsample: {len(emit)} of {len(strong)} gate-passing pairs (seed {args.seed})")
            for theme, count in themes.most_common():
                print(f"  {theme:<36} {count}")
        elif args.select_demo:
            emit, themes, combos = [], set(), {}
            for row in strong:  # already sorted by score
                combo = frozenset((row["toolA"], row["toolB"]))
                if row["theme"] in themes or combos.get(combo, 0) >= args.per_tools:
                    continue
                emit.append(row)
                themes.add(row["theme"])
                combos[combo] = combos.get(combo, 0) + 1
            print(f"\nselect-demo: {len(emit)} pairs (1/theme, <={args.per_tools}/tool pair)")
        if args.sample is None:
            emit = emit[: args.top] if args.top else emit
        out = {
            "source": {"meta": args.meta, "sweep": args.sweep,
                       "pairsScored": len(usable), "pairsFlipping": len(flipping),
                       "pairsPassingGate": len(strong)},
            "rule": {"flip": "min(p_A, p_B) if top tools differ else 0",
                     "score": "flip * jaccard(content words A, content words B)",
                     "minFlip": args.min_flip},
            **({"sample": {"n": len(emit), "seed": args.seed, "population": len(strong),
                           "themeCap": args.theme_cap}}
               if args.sample is not None else {}),
            "pairs": to_pair_records(emit),
        }
        Path(args.emit_pairs).write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote {args.emit_pairs}")


if __name__ == "__main__":
    main()
