#!/usr/bin/env python3
"""Pick the demo's request pairs from the tool-selection sweep, by a fixed rule.

The sweep (``demo/home_repair/sweep_tool_choice.py --scenario tool_selection``)
read the model's first tool on every one-cue training pair of the
``tool_selection`` scenario (325 pairs, 13 themes).  This script turns it into
``pairs.json`` without anyone choosing by hand:

* **flip** = min(p_A, p_B) when the two sides' top tools differ (a pair is as
  good as its less confident side), 0 otherwise; only pairs with flip ≥
  ``--min-flip`` qualify — so "most split" means the cleanest flips, not the
  closest calls;
* **score** = flip × Jaccard(content words of A, content words of B): the
  premise is *one* cue, so two requests that share most of their words and
  still flip rank above two that differ throughout (the highest raw flips are
  pairs whose cue names the tool's domain outright — "in the internal
  knowledge base" vs "from the public web");
* at most one pair per theme and at most ``--per-tools`` pairs per unordered
  tool combination, so the page is not eight variations of
  internal_search-vs-web_search;
* neither request may name a tool outright;
* the sweep readout merges ``file_read|file_write``; pairs whose two sides
  land in the same merged bucket cannot be scored and are skipped (themes
  ``read_vs_write`` / ``create_vs_update`` therefore never qualify from the
  sweep alone);
* cue words are the phrases quoted in the pair's ``signal`` annotation that
  occur in exactly one request, otherwise the spans of a word-level diff of
  the two requests (paths and URLs excluded — identifiers, not cues).

Usage:
    python demo/tool_selection/select_pairs.py [--n 8] [--per-tools 2]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_SWEEP_DIR = _DEMO_DIR.parent / "home_repair" / "output" / "sweep"
_SCENARIO = json.loads((_DEMO_DIR.parents[1] / "scenarios" / "tool_selection.json").read_text())
TOOL_IDS = [t["name"] for t in _SCENARIO["tools"]]
_TOOL_WORDS = set(TOOL_IDS) | {t.replace("_", " ") for t in TOOL_IDS}
_STOP = set(
    "a an the of to for from in on at by with and or our my your this that these those is are be "
    "it its as into up out all any some what which who how please can could would should i me we "
    "you he she they them their there here".split()
)


def names_a_tool(request: str) -> bool:
    low = request.lower()
    return any(word in low for word in _TOOL_WORDS)


def quoted_phrases(signal: str) -> list[str]:
    return [m.strip() for m in re.findall(r"'([^']+)'", signal or "") if m.strip()]


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9.'-]*", text.lower()) if w not in _STOP]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_./:'-]*|[^\sA-Za-z0-9]", text)


def cue_phrases(request: str, other_request: str, limit: int = 3) -> list[str]:
    """Spans of ``request`` that differ from ``other_request`` (word-level diff).

    This is what "the one cue" is operationally: the text one side has where
    the other has something else.  Spans made only of stop words or
    punctuation are dropped; at most ``limit`` spans, longest first.
    """
    import difflib

    a, b = _words(request), _words(other_request)
    other_content = set(content_words(other_request))
    spans = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag not in ("replace", "delete"):
            continue
        words = [w.rstrip(".,;:!?") for w in a[i1:i2]]
        words = [w for w in words if w]
        while words and not re.search(r"[A-Za-z0-9]", words[-1]):
            words = words[:-1]  # trailing punctuation
        while words and not re.search(r"[A-Za-z0-9]", words[0]):
            words = words[1:]
        # keep spans that bring at least one content word the other side lacks
        if any(w.lower() not in _STOP and w.lower() not in other_content for w in words):
            spans.append(" ".join(words).replace(" ,", ","))
    # identifiers (paths, URLs) differ between sides but are not the cue
    spans = [sp for sp in spans if sp.lower() in request.lower() and not re.search(r"[/:]", sp)]
    return sorted(spans, key=lambda sp: -len(sp))[:limit]


def cue_words_for(request: str, other_request: str, signal: str) -> list[str]:
    """Cue words for one side: quoted phrases of the annotation that occur only
    here, otherwise the differing spans of the word-level diff."""
    low, other_low = request.lower(), other_request.lower()
    quoted = [q for q in quoted_phrases(signal) if q.lower() in low and q.lower() not in other_low]
    return quoted[:3] or cue_phrases(request, other_request)


def score_pair(meta: dict, rows_by_request: dict[str, dict]) -> dict | None:
    a = rows_by_request.get(meta["anchor"])
    b = rows_by_request.get(meta["contrast"])
    if not a or not b:
        return None
    tool_a, tool_b = a["toolId"], b["toolId"]
    if not tool_a or not tool_b or tool_a == tool_b:
        flip = 0.0
    else:
        flip = min(float(a["prob"]), float(b["prob"]))
    words_a, words_b = set(content_words(meta["anchor"])), set(content_words(meta["contrast"]))
    jaccard = len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0.0
    return {
        "theme": meta["theme"],
        "anchor": meta["anchor"],
        "contrast": meta["contrast"],
        "signal": meta.get("signal", ""),
        "toolA": tool_a,
        "toolB": tool_b,
        "probA": float(a["prob"]),
        "probB": float(b["prob"]),
        "flip": round(flip, 4),
        "jaccard": round(jaccard, 4),
        "score": round(flip * jaccard, 4),
        "distA": a.get("distribution"),
        "distB": b.get("distribution"),
    }


def select_pairs(
    scored: list[dict], n: int = 8, per_tools: int = 2, min_flip: float = 0.6
) -> list[dict]:
    """Greedy pick by score with flip / theme / tool-combination / naming constraints."""
    chosen: list[dict] = []
    themes: set[str] = set()
    combos: dict[frozenset, int] = {}
    for cand in sorted(scored, key=lambda c: (-c["score"], c["theme"], c["anchor"])):
        if cand["flip"] < min_flip:
            continue
        if cand["theme"] in themes:
            continue
        if names_a_tool(cand["anchor"]) or names_a_tool(cand["contrast"]):
            continue
        combo = frozenset((cand["toolA"], cand["toolB"]))
        if combos.get(combo, 0) >= per_tools:
            continue
        chosen.append(cand)
        themes.add(cand["theme"])
        combos[combo] = combos.get(combo, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


def _title(theme: str) -> str:
    left, _, right = theme.partition("_vs_")
    return f"{left.replace('_', ' ').capitalize()} vs {right.replace('_', ' ')}"


def to_pair_records(chosen: list[dict]) -> list[dict]:
    records = []
    for cand in chosen:
        cues_a = cue_words_for(cand["anchor"], cand["contrast"], cand["signal"])
        cues_b = cue_words_for(cand["contrast"], cand["anchor"], cand["signal"])
        records.append(
            {
                "id": cand["theme"],
                "title": _title(cand["theme"]),
                "cue": f"{' / '.join(cues_a) or '…'} vs {' / '.join(cues_b) or '…'}",
                "signal": cand["signal"],
                "score": cand["score"],
                "flip": cand["flip"],
                "jaccard": cand["jaccard"],
                "a": {
                    "request": cand["anchor"],
                    "cueWords": cues_a,
                    "sweep": {"tool": cand["toolA"], "prob": round(cand["probA"], 4)},
                },
                "b": {
                    "request": cand["contrast"],
                    "cueWords": cues_b,
                    "sweep": {"tool": cand["toolB"], "prob": round(cand["probB"], 4)},
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=str(_SWEEP_DIR / "tool_selection_sweep.json"))
    parser.add_argument("--meta", default=str(_SWEEP_DIR / "tool_selection_meta.json"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--per-tools", type=int, default=2)
    parser.add_argument("--min-flip", type=float, default=0.6)
    parser.add_argument("--output", default=str(_DEMO_DIR / "pairs.json"))
    args = parser.parse_args()

    sweep = json.loads(Path(args.sweep).read_text())
    meta = json.loads(Path(args.meta).read_text())
    rows_by_request = {r["request"]: r for r in sweep["rows"] if not r.get("history")}
    scored = [s for s in (score_pair(m, rows_by_request) for m in meta) if s]
    chosen = select_pairs(scored, n=args.n, per_tools=args.per_tools, min_flip=args.min_flip)
    records = to_pair_records(chosen)
    flipping = sum(1 for s in scored if s["flip"] > 0)
    out = {
        "source": {
            "sweep": str(Path(args.sweep)),
            "meta": str(Path(args.meta)),
            "model": sweep.get("model"),
            "pairsScored": len(scored),
            "pairsFlipping": flipping,
        },
        "rule": {
            "flip": "min(p_A, p_B) if top tools differ else 0",
            "score": "flip * jaccard(content words A, content words B)",
            "n": args.n,
            "perTheme": 1,
            "perToolCombination": args.per_tools,
            "minFlip": args.min_flip,
            "noToolNamed": True,
        },
        "pairs": records,
    }
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(scored)} pairs scored, {flipping} flip; chose {len(records)} -> {args.output}")
    for r in records:
        print(
            f"  {r['score']:.2f} (flip {r['flip']:.2f}, J {r['jaccard']:.2f}) {r['id']:<24} "
            f"{r['a']['sweep']['tool']:<15} -> "
            f"{r['b']['sweep']['tool']:<15} cues {r['a']['cueWords']} / {r['b']['cueWords']}"
        )
        print(f"       A: {r['a']['request']}")
        print(f"       B: {r['b']['request']}")


if __name__ == "__main__":
    main()
