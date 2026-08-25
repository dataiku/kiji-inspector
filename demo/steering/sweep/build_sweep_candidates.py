#!/usr/bin/env python3
"""Stage 1 of finding steering examples: choose which pairs to sweep.

A *steering example* is a one-cue pair -- two near-identical requests where the
model's first tool flips. Finding them takes three stages, all scenario-general:

    1. this script    (CPU)  pairs -> candidates.json + meta.json
    2. sweep_pairs_batched.py (GPU)  -> sweep.jsonl   (the tool readout)
    3. rank_flips.py  (CPU)  -> steering_examples.json

Two source datasets, selected with ``--source``:

``full`` (default)
    ``575-lab/kiji-inspector-pairs``, the SAE training set -- 2,011,672 rows ->
    **1,849,228 clean unique pairs** over 5 scenarios and 37 contrast types.
    Local copy: ``output/pairs``. This is the whole population of candidate
    steering examples.

``demo``
    ``575-lab/kiji-inspector-demo-pairs`` -- 50,076 rows -> 7,467 unique
    tool_selection pairs. The slice the published demo was built from; all
    seven pairs of ``pairs.json`` live here.

Normalisation applied to both:

* rows are deduplicated by ``(scenario, anchor, contrast)`` -- the datasets
  repeat prompt pairs with reworded signals, and sweeping a request twice is
  wasted GPU time;
* 23 rows whose anchor and contrast are the same string are dropped (they can
  never flip), as are 371 tool_selection rows whose ``anchor_tool`` /
  ``contrast_tool`` is a malformed multi-tool string like ``"api_call,
  code_execute"`` (33 such values exist; the intended tool is undefined);
* the demo dataset truncates ``scenario_name`` at the first underscore
  (``"tool"`` / ``"home"``) and prefixes every ``contrast_type`` with the
  remainder, so ``selection_local_vs_remote`` is normalised to the bare theme
  ``local_vs_remote`` that ``select_pairs.py`` and stage 3 key on.

Usage:
    # one scenario of the full dataset
    uv run python demo/steering/sweep/build_sweep_candidates.py \\
        --scenario tool_selection --per-theme 2000

    # every scenario, whole population (3.2M requests -- see the cost note in
    # sweep_pairs_batched.py before committing a GPU to this)
    uv run python demo/steering/sweep/build_sweep_candidates.py \\
        --scenario all --per-theme 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from kiji_inspector.data.contrastive_dataset import ContrastiveDataset

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]

_SOURCES = {
    "full": ("575-lab/kiji-inspector-pairs", _REPO_ROOT / "output" / "pairs"),
    "demo": ("575-lab/kiji-inspector-demo-pairs", None),
}
# The demo dataset truncates scenario_name at the first underscore and moves
# the remainder onto every contrast_type of that scenario.
_SCENARIO_ALIAS = {"tool": "tool_selection", "home": "home_repair"}
_ALIAS_PREFIX = {"tool": "selection_", "home": "repair_"}


def resolve_pairs_dir(source: str, pairs_dir: str | None, cache_dir: str | None) -> str:
    """Local shard directory: an explicit path, the repo copy, or the HF download."""
    if pairs_dir:
        return pairs_dir
    repo_id, local = _SOURCES[source]
    if local is not None and (local / "shard_00000.parquet").exists():
        return str(local)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, repo_type="dataset", cache_dir=cache_dir)


def scenario_of(pair) -> str:
    return _SCENARIO_ALIAS.get(pair.scenario_name, pair.scenario_name)


def theme_of(pair) -> str:
    """Bare theme name, as select_pairs.py and stage 3 expect it."""
    prefix = _ALIAS_PREFIX.get(pair.scenario_name, "")
    if prefix and pair.contrast_type.startswith(prefix):
        return pair.contrast_type[len(prefix) :]
    return pair.contrast_type


def load_scenario_tools() -> dict[str, set[str]]:
    tools = {}
    for path in sorted((_REPO_ROOT / "scenarios").glob("*.json")):
        config = json.loads(path.read_text())
        tools[config["name"]] = {t["name"] for t in config["tools"]}
    return tools


def collect(pairs, scenarios: set[str], valid_tools: dict[str, set[str]]) -> dict[str, dict]:
    """Deduplicated, tool-validated pairs grouped as ``{scenario: {theme: [pair]}}``."""
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, str]] = set()
    dropped = {"identical": 0, "bad_tool": 0, "duplicate": 0}
    for pair in pairs:
        scenario = scenario_of(pair)
        if scenario not in scenarios:
            continue
        if pair.anchor_prompt.strip() == pair.contrast_prompt.strip():
            dropped["identical"] += 1
            continue
        key = (scenario, pair.anchor_prompt, pair.contrast_prompt)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        known = valid_tools.get(scenario)
        if known and (pair.anchor_tool not in known or pair.contrast_tool not in known):
            dropped["bad_tool"] += 1
            continue
        grouped[scenario][theme_of(pair)].append(pair)
    return {s: dict(t) for s, t in grouped.items()}, dropped


def overlap(pair) -> float:
    """Jaccard of the two requests' content words -- ``select_pairs.py``'s own
    tie-breaker, computed here because only the *flip* half of its score needs
    a GPU, so overlap can shape the candidate list in advance.

    This is a band, not a maximum: sorting by it descending is
    counterproductive. The highest-overlap rows of a theme are degenerate
    near-duplicates, while all seven published pairs sit at 0.33-0.56, around
    the per-theme median.
    """
    import sys

    if str(_DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(_DEMO_DIR))
    from select_pairs import content_words

    a, b = set(content_words(pair.anchor_prompt)), set(content_words(pair.contrast_prompt))
    return len(a & b) / len(a | b) if a | b else 0.0


def sample(
    by_theme: dict[str, list],
    per_theme: int | None,
    seed: int,
    band: tuple[float, float] | None,
) -> tuple[dict[str, list[str]], list[dict]]:
    rng = random.Random(seed)
    candidates: dict[str, list[str]] = {}
    meta: list[dict] = []
    for theme in sorted(by_theme):
        chosen = by_theme[theme]
        if band is not None:
            chosen = [p for p in chosen if band[0] <= overlap(p) <= band[1]]
        if per_theme is not None and len(chosen) > per_theme:
            chosen = rng.sample(chosen, per_theme)
        requests: list[str] = []
        for pair in chosen:
            requests.extend((pair.anchor_prompt, pair.contrast_prompt))
            meta.append(
                {
                    "theme": theme,
                    "anchor": pair.anchor_prompt,
                    "contrast": pair.contrast_prompt,
                    "signal": pair.distinguishing_signal,
                    "anchorTool": pair.anchor_tool,
                    "contrastTool": pair.contrast_tool,
                }
            )
        candidates[theme] = requests
    return candidates, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=sorted(_SOURCES), default="full")
    parser.add_argument("--pairs-dir", default=None, help="Override the source with a local dir.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--scenario",
        default="tool_selection",
        help="Scenario name, or 'all' for every scenario in the source.",
    )
    parser.add_argument(
        "--per-theme",
        type=int,
        default=2000,
        help="Pairs sampled per contrast type; 0 keeps all. Default 2000 is a wide but "
        "affordable first pass (~26k pairs / 52k requests for tool_selection's 13 themes).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overlap-band",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Keep only pairs whose content-word Jaccard is in this band before sampling, "
        "e.g. '--overlap-band 0.3 0.7'. Biases toward genuine one-cue edits, but measured "
        "over 10 seeds it does NOT improve recovery of the published seven. Off by default.",
    )
    parser.add_argument("--output-dir", default=str(_DEMO_DIR / "output" / "sweep_candidates"))
    args = parser.parse_args()

    pairs_dir = resolve_pairs_dir(args.source, args.pairs_dir, args.cache_dir)
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    valid_tools = load_scenario_tools()

    available = {scenario_of(p) for p in dataset.pairs}
    wanted = available if args.scenario == "all" else {args.scenario}
    if not wanted & available:
        raise SystemExit(f"No pairs for {args.scenario!r}; source has {sorted(available)}")

    grouped, dropped = collect(dataset.pairs, wanted, valid_tools)
    band = tuple(args.overlap_band) if args.overlap_band else None
    per_theme = None if args.per_theme <= 0 else args.per_theme

    print(f"source: {pairs_dir}")
    print(
        f"dropped {dropped['duplicate']:,} duplicate, {dropped['identical']:,} identical, "
        f"{dropped['bad_tool']:,} malformed-tool rows"
    )
    out_root = Path(args.output_dir)
    total_pairs = total_requests = 0
    for scenario in sorted(grouped):
        by_theme = grouped[scenario]
        candidates, meta = sample(by_theme, per_theme, args.seed, band)
        out_dir = out_root / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidates.json").write_text(json.dumps(candidates, indent=2))
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        n_requests = len({r for v in candidates.values() for r in v})
        total_pairs += len(meta)
        total_requests += n_requests
        pool = sum(len(v) for v in by_theme.values())
        print(
            f"\n{scenario}: {len(by_theme)} themes, {pool:,} clean unique pairs "
            f"-> selected {len(meta):,} pairs / {n_requests:,} unique requests"
        )
        for theme in sorted(by_theme):
            picked = sum(1 for m in meta if m["theme"] == theme)
            print(f"  {theme:<36} {picked:>6,} of {len(by_theme[theme]):>7,}")
        print(f"  -> {out_dir}")
    print(f"\nTOTAL {total_pairs:,} pairs / {total_requests:,} unique requests to sweep")


if __name__ == "__main__":
    main()
