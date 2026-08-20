#!/usr/bin/env python3
"""Build leak-free scenario splits of the existing SAE training shards.

The six ``output/layer_N/activations/shard_000000.npy`` files hold 95,386
decision-token vectors laid out as consecutive (anchor, contrast) duos in
``output/pairs`` parquet order — verified: every duo of ``prompts.json``
matches a parquet ``(anchor_prompt, contrast_prompt)`` row, and the prompt
list is byte-identical across layers.  Prompts repeat across pairs (17,883
unique prompts in 95,386 slots), so holding out *pairs* would leak identical
vectors into training.  This script instead splits at the level of connected
components of the prompt-pair graph (prompts are nodes, duos are edges; 4,785
components, none scenario-mixed), which guarantees no prompt — and therefore
no vector — appears on both sides.

It writes, per layer, training shard directories for three dictionaries
(``home_repair_only``, ``tool_selection_only``, ``joint`` — the joint one
minus both eval sets, so it is a *fair* ceiling, unlike the shipped SAEs
which saw every vector) plus per-scenario eval matrices deduplicated to
unique prompts.  No model, no GPU, no new data: a reindexing of what exists.

Usage:
    uv run python demo/spec_sheet/build_splits.py [--eval-frac 0.1] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
LAYERS = [6, 13, 20, 27, 34, 43]
DICTIONARIES = ("home_repair_only", "tool_selection_only", "joint")
SCENARIOS = ("home_repair", "tool_selection")


def duo_records(prompts: list[str], pair_rows: list[dict]) -> list[dict]:
    """Map each consecutive (anchor, contrast) duo of ``prompts`` to its pair.

    ``pair_rows`` are parquet rows with ``anchor_prompt`` / ``contrast_prompt``
    / ``scenario_name`` / ``contrast_type``.  Every duo must match at least one
    row (duplicate parquet rows are fine — metadata is identical).
    """
    if len(prompts) % 2:
        raise ValueError(
            f"odd number of vectors ({len(prompts)}); expected (anchor, contrast) duos"
        )
    lookup: dict[tuple[str, str], dict] = {}
    for row in pair_rows:
        lookup.setdefault((row["anchor_prompt"], row["contrast_prompt"]), row)
    records = []
    for duo in range(len(prompts) // 2):
        anchor, contrast = prompts[2 * duo], prompts[2 * duo + 1]
        row = lookup.get((anchor, contrast))
        if row is None:
            raise ValueError(f"duo {duo} not found in the pairs parquet: {anchor[:60]!r}…")
        records.append(
            {
                "duo": duo,
                "anchor": anchor,
                "contrast": contrast,
                "scenario": row["scenario_name"],
                "contrastType": row["contrast_type"],
            }
        )
    return records


def prompt_components(records: list[dict]) -> dict[str, str]:
    """Union-find over prompts with duos as edges; returns prompt -> root."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rec in records:
        ra, rb = find(rec["anchor"]), find(rec["contrast"])
        if ra != rb:
            parent[ra] = rb
    return {prompt: find(prompt) for prompt in parent}


def choose_eval_components(
    component_prompts: dict[str, list[str]], eval_frac: float, seed: int
) -> set[str]:
    """Pick component roots until they cover ``eval_frac`` of unique prompts.

    Deterministic: components are sorted by their smallest member prompt and
    shuffled with ``seed``.
    """
    total = sum(len(members) for members in component_prompts.values())
    order = sorted(component_prompts, key=lambda root: min(component_prompts[root]))
    random.Random(seed).shuffle(order)
    chosen: set[str] = set()
    covered = 0
    for root in order:
        if covered >= eval_frac * total:
            break
        chosen.add(root)
        covered += len(component_prompts[root])
    return chosen


def split_row_indices(
    records: list[dict], components: dict[str, str], eval_roots: dict[str, set[str]]
) -> dict[str, list[int]]:
    """Row indices (into the 95,386-row shards) for every split.

    Training splits keep the original duplication (implicit weighting, same
    recipe as the shipped SAEs); eval splits are deduplicated to the first
    row of each unique prompt.
    """
    out: dict[str, list[int]] = {
        "train_home_repair_only": [],
        "train_tool_selection_only": [],
        "train_joint": [],
        "eval_home_repair": [],
        "eval_tool_selection": [],
    }
    seen_eval: set[str] = set()
    for rec in records:
        scenario = rec["scenario"]
        held_out = eval_roots[scenario]
        for offset, prompt in ((0, rec["anchor"]), (1, rec["contrast"])):
            row = 2 * rec["duo"] + offset
            if components[prompt] in held_out:
                if prompt not in seen_eval:
                    seen_eval.add(prompt)
                    out[f"eval_{scenario}"].append(row)
            else:
                out[f"train_{scenario}_only"].append(row)
                out["train_joint"].append(row)
    return out


def component_prompts_by_scenario(
    records: list[dict], components: dict[str, str]
) -> dict[str, dict[str, list[str]]]:
    """{scenario: {root: [unique prompts]}}; raises if a component mixes scenarios."""
    scenario_of: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    for rec in records:
        for prompt in (rec["anchor"], rec["contrast"]):
            root = components[prompt]
            previous = scenario_of.setdefault(root, rec["scenario"])
            if previous != rec["scenario"]:
                raise ValueError(f"component {root[:40]!r}… spans two scenarios")
            members.setdefault(root, set()).add(prompt)
    grouped: dict[str, dict[str, list[str]]] = {s: {} for s in SCENARIOS}
    for root, prompts in members.items():
        grouped[scenario_of[root]][root] = sorted(prompts)
    return grouped


def main() -> None:
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-root", default=str(_REPO_ROOT / "output"))
    parser.add_argument("--pairs-dir", default=str(_REPO_ROOT / "output" / "pairs"))
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default=str(_DEMO_DIR / "output" / "splits"))
    args = parser.parse_args()

    root = Path(args.activations_root)
    prompts = json.loads(
        (root / f"layer_{args.layers[0]}" / "activations" / "prompts.json").read_text()
    )
    parquet_files = sorted(Path(args.pairs_dir).glob("shard_*.parquet"))
    pair_rows = pd.concat([pd.read_parquet(p) for p in parquet_files]).to_dict("records")

    records = duo_records(prompts, pair_rows)
    components = prompt_components(records)
    grouped = component_prompts_by_scenario(records, components)
    eval_roots = {
        scenario: choose_eval_components(grouped[scenario], args.eval_frac, args.seed)
        for scenario in SCENARIOS
    }
    indices = split_row_indices(records, components, eval_roots)

    work = Path(args.work_dir)
    (work / "indices").mkdir(parents=True, exist_ok=True)
    for name, rows in indices.items():
        np.save(work / "indices" / f"{name}.npy", np.asarray(rows, dtype=np.int64))

    for layer in args.layers:
        src_dir = root / f"layer_{layer}" / "activations"
        shard = np.load(src_dir / "shard_000000.npy")
        metadata = json.loads((src_dir / "metadata.json").read_text())
        if shard.shape[0] != len(prompts):
            raise ValueError(f"layer {layer}: {shard.shape[0]} rows vs {len(prompts)} prompts")
        for dictionary in DICTIONARIES:
            rows = indices[f"train_{dictionary}"]
            dst = work / dictionary / f"layer_{layer}" / "activations"
            dst.mkdir(parents=True, exist_ok=True)
            np.save(dst / "shard_000000.npy", shard[rows])
            meta = dict(metadata)
            meta.update({"total_tokens": len(rows), "split": dictionary, "split_seed": args.seed})
            (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
        for scenario in SCENARIOS:
            rows = indices[f"eval_{scenario}"]
            dst = work / "eval" / scenario
            dst.mkdir(parents=True, exist_ok=True)
            np.save(dst / f"layer_{layer}.npy", shard[rows])
        del shard
        print(f"layer {layer}: wrote {', '.join(DICTIONARIES)} + eval matrices")

    for scenario in SCENARIOS:
        rows = indices[f"eval_{scenario}"]
        (work / "eval" / scenario / "prompts.json").write_text(
            json.dumps([prompts[r] for r in rows], indent=0)
        )

    manifest = {
        "seed": args.seed,
        "evalFrac": args.eval_frac,
        "vectors": len(prompts),
        "uniquePrompts": len(set(prompts)),
        "components": {s: len(grouped[s]) for s in SCENARIOS},
        "evalComponents": {s: len(eval_roots[s]) for s in SCENARIOS},
        "rows": {name: len(rows) for name, rows in indices.items()},
        "evalContrastTypes": {
            scenario: dict(
                Counter(
                    rec["contrastType"]
                    for rec in records
                    if rec["scenario"] == scenario
                    and components[rec["anchor"]] in eval_roots[scenario]
                )
            )
            for scenario in SCENARIOS
        },
        "layers": args.layers,
    }
    (work / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("components", "evalComponents", "rows")}, indent=2))
    print(f"Wrote {work}/splits.json")


if __name__ == "__main__":
    main()
