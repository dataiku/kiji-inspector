#!/usr/bin/env python3
"""Assemble ``output/ui_data.json`` for the spec-sheet page from whatever
result files exist, and embed it into ``index.html`` as the first-paint
fallback (same mechanism as the other demo pages).

Sources (each optional — the page hides missing sections):
* ``output/splits/splits.json``                       — split manifest
* ``output/saes/training_summary.json``               — dictionary training log
* ``output/transfer_results.json``                    — EV transfer + matching
* ``output/workbench_results.json``                   — probes + signal recovery
* ``output/population/population_summary.json``       — full-population flip census
* ``../tool_selection/output/steering_layer<L>/steering_results.json``
                                                      — causal battery per layer
* ``output/robustness/<dictionary>/steering_results.json``
                                                      — causal battery per dictionary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_TS_OUTPUT = _DEMO_DIR.parent / "tool_selection" / "output"
_START = "/* MOCK_DATA_START */"
_END = "/* MOCK_DATA_END */"
_MIN_EFFECT = 0.02  # same floor as the tool_selection demo


def summarize_steering(steering: dict) -> dict:
    """Condense one attribute_pairs.py result file into per-side effect counts."""
    sides = []
    set_matched_sides = 0
    ceiling_sides = 0
    for pid, by_side in (steering.get("attribution") or {}).items():
        for side, entry in by_side.items():
            all_rows = entry.get("allRows") or {}
            control = float(entry.get("controlThreshold") or 0.0)
            # ``control`` is drawn to one cue family's mass, so it is the band
            # for a single row.  The all-families arm moves the whole cue set,
            # which carries several times that mass, and needs a band drawn to
            # the set.  Result files written before that arm existed only have
            # the per-family band; fall back to it and say so in the caveat.
            set_raw = entry.get("setControlThreshold")
            set_control = float(control if set_raw is None else set_raw)
            if set_raw is not None:
                set_matched_sides += 1
                if entry.get("setControlMassMatched") is False:
                    ceiling_sides += 1
            all_delta = float(all_rows.get("deltaTarget") or 0.0)
            row_effects = sum(
                1
                for row in entry.get("rows", [])
                if abs(float(row.get("deltaTarget") or 0.0)) > max(control, _MIN_EFFECT)
            )
            sides.append(
                {
                    "pid": pid,
                    "side": side,
                    "targetTool": entry.get("targetTool"),
                    "allDelta": round(all_delta, 4),
                    "allSize": all_rows.get("size"),
                    "control": round(control, 4),
                    "setControl": round(set_control, 4),
                    "setControlMassMatched": entry.get("setControlMassMatched"),
                    "nFamilies": len(entry.get("rows", [])),
                    "familiesBeyondControl": row_effects,
                    "allBeyondControl": abs(all_delta) > max(set_control, _MIN_EFFECT),
                }
            )
    cross = []
    for pid, by_direction in (steering.get("crossPatch") or {}).items():
        for direction, entry in by_direction.items():
            all_rows = entry.get("allRows") or {}
            baseline = entry.get("intoBaselineChoice")
            choice = all_rows.get("choice")
            target = entry.get("targetTool")
            cross.append(
                {
                    "pid": pid,
                    "direction": direction,
                    "targetTool": target,
                    "delta": round(float(all_rows.get("deltaTarget") or 0.0), 4),
                    "choice": choice,
                    "baseline": baseline,
                    "flippedToTarget": bool(
                        choice and target and choice == target and baseline != target
                    ),
                }
            )
    return {
        "layer": steering.get("layer"),
        "saeCheckpoint": steering.get("saeCheckpoint"),
        "thresholdOffset": steering.get("thresholdOffset"),
        "hfFastPath": steering.get("hfFastPath"),
        "sides": sides,
        "nSides": len(sides),
        "sidesAllBeyondControl": sum(1 for s in sides if s["allBeyondControl"]),
        "meanAbsAllDelta": round(
            sum(abs(s["allDelta"]) for s in sides) / len(sides) if sides else 0.0, 4
        ),
        "cross": cross,
        "crossFlips": sum(1 for c in cross if c["flippedToTarget"]),
        "nCross": len(cross),
        "setControlSides": set_matched_sides,
        "setControlCeilings": ceiling_sides,
        "setControlCaveat": _set_control_caveat(len(sides), set_matched_sides, ceiling_sides),
    }


def _set_control_caveat(n_sides: int, set_matched: int, ceilings: int) -> str:
    """What the reader needs to know about the band beside the all-families row."""
    if not n_sides or set_matched < n_sides:
        return (
            "the all-families arm is compared against the max of per-family "
            "mass-matched controls, which are smaller sets; per-family rows are "
            "the like-for-like comparison"
        )
    if ceilings:
        return (
            f"controls are matched to the whole cue set; on {ceilings} of {n_sides} "
            "sides the cue set outweighs every other active feature, so its band "
            "is the whole eligible pool rather than a matched draw"
        )
    return "controls are matched to the whole cue set on every side"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def assemble(
    demo_dir: Path = _DEMO_DIR, ts_output: Path = _TS_OUTPUT, layers=(6, 13, 20, 27, 34, 43)
) -> dict:
    out = demo_dir / "output"
    data: dict = {
        "splits": _load(out / "splits" / "splits.json"),
        "training": _load(out / "saes" / "training_summary.json"),
        "transfer": _load(out / "transfer_results.json"),
        "workbench": _load(out / "workbench_results.json"),
        "population": _load(out / "population" / "population_summary.json"),
        "populationMeta": _load(out / "population" / "readout_meta.json"),
    }
    depth = {}
    for layer in layers:
        steering = _load(ts_output / f"steering_layer{layer}" / "steering_results.json")
        if steering:
            depth[str(layer)] = summarize_steering(steering)
    data["depth"] = depth or None
    robustness = {}
    robustness_dir = out / "robustness"
    if robustness_dir.exists():
        for run_dir in sorted(robustness_dir.iterdir()):
            steering = _load(run_dir / "steering_results.json")
            if steering:
                robustness[run_dir.name] = summarize_steering(steering)
    data["robustness"] = robustness or None
    return data


def embed(html: str, ui_data: dict) -> str:
    start = html.index(_START) + len(_START)
    end = html.index(_END)
    payload = json.dumps(ui_data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</script", "<\\/script")
    return f"{html[:start]} {payload} {html[end:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default=str(_DEMO_DIR / "index.html"))
    parser.add_argument("--output", default=str(_DEMO_DIR / "output" / "ui_data.json"))
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    data = assemble()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    present = [k for k, v in data.items() if v]
    print(f"Wrote {out} (sections: {', '.join(present)})")
    html_path = Path(args.html)
    if not args.no_embed and html_path.exists():
        html_path.write_text(embed(html_path.read_text(), data))
        print(f"Embedded into {html_path}")


if __name__ == "__main__":
    main()
