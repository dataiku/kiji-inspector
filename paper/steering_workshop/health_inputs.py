#!/usr/bin/env python3
"""Extract the dictionary-health inputs from the activation captures.

The health screen is a prerequisite for the paper's whole reading --- a layer
whose code is almost entirely constant has nothing for a cue analysis to work
with, however well it reconstructs --- so it should be reproducible from the
published artifacts rather than taken on trust from the report.

The captures themselves are 478 MB across the grid and stay unpublished. But
the screen needs very little of them: per pair prompt, which features are
active and the L0, plus the per-prompt explained variance. That is four orders
of magnitude smaller (0.13 MB for the largest scenario against 248 MB), so
there is no reason not to ship it.

Only positive activations count as active, matching the screen itself; storing
the indices alone means the file cannot be used to reconstruct activations,
only to recount the code.

Usage::

    python paper/steering_workshop/health_inputs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEER = ROOT / "demo" / "steering"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
NAME = "health_inputs.json"


def compact(evaluation: dict) -> dict:
    """The subset of a capture the health screen actually reads."""
    names = [p.get("step") if isinstance(p, dict) else p for p in evaluation["prompts"]]
    pair_idx = [i for i, n in enumerate(names) if str(n).endswith(("_A", "_B"))]
    return {
        "note": "dictionary-health inputs only; positive-activation feature ids, "
                "not activations, so no capture can be reconstructed from this",
        "pairPrompts": [names[i] for i in pair_idx],
        "layers": [
            {
                "layer": blk["layer"],
                # restricted to the pair prompts, as the screen reads it
                "l0": [blk["l0"][i] for i in pair_idx],
                # kept over every prompt, as the screen reads it
                "explainedVariance": blk["explained_variance"],
                "activeFeatures": [
                    sorted(f["index"] for f in blk["active_features"][i] if f["activation"] > 0)
                    for i in pair_idx
                ],
            }
            for blk in evaluation["layers"]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = ap.parse_args()

    written = 0
    for capture in sorted(STEER.glob("*/output/capture/evaluation.json")):
        scenario = capture.relative_to(STEER).parts[0]
        out = args.artifacts / scenario / "capture" / NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(compact(json.loads(capture.read_text())), indent=1) + "\n")
        print(f"  {scenario:28s} {capture.stat().st_size / 1e6:8.1f} MB -> "
              f"{out.stat().st_size / 1e6:6.3f} MB")
        written += 1
    print(f"wrote {written} health-input files")


if __name__ == "__main__":
    main()
