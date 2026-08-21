#!/usr/bin/env python3
"""Train the spec-sheet SAEs from the split shards ``build_splits.py`` wrote.

Four dictionaries per layer, all with the exact configuration of the shipped
six-layer run (d_sae 10752, batch 256, 10 epochs, target L0 75, auto-calibrated
thresholds, no resampling — ``output/layer_43/sae_checkpoints/config.json``):

* ``home_repair_only``    — trained on home_repair training vectors only
* ``tool_selection_only`` — trained on tool_selection training vectors only
* ``joint``               — both scenarios minus both eval sets (fair ceiling;
                            the shipped SAEs saw every vector, so they cannot
                            give an honest held-out number)
* ``joint_seed123``       — same data as ``joint``, training seed 123: how much
                            of the dictionary is stable under retraining?

Checkpoints land in ``output/saes/<dictionary>/layer_<N>/sae_checkpoints/`` —
the same layout as ``output/``, so intervention scripts accept them via
``--sae-local-dir``.

Usage:
    uv run python demo/spec_sheet/train_split_saes.py [--layers 43 27] [--dictionaries joint]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent

# (dictionary name, shard directory under splits/, training seed)
DICTIONARIES: list[tuple[str, str, int]] = [
    ("home_repair_only", "home_repair_only", 42),
    ("tool_selection_only", "tool_selection_only", 42),
    ("joint", "joint", 42),
    ("joint_seed123", "joint", 123),
]


def canonical_config(checkpoint_dir: str, seed: int):
    """The shipped six-layer run's configuration, with only output/seed varied."""
    from kiji_inspector.training import SAETrainingConfig

    return SAETrainingConfig(
        d_sae=10752,
        batch_size=256,
        learning_rate=3e-4,
        l1_coefficient=5e-3,
        target_l0=75.0,
        l1_max=0.1,
        auto_calibrate_threshold=True,
        resample_dead_features=False,
        num_epochs=10,
        output_dir=checkpoint_dir,
        seed=seed,
    )


def main() -> None:
    from kiji_inspector.training import train_sae

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default=str(_DEMO_DIR / "output" / "splits"))
    parser.add_argument("--output-dir", default=str(_DEMO_DIR / "output" / "saes"))
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 27, 34, 43])
    parser.add_argument("--dictionaries", nargs="+", default=[name for name, _, _ in DICTIONARIES])
    args = parser.parse_args()

    splits = Path(args.splits_dir)
    out_root = Path(args.output_dir)
    chosen = [entry for entry in DICTIONARIES if entry[0] in set(args.dictionaries)]
    unknown = set(args.dictionaries) - {name for name, _, _ in DICTIONARIES}
    if unknown:
        raise SystemExit(f"unknown dictionaries: {sorted(unknown)}")

    summary_path = out_root / "training_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for name, shard_dir, seed in chosen:
        for layer in args.layers:
            activations = splits / shard_dir / f"layer_{layer}" / "activations"
            checkpoint_dir = out_root / name / f"layer_{layer}" / "sae_checkpoints"
            final = checkpoint_dir / "sae_final.pt"
            if final.exists():
                print(f"skip {name} layer {layer} (exists)")
                continue
            started = time.time()
            print(f"=== {name} layer {layer} (seed {seed}) ===", flush=True)
            train_sae(
                activations_dir=str(activations),
                config=canonical_config(str(checkpoint_dir), seed),
            )
            summary.append(
                {
                    "dictionary": name,
                    "layer": layer,
                    "seed": seed,
                    "seconds": round(time.time() - started, 1),
                }
            )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
