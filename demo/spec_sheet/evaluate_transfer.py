#!/usr/bin/env python3
"""Cross-scenario transfer evaluation of the spec-sheet SAE dictionaries.

For every dictionary (``home_repair_only``, ``tool_selection_only``, ``joint``,
``joint_seed123`` from ``train_split_saes.py``, plus the *shipped* six-layer
SAEs in ``output/`` — flagged, because they saw every eval vector during
training) and every layer, this script measures on the two held-out eval sets:

* **ev** — explained variance of the raw-space reconstruction against the
  eval set's per-dimension mean (shift/scale invariant, so identical in the
  SAE's normalized space);
* **cosineMean**, **l0Mean**, **featuresUsed** — reconstruction direction,
  sparsity, and how much of the dictionary the domain exercises;
* **evAffineAligned** — the norm-matched control: eval vectors are first
  affine-aligned (per-dimension mean + global RMS) to the dictionary's
  training distribution.  If cross-domain EV recovers here, the transfer gap
  is first/second-moment shift, not dictionary content;
* **PCA-75 baseline** — a rank-``target_l0`` PCA fit on each training split,
  evaluated on both domains: the dense linear reference that separates "the
  domains occupy different subspaces" from "the SAE specifically fails".

It also matches features across dictionaries two ways, at two firing-rate
cuts (dead and ultra-rare rows sit at their initialization, and identical
seeds share initialization, so an uncut same-seed comparison is trivially
inflated — cut 0 matches at ~0.9 decoder cosine for that reason alone):

* **decoder cosine** — do the two dictionaries use the same *directions*?
* **activation correlation** on the union of both eval sets — do their
  features fire on the same *prompts* (functional identity), with a
  prompt-permutation null?

The two can disagree: across retraining seeds the decoder directions are
nearly disjoint while about half the frequently-firing features keep a
functional counterpart.

Usage:
    uv run python demo/spec_sheet/evaluate_transfer.py [--layers 43 27]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
SCENARIOS = ("home_repair", "tool_selection")
NEW_DICTIONARIES = ("home_repair_only", "tool_selection_only", "joint", "joint_seed123")
TRAIN_SPLIT_OF = {
    "home_repair_only": "home_repair_only",
    "tool_selection_only": "tool_selection_only",
    "joint": "joint",
    "joint_seed123": "joint",
}
MATCH_PAIRS = [
    ("joint", "joint_seed123"),
    ("home_repair_only", "tool_selection_only"),
    ("tool_selection_only", "home_repair_only"),
    ("home_repair_only", "joint"),
    ("tool_selection_only", "joint"),
    ("shipped", "joint"),
]


def explained_variance(x, reconstruction) -> float:
    """1 - residual/variance with per-dimension centering over the eval set."""
    centered = x - x.mean(dim=0, keepdim=True)
    total = float(centered.square().sum().item())
    if total <= 0:
        return 0.0
    return 1.0 - float((x - reconstruction).square().sum().item()) / total


def rms_scale_of(x) -> float:
    """The trainer's scalar scale: sqrt(mean over dims of per-dim variance)."""
    return float(x.var(dim=0, unbiased=False).mean().sqrt().item())


def affine_align(x, mean_to, scale_to):
    """Map x's per-dimension mean and global RMS onto the target statistics."""
    mean_from = x.mean(dim=0, keepdim=True)
    scale_from = rms_scale_of(x)
    if scale_from <= 0:
        return x
    return (x - mean_from) * (scale_to / scale_from) + mean_to


def pca_fit(train, k: int):
    """Rank-k PCA on the training matrix; returns (mean, components [k, d])."""
    import torch

    mean = train.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(train - mean, q=min(k + 16, train.shape[1]), niter=4)
    return mean, v[:, :k].T.contiguous()


def pca_reconstruct(x, mean, components):
    centered = x - mean
    return centered @ components.T @ components + mean


def match_features(decoder_a, decoder_b) -> dict:
    """Best decoder-cosine match in B for every row of A (rows are features)."""
    import torch

    a = torch.nn.functional.normalize(decoder_a.float(), dim=1)
    b = torch.nn.functional.normalize(decoder_b.float(), dim=1)
    best = (a @ b.T).max(dim=1).values
    return {
        "n": int(a.shape[0]),
        "meanMaxCosine": round(float(best.mean().item()), 4),
        "fracAtLeast07": round(float((best >= 0.7).float().mean().item()), 4),
        "fracAtLeast09": round(float((best >= 0.9).float().mean().item()), 4),
    }


def functional_match(features_a, features_b) -> dict:
    """Best activation-correlation match in B for every feature (column) of A."""

    def znorm(matrix):
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        return centered / (centered.std(dim=0, keepdim=True) + 1e-6)

    n = features_a.shape[0]
    corr = znorm(features_a).T @ znorm(features_b) / max(n - 1, 1)
    best = corr.max(dim=1).values
    return {
        "n": int(features_a.shape[1]),
        "meanBestCorr": round(float(best.mean().item()), 4),
        "fracAtLeast07": round(float((best >= 0.7).float().mean().item()), 4),
        "fracAtLeast09": round(float((best >= 0.9).float().mean().item()), 4),
    }


def main() -> None:
    import numpy as np
    import torch

    from kiji_inspector.core.sae_core import JumpReLUSAE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default=str(_DEMO_DIR / "output" / "splits"))
    parser.add_argument("--saes-dir", default=str(_DEMO_DIR / "output" / "saes"))
    parser.add_argument("--shipped-dir", default=str(_REPO_ROOT / "output"))
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 27, 34, 43])
    parser.add_argument("--pca-rank", type=int, default=75)
    parser.add_argument("--output", default=str(_DEMO_DIR / "output" / "transfer_results.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = Path(args.splits_dir)
    saes_root = Path(args.saes_dir)
    torch.manual_seed(0)

    def checkpoint_dir(dictionary: str, layer: int) -> Path:
        root = Path(args.shipped_dir) if dictionary == "shipped" else saes_root / dictionary
        return root / f"layer_{layer}" / "sae_checkpoints"

    results: dict = {
        "definitions": {
            "ev": "1 - sum((x - x_hat)^2) / sum((x - mean_eval)^2), raw space, per-dim mean",
            "evAffineAligned": "same, after aligning eval mean+RMS to the training stats",
            "pcaRank": args.pca_rank,
            "shippedCaveat": "the shipped SAEs saw every eval vector during training",
        },
        "layers": {},
    }

    for layer in args.layers:
        evals = {
            scenario: torch.from_numpy(np.load(splits / "eval" / scenario / f"layer_{layer}.npy"))
            .float()
            .to(device)
            for scenario in SCENARIOS
        }

        train_stats: dict[str, tuple] = {}
        pca_models: dict[str, tuple] = {}
        for split in ("home_repair_only", "tool_selection_only", "joint"):
            shard = np.load(splits / split / f"layer_{layer}" / "activations" / "shard_000000.npy")
            train = torch.from_numpy(shard).float().to(device)
            train_stats[split] = (train.mean(dim=0, keepdim=True), rms_scale_of(train))
            pca_models[split] = pca_fit(train, args.pca_rank)
            del train, shard

        layer_out: dict = {"dictionaries": {}, "pcaBaseline": {}, "matching": {}}

        for split, (mean, components) in pca_models.items():
            layer_out["pcaBaseline"][split] = {
                scenario: round(
                    explained_variance(
                        evals[scenario], pca_reconstruct(evals[scenario], mean, components)
                    ),
                    4,
                )
                for scenario in SCENARIOS
            }

        saes: dict[str, object] = {}
        for dictionary in (*NEW_DICTIONARIES, "shipped"):
            ckpt = checkpoint_dir(dictionary, layer) / "sae_final.pt"
            if not ckpt.exists():
                continue
            sae = JumpReLUSAE.from_pretrained(str(ckpt), device=device).float()
            saes[dictionary] = sae
            entry: dict = {"sawEval": dictionary == "shipped", "evalSets": {}}
            train_split = TRAIN_SPLIT_OF.get(dictionary)
            for scenario in SCENARIOS:
                x = evals[scenario]
                with torch.no_grad():
                    features = sae.encode(sae.normalize_input(x))
                    reconstruction = sae.denormalize_output(sae.decode(features))
                    row = {
                        "ev": round(explained_variance(x, reconstruction), 4),
                        "cosineMean": round(
                            float(
                                torch.nn.functional.cosine_similarity(
                                    x - x.mean(dim=0, keepdim=True),
                                    reconstruction - x.mean(dim=0, keepdim=True),
                                    dim=1,
                                )
                                .mean()
                                .item()
                            ),
                            4,
                        ),
                        "l0Mean": round(float((features > 0).float().sum(dim=1).mean().item()), 2),
                        "featuresUsed": int(((features > 0).any(dim=0)).sum().item()),
                        "nEval": int(x.shape[0]),
                    }
                    if train_split is not None:
                        mean_to, scale_to = train_stats[train_split]
                        aligned = affine_align(x, mean_to, scale_to)
                        f2 = sae.encode(sae.normalize_input(aligned))
                        r2 = sae.denormalize_output(sae.decode(f2))
                        row["evAffineAligned"] = round(explained_variance(aligned, r2), 4)
                entry["evalSets"][scenario] = row
            rates = np.load(checkpoint_dir(dictionary, layer) / "firing_rates.npy")
            entry["aliveTrain"] = int((rates > 0).sum())
            layer_out["dictionaries"][dictionary] = entry

        union = torch.cat([evals[s] for s in SCENARIOS])
        feature_cache: dict[str, object] = {}
        rate_cache: dict[str, object] = {}
        with torch.no_grad():
            for dictionary, sae in saes.items():
                feature_cache[dictionary] = sae.encode(sae.normalize_input(union))
                rate_cache[dictionary] = np.load(
                    checkpoint_dir(dictionary, layer) / "firing_rates.npy"
                )
        permutation = torch.randperm(union.shape[0], device=union.device)
        for name_a, name_b in MATCH_PAIRS:
            if name_a not in saes or name_b not in saes:
                continue
            entry = {}
            for cut in (0.001, 0.01):
                mask_a = torch.from_numpy(rate_cache[name_a] >= cut).to(device)
                mask_b = torch.from_numpy(rate_cache[name_b] >= cut).to(device)
                if not bool(mask_a.any()) or not bool(mask_b.any()):
                    continue
                dec_a = saes[name_a].W_dec.detach()[mask_a]
                dec_b = saes[name_b].W_dec.detach()[mask_b]
                features_a = feature_cache[name_a][:, mask_a]
                features_b = feature_cache[name_b][:, mask_b]
                entry[f"rateAtLeast{cut}"] = {
                    "decoder": match_features(dec_a, dec_b),
                    "decoderNull": match_features(dec_a, torch.randn_like(dec_b)),
                    "functional": functional_match(features_a, features_b),
                    "functionalNull": functional_match(features_a, features_b[permutation]),
                }
            layer_out["matching"][f"{name_a}->{name_b}"] = entry
        del saes, evals, pca_models
        torch.cuda.empty_cache() if device == "cuda" else None

        results["layers"][str(layer)] = layer_out
        print(f"layer {layer}: done")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
