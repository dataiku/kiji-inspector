#!/usr/bin/env python3
"""Probe and label-alignment experiments over the existing 95K-vector shards.

Everything here is computed from data already on disk — the SAE training
activations (captured with the canonical vLLM backend), the pairs parquet,
and the shipped feature labels.  No model forward passes, no new prompts.

Two experiments:

1. **Tool probes** (per layer, per scenario): predict a prompt's tool label
   (the parquet's ``anchor_tool`` / ``contrast_tool``) from
   (a) the fair ``joint`` split-SAE's features (the SAE and the probe both
   never saw the eval components), (b) the raw residual vector, and
   (c) a bag-of-words baseline that sees only the request text.  Train on the
   training-split unique prompts, test on the held-out eval prompts — one
   leak-free split, same for every representation.  The question: do SAE
   features carry the decision signal beyond wording, and how much of the
   raw residual's signal do they keep?

2. **Blind signal recovery** (per layer, per scenario): for held-out pairs,
   do the top anchor-side / contrast-side feature *labels* share content
   words with the matching clause of the pair's ``distinguishing_signal``
   annotation — against a shuffled-signal null?  Uses the shipped SAEs and
   their labels (the only labeled dictionaries; they saw these vectors, so
   this is a descriptive claim about label alignment, not generalization).
   Caveat printed into the output: pairs, labels, and signals all come from
   the same LLM family, so "meaning" here partially inherits the generator's.

Usage:
    uv run python demo/spec_sheet/feature_workbench.py [--layers 43 27]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
SCENARIOS = ("home_repair", "tool_selection")

_STOP = set(
    "a an the of to for from in on at by with and or our my your this that these those is are be "
    "it its as into up out all any some what which who how please can could would should i me we "
    "you he she they them their there here while whereas anchor contrast request requests requires "
    "require asks ask asking needs need wants want seeks seek user".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9'-]{2,}", (text or "").lower()) if w not in _STOP}


def signal_sides(signal: str) -> tuple[str, str]:
    """Split a ``distinguishing_signal`` sentence into anchor / contrast clauses."""
    parts = re.split(r"\bwhile\b|\bwhereas\b", signal or "", maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0], parts[1]
    return signal or "", signal or ""


def prompt_tool_map(pair_rows: list[dict]) -> tuple[dict[str, str], int]:
    """prompt -> tool from the parquet; prompts with conflicting labels dropped."""
    tools: dict[str, set[str]] = {}
    for row in pair_rows:
        tools.setdefault(row["anchor_prompt"], set()).add(row["anchor_tool"])
        tools.setdefault(row["contrast_prompt"], set()).add(row["contrast_tool"])
    consistent = {p: next(iter(t)) for p, t in tools.items() if len(t) == 1}
    return consistent, sum(1 for t in tools.values() if len(t) > 1)


def top_side_features(diff, k: int = 3) -> tuple[list[int], list[int]]:
    """Indices of the k most anchor-side (positive) and contrast-side features."""
    import numpy as np

    order = np.argsort(diff)
    anchor = [int(i) for i in order[::-1][:k] if diff[i] > 0]
    contrast = [int(i) for i in order[:k] if diff[i] < 0]
    return anchor, contrast


def label_overlap(feature_indices: list[int], labels: dict, clause_words: set[str]) -> int:
    """How many of the features have a label/description sharing a clause word."""
    hits = 0
    for index in feature_indices:
        entry = labels.get(str(index))
        if not entry:
            continue
        words = content_words(f"{entry.get('label', '')} {entry.get('description', '')}")
        if words & clause_words:
            hits += 1
    return hits


def bow_features(texts: list[str], vocabulary: list[str]):
    import numpy as np

    index = {w: i for i, w in enumerate(vocabulary)}
    x = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row, text in enumerate(texts):
        for w in content_words(text):
            col = index.get(w)
            if col is not None:
                x[row, col] = 1.0
    return x


def build_vocabulary(texts: list[str], min_count: int = 5, cap: int = 4096) -> list[str]:
    from collections import Counter

    counts = Counter(w for t in texts for w in content_words(t))
    frequent = [w for w, c in counts.most_common() if c >= min_count]
    return sorted(frequent[:cap])


def fit_probe(x_train, y_train, x_test, y_test) -> dict:
    """Multinomial logistic probe (torch LBFGS — GPU when available).

    L2 strength matches sklearn's ``LogisticRegression(C=1.0)`` convention
    (0.5·||w||² against a summed data term, i.e. λ = 1/n on the mean loss).
    """
    import numpy as np
    import torch
    from sklearn.metrics import f1_score

    classes = sorted(set(y_train))
    index = {c: i for i, c in enumerate(classes)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device)
    yt = torch.tensor([index[c] for c in y_train], device=device)
    weights = torch.zeros(xt.shape[1], len(classes), device=device, requires_grad=True)
    bias = torch.zeros(len(classes), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias], max_iter=300, tolerance_grad=1e-7, line_search_fn="strong_wolfe"
    )
    l2 = 0.5 / xt.shape[0]

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ weights + bias, yt)
        loss = loss + l2 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = torch.from_numpy(np.asarray(x_test, dtype=np.float32)).to(device) @ weights + bias
        predictions = np.array([classes[i] for i in logits.argmax(dim=1).cpu().numpy()])
    y_test = np.asarray(y_test)
    return {
        "accuracy": round(float((predictions == y_test).mean()), 4),
        "macroF1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "nTrain": int(len(y_train)),
        "nTest": int(len(y_test)),
        "nClasses": int(len(classes)),
    }


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch

    from kiji_inspector.core.sae_core import JumpReLUSAE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-root", default=str(_REPO_ROOT / "output"))
    parser.add_argument("--pairs-dir", default=str(_REPO_ROOT / "output" / "pairs"))
    parser.add_argument("--splits-dir", default=str(_DEMO_DIR / "output" / "splits"))
    parser.add_argument("--saes-dir", default=str(_DEMO_DIR / "output" / "saes"))
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 27, 34, 43])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument("--output", default=str(_DEMO_DIR / "output" / "workbench_results.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.activations_root)
    splits = Path(args.splits_dir)
    rng = np.random.default_rng(0)

    prompts = json.loads(
        (root / f"layer_{args.layers[0]}" / "activations" / "prompts.json").read_text()
    )
    parquet_files = sorted(Path(args.pairs_dir).glob("shard_*.parquet"))
    pair_rows = pd.concat([pd.read_parquet(p) for p in parquet_files]).to_dict("records")
    tool_of, conflicting = prompt_tool_map(pair_rows)

    first_row: dict[str, int] = {}
    for row, prompt in enumerate(prompts):
        first_row.setdefault(prompt, row)

    eval_prompts = {
        scenario: set(json.loads((splits / "eval" / scenario / "prompts.json").read_text()))
        for scenario in SCENARIOS
    }
    scenario_of: dict[str, str] = {}
    signal_of_duo: dict[tuple[str, str], str] = {}
    for row in pair_rows:
        scenario_of[row["anchor_prompt"]] = row["scenario_name"]
        scenario_of[row["contrast_prompt"]] = row["scenario_name"]
        signal_of_duo.setdefault(
            (row["anchor_prompt"], row["contrast_prompt"]), row["distinguishing_signal"]
        )

    datasets: dict[str, dict[str, list]] = {}
    for scenario in SCENARIOS:
        train_p, test_p = [], []
        for prompt in first_row:
            if scenario_of.get(prompt) != scenario or prompt not in tool_of:
                continue
            (test_p if prompt in eval_prompts[scenario] else train_p).append(prompt)
        datasets[scenario] = {"train": train_p, "test": test_p}

    eval_duos: dict[str, list[tuple[str, str]]] = {s: [] for s in SCENARIOS}
    seen_duo: set[tuple[str, str]] = set()
    for duo in range(len(prompts) // 2):
        anchor, contrast = prompts[2 * duo], prompts[2 * duo + 1]
        key = (anchor, contrast)
        scenario = scenario_of[anchor]
        if anchor in eval_prompts[scenario] and key not in seen_duo:
            seen_duo.add(key)
            eval_duos[scenario].append(key)

    results: dict = {
        "caveats": {
            "signalRecovery": (
                "shipped SAEs + labels (only labeled dictionaries; they saw these vectors); "
                "pairs, labels and signals share an LLM family"
            ),
            "conflictingToolPrompts": conflicting,
        },
        "bow": {},
        "layers": {},
    }

    vocab = {s: build_vocabulary(datasets[s]["train"]) for s in SCENARIOS}
    bow_probe = {}
    for scenario in SCENARIOS:
        data = datasets[scenario]
        y_train = np.array([tool_of[p] for p in data["train"]])
        y_test = np.array([tool_of[p] for p in data["test"]])
        bow_probe[scenario] = fit_probe(
            bow_features(data["train"], vocab[scenario]),
            y_train,
            bow_features(data["test"], vocab[scenario]),
            y_test,
        )
        majority = max(set(y_train), key=list(y_train).count)
        bow_probe[scenario]["majorityAccuracy"] = round(float((y_test == majority).mean()), 4)
        bow_probe[scenario]["vocabulary"] = len(vocab[scenario])
    results["bow"] = bow_probe

    for layer in args.layers:
        shard = np.load(root / f"layer_{layer}" / "activations" / "shard_000000.npy")
        layer_out: dict = {"probes": {}, "signalRecovery": {}}

        joint_ckpt = Path(args.saes_dir) / "joint" / f"layer_{layer}" / "sae_checkpoints"
        sae_joint = JumpReLUSAE.from_pretrained(
            str(joint_ckpt / "sae_final.pt"), device=device
        ).float()
        alive = np.load(joint_ckpt / "firing_rates.npy") > 0

        def encode(rows: list[int], sae, mask, shard=shard) -> np.ndarray:
            out = []
            with torch.no_grad():
                for start in range(0, len(rows), 4096):
                    block = torch.from_numpy(shard[rows[start : start + 4096]]).float().to(device)
                    feats = sae.encode(sae.normalize_input(block))
                    out.append(feats[:, mask].cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, int(mask.sum())), dtype=np.float32)

        for scenario in SCENARIOS:
            data = datasets[scenario]
            rows_train = [first_row[p] for p in data["train"]]
            rows_test = [first_row[p] for p in data["test"]]
            y_train = np.array([tool_of[p] for p in data["train"]])
            y_test = np.array([tool_of[p] for p in data["test"]])

            x_train = encode(rows_train, sae_joint, alive)
            x_test = encode(rows_test, sae_joint, alive)
            scale = x_train.std(axis=0) + 1e-6
            probes = {"saeFeatures": fit_probe(x_train / scale, y_train, x_test / scale, y_test)}
            probes["saeFeatures"]["nFeatures"] = int(alive.sum())

            r_train = shard[rows_train].astype(np.float32)
            r_test = shard[rows_test].astype(np.float32)
            mean, std = r_train.mean(axis=0), r_train.std(axis=0) + 1e-6
            probes["residual"] = fit_probe(
                (r_train - mean) / std, y_train, (r_test - mean) / std, y_test
            )
            layer_out["probes"][scenario] = probes

        labels_path = root / f"layer_{layer}" / "activations" / "feature_descriptions.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        shipped = JumpReLUSAE.from_pretrained(
            str(root / f"layer_{layer}" / "sae_checkpoints" / "sae_final.pt"), device=device
        ).float()
        for scenario in SCENARIOS:
            duos = eval_duos[scenario]
            if not duos:
                continue
            anchor_rows = [first_row[a] for a, _ in duos]
            contrast_rows = [first_row[c] for _, c in duos]
            with torch.no_grad():
                fa = (
                    shipped.encode(
                        shipped.normalize_input(
                            torch.from_numpy(shard[anchor_rows]).float().to(device)
                        )
                    )
                    .cpu()
                    .numpy()
                )
                fc = (
                    shipped.encode(
                        shipped.normalize_input(
                            torch.from_numpy(shard[contrast_rows]).float().to(device)
                        )
                    )
                    .cpu()
                    .numpy()
                )

            signals = [signal_of_duo[d] for d in duos]

            def score(assigned_signals: list[str], duos=duos, fa=fa, fc=fc, labels=labels) -> float:
                hits = 0
                for i in range(len(duos)):
                    anchor_clause, contrast_clause = signal_sides(assigned_signals[i])
                    top_a, top_c = top_side_features(fa[i] - fc[i], args.top_k)
                    if label_overlap(top_a, labels, content_words(anchor_clause)) > 0:
                        hits += 1
                    if label_overlap(top_c, labels, content_words(contrast_clause)) > 0:
                        hits += 1
                return hits / (2 * len(duos))

            observed = score(signals)
            null_scores = []
            for _ in range(args.null_draws):
                shuffled = list(signals)
                rng.shuffle(shuffled)
                null_scores.append(score(shuffled))
            layer_out["signalRecovery"][scenario] = {
                "nPairs": len(duos),
                "topK": args.top_k,
                "hitRate": round(observed, 4),
                "nullMean": round(float(np.mean(null_scores)), 4),
                "nullStd": round(float(np.std(null_scores)), 4),
            }

        del shard
        results["layers"][str(layer)] = layer_out
        print(f"layer {layer}: done")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
