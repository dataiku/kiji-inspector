"""
Baseline comparison methods for SAE-based tool-selection interpretability.

Provides two baselines that operate on the same raw activations used by the SAE:
  1. Linear probe (logistic regression) -- supervised tool prediction
  2. PCA + k-means clustering -- unsupervised structure discovery

These baselines answer the reviewer question: "Is the SAE doing meaningful
work beyond what simpler methods achieve?"

Usage:
    python baselines.py <activations_dir> <pairs_dir> [output_dir]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Cap BLAS threads before importing numpy/sklearn to avoid OpenBLAS OOM
# when the system has many cores.
_N_CORES = os.environ.get("BASELINES_N_CORES", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_CORES)
os.environ.setdefault("MKL_NUM_THREADS", _N_CORES)
os.environ.setdefault("OMP_NUM_THREADS", _N_CORES)

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_activations_with_labels(
    activations_dir: str | Path,
    pairs_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load raw activations and join with tool labels from contrastive pairs.

    Args:
        activations_dir: Directory with shard_*.npy, prompts.json, metadata.json.
        pairs_dir: Directory with shard_*.parquet contrastive pairs.

    Returns:
        activations: (N, d_model) float32 array.
        labels: (N,) integer-encoded tool labels.
        pair_ids: (N,) integer pair indices (for group-aware splitting).
        label_names: List of tool name strings (index = label int).
    """
    from kiji_inspector.data.contrastive_dataset import ContrastiveDataset

    activations_dir = Path(activations_dir)
    pairs_dir = Path(pairs_dir)

    # Load prompts.json (same order as activation shards)
    with open(activations_dir / "prompts.json") as f:
        all_prompts: list[str] = json.load(f)

    # Load activation shards
    shard_paths = sorted(activations_dir.glob("shard_*.npy"))
    shards = [np.load(sp) for sp in tqdm(shard_paths, desc="Loading shards", unit="shard")]
    all_activations = np.concatenate(shards, axis=0).astype(np.float32)
    del shards

    # prompts.json is the source of truth for ordering.  Activation shards
    # may contain extra rows from multiple extraction runs — truncate to
    # match the prompt list.
    n_prompts = len(all_prompts)
    if all_activations.shape[0] > n_prompts:
        print(
            f"  Note: {all_activations.shape[0]} activation rows but {n_prompts} prompts. "
            f"Using first {n_prompts} rows."
        )
        all_activations = all_activations[:n_prompts]
    elif all_activations.shape[0] < n_prompts:
        print(
            f"  Note: {all_activations.shape[0]} activation rows but {n_prompts} prompts. "
            f"Truncating prompt list to {all_activations.shape[0]}."
        )
        all_prompts = all_prompts[: all_activations.shape[0]]

    # Load contrastive pairs and build prompt -> (tool, pair_index) mapping
    dataset = ContrastiveDataset.from_parquet(pairs_dir)

    prompt_to_tool: dict[str, str] = {}
    prompt_to_pair: dict[str, int] = {}
    for i, pair in enumerate(dataset.pairs):
        # Normalize compound tool labels ("api_call, code_execute" -> "api_call")
        # The LLM generator sometimes produces multi-tool answers.
        anchor_tool = pair.anchor_tool.split(",")[0].strip()
        contrast_tool = pair.contrast_tool.split(",")[0].strip()
        prompt_to_tool[pair.anchor_prompt] = anchor_tool
        prompt_to_tool[pair.contrast_prompt] = contrast_tool
        prompt_to_pair[pair.anchor_prompt] = i
        prompt_to_pair[pair.contrast_prompt] = i

    # Join: keep only prompts that have tool labels
    keep_indices = []
    tools = []
    pair_indices = []
    for idx, prompt in enumerate(all_prompts):
        if prompt in prompt_to_tool:
            keep_indices.append(idx)
            tools.append(prompt_to_tool[prompt])
            pair_indices.append(prompt_to_pair[prompt])

    activations = all_activations[keep_indices]
    del all_activations

    # Encode tool names as integers
    unique_tools = sorted(set(tools))
    tool_to_int = {t: i for i, t in enumerate(unique_tools)}
    labels = np.array([tool_to_int[t] for t in tools], dtype=np.int64)
    pair_ids = np.array(pair_indices, dtype=np.int64)

    print(f"  Loaded {len(labels)} labeled activations across {len(unique_tools)} tools")
    print(f"  Tools: {unique_tools}")

    return activations, labels, pair_ids, unique_tools


# ---------------------------------------------------------------------------
# Linear probe baseline
# ---------------------------------------------------------------------------


def _train_fold(fold, train_idx, test_idx, activations, labels, n_splits):
    """Train and evaluate a single CV fold (for parallel execution)."""
    # Limit BLAS threads per worker so parallel folds don't over-subscribe
    import os

    per_fold = max(1, int(_N_CORES) // n_splits)
    os.environ["OPENBLAS_NUM_THREADS"] = str(per_fold)
    os.environ["MKL_NUM_THREADS"] = str(per_fold)
    os.environ["OMP_NUM_THREADS"] = str(per_fold)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    X_train, X_test = activations[train_idx], activations[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="saga")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"    Fold {fold + 1}/{n_splits}: accuracy={acc:.4f}, macro_f1={f1:.4f}")
    return acc, f1


def run_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    label_names: list[str],
    n_splits: int = 5,
) -> dict:
    """Train a logistic regression on raw activations to predict tool choice.

    Uses GroupKFold to prevent data leakage (both sides of a pair stay in
    the same fold).  Folds run in parallel via joblib.

    Returns:
        Dict with accuracy, macro_f1, per-fold results, and CIs.
    """
    from joblib import Parallel, delayed
    from scipy.stats import sem as _sem
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(activations, labels, pair_ids))

    # Run folds in parallel.  n_jobs = n_splits (typically 5) so each fold
    # gets ~3 BLAS threads on a 16-core machine.  Use "loky" backend to
    # avoid GIL contention.
    n_jobs = min(n_splits, int(_N_CORES))
    print(f"    Running {n_splits} folds with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_train_fold)(fold, train_idx, test_idx, activations, labels, n_splits)
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    fold_accuracies = [r[0] for r in results]
    fold_f1s = [r[1] for r in results]

    acc_arr = np.array(fold_accuracies)
    f1_arr = np.array(fold_f1s)

    result = {
        "method": "linear_probe",
        "model": "LogisticRegression(saga)",
        "n_splits": n_splits,
        "accuracy": {
            "mean": round(float(acc_arr.mean()), 4),
            "std": round(float(acc_arr.std(ddof=1)), 4),
            "sem": round(float(_sem(acc_arr)), 4),
            "per_fold": [round(v, 4) for v in fold_accuracies],
        },
        "macro_f1": {
            "mean": round(float(f1_arr.mean()), 4),
            "std": round(float(f1_arr.std(ddof=1)), 4),
            "sem": round(float(_sem(f1_arr)), 4),
            "per_fold": [round(v, 4) for v in fold_f1s],
        },
        "n_samples": len(labels),
        "n_classes": len(label_names),
        "label_names": label_names,
    }

    print(
        f"  Linear probe: accuracy={result['accuracy']['mean']:.4f} "
        f"+/- {result['accuracy']['sem']:.4f} (SEM), "
        f"macro_f1={result['macro_f1']['mean']:.4f}"
    )

    return result


# ---------------------------------------------------------------------------
# PCA + k-means baseline
# ---------------------------------------------------------------------------


def run_pca_kmeans(
    activations: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    n_components: int = 50,
    n_clusters: int | None = None,
) -> dict:
    """PCA dimensionality reduction + k-means clustering.

    Measures how well unsupervised clusters align with tool labels.

    Returns:
        Dict with cluster purity, NMI, ARI, and PCA variance explained.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler

    if n_clusters is None:
        n_clusters = len(label_names)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    variance_explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA: {n_components} components explain {variance_explained:.1%} of variance")

    # k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Cluster purity
    purity = _cluster_purity(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    result = {
        "method": "pca_kmeans",
        "n_components": n_components,
        "variance_explained": round(variance_explained, 4),
        "n_clusters": n_clusters,
        "cluster_purity": round(purity, 4),
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "n_samples": len(labels),
        "n_classes": len(label_names),
    }

    print(f"  PCA+k-means: purity={purity:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")

    return result


def _cluster_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Cluster purity: fraction of samples assigned to the majority class in each cluster."""
    correct = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        most_common = np.bincount(true_labels[mask]).argmax()
        correct += (true_labels[mask] == most_common).sum()
    return correct / len(true_labels)


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------


def run_all_baselines(
    activations_dir: str | Path,
    pairs_dir: str | Path,
    output_dir: str | Path = "baselines_output",
) -> dict:
    """Run all baselines and save a unified report.

    Args:
        activations_dir: Step 2 output (shard_*.npy + prompts.json).
        pairs_dir: Step 1 output (shard_*.parquet contrastive pairs).
        output_dir: Where to save baselines_report.json.

    Returns:
        Combined results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("  Baseline Comparisons")
    print("=" * 70)

    t0 = time.time()
    activations, labels, pair_ids, label_names = load_activations_with_labels(
        activations_dir, pairs_dir
    )
    t_load = time.time() - t0
    print(f"  Data loading: {t_load:.1f}s")

    # Run sequentially — both methods use BLAS-level parallelism internally,
    # so threading on top doubles thread counts and can OOM OpenBLAS.
    print("\n  Running linear probe...")
    t0 = time.time()
    linear_result = run_linear_probe(activations, labels, pair_ids, label_names)
    t_probe = time.time() - t0
    print(f"  Linear probe: {t_probe:.1f}s")

    print("\n  Running PCA + k-means...")
    t0 = time.time()
    pca_result = run_pca_kmeans(activations, labels, label_names)
    t_pca = time.time() - t0
    print(f"  PCA + k-means: {t_pca:.1f}s")

    t_elapsed = time.time() - t_total

    report = {
        "linear_probe": linear_result,
        "pca_kmeans": pca_result,
        "timing": {
            "data_loading_s": round(t_load, 1),
            "linear_probe_s": round(t_probe, 1),
            "pca_kmeans_s": round(t_pca, 1),
            "total_s": round(t_elapsed, 1),
        },
    }

    report_path = output_dir / "baselines_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Saved baselines report: {report_path}")
    print(f"  Total elapsed: {t_elapsed:.1f}s ({t_elapsed / 60:.1f}m)")
    print("=" * 70)

    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python baselines.py <activations_dir> <pairs_dir> [output_dir]")
        sys.exit(1)

    act_dir = sys.argv[1]
    pairs = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "baselines_output"
    run_all_baselines(act_dir, pairs, out)
