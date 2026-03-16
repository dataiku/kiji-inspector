"""Shared statistical utility functions.

Consolidates confidence interval and hypothesis testing helpers
previously duplicated across sae_trainer, fuzzing_evaluator, and ablation.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci_mean(
    data: np.ndarray | list[float], n_bootstrap: int = 10_000, ci: float = 0.95
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.asarray(data)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = 1 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def wilson_score_ci(successes: int, total: int, ci: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    from scipy.stats import norm

    if total == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - ci) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def clopper_pearson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact confidence interval for a proportion."""
    from scipy.stats import beta

    if total == 0:
        return (0.0, 0.0)
    lo = beta.ppf(alpha / 2, successes, total - successes + 1) if successes > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, successes + 1, total - successes) if successes < total else 1.0
    return (float(lo), float(hi))


def binomial_p_value(n_correct: int, n_total: int, baseline: float = 0.5) -> float:
    """One-sided binomial test: is accuracy significantly above baseline?"""
    from scipy.stats import binomtest

    if n_total == 0:
        return 1.0
    result = binomtest(n_correct, n_total, baseline, alternative="greater")
    return float(result.pvalue)
