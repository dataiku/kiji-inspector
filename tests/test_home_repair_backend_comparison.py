from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def comparison():
    directory = Path(__file__).parents[1] / "demo" / "home_repair"
    path = directory / "compare_sae_backends.py"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("compare_sae_backends", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_vector_parity_identical_and_scaled(comparison):
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    identical = comparison._vector_parity(values, values)
    scaled = comparison._vector_parity(values, values * 2)

    assert identical["mean_cosine_similarity"] == pytest.approx(1)
    assert identical["mean_relative_l2_error"] == pytest.approx(0)
    assert scaled["mean_cosine_similarity"] == pytest.approx(1)
    assert scaled["mean_rms_ratio_vllm_over_hf"] == pytest.approx(2)
    assert scaled["best_fit_scale_vllm_from_hf"] == pytest.approx(2)
    assert scaled["relative_error_after_best_fit_scale"] == pytest.approx(0)


def test_feature_rank_parity(comparison):
    hf = {
        "layer": 27,
        "top_features": [[{"index": 1}, {"index": 2}, {"index": 3}]],
    }
    vllm = {
        "layer": 27,
        "top_features": [[{"index": 1}, {"index": 2}, {"index": 4}]],
    }

    result = comparison._feature_rank_parity(hf, vllm)

    assert result["mean_top_k_jaccard"] == pytest.approx(0.5)
    assert 0 < result["mean_rbo"] < 1
