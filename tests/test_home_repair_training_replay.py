from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def replay():
    directory = Path(__file__).parents[1] / "demo" / "home_repair"
    path = directory / "replay_training_activations.py"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("replay_training_activations", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_pair_selection_spans_full_range(replay):
    assert replay._select_pair_indices(47_693, 6) == [
        0,
        9_538,
        19_077,
        28_615,
        38_154,
        47_692,
    ]


def test_pair_delta_parity(replay):
    stored = np.array(
        [[2.0, 0.0], [0.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    identical = replay._pair_delta_parity(stored, stored)
    scaled = replay._pair_delta_parity(stored, stored * 2)

    assert identical["mean_cosine_similarity"] == pytest.approx(1)
    assert identical["mean_relative_l2_error"] == pytest.approx(0)
    assert scaled["mean_cosine_similarity"] == pytest.approx(1)
    assert scaled["mean_relative_l2_error"] == pytest.approx(1)
