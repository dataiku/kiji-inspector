from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def evaluator():
    path = Path(__file__).parents[1] / "demo" / "home_repair" / "evaluate_sae_layers.py"
    spec = importlib.util.spec_from_file_location("evaluate_sae_layers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_similarity_metrics(evaluator):
    assert evaluator._cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(
        0
    )
    assert evaluator._cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        1
    )
    assert evaluator._jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert evaluator._truncated_rbo([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
    assert evaluator._truncated_rbo([1, 2, 3], [4, 5, 6]) == pytest.approx(0)


def test_prompt_metadata_has_one_initial_decision_per_problem(evaluator):
    metadata = evaluator._prompt_metadata()

    assert len(metadata) == 3
    assert len({item["problem"] for item in metadata}) == 3
    assert all(item["step"].endswith("_InitialDecision") for item in metadata)
    assert {item["tool"] for item in metadata} <= {"PartsSearch", "TutorialSearch", "ProQuote"}


def test_prompt_metadata_with_probes_keeps_base_rows_first(evaluator):
    metadata = evaluator._prompt_metadata(include_contrasts=True, include_probes=True)

    assert evaluator._base_rows(metadata) == [0, 1, 2]
    assert len(metadata) == 3 + 3 + 9 + 6 + 3  # + one open (no-ask) request per problem
    assert {item["kind"] for item in metadata} == {
        "base",
        "contrast",
        "paraphrase",
        "control",
        "open",
    }


def test_problem_separation_compares_every_distinct_problem(evaluator):
    features = np.eye(3, dtype=np.float32)
    rankings = [[0], [1], [2]]
    metadata = [
        {"step": "a", "problem": "a", "tool": "PartsSearch"},
        {"step": "b", "problem": "b", "tool": "TutorialSearch"},
        {"step": "c", "problem": "c", "tool": "TutorialSearch"},
    ]

    result = evaluator._matched_separation(features, rankings, metadata)

    assert result["num_pairs"] == 3
    assert result["mean_cosine_distance"] == pytest.approx(1)
    assert result["mean_top_k_jaccard"] == pytest.approx(0)


def test_scenario_metrics_distinguish_correct_and_contaminating_labels(evaluator):
    features = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    rankings = [[0, 1, 2]]
    metadata = [
        {
            "step": "dishwasher_test",
            "problem": "dishwasher_leak",
            "tool": "ManualCheck",
        }
    ]
    labels = {
        "0": {"label": "Dishwasher door gasket replacement"},
        "1": {"label": "Gas appliance safety warnings"},
        "2": {"label": "Generic repair request"},
    }

    result = evaluator._scenario_label_metrics(features, rankings, metadata, labels)

    assert result["correct_mass_share"] == pytest.approx(0.5)
    assert result["contamination_mass_share"] == pytest.approx(1 / 3)
    assert result["scenario_purity"] == pytest.approx(0.6)
    assert result["prompts_with_correct_feature"] == 1
