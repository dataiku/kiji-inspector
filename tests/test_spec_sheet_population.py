from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def report():
    directory = Path(__file__).parents[1] / "demo" / "spec_sheet"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "population_report", directory / "population_report.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_flip_of(report):
    a = {"toolId": "web_search", "prob": 0.9}
    b = {"toolId": "internal_search", "prob": 0.7}
    assert report.flip_of(a, b) == pytest.approx(0.7)
    assert report.flip_of(a, {"toolId": "web_search", "prob": 0.6}) == 0.0
    assert report.flip_of(a, None) == 0.0
    assert report.flip_of(a, {"toolId": None, "prob": 0.5}) == 0.0


def _pair(
    anchor, contrast, tool_a="web_search", tool_b="internal_search", ct="internal_vs_external"
):
    return {
        "scenario_name": "tool_selection",
        "anchor_prompt": anchor,
        "contrast_prompt": contrast,
        "anchor_tool": tool_a,
        "contrast_tool": tool_b,
        "contrast_type": ct,
    }


def test_census_counts_flips_strata_and_agreement(report):
    pair_rows = [
        _pair("p1", "p2"),
        _pair("p1", "p2"),  # duplicate parquet row collapses
        _pair("p3", "p4", ct="query_vs_mutate"),
        {**_pair("hr1", "hr2"), "scenario_name": "home_repair"},  # other scenario ignored
        _pair("p5", "missing"),  # side without readout skipped
    ]
    decisions = {
        "p1": {"toolId": "web_search", "prob": 0.95},
        "p2": {"toolId": "internal_search", "prob": 0.8},
        "p3": {"toolId": "database_query", "prob": 0.9},
        "p4": {"toolId": "database_query", "prob": 0.85},
        "p5": {"toolId": "web_search", "prob": 0.9},
    }
    records, summary = report.census(pair_rows, decisions, sae_prompts={"p1", "p2"})
    assert len(records) == 2
    assert summary["overall"]["pairs"] == 2
    assert summary["overall"]["flipping"] == 1
    assert summary["overall"]["flipAtLeast06"] == 1
    flip_record = next(r for r in records if r["anchor"] == "p1")
    assert flip_record["flip"] == pytest.approx(0.8)
    assert flip_record["seenBySae"] is True
    assert flip_record["agreeA"] and flip_record["agreeB"]
    non_flip = next(r for r in records if r["anchor"] == "p3")
    assert non_flip["flip"] == 0.0 and non_flip["seenBySae"] is False
    assert summary["seen"]["pairs"] == 1 and summary["unseenBySae"]["pairs"] == 1
    assert summary["byContrastType"]["internal_vs_external"]["flipping"] == 1
    assert "0.9" in summary["reliabilityVsIntent"]
