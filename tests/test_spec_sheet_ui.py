from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_DEMO_DIR = Path(__file__).parents[1] / "demo" / "spec_sheet"


@pytest.fixture(scope="module")
def ui():
    sys.path.insert(0, str(_DEMO_DIR))
    try:
        spec = importlib.util.spec_from_file_location("build_ui", _DEMO_DIR / "build_ui.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _steering_fixture():
    return {
        "layer": 43,
        "saeCheckpoint": "x/sae_final.pt",
        "thresholdOffset": 1.13,
        "hfFastPath": True,
        "attribution": {
            "pair1": {
                "a": {
                    "targetTool": "web_search",
                    "allRows": {"deltaTarget": -0.41, "size": 21},
                    "controlThreshold": 0.03,
                    "rows": [
                        {"deltaTarget": -0.30},  # beyond max(control, 0.02)
                        {"deltaTarget": -0.01},  # inside the floor
                    ],
                },
                "b": {
                    "targetTool": "internal_search",
                    "allRows": {"deltaTarget": -0.005, "size": 3},
                    "controlThreshold": 0.001,
                    "rows": [{"deltaTarget": 0.004}],
                },
            }
        },
        "crossPatch": {
            "pair1": {
                "a_into_b": {
                    "targetTool": "web_search",
                    "intoBaselineChoice": "internal_search",
                    "allRows": {"deltaTarget": 0.55, "choice": "web_search"},
                },
                "b_into_a": {
                    "targetTool": "internal_search",
                    "intoBaselineChoice": "web_search",
                    "allRows": {"deltaTarget": 0.01, "choice": "web_search"},
                },
            }
        },
    }


def test_summarize_steering_counts(ui):
    summary = ui.summarize_steering(_steering_fixture())
    assert summary["nSides"] == 2
    # side a: |−0.41| > max(0.03, 0.02); side b: |−0.005| below the 0.02 floor
    assert summary["sidesAllBeyondControl"] == 1
    side_a = next(s for s in summary["sides"] if s["side"] == "a")
    assert side_a["familiesBeyondControl"] == 1
    assert side_a["allSize"] == 21
    assert summary["nCross"] == 2
    # a_into_b flips the choice to the target; b_into_a keeps web_search
    assert summary["crossFlips"] == 1
    assert summary["meanAbsAllDelta"] == pytest.approx((0.41 + 0.005) / 2, abs=1e-4)
    assert "setControlCaveat" in summary


def _set_matched_fixture():
    """One side where the set-level band is the one that decides the verdict."""
    return {
        "layer": 43,
        "attribution": {
            "pair1": {
                # the cue set moves 0.09; a band drawn to one family says that
                # clears control, a band drawn to the whole set says it does not
                "a": {
                    "targetTool": "web_search",
                    "allRows": {"deltaTarget": -0.09, "size": 6},
                    "controlThreshold": 0.03,
                    "setControlThreshold": 0.12,
                    "setControlMassMatched": True,
                    "rows": [{"deltaTarget": -0.05}],
                },
                # ceiling: the pool could not reach the cue set's mass
                "b": {
                    "targetTool": "internal_search",
                    "allRows": {"deltaTarget": -0.40, "size": 4},
                    "controlThreshold": 0.02,
                    "setControlThreshold": 0.02,
                    "setControlMassMatched": False,
                    "rows": [{"deltaTarget": -0.35}],
                },
            }
        },
    }


def test_all_families_arm_is_judged_against_the_set_matched_band(ui):
    summary = ui.summarize_steering(_set_matched_fixture())
    side_a = next(s for s in summary["sides"] if s["side"] == "a")
    # 0.09 clears the per-family band (0.03) but not the set-matched one (0.12)
    assert side_a["control"] == 0.03 and side_a["setControl"] == 0.12
    assert side_a["allBeyondControl"] is False
    # the row itself is still judged against the per-family band, which is the
    # like-for-like comparison at that granularity
    assert side_a["familiesBeyondControl"] == 1
    side_b = next(s for s in summary["sides"] if s["side"] == "b")
    assert side_b["allBeyondControl"] is True
    assert summary["sidesAllBeyondControl"] == 1


def test_ceiling_draws_are_counted_and_named(ui):
    summary = ui.summarize_steering(_set_matched_fixture())
    assert summary["setControlSides"] == 2 and summary["setControlCeilings"] == 1
    assert "1 of 2" in summary["setControlCaveat"]
    assert "whole eligible pool" in summary["setControlCaveat"]


def test_results_without_a_set_matched_arm_fall_back_and_say_so(ui):
    summary = ui.summarize_steering(_steering_fixture())
    assert summary["setControlSides"] == 0
    assert "per-family" in summary["setControlCaveat"]
    side_a = next(s for s in summary["sides"] if s["side"] == "a")
    assert side_a["setControl"] == side_a["control"]


def test_summarize_steering_empty(ui):
    summary = ui.summarize_steering({"layer": 6})
    assert summary["nSides"] == 0 and summary["nCross"] == 0
    assert summary["meanAbsAllDelta"] == 0.0


def test_embed_roundtrip(ui):
    html = "<script>const D = /* MOCK_DATA_START */ null /* MOCK_DATA_END */;</script>"
    once = ui.embed(html, {"a": 1, "s": "</script>"})
    assert '"a":1' in once and "<\\/script" in once
    # embedding again replaces, not accumulates
    twice = ui.embed(once, {"a": 2})
    assert '"a":2' in twice and '"a":1' not in twice


def test_index_html_has_markers_and_guards():
    html = (_DEMO_DIR / "index.html").read_text()
    assert "/* MOCK_DATA_START */" in html and "/* MOCK_DATA_END */" in html
    for renderer in (
        "renderMethod",
        "renderTransfer",
        "renderStability",
        "renderProbes",
        "renderSignal",
        "renderPopulation",
        "renderDepth",
        "renderRobustness",
    ):
        assert renderer in html
    # every section starts hidden and is only shown when its data exists
    assert html.count('class="card hidden"') >= 7
