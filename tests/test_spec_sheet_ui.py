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
