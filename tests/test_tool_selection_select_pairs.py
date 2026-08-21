from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sel():
    directory = Path(__file__).parents[1] / "demo" / "tool_selection"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("select_pairs", directory / "select_pairs.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _row(request, tool, prob):
    return {"request": request, "toolId": tool, "prob": prob, "distribution": {tool: prob}}


def test_score_pair_combines_flip_and_overlap(sel):
    meta = {
        "theme": "local_vs_remote",
        "anchor": "Read the local config file now.",
        "contrast": "Fetch the remote config file now.",
        "signal": "x",
    }
    rows = {
        meta["anchor"]: _row(meta["anchor"], "file_read", 0.9),
        meta["contrast"]: _row(meta["contrast"], "api_call", 0.7),
    }
    s = sel.score_pair(meta, rows)
    assert s["flip"] == pytest.approx(0.7)
    # content words: {read, local, config, file, now} vs {fetch, remote, config, file, now}
    assert s["jaccard"] == pytest.approx(3 / 7, abs=1e-4)
    assert s["score"] == pytest.approx(0.7 * 3 / 7, abs=1e-4)
    rows[meta["contrast"]] = _row(meta["contrast"], "file_read", 0.99)
    assert sel.score_pair(meta, rows)["flip"] == 0.0  # same tool: no flip
    assert sel.score_pair(meta, {}) is None


def test_select_pairs_respects_constraints(sel):
    def cand(theme, a, b, ta, tb, flip, jac):
        return {
            "theme": theme,
            "anchor": a,
            "contrast": b,
            "toolA": ta,
            "toolB": tb,
            "flip": flip,
            "jaccard": jac,
            "score": flip * jac,
            "signal": "",
            "probA": flip,
            "probB": flip,
        }

    scored = [
        cand("t1", "a1", "b1", "internal_search", "web_search", 1.0, 0.6),
        cand("t1", "a1b", "b1b", "internal_search", "web_search", 1.0, 0.5),  # same theme
        cand("t2", "a2", "b2", "internal_search", "web_search", 0.9, 0.6),
        cand("t3", "a3", "b3", "web_search", "internal_search", 0.9, 0.5),  # 3rd of combo
        cand("t4", "use the web search tool", "b4", "file_read", "api_call", 0.9, 0.9),  # names
        cand("t5", "a5", "b5", "file_read", "api_call", 0.5, 0.9),  # flip too low
        cand("t6", "a6", "b6", "file_read", "api_call", 0.7, 0.3),
    ]
    chosen = sel.select_pairs(scored, n=8, per_tools=2, min_flip=0.6)
    assert [c["theme"] for c in chosen] == ["t1", "t2", "t6"]
    assert len(sel.select_pairs(scored, n=1)) == 1


def test_cue_phrases_are_the_differing_spans(sel):
    a = "Please read the contents of the local configuration file at /etc/app/config.json."
    b = "Please fetch the contents of the remote configuration file hosted at https://x/config.json."
    cues_a = sel.cue_phrases(a, b)
    cues_b = sel.cue_phrases(b, a)
    assert "local" in cues_a and "read" in cues_a
    assert "remote" in cues_b and all(c.lower() in b.lower() for c in cues_b)
    # trailing punctuation and stop-word-only spans are dropped
    assert not any(c.endswith(".") for c in cues_a + cues_b)
    assert sel.cue_phrases("Check the value.", "Check the value.") == []
    # quoted annotation phrases win when they occur on exactly one side
    assert sel.cue_words_for(a, b, "needs 'local' access vs 'remote'") == ["local"]


def test_committed_pairs_match_the_rule(sel):
    """pairs.json must be reproducible from the sweep files with the default rule."""
    import json

    committed = json.loads((Path(sel._DEMO_DIR) / "pairs.json").read_text())
    sweep_path = Path(sel._SWEEP_DIR) / "tool_selection_sweep.json"
    if not sweep_path.exists():
        pytest.skip("sweep output not present")
    sweep = json.loads(sweep_path.read_text())
    meta = json.loads((Path(sel._SWEEP_DIR) / "tool_selection_meta.json").read_text())
    rows = {r["request"]: r for r in sweep["rows"] if not r.get("history")}
    scored = [s for s in (sel.score_pair(m, rows) for m in meta) if s]
    rule = committed["rule"]
    chosen = sel.select_pairs(
        scored, n=rule["n"], per_tools=rule["perToolCombination"], min_flip=rule["minFlip"]
    )
    regenerated = sel.to_pair_records(chosen)
    assert [p["id"] for p in regenerated] == [p["id"] for p in committed["pairs"]]
    assert [p["a"]["request"] for p in regenerated] == [
        p["a"]["request"] for p in committed["pairs"]
    ]
