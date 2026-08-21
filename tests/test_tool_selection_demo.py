from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ts():
    directory = Path(__file__).parents[1] / "demo" / "tool_selection"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "tool_selection_demo", directory / "tool_selection_demo.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class FakeTokenizer:
    """Space-prefixed tool names; file_read / file_write share their first token.

    Also offers capitalised surface forms (" API call") and one fragment
    alias (" W" for "WEB search") that the tree must ignore.
    """

    _TABLE = {
        " internal_search": [10, 99],
        " web_search": [11, 99],
        " file_read": [12, 20],
        " file_write": [12, 21],
        " database_query": [13, 98],
        " api_call": [14, 97],
        " API call": [30, 50],
        " code_execute": [15, 96, 95],
        " delegate_agent": [16, 94],
        " WEB search": [40, 11],
    }
    _DECODE = {
        10: " internal",
        11: " web",
        12: " file",
        13: " database",
        14: " api",
        15: " code",
        16: " delegate",
        30: " API",
        40: " W",
    }

    def encode(self, text, add_special_tokens=False):
        if text in self._TABLE:
            return list(self._TABLE[text])
        # other surface forms: a fragment first token that must be ignored
        return [77, 78]

    def decode(self, ids):
        return self._DECODE.get(ids[0], "?")


def test_pairs_and_prompts_are_well_formed(ts):
    prompts = ts.decision_prompts()
    assert len(prompts) == 2 * len(ts.PAIRS)
    assert [p["side"] for p in prompts[:2]] == ["a", "b"]
    assert len({p["request"] for p in prompts}) == len(prompts)
    tool_words = {t.replace("_", " ") for t in ts.TOOL_IDS} | set(ts.TOOL_IDS)
    for pair in ts.PAIRS:
        assert pair["a"]["request"] != pair["b"]["request"]
        for side in ts.SIDES:
            request = pair[side]["request"].lower()
            # Neither side names a tool outright.
            assert not any(word in request for word in tool_words)
            for cue in pair[side]["cueWords"]:
                assert cue.lower() in request
            # The sweep readout merges file_read|file_write; either form is fine.
            for part in pair[side]["sweep"]["tool"].split("|"):
                assert part in ts.TOOL_IDS
        assert pair["a"]["sweep"]["tool"] != pair["b"]["sweep"]["tool"]
    # Probes: paraphrases avoid the side's cue words; keyword controls carry one
    # of them inside the other side's request.
    all_prompts = ts.decision_prompts(include_probes=True)
    assert all_prompts[: len(prompts)] == prompts
    probes = all_prompts[len(prompts) :]
    assert probes and len({p["request"] for p in all_prompts}) == len(all_prompts)
    by_id = {p["id"]: p for p in ts.PAIRS}
    for probe in probes:
        pair = by_id[probe["pair"]]
        side = pair[probe["side"]]
        other = pair["b" if probe["side"] == "a" else "a"]
        text = probe["request"].lower()
        assert probe["base"] == f"{probe['pair']}_{probe['side'].upper()}"
        assert not any(word in text for word in tool_words)
        if probe["kind"] == "paraphrase":
            assert probe["step"].endswith(("_P1", "_P2"))
            assert not any(cue.lower() in text for cue in side["cueWords"]), probe["step"]
            assert text != side["request"].lower()
        else:
            assert probe["kind"] == "keyword" and probe["step"].endswith("_K")
            assert probe["cue"].lower() in text and probe["cue"].lower() in side["request"].lower()
            # built from the other side's request: shares most of its words
            ow = set(other["request"].lower().split())
            assert len(ow & set(text.split())) >= len(ow) * 0.6, probe["step"]
    # Pairs come from the sweep by rule, not by hand.
    assert ts.PAIR_SELECTION["rule"]["perTheme"] == 1
    assert len({p["id"] for p in ts.PAIRS}) == len(ts.PAIRS)


def test_tool_token_tree_and_distribution(ts):
    tree = ts.tool_token_tree(FakeTokenizer(), ts.TOOLS)
    assert tree["first"][12] == {"file_read": [20], "file_write": [21]}
    assert tree["shared"] == [12]
    assert ts.second_token_ids(tree, 12) == [20, 21]
    assert tree["first"][30] == {"api_call": [50]}  # " API call" counts toward api_call
    assert 40 not in tree["first"] and 77 not in tree["first"]  # fragments ignored

    first = {10: math.log(0.3), 12: math.log(0.5), 14: math.log(0.05), 30: math.log(0.05)}
    second = {12: {20: math.log(0.2), 21: math.log(0.6)}}
    decision = ts.distribution_from_tree(first, second, tree)
    assert decision["toolId"] == "file_write"
    assert decision["raw"]["file_read"] == pytest.approx(0.125)
    assert decision["raw"]["file_write"] == pytest.approx(0.375)
    assert decision["raw"]["api_call"] == pytest.approx(0.1)  # both surface forms
    assert decision["coverage"] == pytest.approx(0.9)
    assert decision["distribution"]["internal_search"] == pytest.approx(0.3 / 0.9, abs=1e-3)
    # Missing second-token readout splits the shared mass evenly.
    even = ts.distribution_from_tree(first, {}, tree)
    assert even["raw"]["file_read"] == pytest.approx(0.25)


def test_side_specific_rows_and_pair_analysis(ts):
    labels = {
        "1": "Live stock price lookup",
        "2": "Live stock price lookup query",
        "3": "Shared thing",
        "4": "Cached report",
    }
    a = [(1, 5.0), (2, 2.0), (3, 4.0)]
    b = [(3, 3.0), (4, 6.0), (1, 1.0)]

    rows = ts.side_specific_rows(a, b, labels)
    assert rows[0]["index"] == 1 and rows[0]["delta"] == pytest.approx(4.0)
    assert rows[0]["merged"] == [2]  # near-duplicate label merged into the family
    assert rows[0]["other"] == pytest.approx(rows[0]["activation"] - rows[0]["delta"])
    assert [r["index"] for r in rows] == [1, 3]  # 3 is stronger on A by 1.0
    analysis = ts.pair_feature_analysis(a, b, labels)
    assert [r["index"] for r in analysis["bFeatures"]] == [4]
    assert analysis["shared"][0]["index"] == 3
    assert analysis["numActive"] == {"a": 3, "b": 3}
    assert 0 < analysis["overlap"]["jaccard"] < 1


def _report(ts):
    prompts = ts.decision_prompts()
    active, decisions = [], []
    for p in prompts:
        rows = [{"index": 100, "activation": 6.0, "label": "Generic request"}]
        if p["side"] == "a":
            rows.append({"index": 1, "activation": 5.0, "label": "Cue A feature"})
        else:
            rows.append({"index": 2, "activation": 4.0, "label": "Cue B feature"})
        active.append(rows)
        tool = "internal_search" if p["side"] == "a" else "web_search"
        decisions.append(
            {
                "step": p["step"],
                "toolId": tool,
                "display": tool,
                "prob": 0.9,
                "distribution": {tool: 0.9},
                "coverage": 0.95,
            }
        )
    return {
        "model": "/models/x",
        "backend": "vllm",
        "layer": 27,
        "layers": [{"layer": 27, "d_sae": 10752, "active_features": active}],
        "prompts": prompts,
        "decisions": decisions,
    }


def test_build_ui_data_attaches_causal_and_withholds_on_disagreement(ts):
    report = _report(ts)
    pid = ts.PAIRS[0]["id"]
    steering = {
        "hfFastPath": True,
        "parity": {
            f"{pid}_A": {
                "cosine": 0.999,
                "baselineDistributionHf": {"internal_search": 0.9},
                "baselineDistributionVllm": {"internal_search": 0.9},
            }
        },
        "attribution": {
            pid: {
                "a": {
                    "targetTool": "internal_search",
                    "otherTool": "web_search",
                    "hfChoice": "internal_search",
                    "controlThreshold": 0.02,
                    "rows": [
                        {"index": 1, "deltaTarget": -0.3, "deltaOther": 0.2, "hfActivation": 4.0}
                    ],
                    "allRows": {"deltaTarget": -0.3, "size": 1},
                },
                "b": {
                    "targetTool": "web_search",
                    "otherTool": "internal_search",
                    "hfChoice": "internal_search",
                    "controlThreshold": 0.02,
                    "rows": [{"index": 2, "deltaTarget": -0.1, "hfActivation": 3.0}],
                },
            }
        },
        "crossPatch": {
            pid: {
                "a_into_b": {
                    "fromSide": "a",
                    "intoSide": "b",
                    "targetTool": "internal_search",
                    "intoBaselineChoice": "web_search",
                    "rows": [],
                    "allRows": {"deltaTarget": 0.0},
                }
            }
        },
    }
    ui = ts.build_ui_data(report, steering)
    pair = ui["pairs"][0]
    assert pair["id"] == pid and pair["flipped"] is True
    assert pair["a"]["features"][0]["index"] == 1
    assert pair["a"]["features"][0]["causal"]["deltaTarget"] == -0.3
    assert pair["a"]["features"][0]["causal"]["descriptive"] is False
    assert pair["a"]["attribution"]["targetTool"] == "internal_search"
    assert "causal" not in pair["b"]["features"][0]
    assert pair["b"]["causalWithheld"]["hfChoice"] == "internal_search"
    assert pair["crossPatch"]["a_into_b"]["fromSide"] == "a"
    assert pair["crossPatch"]["a_into_b"]["intoBaselineMismatch"] is False
    # Under 2 pp is descriptive even when it beats a tiny random band.
    probe = [{"index": 100, "label": "x"}]
    ts.attach_causal(
        probe, {"controlThreshold": 0.001, "rows": [{"index": 100, "deltaTarget": -0.015}]}
    )
    assert probe[0]["causal"]["descriptive"] is True
    assert pair["shared"][0]["index"] == 100
    assert ui["scenario"]["tools"][0]["id"] == "internal_search"
    assert ui["pairSelection"]["rule"]["perTheme"] == 1
    assert ui["runMetadata"]["saeLayer"] == 27
    assert ui["runMetadata"]["hf"]["fastPath"] is True
    assert ui["runMetadata"]["hf"]["baselineAgreement"] == 1
    assert "hf" not in ts.build_ui_data(report, None)["runMetadata"]
    # Other pairs carry no causal data and no cross-patch.
    assert "attribution" not in ui["pairs"][1]["a"] and ui["pairs"][1]["crossPatch"] is None

    # Trace results (trace_pairs.py) are sliced per pair and surfaced as-is.
    trace = {
        "layer": 27,
        "compareLayers": [43],
        "scales": [0, 1],
        "genTokens": 8,
        "positions": {f"{pid}_A": {"step": f"{pid}_A", "tokens": ["a", "b"], "layers": {"27": {}}}},
        "dose": {pid: {"a_into_b": {"fromSide": "a", "intoSide": "b", "allRows": []}}},
        "generations": {pid: {"a_into_b": {"baseline": "x", "steered": "y"}}},
    }
    with_trace = ts.build_ui_data(report, steering, trace=trace)
    first = with_trace["pairs"][0]
    assert first["positions"]["a"]["tokens"] == ["a", "b"] and "b" not in first["positions"]
    assert first["dose"]["a_into_b"]["fromSide"] == "a"
    assert first["generations"]["a_into_b"]["steered"] == "y"
    assert with_trace["pairs"][1]["positions"] == {} and with_trace["pairs"][1]["dose"] is None
    assert with_trace["runMetadata"]["trace"]["compareLayers"] == [43]
    assert "trace" not in ui["runMetadata"]
    with pytest.raises(ValueError):
        ts.build_ui_data(report, steering, trace={"layer": 43})  # trace layer mismatch

    with pytest.raises(ValueError):
        ts.build_ui_data(report, {"layer": 34, "attribution": {}})  # layer mismatch
    with pytest.raises(ValueError):
        ts.build_ui_data(report, None, layer=43)  # layer not captured
    report["prompts"][0]["request"] = "something else"
    with pytest.raises(ValueError):
        ts.build_ui_data(report, None)


def test_html_guards_and_embeds(ts):
    html = (Path(__file__).parents[1] / "demo" / "tool_selection" / "index.html").read_text()
    assert "/* MOCK_DATA_START */" in html and "/* MOCK_DATA_END */" in html
    assert "renderCross" in html and "causalWithheld" in html
    assert "hfParityNote(rm.hf)" in html
    for fn in ("renderPositions", "renderDose", "renderGenerations", "renderProbes", "doseSvg"):
        assert f"function {fn}(" in html and f"${{{fn}(" in html or fn == "doseSvg"
    assert "fetch('output/ui_data.json'" in html


def test_probe_evidence_and_probe_block(ts):
    labels = {"1": "Cue A feature", "2": "Cue B feature", "100": "Generic request"}
    rows = ts.side_specific_rows([(1, 5.0), (100, 6.0)], [(2, 4.0), (100, 6.0)], labels)
    base = [(1, 5.0), (100, 6.0)]
    other = [(2, 4.0), (100, 6.0)]
    ev = ts.probe_evidence(
        rows,
        base,
        other,
        [(1, 2.5), (100, 6.0)],
        {"display": "internal_search"},
        {"display": "internal_search"},
    )
    assert ev["familiesFiring"] == 1 and ev["familiesTotal"] == 1
    assert ev["families"][0]["activation"] == 2.5 and ev["families"][0]["base"] == 5.0
    assert ev["sameTool"] is True and ev["cosineToBase"] > ev["cosineToOther"]
    silent = ts.probe_evidence(rows, base, other, [(2, 4.0), (100, 6.0)], None, None)
    assert silent["familiesFiring"] == 0 and silent["sameTool"] is None

    # A report that also captured the probes surfaces them per side.
    prompts = ts.decision_prompts(include_probes=True)
    active, decisions = [], []
    for p in prompts:
        feats = [{"index": 100, "activation": 6.0, "label": "Generic request"}]
        if p["side"] == "a" and p["kind"] != "keyword":
            feats.append({"index": 1, "activation": 5.0, "label": "Cue A feature"})
        if p["side"] == "b" and p["kind"] != "keyword":
            feats.append({"index": 2, "activation": 4.0, "label": "Cue B feature"})
        active.append(feats)
        tool = "internal_search" if p["side"] == "a" else "web_search"
        decisions.append({"step": p["step"], "toolId": tool, "display": tool, "prob": 0.9})
    report = {
        "model": "/models/x",
        "backend": "vllm",
        "layer": 27,
        "layers": [{"layer": 27, "d_sae": 10752, "active_features": active}],
        "prompts": prompts,
        "decisions": decisions,
    }
    ui = ts.build_ui_data(report, None)
    probed = [pr for pr in ui["pairs"] if "probes" in pr["a"]]
    assert probed
    block = probed[0]["a"]["probes"]
    assert len(block["paraphrases"]) == 2 and block["paraphrases"][0]["kind"] == "paraphrase"
    assert block["paraphrases"][0]["familiesFiring"] == 1
    assert block["keyword"]["kind"] == "keyword" and block["keyword"]["familiesFiring"] == 0
    assert block["keyword"]["cue"]
    # A base-only report passes the prompt check and carries no probes.
    base_only = dict(report, prompts=ts.decision_prompts())
    base_only["layers"] = [
        {"layer": 27, "d_sae": 10752, "active_features": active[: len(base_only["prompts"])]}
    ]
    assert "probes" not in ts.build_ui_data(base_only, None)["pairs"][0]["a"]


def test_html_spec_sheet_strip_guarded():
    html = (Path(__file__).parents[1] / "demo" / "tool_selection" / "index.html").read_text()
    script = html[html.index("<script>") :]
    code = (
        script[: script.index("/* MOCK_DATA_START */")]
        + script[script.index("/* MOCK_DATA_END */") :]
    )
    assert 'id="spec-sheet"' in html
    assert "function renderSpecSheet(" in code
    assert "renderSpecSheet(DATA.specSheet)" in code
    for topic in ("population", "robustness", "stability", "probes", "depth"):
        assert f"spec.{topic}" in code
