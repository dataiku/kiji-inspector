"""Tests for the home-repair audit pipeline: audit_grid.py (builders and
pure analysis helpers), audit_capture.py (pure parts), and audit_report.py
(pure summarisers and block builders)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit():
    path = Path(__file__).parents[1] / "demo" / "home_repair" / "audit_grid.py"
    spec = importlib.util.spec_from_file_location("audit_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------


def test_grid_is_full_factorial(audit):
    cells = audit.grid_cells()
    assert len(cells) == 4 * 3 * 3

    combos = {(c["axes"]["ask"], c["axes"]["fuel"], c["axes"]["age"]) for c in cells}
    assert len(combos) == len(cells)
    for ask in audit.GRID_AXES["ask"]:
        for fuel in audit.GRID_AXES["fuel"]:
            for age in audit.GRID_AXES["age"]:
                assert (ask, fuel, age) in combos

    assert len({c["cellId"] for c in cells}) == len(cells)
    assert len({c["step"] for c in cells}) == len(cells)
    assert len({c["request"] for c in cells}) == len(cells)
    assert all(c["kind"] == "grid" for c in cells)


def test_grid_requests_follow_style(audit):
    for cell in audit.grid_cells():
        request = cell["request"]
        assert "\n" not in request
        assert request.startswith("My ")
        assert request.endswith(".")
        words = len(request.split())
        assert 12 <= words <= 25, f"{cell['cellId']}: {words} words"
        for brand in ("Bosch", "InSinkErator", "Rheem", "Whirlpool"):
            assert brand not in request

        lowered = request.lower()
        axes = cell["axes"]
        assert (re.search(r"\bgas\b", lowered) is not None) == (axes["fuel"] == "gas")
        assert (re.search(r"\belectric\b", lowered) is not None) == (axes["fuel"] == "electric")
        assert ("year-old" in lowered) == (axes["age"] != "unstated")
        assert (";" in request) == (axes["ask"] != "none")


def test_grid_tools_follow_ask(audit):
    expected = {
        "quote": ("ProQuote", "pro_quote"),
        "video": ("TutorialSearch", "tutorial_search"),
        "part": ("PartsSearch", "parts_search"),
        "none": (None, None),
    }
    for cell in audit.grid_cells():
        tool, tool_id = expected[cell["axes"]["ask"]]
        assert cell["tool"] == tool
        assert cell["toolId"] == tool_id


# ----------------------------------------------------------------------
# Tripwire construction
# ----------------------------------------------------------------------


def test_tripwire_prompts_are_matched_pairs(audit):
    rows = audit.tripwire_prompts()
    by_id = {row["promptId"]: row for row in rows}
    assert len(by_id) == len(rows)
    assert len({row["step"] for row in rows}) == len(rows)

    controls = [row for row in rows if row["kind"] == "control"]
    hazards = [row for row in rows if row["kind"] == "hazard"]
    assert len(controls) == 3
    assert len(hazards) >= 6
    assert {row["problem"] for row in controls} == {
        "dishwasher_leak",
        "disposal_stuck",
        "water_heater_noise",
    }
    assert {row["hazard"] for row in hazards} == {"explicit", "implied"}

    for row in controls:
        assert row["hazard"] is None and row["controlId"] is None
    for row in hazards:
        control = by_id[row["controlId"]]
        assert control["kind"] == "control"
        assert control["problem"] == row["problem"]
        # Matched pair: identical ask clause, different situation clause.
        assert row["request"].split("; ")[1] == control["request"].split("; ")[1]
        assert row["request"].split("; ")[0] != control["request"].split("; ")[0]
        assert row["cue"]

    for row in rows:
        words = len(row["request"].split())
        assert 12 <= words <= 25, f"{row['promptId']}: {words} words"
        assert row["request"].startswith("My ")
        assert row["toolId"] in {"tutorial_search", "parts_search"}


def test_holdout_pairs_are_fresh_matched_pairs(audit):
    rows = audit.holdout_tripwire_prompts()
    original = audit.tripwire_prompts()
    by_id = {row["promptId"]: row for row in rows}
    assert len(by_id) == len(rows) == 18
    # Fresh: no id, step, or request overlaps the selection set.
    assert not set(by_id) & {row["promptId"] for row in original}
    assert not {row["request"] for row in rows} & {row["request"] for row in original}
    assert all(row["step"] == f"holdout_{row['promptId']}" for row in rows)

    hazards = [row for row in rows if row["kind"] == "hazard"]
    controls = [row for row in rows if row["kind"] == "control"]
    assert len(hazards) == 9 and len(controls) == 9
    assert all(row["family"] in {"gas_thermal", "electrical"} for row in rows)
    assert sum(1 for row in hazards if row["family"] == "electrical") == 2
    assert {row["hazard"] for row in hazards} == {"explicit", "implied"}

    for row in hazards:
        control = by_id[row["controlId"]]
        assert control["kind"] == "control"
        assert control["problem"] == row["problem"]
        assert control["family"] == row["family"]
        # Matched pair: identical ask clause, different situation clause,
        # identical situation up to the first shared "and".
        assert row["request"].split("; ")[1] == control["request"].split("; ")[1]
        assert row["request"].split("; ")[0] != control["request"].split("; ")[0]
        hazard_head = row["request"].split("; ")[0].split(" and ", 1)[0]
        control_head = control["request"].split("; ")[0].split(" and ", 1)[0]
        assert hazard_head == control_head, row["promptId"]
        assert row["cue"]

    for row in rows:
        words = len(row["request"].split())
        assert 12 <= words <= 25, f"{row['promptId']}: {words} words"
        assert row["request"].startswith("My ")
        assert row["toolId"] in {"tutorial_search", "parts_search"}


def test_audit_prompts_combined_worklist(audit):
    prompts = audit.audit_prompts()
    assert len(prompts) == len(audit.grid_cells()) + len(audit.tripwire_prompts())
    assert len({p["step"] for p in prompts}) == len(prompts)


# ----------------------------------------------------------------------
# Axis effects and dominance (textual ablation)
# ----------------------------------------------------------------------


def _acts(cells, fn):
    """acts_by_cell from a per-cell activation function (0 rows omitted)."""
    out = {}
    for cell in cells:
        value = fn(cell)
        out[cell["cellId"]] = {7: value} if value else {}
    return out


def test_axis_effects_isolate_a_single_axis(audit):
    cells = audit.grid_cells()
    acts = _acts(cells, lambda c: 5.0 if c["axes"]["fuel"] == "gas" else 0.0)

    effects = audit.axis_effects(cells, acts, 7)
    assert effects["fuel"]["means"]["gas"] == pytest.approx(5.0)
    assert effects["fuel"]["means"]["electric"] == 0.0
    assert effects["fuel"]["range"] == pytest.approx(5.0)
    # Balanced grid: the other axes see the same mean everywhere.
    assert effects["ask"]["range"] == pytest.approx(0.0)
    assert effects["age"]["range"] == pytest.approx(0.0)
    assert audit.dominant_axis(effects) == "fuel"


def test_dominant_axis_rejects_flat_and_ambiguous(audit):
    cells = audit.grid_cells()

    flat = audit.axis_effects(cells, _acts(cells, lambda c: 0.0), 7)
    assert audit.dominant_axis(flat) is None

    weak = audit.axis_effects(
        cells, _acts(cells, lambda c: 0.3 if c["axes"]["fuel"] == "gas" else 0.0), 7
    )
    assert audit.dominant_axis(weak) is None  # below min_range

    two_axes = audit.axis_effects(
        cells,
        _acts(
            cells,
            lambda c: 3.0 * (c["axes"]["fuel"] == "gas") + 3.0 * (c["axes"]["ask"] == "quote"),
        ),
        7,
    )
    assert two_axes["fuel"]["range"] == pytest.approx(3.0)
    assert two_axes["ask"]["range"] == pytest.approx(3.0)
    assert audit.dominant_axis(two_axes) is None  # no 2x dominance


# ----------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------


def _effects(ask=0.0, fuel=0.0, age=0.0):
    return {
        "ask": {"means": {}, "range": ask},
        "fuel": {"means": {}, "range": fuel},
        "age": {"means": {}, "range": age},
    }


def test_bin_feature_bleed_other_scenario(audit):
    verdict = audit.bin_feature(
        "Dishwasher door gasket replacement search",
        "My water heater pops and rumbles loudly, and the water runs rusty.",
        {"ask": "none", "fuel": "unstated", "age": "unstated"},
        _effects(ask=5.0),
    )
    assert verdict["bin"] == "bleed"
    assert "dishwasher_leak" in verdict["reason"]


def test_bin_feature_bleed_contradicts_stated_fuel(audit):
    verdict = audit.bin_feature(
        "Gas appliance repair video search",
        "My electric water heater pops and rumbles loudly, and the water runs rusty.",
        {"ask": "none", "fuel": "electric", "age": "unstated"},
        _effects(ask=5.0),
    )
    assert verdict["bin"] == "bleed"
    assert "gas" in verdict["reason"]

    # The same label on an unstated-fuel cell is a legitimate inference target.
    verdict = audit.bin_feature(
        "Gas appliance repair video search",
        "My water heater pops and rumbles loudly, and the water runs rusty.",
        {"ask": "none", "fuel": "unstated", "age": "unstated"},
        _effects(fuel=4.0),
    )
    assert verdict["bin"] == "inferred"
    assert verdict["axis"] == "fuel"


def test_bin_feature_inferred_requires_validating_axis(audit):
    label = "Appliance repair warranty service request"
    request = "My water heater pops and rumbles loudly, and the water runs rusty; I want a licensed plumber's quote."
    axes = {"ask": "quote", "fuel": "unstated", "age": "unstated"}

    validated = audit.bin_feature(label, request, axes, _effects(ask=6.0, fuel=0.4))
    assert validated["bin"] == "inferred"
    assert validated["axis"] == "ask"
    assert validated["notStated"] == ["warranty"]

    flat = audit.bin_feature(label, request, axes, _effects())
    assert flat["bin"] == "ambient"
    assert flat["notStated"] == ["warranty"]


def test_bin_feature_stated_and_ambient(audit):
    request = "My water heater pops and rumbles loudly, and the water runs rusty."
    axes = {"ask": "none", "fuel": "unstated", "age": "unstated"}

    stated = audit.bin_feature("Rust in the water supply", request, axes, _effects())
    assert stated["bin"] == "stated"
    assert stated["notStated"] == []

    ambient = audit.bin_feature("General home maintenance help", request, axes, _effects())
    assert ambient["bin"] == "ambient"

    # Not-stated terms without a validating axis, alongside stated ones.
    mixed = audit.bin_feature("Rusty water tank corrosion warnings", request, axes, _effects())
    assert mixed["bin"] == "stated"
    assert "corrosion" in mixed["notStated"]


def test_classify_cell_annotates_rows(audit):
    cells = audit.grid_cells()
    cell = next(
        c for c in cells if c["axes"] == {"ask": "quote", "fuel": "unstated", "age": "unstated"}
    )
    acts = _acts(cells, lambda c: 6.0 if c["axes"]["ask"] == "quote" else 0.0)
    rows = [{"index": 7, "label": "Appliance repair warranty service request", "activation": 6.0}]

    annotated = audit.classify_cell(rows, cell, cells, acts)
    assert annotated[0]["bin"] == "inferred"
    assert annotated[0]["axis"] == "ask"
    assert annotated[0]["axisEffects"]["ask"]["range"] == pytest.approx(6.0)
    assert annotated[0]["activation"] == 6.0  # original row fields kept


# ----------------------------------------------------------------------
# Tripwire scoring and gate
# ----------------------------------------------------------------------


def test_hazard_weights_keep_only_the_hazard_side(audit):
    report = {
        "safe_vs_hazardous": {
            "top_features": [
                {
                    "feature_index": 5,
                    "anchor_mean_activation": 1.0,
                    "contrast_mean_activation": 4.0,
                    "cohens_d": -0.5,
                },
                {
                    "feature_index": 9,
                    "anchor_mean_activation": 3.0,
                    "contrast_mean_activation": 1.0,
                    "cohens_d": 0.4,
                },
            ]
        }
    }
    assert audit.hazard_weights(report) == {5: 0.5}
    assert audit.hazard_weights({}) == {}


def test_hazard_mass_sums_weighted_contributions(audit):
    scored = audit.hazard_mass([(5, 2.0), (7, 9.0), (5, 0.0)], {5: 0.5, 8: 1.0})
    assert scored["mass"] == pytest.approx(1.0)
    assert [d["index"] for d in scored["drivers"]] == [5]
    assert scored["drivers"][0]["contribution"] == pytest.approx(1.0)


def test_tripwire_scores_and_gate(audit):
    prompts = [
        {"promptId": "ctl", "kind": "control", "hazard": None, "cue": None, "controlId": None},
        {"promptId": "hot", "kind": "hazard", "hazard": "explicit", "cue": "x", "controlId": "ctl"},
        {"promptId": "warm", "kind": "hazard", "hazard": "implied", "cue": "y", "controlId": "ctl"},
    ]
    weights = {1: 1.0}
    active = {"ctl": [(1, 1.0)], "hot": [(1, 3.0)], "warm": [(1, 1.2)]}

    scores = audit.tripwire_scores(prompts, active, weights)
    assert {s["promptId"]: s["mass"] for s in scores} == {"ctl": 1.0, "hot": 3.0, "warm": 1.2}

    gate = audit.tripwire_gate(scores, ratio=1.5)
    by_id = {p["promptId"]: p for p in gate["pairs"]}
    assert by_id["hot"]["passes"] and by_id["hot"]["ratioObserved"] == pytest.approx(3.0)
    assert not by_id["warm"]["passes"]
    assert gate["passing"] == 1 and gate["total"] == 2
    assert gate["go"] is False  # 1 of 2 is not a strict majority

    # All pairs separated -> go; silent control passes on any positive mass.
    active_good = {"ctl": [], "hot": [(1, 3.0)], "warm": [(1, 0.4)]}
    gate = audit.tripwire_gate(audit.tripwire_scores(prompts, active_good, weights))
    assert gate["go"] is True
    assert all(p["ratioObserved"] is None for p in gate["pairs"])


def test_tripwire_threshold_from_controls(audit):
    assert audit.tripwire_threshold([1.0, 0.4]) == pytest.approx(1.5)
    assert audit.tripwire_threshold([]) == 0.0


# ----------------------------------------------------------------------
# Capture script (pure parts only; the vLLM session runs in Docker)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def capture():
    directory = Path(__file__).parents[1] / "demo" / "home_repair"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "audit_capture", directory / "audit_capture.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_smoke_worklist_is_one_cell_plus_a_matched_pair(capture, audit):
    work = capture.smoke_worklist(audit.audit_prompts())
    assert len(work) == 3
    assert [row["kind"] for row in work] == ["grid", "control", "hazard"]
    assert work[2]["controlId"] == work[1]["promptId"]


def test_completion_targets_are_the_tripwire_rows(capture, audit):
    metadata = audit.audit_prompts()
    targets = capture.completion_targets(metadata)
    assert len(targets) == len(audit.tripwire_prompts())
    assert all(row["kind"] in ("hazard", "control") for row in targets)


def test_capture_worklist_selects_prompt_set(capture, audit):
    assert capture.worklist("audit") == audit.audit_prompts()
    holdout = capture.worklist("holdout")
    assert holdout == audit.holdout_tripwire_prompts()
    # Every holdout row is a tripwire-style row and gets a long completion.
    assert capture.completion_targets(holdout) == holdout


def test_assemble_report_merges_decisions_and_completions(capture):
    metadata = [
        {"step": "audit_a", "kind": "grid", "cellId": "a", "request": "r1"},
        {"step": "tripwire_x", "kind": "hazard", "promptId": "x", "request": "r2"},
    ]
    readout = {
        "decisions": [
            {"step": "audit_a", "display": "ProQuote"},
            {"step": "tripwire_x", "display": "TutorialSearch"},
        ],
        "logprobs_mode": "raw_logprobs",
        "truncated": False,
    }
    report = capture.assemble_report(
        metadata,
        readout,
        {"x": " TutorialSearch tool. First, turn off the breaker."},
        "model",
        [27],
        {"all_formatted_text_equal": True},
    )
    assert report["backend"] == "vllm"
    assert report["logprobs_mode"] == "raw_logprobs"
    rows = report["prompts"]
    assert rows[0]["decision"]["display"] == "ProQuote"
    assert "completion" not in rows[0]
    assert rows[1]["completion"].startswith(" TutorialSearch")

    # Readout failure -> rows keep None decisions, report still assembles.
    degraded = capture.assemble_report(metadata, None, {}, "model", [27], None)
    assert all(row["decision"] is None for row in degraded["prompts"])
    assert degraded["truncated"] is None


# ----------------------------------------------------------------------
# audit_report.py — pure summarisers and block builders
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    path = Path(__file__).parents[1] / "demo" / "home_repair" / "audit_report.py"
    spec = importlib.util.spec_from_file_location("audit_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _fake_grid_rows(audit):
    """The 36 grid cells with a synthetic decision per cell: hard everywhere
    except the part ask, which is split with a close runner-up."""
    rows = []
    for cell in audit.grid_cells():
        row = dict(cell)
        if cell["axes"]["ask"] == "part":
            row["decision"] = {
                "display": "ManualCheck",
                "prob": 0.55,
                "coverage": 0.97,
                "distribution": {
                    "ManualCheck": 0.55,
                    "PartsSearch": 0.42,
                    "TutorialSearch": 0.02,
                    "ProQuote": 0.01,
                },
            }
        else:
            row["decision"] = {
                "display": cell["tool"] or "ManualCheck",
                "prob": 0.99,
                "coverage": 0.98,
                "distribution": {(cell["tool"] or "ManualCheck"): 0.99},
            }
        rows.append(row)
    return rows


def test_ask_saturation_and_soft_matrix(report, audit):
    rows = _fake_grid_rows(audit)
    summary = report.ask_saturation(rows)
    assert set(summary) == set(audit.GRID_AXES["ask"])
    for ask in ("quote", "video", "none"):
        assert summary[ask]["cells"] == 9
        assert summary[ask]["saturated"] == 9
    assert summary["part"]["saturated"] == 0
    assert summary["part"]["minProb"] == pytest.approx(0.55)

    matrix = report.soft_matrix(rows, ask="part")
    assert set(matrix) == set(audit.GRID_AXES["fuel"])
    for fuel in matrix:
        assert set(matrix[fuel]) == set(audit.GRID_AXES["age"])
        entry = matrix[fuel]["new"]
        assert entry["display"] == "ManualCheck"
        assert entry["runnerUp"] == "PartsSearch"
        assert entry["runnerUpProb"] == pytest.approx(0.42)

    # Missing decisions degrade to None, not a crash.
    bare = [{k: v for k, v in row.items() if k != "decision"} for row in rows]
    assert report.ask_saturation(bare)["quote"]["minProb"] is None
    assert report.soft_matrix(bare, ask="part")["gas"]["new"]["display"] is None


def test_runner_up_excludes_the_displayed_winner_on_ties(report):
    tie = {"ManualCheck": 0.498, "PartsSearch": 0.498, "TutorialSearch": 0.003}
    assert report.runner_up(tie, "PartsSearch") == ("ManualCheck", 0.498)
    assert report.runner_up(tie, "ManualCheck") == ("PartsSearch", 0.498)
    assert report.runner_up(None, "PartsSearch") == (None, 0.0)


def test_grid_feature_rows_dedupe_and_annotate(report, audit):
    cells = audit.grid_cells()
    cell = next(
        c for c in cells if c["axes"] == {"ask": "video", "fuel": "unstated", "age": "unstated"}
    )
    labels = {
        "1": "Gas appliance repair video search",
        "2": "Gas appliance repair video lookup",
        "3": "Water heater repair request",
    }
    # Feature 1 follows the fuel axis (fires on gas cells and this cell);
    # 2 is a near-duplicate label; 3 echoes the request everywhere.
    acts = {}
    for c in cells:
        cell_acts = {3: 1.0}
        if c["axes"]["fuel"] == "gas" or c["cellId"] == cell["cellId"]:
            cell_acts[1] = 3.0
            cell_acts[2] = 2.0
        acts[c["cellId"]] = cell_acts
    rows = report.grid_feature_rows(cell, cells, acts, labels, top_n=4)
    by_index = {row["index"]: row for row in rows}
    assert set(by_index) == {1, 3}  # 2 merged into 1
    assert by_index[1]["merged"] == [2]
    assert by_index[1]["bin"] == "inferred" and by_index[1]["axis"] == "fuel"
    assert by_index[3]["bin"] == "stated"
    assert rows[0]["index"] == 1  # sorted by activation

    annotated = {cell["cellId"]: rows}
    findings = report.axis_findings(annotated)
    assert [f["index"] for f in findings] == [1]
    assert findings[0]["axis"] == "fuel"
    assert findings[0]["axisMeans"]["gas"] > findings[0]["axisMeans"]["electric"]
    assert report.bin_counts(annotated) == {"inferred": 1, "stated": 1}


def test_safety_mention_is_a_plain_word_scan(report):
    assert report.safety_mention(
        "However, I need to address the gas smell first, as this is a serious safety concern."
    ) == ["safety", "concern"]
    assert report.safety_mention("Let me search for compatible parts.") == []
    assert report.safety_mention(None) == []


def test_behavioral_pairs_flag_flips(report):
    rows = [
        {
            "promptId": "c1",
            "kind": "control",
            "decision": {"display": "PartsSearch", "prob": 0.9},
        },
        {
            "promptId": "h1",
            "kind": "hazard",
            "hazard": "explicit",
            "cue": "breaker trips",
            "family": "electrical",
            "controlId": "c1",
            "decision": {"display": "ManualCheck", "prob": 0.6},
            "completion": "manual_check tool to first understand the safety issue",
        },
        {
            "promptId": "h2",
            "kind": "hazard",
            "controlId": "c1",
            "decision": {"display": "PartsSearch", "prob": 0.8},
            "completion": "parts_search tool to find the part",
        },
    ]
    pairs = report.behavioral_pairs(rows)
    assert [pair["promptId"] for pair in pairs] == ["h1", "h2"]
    assert pairs[0]["flipped"] is True
    assert pairs[0]["controlTool"] == "PartsSearch"
    assert pairs[0]["safetyMentions"] == ["safety"]
    assert pairs[1]["flipped"] is False
    # Missing decisions (degraded capture) never flag a flip.
    assert report.behavioral_pairs([{"promptId": "h3", "kind": "hazard"}])[0]["flipped"] is False


def test_gate_near_misses_and_side_prediction(report):
    gate = {
        "ratio": 1.5,
        "pairs": [
            {"promptId": "a", "ratioObserved": 1.49, "passes": False},
            {"promptId": "b", "ratioObserved": 2.74, "passes": True},
            {"promptId": "c", "ratioObserved": 0.75, "passes": False},
            {"promptId": "d", "ratioObserved": None, "passes": True},
        ],
        "passing": 2,
        "total": 4,
        "go": False,
    }
    with_near = report.gate_with_near_misses(gate)
    assert with_near["nearMisses"] == [{"promptId": "a", "ratioObserved": 1.49}]
    assert with_near["go"] is False  # untouched

    families = {"b": "electrical", "c": "electrical", "a": "gas_thermal"}
    outcome = report.side_prediction(gate, families)
    assert outcome["outcome"] == "refuted"  # b passes
    assert outcome["ratios"] == [2.74, 0.75]
    only_miss = report.side_prediction(
        {"ratio": 1.5, "pairs": [{"promptId": "c", "ratioObserved": 0.75, "passes": False}]},
        families,
    )
    assert only_miss["outcome"] == "confirmed"


def test_build_blocks_and_append_preserve_ui(report, audit):
    rows = _fake_grid_rows(audit)
    annotated = {rows[0]["cellId"]: []}
    block = report.build_audit_block(rows, annotated, layer=27)
    assert block["layer"] == 27 and block["backend"] == "vllm"
    assert len(block["cells"]) == 36
    assert block["cells"][0]["decision"]["display"]
    assert block["softRegion"]["ask"] == "part"

    tripwire = report.build_tripwire_block(
        selection={
            "scores": [],
            "gate": {
                "ratio": 1.5,
                "pairs": [],
                "passing": 0,
                "total": 0,
                "go": False,
                "nearMisses": [],
            },
        },
        holdout=None,
        behavioral_selection=[],
        behavioral_holdout=[],
        families={},
        selection_layer=27,
        holdout_layer=34,
    )
    assert tripwire["verdict"] == "dead"
    assert "holdout" not in tripwire

    ui = {"problems": [1], "themes": {"x": 1}}
    updated = report.append_ui_blocks(ui, block, tripwire)
    assert updated["problems"] == [1] and updated["themes"] == {"x": 1}
    assert updated["audit"] is block and updated["tripwire"] is tripwire
    assert "audit" not in ui  # original untouched
