from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def home_repair_demo():
    demo_path = Path(__file__).parents[1] / "demo" / "home_repair" / "home_repair_demo.py"
    spec = importlib.util.spec_from_file_location("home_repair_demo", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_contrastive_feature_map_uses_activations_directory(home_repair_demo, tmp_path):
    report_dir = tmp_path / "layer_27" / "activations"
    report_dir.mkdir(parents=True)
    report = {
        "_summary": {"unique_features": 1},
        "diy_vs_professional": {
            "top_features": [
                {
                    "feature_index": 42,
                    "rank": 0,
                    "cohens_d": 1.25,
                    "anchor_mean_activation": 2.0,
                    "contrast_mean_activation": 0.5,
                }
            ]
        },
    }
    (report_dir / "contrastive_features.json").write_text(json.dumps(report))

    feature_map = home_repair_demo._load_contrastive_feature_map(str(tmp_path), 27)

    assert feature_map == {
        42: [
            {
                "theme": "diy_vs_professional",
                "rank": 0,
                "cohens_d": 1.25,
                "direction": "anchor",
            }
        ]
    }


def test_analyze_activations_normalizes_input_and_derives_layer_key(home_repair_demo, monkeypatch):
    class FakeSAE(torch.nn.Module):
        d_model = 2

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.threshold = torch.nn.Parameter(torch.ones(3))
            self.normalized_input = None

        def normalize_input(self, x):
            self.normalized_input = x - 10
            return self.normalized_input

        def encode(self, x):
            assert torch.equal(x, self.normalized_input)
            return torch.tensor([[2.0, 0.0, 1.0]], device=x.device)

    sae = FakeSAE()
    descriptions = {
        "0": {"label": "DIY repair", "description": "A safe repair", "confidence": "high"},
        "2": {"label": "Parts search", "description": "Find a part", "confidence": "high"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        home_repair_demo,
        "_load_sae_local",
        lambda output_dir, layer, device: (sae, descriptions),
    )
    monkeypatch.setattr(home_repair_demo, "_load_contrastive_feature_map", lambda *_: {})

    result = home_repair_demo.analyze_activations(
        activation_log=[("step", {"residual_27": np.array([11.0, 12.0], dtype=np.float32)})],
        sae_repo_id="unused",
        sae_layer=27,
        sae_local_dir="output",
    )

    assert result["sae_layer_key"] == "residual_27"
    assert result["sae_threshold_offset"] == home_repair_demo._HF_THRESHOLD_OFFSET
    assert torch.allclose(
        sae.threshold,
        torch.full((3,), 1.0 + home_repair_demo._HF_THRESHOLD_OFFSET),
    )
    assert torch.equal(sae.normalized_input, torch.tensor([[1.0, 2.0]]))
    assert result["steps"][0]["sae_features"]["num_active"] == 2
    assert result["steps"][0]["sae_features"]["top_features"][0]["label"] == "DIY repair"
    assert [row["index"] for row in result["steps"][0]["sae_features"]["active_features"]] == [
        0,
        2,
    ]
    assert result["steps"][0]["sae_features"]["theme_evidence"] is None
    assert result["backend"] == "hf"


def test_initial_tool_decisions_match_short_training_style(home_repair_demo):
    requests = []
    for prompt in home_repair_demo.decision_prompts(include_contrasts=True, include_probes=True):
        tool_name, request = prompt["tool"], prompt["request"]
        requests.append(request)
        if prompt["kind"] == "open":
            # The situation alone names no tool.
            assert tool_name is None and prompt["toolId"] is None
        else:
            assert tool_name in home_repair_demo._TOOLS
            assert prompt["toolId"] in home_repair_demo._TOOL_ID_TO_DISPLAY
        assert "\n" not in request
        assert "Problem:" not in request
        assert "Details:" not in request
        if prompt["kind"] in ("base", "contrast", "control", "open"):
            assert request.startswith("My ")
        assert (7 if prompt["kind"] == "open" else 12) <= len(request.split()) <= 32
        assert not any(brand in request for brand in ("Bosch", "InSinkErator", "Badger", "Rheem"))

    assert len(set(requests)) == len(requests)


def test_initial_decisions_express_the_expected_tool_capability(home_repair_demo):
    for problem in home_repair_demo._PROBLEMS:
        tool_name, request = home_repair_demo._initial_tool_decision(problem)
        request = request.lower()
        if tool_name == "PartsSearch":
            assert any(term in request for term in ("part", "replac"))
        elif tool_name == "TutorialSearch":
            assert any(term in request for term in ("guide", "how do i", "video", "tutorial"))
        elif tool_name == "ProQuote":
            assert any(term in request for term in ("quote", "professional", "plumber"))
        else:
            pytest.fail(f"Unexpected initial tool in fixed demo: {tool_name}")


def test_evaluation_report_adapts_to_demo_analysis(home_repair_demo):
    prompts = []
    for problem in home_repair_demo._PROBLEMS:
        tool_name, request = home_repair_demo._initial_tool_decision(problem)
        prompts.append(
            {
                "step": f"{problem['id']}_InitialDecision",
                "problem": problem["id"],
                "tool": tool_name,
                "request": request,
            }
        )
    report = {
        "model": "/models/nemotron",
        "threshold_offset": 0.0,
        "prompts": prompts,
        "layers": [
            {
                "layer": 27,
                "d_sae": 10_752,
                "l0": {"per_prompt": [57, 77, 72]},
                "top_features": [
                    [{"index": index, "activation": 3.0, "label": f"Feature {index}"}]
                    for index in range(3)
                ],
            }
        ],
    }

    analysis = home_repair_demo.analysis_from_evaluation_report(report, 27)

    assert analysis["sae_layer_key"] == "residual_27"
    assert [step["step"] for step in analysis["steps"]] == [prompt["step"] for prompt in prompts]
    assert [step["sae_features"]["num_active"] for step in analysis["steps"]] == [
        57,
        77,
        72,
    ]


def test_ui_preserves_prompt_unique_feature_strength(home_repair_demo):
    steps = []
    for index, problem in enumerate(home_repair_demo._PROBLEMS):
        steps.append(
            {
                "step": f"{problem['id']}_InitialDecision",
                "sae_features": {
                    "top_features": [
                        {
                            "index": index,
                            "activation": 3.0,
                            "label": f"Unique feature {index}",
                        }
                    ]
                },
            }
        )

    ui_data = home_repair_demo.build_ui_data(
        analysis={"steps": steps},
        per_problem={},
        final_recommendation="",
    )

    for problem in home_repair_demo._PROBLEMS:
        row = ui_data["decisionFeatures"][problem["id"]]["features"][0]
        # A feature unique to one prompt keeps its full activation as deviation
        # (baseline over three prompts = 1.0), so it ranks first.
        assert row["activation"] == 3.0
        assert row["delta"] == pytest.approx(2.0)
        assert row["share"] == 1.0
        assert "strength" not in row
    assert "comparison" not in ui_data
    assert "contrasts" not in ui_data


def test_decision_prompts_are_base_first_then_contrasts(home_repair_demo):
    base = home_repair_demo.decision_prompts()
    full = home_repair_demo.decision_prompts(include_contrasts=True)

    assert [p["kind"] for p in base] == ["base"] * 3
    assert full[:3] == base
    assert [p["kind"] for p in full[3:]] == ["contrast"] * 3
    assert [p["step"] for p in full[3:]] == [
        f"{problem['id']}_Contrast" for problem in home_repair_demo._PROBLEMS
    ]
    assert all(p["changed"] for p in full[3:])
    assert all(p["toolId"] == home_repair_demo._DISPLAY_TO_TOOL_ID[p["tool"]] for p in full)


def test_decision_prompts_probes_follow_contrasts(home_repair_demo):
    full = home_repair_demo.decision_prompts(include_contrasts=True)
    probed = home_repair_demo.decision_prompts(include_contrasts=True, include_probes=True)

    assert probed[: len(full)] == full
    kinds = [p["kind"] for p in probed[len(full) :]]
    assert kinds == ["paraphrase"] * 9 + ["control"] * 6 + ["open"] * 3
    opens = [p for p in probed if p["kind"] == "open"]
    assert [p["step"] for p in opens] == [f"{pr['id']}_Open" for pr in home_repair_demo._PROBLEMS]
    # The open request is the base situation with the ask removed: no
    # instruction verbs, and a prefix of the base request's situation.
    for item in opens:
        base = next(p for p in full if p["problem"] == item["problem"] and p["kind"] == "base")
        assert base["request"].lower().startswith(item["request"].lower().rstrip(".")[:20])
        assert not any(
            verb in item["request"].lower()
            for verb in ("find", "show", "get me", "quote", "video", "search", "price")
        )
    assert [p["step"] for p in probed if p["kind"] == "paraphrase"][:3] == [
        "dishwasher_leak_Paraphrase1",
        "dishwasher_leak_Paraphrase2",
        "dishwasher_leak_Paraphrase3",
    ]
    controls = [p for p in probed if p["kind"] == "control"]
    assert {c["direction"] for c in controls} == {"added", "removed"}
    for control in controls:
        assert control["keyword"] and control["note"] and control["targetWords"]
        base = next(p for p in full if p["problem"] == control["problem"] and p["kind"] == "base")
        present = control["keyword"].lower() in control["request"].lower()
        assert present == (control["direction"] == "added")
        assert (control["keyword"].lower() in base["request"].lower()) == (
            control["direction"] == "removed"
        )
    # Paraphrases avoid the base request's key terms (they test meaning, not words).
    for problem in home_repair_demo._PROBLEMS:
        key_terms = {
            "dishwasher_leak": ("gasket", "replacement", "availability"),
            "disposal_stuck": ("hums", "step-by-step", "hex wrench", "unjam"),
            "water_heater_noise": ("licensed plumber's quote", "pops and rumbles", "rusty"),
        }[problem["id"]]
        for paraphrase in problem["paraphrases"]:
            assert not any(term in paraphrase.lower() for term in key_terms), paraphrase


def test_probe_helpers_overlap_family_and_label_mentions(home_repair_demo):
    left = [(1, 2.0), (2, 1.0), (3, 0.0)]
    right = [(1, 2.0), (4, 1.0)]
    overlap = home_repair_demo.active_overlap(left, right)
    assert overlap["jaccard"] == pytest.approx(1 / 3, abs=1e-3)
    assert overlap["cosine"] == pytest.approx(4.0 / (math.sqrt(5.0) * math.sqrt(5.0)), abs=1e-3)
    assert home_repair_demo.active_overlap([], right) == {"jaccard": 0.0, "cosine": 0.0}

    row = {"index": 7, "merged": [8, 9]}
    assert home_repair_demo.row_family(row) == [7, 8, 9]
    assert home_repair_demo.family_activation({8: 3.0, 9: 1.0}, [7, 8, 9]) == 3.0
    assert home_repair_demo.family_activation([(1, 2.0)], [7]) == 0.0
    assert home_repair_demo.label_mentions("Gas appliance safety warnings", ("gas",))
    assert not home_repair_demo.label_mentions("Gasket replacement", ("gas",))


def test_paraphrase_evidence_counts_surviving_rows_and_calibrates_overlap(home_repair_demo):
    rows = [
        {"index": 1, "label": "Gasket search", "merged": [2]},
        {"index": 3, "label": "Part lookup", "merged": []},
    ]
    base = [(1, 5.0), (2, 1.0), (3, 4.0), (9, 1.0)]
    paraphrases = [
        {
            "step": "p1",
            "request": "seal search",
            "active": [(2, 3.0), (9, 1.0)],
            "modelChoice": {"toolId": "parts_search"},
        },
        {
            "step": "p2",
            "request": "seal cost",
            "active": [(1, 4.0), (3, 2.0)],
            "modelChoice": {"toolId": "manual_check"},
        },
    ]
    others = [{"step": "o", "label": "other", "request": "x", "active": [(9, 1.0), (20, 2.0)]}]

    evidence = home_repair_demo.paraphrase_evidence(
        rows, base, paraphrases, others, base_tool_id="parts_search"
    )

    assert [r["fires"] for r in evidence["paraphrases"][0]["rows"]] == [True, False]
    assert evidence["paraphrases"][0]["rows"][0]["activation"] == 3.0  # merged twin counts
    assert evidence["paraphrases"][0]["rows"][0]["ratio"] == pytest.approx(0.6)  # vs base 5.0
    assert evidence["paraphrases"][1]["rows"][1]["fires"]  # 2.0 / 4.0 reaches the half mark
    assert [r["firesIn"] for r in evidence["rowSummary"]] == [2, 1]
    assert evidence["rowsFiringInAll"] == 1
    assert evidence["sameTool"] == 1
    assert evidence["comparisons"][0]["rowsFiring"] == 0
    assert [c["active"] for c in evidence["comparisons"][0]["rows"]] == [False, False]
    assert evidence["rowSummary"][0]["firesInComparisons"] == 0
    # A weak echo (below half the base strength) is active but does not count as firing.
    weak = home_repair_demo.paraphrase_evidence(
        rows, base, [{"step": "w", "request": "x", "active": [(1, 1.0)]}], []
    )
    cell = weak["paraphrases"][0]["rows"][0]
    assert cell["active"] and not cell["fires"] and cell["ratio"] == pytest.approx(0.2)
    assert evidence["comparisons"][0]["overlap"]["jaccard"] == pytest.approx(1 / 5)
    assert evidence["meanJaccard"] > evidence["comparisons"][0]["overlap"]["jaccard"]


def test_keyword_control_evidence_added_and_removed(home_repair_demo):
    labels = {
        "1": "Appliance repair warranty service request",
        "2": "Warranty coverage inquiry",
        "3": "Gasket replacement search",
        "4": "Gas appliance safety warnings",
    }
    active_by_step = {
        "a_InitialDecision": [(3, 6.0), (4, 5.0)],
        "b_InitialDecision": [(1, 8.0), (2, 2.0)],
    }
    rows = [{"index": 3, "label": labels["3"], "merged": []}]
    added = {
        "step": "a_Control1",
        "request": "warranty expired; find a gasket",
        "direction": "added",
        "keyword": "warranty",
        "targetWords": ["warranty"],
        "note": "n",
        "modelChoice": None,
    }
    quiet = home_repair_demo.keyword_control_evidence(
        added,
        active_by_step["a_InitialDecision"],
        [(3, 5.5), (4, 5.0)],
        active_by_step,
        labels,
        rows,
    )
    assert [t["index"] for t in quiet["targets"]] == [1, 2]  # ranked by strongest capture
    assert quiet["targets"][0]["peakStep"] == "b_InitialDecision"
    assert quiet["verdict"] == "stayed quiet" and quiet["responding"] == 0
    assert quiet["snapshotRows"][0] == {"index": 3, "base": 6.0, "control": 5.5}

    fired = home_repair_demo.keyword_control_evidence(
        added,
        active_by_step["a_InitialDecision"],
        [(3, 5.5), (1, 4.0)],
        active_by_step,
        labels,
        rows,
    )
    assert fired["verdict"] == "fired" and fired["responding"] == 1
    assert fired["targets"][0]["delta"] == 4.0

    removed = {**added, "direction": "removed", "keyword": "gas", "targetWords": ["gas"]}
    still = home_repair_demo.keyword_control_evidence(
        removed,
        active_by_step["a_InitialDecision"],
        [(3, 6.0), (4, 4.5)],
        active_by_step,
        labels,
        rows,
    )
    assert [t["index"] for t in still["targets"]] == [4]
    assert still["verdict"] == "still fire"
    off = home_repair_demo.keyword_control_evidence(
        removed, active_by_step["a_InitialDecision"], [(3, 6.0)], active_by_step, labels, rows
    )
    assert off["verdict"] == "turned off"


def test_attach_attribution_marks_rows_below_control_as_descriptive(home_repair_demo):
    rows = [{"index": 1, "label": "a"}, {"index": 2, "label": "b"}, {"index": 3, "label": "c"}]
    attribution = {
        "targetTool": "ProQuote",
        "controlThreshold": 0.05,
        "rows": [
            {"index": 1, "deltaTarget": -0.2, "hfActivation": 3.0, "argmaxChanged": True},
            {"index": 2, "deltaTarget": -0.01, "hfActivation": 0.0, "inactiveUnderHf": True},
        ],
    }
    home_repair_demo.attach_attribution(rows, attribution)
    assert rows[0]["causal"]["descriptive"] is False and rows[0]["causal"]["argmaxChanged"]
    assert rows[1]["causal"]["descriptive"] is True and rows[1]["causal"]["inactiveUnderHf"]
    assert "causal" not in rows[2]
    home_repair_demo.attach_attribution(rows, None)  # no-op
    # A tiny random band must not promote a sub-2 pp effect to load-bearing.
    tiny = [{"index": 1, "label": "a"}]
    home_repair_demo.attach_attribution(
        tiny, {"controlThreshold": 0.003, "rows": [{"index": 1, "deltaTarget": -0.015}]}
    )
    assert tiny[0]["causal"]["descriptive"] is True


def test_contrastive_theme_evidence_sides_shrinkage_and_insufficiency(home_repair_demo):
    contrastive_map = {
        1: [{"theme": "safe_vs_hazardous", "cohens_d": 0.5, "direction": "contrast"}],
        2: [{"theme": "safe_vs_hazardous", "cohens_d": 0.5, "direction": "contrast"}],
        3: [{"theme": "safe_vs_hazardous", "cohens_d": 1.0, "direction": "anchor"}],
        4: [{"theme": "urgent_vs_planned", "cohens_d": 2.0, "direction": "anchor"}],
    }
    labels = {"1": {"label": "Gas valve"}, "2": "Gas pilot", "3": {"label": "Allen wrench"}}
    active = [(1, 4.0), (2, 4.0), (3, 2.0), (4, 1.0), (99, 9.0)]

    evidence = home_repair_demo.contrastive_theme_evidence(
        active, contrastive_map, labels, shrink=0.0
    )

    hazard = evidence["safe_vs_hazardous"]
    assert hazard["anchorSide"] == "safe" and hazard["contrastSide"] == "hazardous"
    assert hazard["contrastMass"] == pytest.approx(4.0)  # 4*0.5 + 4*0.5
    assert hazard["anchorMass"] == pytest.approx(2.0)
    assert hazard["position"] == pytest.approx(4.0 / 6.0, abs=1e-3)
    assert hazard["nFeatures"] == 3
    assert hazard["coverage"] == pytest.approx(10.0 / 20.0)
    assert not hazard["insufficient"]
    assert [row["label"] for row in hazard["drivers"]["contrast"]] == ["Gas valve", "Gas pilot"]
    assert hazard["drivers"]["anchor"][0]["label"] == "Allen wrench"

    urgent = evidence["urgent_vs_planned"]
    assert urgent["insufficient"]  # one feature only
    assert urgent["position"] == pytest.approx(0.0)  # no shrinkage requested

    shrunk = home_repair_demo.contrastive_theme_evidence(
        active, contrastive_map, labels, shrink=2.0
    )
    assert 0.0 < shrunk["urgent_vs_planned"]["position"] < 0.5
    assert shrunk["cheap_fix_vs_replacement"]["position"] == pytest.approx(0.5)
    assert shrunk["cheap_fix_vs_replacement"]["insufficient"]


def test_filter_contrastive_map_keeps_home_repair_themes_only(home_repair_demo):
    contrastive_map = {
        1: [
            {"theme": "read_vs_write", "cohens_d": 1.0, "direction": "anchor"},
            {"theme": "diy_vs_professional", "cohens_d": 1.0, "direction": "anchor"},
        ],
        2: [{"theme": "cached_vs_live", "cohens_d": 1.0, "direction": "anchor"}],
    }

    filtered = home_repair_demo.filter_contrastive_map(contrastive_map)

    assert list(filtered) == [1]
    assert [entry["theme"] for entry in filtered[1]] == ["diy_vs_professional"]


class _FakeTokenizer:
    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        return self.table[text]

    def decode(self, ids):
        for text, encoded in self.table.items():
            if encoded[:1] == list(ids):
                return text.split("_")[0]
        return ""


def test_tool_first_token_ids_requires_distinct_first_tokens(home_repair_demo):
    tools = home_repair_demo._DECISION_TOOLS
    table = {
        " manual_check": [10, 1],
        " parts_search": [11, 2],
        " tutorial_search": [12, 2],
        " pro_quote": [13, 3, 4],
    }

    ids = home_repair_demo.tool_first_token_ids(_FakeTokenizer(table), tools)

    assert ids == {"manual_check": 10, "parts_search": 11, "tutorial_search": 12, "pro_quote": 13}

    table[" pro_quote"] = [12, 3]
    with pytest.raises(ValueError):
        home_repair_demo.tool_first_token_ids(_FakeTokenizer(table), tools)


def test_decision_from_logprobs_renormalises_and_flags_truncation(home_repair_demo):
    import math

    tool_to_token = {"manual_check": 10, "parts_search": 11, "tutorial_search": 12, "pro_quote": 13}
    logprobs = {11: math.log(0.6), 12: math.log(0.2), 99: math.log(0.1)}

    decision = home_repair_demo.decision_from_logprobs(
        logprobs,
        tool_to_token,
        sampled_id=11,
        completion=" parts_search",
    )

    assert decision["toolId"] == "parts_search"
    assert decision["display"] == "PartsSearch"
    assert decision["distribution"]["PartsSearch"] == pytest.approx(0.75)
    assert decision["distribution"]["TutorialSearch"] == pytest.approx(0.25)
    assert decision["distribution"]["ProQuote"] == 0.0
    assert decision["coverage"] == pytest.approx(0.8)
    assert "matchesExpected" not in decision  # the ask names the tool; nothing is "expected"
    assert not decision["lowCoverage"] and not decision["truncated"]

    truncated = home_repair_demo.decision_from_logprobs(
        {12: math.log(0.3)}, tool_to_token, truncated=True
    )
    assert truncated["toolId"] == "tutorial_search"
    assert truncated["truncated"] and truncated["lowCoverage"]


def test_dedupe_feature_rows_merges_split_labels(home_repair_demo):
    rows = [
        {
            "index": 1,
            "label": "Home appliance repair guide request",
            "activation": 7.0,
            "delta": 3.0,
        },
        {
            "index": 2,
            "label": "Home appliance repair guide search",
            "activation": 6.0,
            "delta": 2.5,
        },
        {
            "index": 3,
            "label": "Gas appliance pilot and valve repair",
            "activation": 5.0,
            "delta": 5.0,
        },
        {
            "index": 4,
            "label": "Home appliance repair guide request",
            "activation": 8.0,
            "delta": 1.0,
        },
    ]

    deduped = home_repair_demo.dedupe_feature_rows(rows)

    assert [row["index"] for row in deduped] == [4, 3]
    assert sorted(deduped[0]["merged"]) == [1, 2]
    assert deduped[0]["activation"] == 8.0
    assert deduped[1].get("merged", []) == []


def test_not_stated_in_request_flags_specific_terms_only(home_repair_demo):
    request = "My water heater pops while heating; can you find a guide to flush sediment safely?"

    assert home_repair_demo.not_stated_in_request(
        "Gas appliance pilot and valve repair", request
    ) == [
        "gas",
        "pilot",
        "valve",
    ]
    assert (
        home_repair_demo.not_stated_in_request("Home appliance repair guide request", request) == []
    )
    assert (
        home_repair_demo.not_stated_in_request(
            "Appliance door gasket leak diagnosis", "My dishwasher door gasket leaks"
        )
        == []
    )
    assert "gas" not in home_repair_demo.not_stated_in_request(
        "Door gasket failure", "My dishwasher leaks"
    )


def test_feature_rows_rank_by_deviation_and_expose_activation(home_repair_demo):
    labels = {
        "1": {"label": "Home appliance repair and maintenance"},
        "2": {"label": "Gas valve repair"},
    }
    active = [(1, 10.0), (2, 4.0), (3, 1.0)]
    baseline = {1: 9.5, 2: 1.0}

    rows = home_repair_demo.feature_rows(active, baseline, labels, "My heater pops", top_n=2)

    assert [row["index"] for row in rows] == [2, 3]
    assert rows[0]["activation"] == 4.0 and rows[0]["delta"] == pytest.approx(3.0)
    assert rows[0]["share"] == pytest.approx(0.4)
    assert rows[0]["notStated"] == ["gas", "valve"]
    assert rows[1]["label"] == "Feature #3"


def test_contrast_diff_reports_gained_lost_and_shifted(home_repair_demo):
    labels = {
        "1": {"label": "Gas valve"},
        "2": {"label": "Warranty covered repair"},
        "3": {"label": "Leak"},
    }
    base = [(1, 5.0), (3, 2.0)]
    variant = [(2, 4.0), (3, 3.0)]

    diff = home_repair_demo.contrast_diff(
        base, variant, labels, "My electric heater is under warranty"
    )

    assert [row["index"] for row in diff["gained"]] == [2]
    assert diff["gained"][0]["notStated"] == []
    assert [row["index"] for row in diff["lost"]] == [1]
    assert diff["lost"][0]["notStated"] == ["gas", "valve"]
    assert diff["shifted"][0]["index"] == 3 and diff["shifted"][0]["delta"] == pytest.approx(1.0)


def test_also_fired_flags_other_scenarios(home_repair_demo):
    rows = [
        {"index": 1, "label": "Dishwasher drainage troubleshooting"},
        {"index": 2, "label": "Gas appliance repair"},
        {"index": 3, "label": "Home appliance repair"},
    ]

    flagged = home_repair_demo.also_fired(rows, "water_heater_noise")

    assert [row["index"] for row in flagged] == [1]
    assert flagged[0]["otherScenarios"] == ["dishwasher_leak"]


def _synthetic_report(
    home_repair_demo, with_contrasts=True, with_decisions=True, with_probes=False
):
    prompts = home_repair_demo.decision_prompts(
        include_contrasts=with_contrasts, include_probes=with_probes
    )
    problem_ids = [problem["id"] for problem in home_repair_demo._PROBLEMS]
    active = []
    top = []
    evidence = []
    for offset, prompt in enumerate(prompts):
        rows = [
            {"index": 100, "activation": 9.0, "label": "Home appliance repair and maintenance"},
            {"index": offset, "activation": 5.0 + offset, "label": f"Specific feature {offset}"},
        ]
        if prompt["kind"] == "contrast":
            rows.append({"index": 50, "activation": 2.0, "label": "Warranty covered repair"})
        if prompt["kind"] == "paraphrase":
            # Paraphrases re-fire the problem's base-specific feature.
            base_offset = problem_ids.index(prompt["problem"])
            rows.append(
                {
                    "index": base_offset,
                    "activation": 4.0,
                    "label": f"Specific feature {base_offset}",
                }
            )
        if prompt["kind"] == "control" and prompt["direction"] == "added":
            rows.append({"index": 50, "activation": 0.3, "label": "Warranty covered repair"})
        if prompt["kind"] == "open":
            # Without the ask: the base-specific feature drops to a weak echo and
            # a diagnostic feature appears.
            base_offset = problem_ids.index(prompt["problem"])
            rows.append(
                {
                    "index": base_offset,
                    "activation": 1.0,
                    "label": f"Specific feature {base_offset}",
                }
            )
            rows.append({"index": 70, "activation": 3.0, "label": "Appliance diagnostic lookup"})
        active.append(rows)
        top.append(rows[:2])
        evidence.append(
            {
                theme: {
                    "anchorSide": sides[0],
                    "contrastSide": sides[1],
                    "position": 0.5,
                    "insufficient": True,
                    "nFeatures": 0,
                    "coverage": 0.0,
                    "anchorMass": 0.0,
                    "contrastMass": 0.0,
                    "anchorShare": 0.0,
                    "contrastShare": 0.0,
                    "drivers": {"anchor": [], "contrast": []},
                }
                for theme, sides in home_repair_demo._THEME_SIDES.items()
            }
        )
    report = {
        "model": "/models/nemotron",
        "threshold_offset": 0.0,
        "backend": "vllm",
        "logprobs_mode": "raw_logprobs",
        "prompts": prompts,
        "layers": [
            {
                "layer": 27,
                "d_sae": 10_752,
                "l0": {"per_prompt": [len(rows) for rows in active]},
                "top_features": top,
                "active_features": active,
                "theme_evidence": evidence,
            }
        ],
    }
    if with_decisions:
        report["decisions"] = [
            {
                "step": prompt["step"],
                "toolId": prompt["toolId"] or "manual_check",
                "display": prompt["tool"] or "ManualCheck",
                "prob": 0.9,
                "distribution": {(prompt["tool"] or "ManualCheck"): 0.9},
                "coverage": 0.95,
            }
            for prompt in prompts
        ]
    return report


def test_new_evaluation_report_carries_decisions_and_contrasts(home_repair_demo):
    report = _synthetic_report(home_repair_demo)

    analysis = home_repair_demo.analysis_from_evaluation_report(report, 27)
    assert analysis["backend"] == "vllm"
    assert [step["kind"] for step in analysis["steps"]] == ["base"] * 3 + ["contrast"] * 3
    assert analysis["steps"][0]["decision"]["display"] == "PartsSearch"

    steering = {
        "backend": "hf",
        "hfFastPath": True,
        "layer": 27,
        "parity": {
            "water_heater_noise_InitialDecision": {
                "cosine": 0.998,
                "baselineDistributionHf": {"TutorialSearch": 0.8, "ProQuote": 0.2},
                "baselineDistributionVllm": {"TutorialSearch": 0.7, "ProQuote": 0.3},
            },
            "dishwasher_leak_InitialDecision": {
                "cosine": 0.992,
                "baselineDistributionHf": {"PartsSearch": 0.4, "ManualCheck": 0.6},
                "baselineDistributionVllm": {"PartsSearch": 0.9, "ManualCheck": 0.1},
            },
        },
        "experiments": [
            {
                "problem": "water_heater_noise",
                "mode": "ablate",
                "features": [{"index": 2, "label": "Specific feature 2"}],
                "baseline": {"TutorialSearch": 0.8, "ProQuote": 0.2},
                "intervened": {"TutorialSearch": 0.9, "ProQuote": 0.1},
            }
        ],
    }
    ui_data = home_repair_demo.build_ui_data(
        analysis=analysis, per_problem={}, final_recommendation="", steering=steering
    )

    water = ui_data["decisionFeatures"]["water_heater_noise"]
    assert water["modelChoice"]["display"] == "ProQuote"
    assert water["features"][0]["index"] == 2  # prompt-specific feature ranks first
    assert water["sharedAcrossProblems"][0]["index"] == 100
    assert water["themeEvidence"]["safe_vs_hazardous"]["insufficient"] is True
    assert ui_data["themes"][0]["evidence"]["water_heater_noise"]["position"] == 0.5
    assert "leftLabel" in ui_data["themes"][0] and "markers" not in ui_data["themes"][0]

    contrast = ui_data["contrasts"]["water_heater_noise"]
    assert contrast["changed"].startswith("ask:")
    assert [row["index"] for row in contrast["gained"]] == [5, 50]
    assert [row["index"] for row in contrast["lost"]] == [2]
    assert contrast["modelChoice"]["prob"] == 0.9

    # The ablate/clamp experiments stay on disk; only attribution reaches the page.
    assert "steering" not in ui_data and "steeringMetadata" not in ui_data
    assert ui_data["runMetadata"]["backend"] == "vllm"
    assert ui_data["runMetadata"]["logprobsMode"] == "raw_logprobs"
    assert ui_data["runMetadata"]["hf"] == {
        "fastPath": True,
        "prompts": 2,
        "cosineMean": 0.995,
        "cosineMin": 0.992,
        "baselineAgreement": 1,
        "baselineCompared": 2,
    }
    assert ui_data["problems"][2]["safety"] == {"label": "High", "level": "red"}


def test_probe_report_yields_paraphrase_and_control_blocks(home_repair_demo):
    report = _synthetic_report(home_repair_demo, with_probes=True)
    analysis = home_repair_demo.analysis_from_evaluation_report(report, 27)
    assert [s["kind"] for s in analysis["steps"]].count("paraphrase") == 9

    steering = {
        "backend": "hf",
        "experiments": [],
        "attribution": {
            "dishwasher_leak": {
                "step": "dishwasher_leak_InitialDecision",
                "targetTool": "PartsSearch",
                "controlThreshold": 0.02,
                "rows": [{"index": 0, "deltaTarget": -0.3, "hfActivation": 2.0}],
                "allRows": {"deltaTarget": -0.4},
            }
        },
    }
    ui_data = home_repair_demo.build_ui_data(
        analysis=analysis, per_problem={}, final_recommendation="", steering=steering
    )

    probes = ui_data["probes"]
    assert set(probes) == {p["id"] for p in home_repair_demo._PROBLEMS}
    dish = probes["dishwasher_leak"]
    assert "expectedTool" not in dish and dish["baseTool"] == "PartsSearch"
    para = dish["paraphrase"]
    assert len(para["paraphrases"]) == 3 and para["sameTool"] == 3
    assert para["rowSummary"][0]["index"] == 0 and para["rowSummary"][0]["firesIn"] == 3
    assert [c["label"] for c in para["comparisons"]][0] == "same situation, different ask"
    assert len(para["comparisons"]) == 3
    controls = dish["controls"]
    assert [c["keyword"] for c in controls] == ["warranty", "professional"]
    assert controls[0]["targets"][0]["index"] == 50  # the warranty-labelled feature
    assert controls[0]["targets"][0]["peakStep"].endswith("_Contrast")
    assert controls[0]["verdict"] == "stayed quiet"  # 0.5 < quarter of the 2.0 peak
    assert controls[0]["modelChoice"]["display"] == "PartsSearch"
    water = probes["water_heater_noise"]["controls"][0]
    assert water["direction"] == "removed" and water["keyword"] == "gas"

    # The open (no-ask) request: its own readout, the snapshot rows followed
    # into it, and the features gained / lost without the ask.
    open_block = ui_data["openRequests"]["dishwasher_leak"]
    assert open_block["request"] == home_repair_demo._PROBLEMS[0]["open"]["request"]
    assert open_block["modelChoice"]["display"] == "ManualCheck"
    assert open_block["baseChoice"]["display"] == "PartsSearch"
    # Rows follow the snapshot: the specific feature falls to a weak echo
    # (1.0 / 5.0 < half), the generic one holds.
    assert [c["fires"] for c in open_block["rows"]] == [False, True]
    assert open_block["rows"][0]["label"] == "Specific feature 0"
    assert open_block["rows"][0]["ratio"] == pytest.approx(0.2)
    assert open_block["rowsFiring"] == 1
    assert 70 in [row["index"] for row in open_block["gained"]]
    assert open_block["lost"] == [] and open_block["shifted"][0]["index"] == 0
    assert 0.0 < open_block["overlap"]["jaccard"] < 1.0

    features = ui_data["decisionFeatures"]["dishwasher_leak"]["features"]
    assert features[0]["causal"]["deltaTarget"] == -0.3
    assert features[0]["causal"]["descriptive"] is False
    attribution = ui_data["decisionFeatures"]["dishwasher_leak"]["attribution"]
    assert attribution["allRows"]["deltaTarget"] == -0.4 and "rows" not in attribution
    assert "steering" not in ui_data

    # When the HF baseline picks a different tool than vLLM, the causal column
    # is withheld rather than shown with a caveat.
    steering["attribution"]["dishwasher_leak"]["hfChoice"] = "ManualCheck"
    ui_data = home_repair_demo.build_ui_data(
        analysis=analysis, per_problem={}, final_recommendation="", steering=steering
    )
    dish = ui_data["decisionFeatures"]["dishwasher_leak"]
    assert all("causal" not in row for row in dish["features"])
    assert "attribution" not in dish
    assert dish["causalWithheld"] == {
        "reason": "hf_baseline_disagrees",
        "hfChoice": "ManualCheck",
        "vllmChoice": "PartsSearch",
    }
    assert home_repair_demo.attribution_backends_agree({"hfChoice": "ProQuote"}, None)
    assert not home_repair_demo.attribution_backends_agree(None, {"display": "ProQuote"})
    assert "attribution" not in ui_data["decisionFeatures"]["disposal_stuck"]


def test_evaluation_report_rejects_mismatched_prompts(home_repair_demo):
    report = _synthetic_report(home_repair_demo)
    report["prompts"][1]["request"] = "My dishwasher is fine."

    with pytest.raises(ValueError):
        home_repair_demo.analysis_from_evaluation_report(report, 27)


def test_html_reads_new_schema_and_guards_optional_sections():
    html_path = Path(__file__).parents[1] / "demo" / "home_repair" / "index.html"
    html = html_path.read_text()

    script = html[html.index("<script>") :]
    # Judge the code, not the embedded data snapshot.
    code = (
        script[: script.index("/* MOCK_DATA_START */")]
        + script[script.index("/* MOCK_DATA_END */") :]
    )
    assert "DATA.comparison" not in script
    assert "ui.comparison" not in script
    assert "themes[*].evidence" in html or ".evidence" in script
    assert "DATA.contrasts?.[" in script or "DATA.contrasts && DATA.contrasts[" in script
    assert "DATA.steering" not in script and "renderSteering" not in script
    assert "DATA.probes?.[" in script
    assert "DATA.openRequests?.[" in script and "renderOpenRequest" in script
    assert "matchesExpected" not in code and "expectedTool" not in code
    assert "causal" in script
    assert "modelChoice" in script
    assert "strengthLabel" not in script
    assert "safely('comparison', renderComparison);" in html
    assert html.index("safely('comparison', renderComparison);") < html.index(
        "safely('problem detail', renderDetail);"
    )
    assert "clearInterval" in script


def test_final_recommendation_is_complete_and_tool_grounded(home_repair_demo, monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.decisions = []

        def record_tool_decision(self, *decision):
            self.decisions.append(decision)

        def _build_prompt(self, _system, user):
            return user

        def generate(self, prompt, step_label, max_tokens):
            return f"analysis for {step_label} " + ("detail " * 300)

    monkeypatch.setattr(home_repair_demo, "_get_tool_result", lambda *_: "{}")
    engine = FakeEngine()

    final_recommendation, _ = home_repair_demo.run_home_repair_analysis(engine)

    assert len(engine.decisions) == len(home_repair_demo._PROBLEMS)
    assert [step for step, _ in engine.decisions] == [
        f"{problem['id']}_InitialDecision" for problem in home_repair_demo._PROBLEMS
    ]

    for problem in home_repair_demo._PROBLEMS:
        assert f"## {problem['summary']}" in final_recommendation
        assert home_repair_demo._GROUNDED_RATIONALES[problem["id"]] in final_recommendation
        assert home_repair_demo._PARTS_DATA[problem["id"]]["diy_cost_range"] in final_recommendation
        assert home_repair_demo._MANUAL_DATA[problem["id"]]["safety"] in final_recommendation
        assert home_repair_demo._PRO_QUOTE_DATA[problem["id"]]["urgency"] in final_recommendation
    assert "## Priority order" in final_recommendation


def test_ui_recommendations_use_grounded_results_not_generated_analysis(home_repair_demo):
    hallucinated = {
        problem["id"]: "[ProQuote] Faulty heating element. Answer is 15."
        for problem in home_repair_demo._PROBLEMS
    }

    ui_data = home_repair_demo.build_ui_data(
        analysis={"steps": []},
        per_problem=hallucinated,
        final_recommendation="grounded report",
        model_name="/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp",
        sae_layer=27,
        threshold_offset=1.12890625,
    )

    for problem in home_repair_demo._PROBLEMS:
        recommendation = ui_data["recommendations"][problem["id"]]
        assert recommendation["rationale"] == home_repair_demo._GROUNDED_RATIONALES[problem["id"]]
        assert "heating element" not in recommendation["rationale"]
        decision = ui_data["decisionFeatures"][problem["id"]]
        ask_tool, ask_request = home_repair_demo._initial_tool_decision(problem)
        assert "expectedTool" not in decision and decision["askTool"] == ask_tool
        assert decision["request"] == ask_request
        assert decision["features"] == []

    assert "saeFeatures" not in ui_data

    assert ui_data["runMetadata"] == {
        "model": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp",
        "saeLayer": 27,
        "thresholdOffset": 1.12890625,
    }
    for problem in home_repair_demo._PROBLEMS:
        decision = ui_data["decisionFeatures"][problem["id"]]
        assert decision["modelChoice"] is None
        assert decision["themeEvidence"] is None
    assert "steering" not in ui_data


def test_generation_prompt_disables_thinking(home_repair_demo):
    class FakeTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "formatted"

    engine = home_repair_demo.HFEngine.__new__(home_repair_demo.HFEngine)
    engine.tokenizer = FakeTokenizer()
    engine.generation_template_kwargs = {"enable_thinking": False}

    assert engine._build_prompt("system", "user") == "formatted"
    assert engine.tokenizer.kwargs["enable_thinking"] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>private reasoning</think>Final answer", "Final answer"),
        ("private reasoning</think>Final answer", "Final answer"),
        ("Final answer", "Final answer"),
    ],
)
def test_strip_thinking(home_repair_demo, text, expected):
    assert home_repair_demo._strip_thinking(text) == expected


def test_injection_summary_condenses_and_withholds(home_repair_demo):
    block = {
        "hfChoice": "ManualCheck",
        "targetTool": "PartsSearch",
        "rows": [
            {"index": 1, "label": "Gasket search", "deltaTarget": 0.002},
            {"index": 2, "label": "Part lookup", "deltaTarget": 0.01},
        ],
        "allRows": {
            "size": 5,
            "deltaTarget": 0.012,
            "argmaxChanged": False,
            "choice": "ManualCheck",
        },
        "allBase": {
            "size": 80,
            "deltaTarget": 0.02,
            "argmaxChanged": False,
            "choice": "ManualCheck",
        },
        "controlThreshold": 0.001,
    }
    summary = home_repair_demo.injection_summary(block, {"display": "ManualCheck"})
    assert summary["withheld"] is False and summary["bestRow"]["index"] == 2
    assert summary["allBase"] == {
        "size": 80,
        "deltaTarget": 0.02,
        "argmaxChanged": False,
        "choice": "ManualCheck",
    }
    withheld = home_repair_demo.injection_summary(block, {"display": "PartsSearch"})
    assert withheld == {"withheld": True, "hfChoice": "ManualCheck", "vllmChoice": "PartsSearch"}
    assert home_repair_demo.injection_summary(None, None) is None


def test_hf_parity_summary_and_fast_path_detection(home_repair_demo):
    assert home_repair_demo.hf_parity_summary(None) is None
    assert home_repair_demo.hf_parity_summary({"parity": {}}) is None
    summary = home_repair_demo.hf_parity_summary(
        {"parity": {"a": {"cosine": 0.9}, "b": {}}}  # no baselines, no fast-path flag
    )
    assert summary == {
        "fastPath": None,
        "prompts": 2,
        "cosineMean": 0.9,
        "cosineMin": 0.9,
        "baselineAgreement": None,
        "baselineCompared": 0,
    }

    import sys
    import types

    import torch

    # A model whose layer class lives in a module carrying transformers' flag.
    fake_mod = types.ModuleType("fake_mamba_modeling")
    fake_mod.is_fast_path_available = False
    sys.modules["fake_mamba_modeling"] = fake_mod
    try:
        layer_cls = type("FakeMixer", (torch.nn.Module,), {"__module__": "fake_mamba_modeling"})
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), layer_cls())
        assert home_repair_demo.hf_fast_path_status(model) is False
        fake_mod.is_fast_path_available = True
        assert home_repair_demo.hf_fast_path_status(model) is True
    finally:
        del sys.modules["fake_mamba_modeling"]
    # Architectures without Mamba layers report None.
    assert home_repair_demo.hf_fast_path_status(torch.nn.Linear(2, 2)) is None


def _spec_sheet_fixture():
    return {
        "depth": {
            "43": {"sidesAllBeyondControl": 8, "nSides": 14, "crossFlips": 1},
            "6": {"sidesAllBeyondControl": 1, "nSides": 14, "crossFlips": 0},
        },
        "transfer": {
            "layers": {
                "43": {
                    "matching": {
                        "joint->joint_seed123": {
                            "rateAtLeast0.01": {
                                "decoder": {"fracAtLeast07": 0.0011},
                                "functional": {"fracAtLeast07": 0.6266},
                            }
                        }
                    }
                }
            }
        },
        "workbench": {
            "bow": {
                "home_repair": {"accuracy": 0.8634},
                "tool_selection": {"accuracy": 0.8993},
            },
            "layers": {
                "27": {
                    "probes": {
                        "home_repair": {
                            "saeFeatures": {"accuracy": 0.7681},
                            "residual": {"accuracy": 0.8703},
                        }
                    }
                },
                "43": {
                    "probes": {
                        "tool_selection": {
                            "saeFeatures": {"accuracy": 0.8907},
                            "residual": {"accuracy": 0.9324},
                        }
                    }
                },
            },
        },
        "population": {"overall": {"pairs": 7478, "flipping": 3342, "flipAtLeast06": 2303}},
        "robustness": {
            "joint_seed123_layer43": {"sidesAllBeyondControl": 11, "nSides": 14, "crossFlips": 5}
        },
    }


def test_spec_sheet_note_condenses_per_scenario(home_repair_demo):
    hr_note = home_repair_demo.spec_sheet_note(_spec_sheet_fixture(), "home_repair", 27)
    assert [d["layer"] for d in hr_note["depth"]] == [6, 43]  # sorted numerically
    assert hr_note["probes"] == {
        "layer": 27,
        "sae": 0.7681,
        "residual": 0.8703,
        "bow": 0.8634,
    }
    assert hr_note["stability"]["functionalFrac07"] == 0.6266
    assert "population" not in hr_note and "robustness" not in hr_note
    assert hr_note["link"].endswith("spec_sheet/index.html")

    ts_note = home_repair_demo.spec_sheet_note(_spec_sheet_fixture(), "tool_selection", 43)
    assert ts_note["population"]["pairs"] == 7478
    assert ts_note["robustness"][0]["beyond"] == 11
    assert ts_note["probes"]["sae"] == 0.8907

    assert home_repair_demo.spec_sheet_note(None, "home_repair", 27) is None
    assert home_repair_demo.spec_sheet_note({}, "home_repair", 27) is None


def test_attach_spec_sheet_is_optional(home_repair_demo, tmp_path):
    spec_path = tmp_path / "ui_data.json"
    spec_path.write_text(json.dumps(_spec_sheet_fixture()))
    data = home_repair_demo.attach_spec_sheet({}, "tool_selection", 43, path=spec_path)
    assert data["specSheet"]["population"]["flipping"] == 3342
    untouched = home_repair_demo.attach_spec_sheet({}, "home_repair", 27, path=tmp_path / "no.json")
    assert "specSheet" not in untouched


def test_html_spec_sheet_strip_guarded():
    html = (Path(__file__).parents[1] / "demo" / "home_repair" / "index.html").read_text()
    script = html[html.index("<script>") :]
    code = (
        script[: script.index("/* MOCK_DATA_START */")]
        + script[script.index("/* MOCK_DATA_END */") :]
    )
    assert 'id="spec-sheet-section"' in html and 'style="display:none"' in html
    assert "function renderSpecSheet(" in code
    assert "DATA.specSheet" in code  # renders only when the block exists
    assert "safely('spec sheet', renderSpecSheet)" in code
