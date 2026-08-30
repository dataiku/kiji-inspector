from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def steering():
    directory = Path(__file__).parents[1] / "demo" / "home_repair"
    path = directory / "steer_tool_choice.py"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("steer_tool_choice", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class FakeSAE(torch.nn.Module):
    """Two-feature linear SAE with an exact decoder and a lossy encoder."""

    def __init__(self):
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
        self.b_dec = torch.nn.Parameter(torch.zeros(3))
        self.rms_scale = 4.0
        self.mean_vec = torch.tensor([1.0, 1.0, 1.0])

    def normalize_input(self, x):
        return (x - self.mean_vec.to(x)) / self.rms_scale

    def denormalize_output(self, x):
        return x * self.rms_scale + self.mean_vec.to(x)

    def encode(self, x):
        # feature 0 = normalized dim 0, feature 1 = normalized dim 1 / 2 (thresholded at 0)
        feats = torch.stack([x[:, 0], x[:, 1] / 2.0], dim=1)
        return torch.clamp(feats, min=0.0)

    def decode(self, feats):
        return feats @ self.W_dec + self.b_dec


class FakeLayer(torch.nn.Module):
    def forward(self, hidden_states, *rest, **kwargs):
        return hidden_states


def _run_hook(hook, hidden, use_kwargs=False):
    layer = FakeLayer()
    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        if use_kwargs:
            return layer(hidden_states=hidden)
        return layer(hidden)
    finally:
        handle.remove()


def test_delta_mode_ablation_removes_exactly_the_decoder_direction(steering):
    sae = FakeSAE()
    # last token: normalized = (2, 4, 0.5) -> features (2, 2)
    hidden = torch.tensor([[[0.0, 0.0, 0.0], [9.0, 17.0, 3.0]]])
    record = {}

    out = _run_hook(steering.make_feature_edit_hook(sae, {1: None}, "delta", record=record), hidden)

    assert torch.allclose(out[0, 0], hidden[0, 0])  # earlier tokens untouched
    # feature 1 activation 2, W_dec[1] = (0,2,0), scale 4 -> subtract 16 from dim 1
    assert torch.allclose(out[0, 1], torch.tensor([9.0, 1.0, 3.0]))
    assert record["pre_edit"] == {1: 2.0}


def test_delta_mode_clamp_and_noop(steering):
    sae = FakeSAE()
    hidden = torch.tensor([[[9.0, 17.0, 3.0]]])

    clamped = _run_hook(steering.make_feature_edit_hook(sae, {0: 5.0}, "delta"), hidden)
    # feature 0 activation 2 -> 5: +3 * 4 * (1,0,0)
    assert torch.allclose(clamped[0, 0], torch.tensor([21.0, 17.0, 3.0]))

    noop = _run_hook(steering.make_feature_edit_hook(sae, {0: 2.0}, "delta"), hidden)
    assert torch.allclose(noop, hidden)

    untouched = _run_hook(steering.make_feature_edit_hook(sae, {}, "delta"), hidden)
    assert torch.allclose(untouched, hidden)


def test_replace_mode_reconstructs_and_handles_kwargs(steering):
    sae = FakeSAE()
    hidden = torch.tensor([[[9.0, 17.0, 3.0]]])

    recon_only = _run_hook(steering.make_feature_edit_hook(sae, {}, "replace"), hidden, True)
    # decode((2,2)) = (2, 4, 0) -> denormalize -> (9, 17, 1): dim 2 is lost by the SAE
    assert torch.allclose(recon_only[0, 0], torch.tensor([9.0, 17.0, 1.0]))

    ablated = _run_hook(steering.make_feature_edit_hook(sae, {1: None}, "replace"), hidden, True)
    assert torch.allclose(ablated[0, 0], torch.tensor([9.0, 1.0, 1.0]))

    with pytest.raises(ValueError):
        steering.make_feature_edit_hook(sae, {}, "bogus")


def test_select_steering_features_ranks_side_specific_features(steering):
    contrastive_map = {
        1: [{"theme": "safe_vs_hazardous", "cohens_d": 0.5, "direction": "contrast"}],
        2: [
            {"theme": "diy_vs_professional", "cohens_d": 0.7, "direction": "contrast"},
            {"theme": "safe_vs_hazardous", "cohens_d": 0.2, "direction": "contrast"},
        ],
        3: [{"theme": "safe_vs_hazardous", "cohens_d": 3.0, "direction": "anchor"}],
        4: [{"theme": "urgent_vs_planned", "cohens_d": 3.0, "direction": "contrast"}],
    }
    active = [(1, 4.0), (2, 2.0), (3, 9.0), (4, 9.0)]

    rows = steering.select_steering_features(
        active, contrastive_map, ("safe_vs_hazardous", "diy_vs_professional"), "contrast", k=5
    )

    assert [row["index"] for row in rows] == [1, 2]
    assert rows[0]["weight"] == pytest.approx(2.0)
    assert rows[1]["theme"] == "diy_vs_professional"  # best of the two entries kept


def test_distribution_from_logits_uses_full_softmax(steering):
    logits = torch.full((20,), -10.0)
    logits[10] = 2.0
    logits[11] = 1.0
    logits[5] = 3.0  # a non-tool token wins the argmax
    tool_to_token = {"manual_check": 10, "parts_search": 11, "tutorial_search": 12, "pro_quote": 13}

    reading = steering.distribution_from_logits(logits, tool_to_token)

    assert reading["toolId"] == "manual_check"
    assert reading["distribution"]["ManualCheck"] == pytest.approx(0.7311, abs=1e-3)
    assert reading["sampledTool"] is None
    assert reading["coverage"] < 1.0


def test_hazard_experiments_pair_ablation_with_clamp(steering):
    contrastive_map = {
        7: [{"theme": "safe_vs_hazardous", "cohens_d": 0.6, "direction": "contrast"}],
    }
    active = {"water_heater_noise": [(7, 5.0), (8, 1.0)], "disposal_stuck": [(8, 2.0)]}

    experiments = steering.hazard_experiments(active, contrastive_map, {"7": "Gas valve"})

    assert [e["id"] for e in experiments] == ["water_heater_ablate_hazard", "disposal_clamp_hazard"]
    assert experiments[0]["mode"] == "ablate" and experiments[0]["features"][0]["target"] == 0.0
    assert experiments[1]["mode"] == "clamp" and experiments[1]["features"][0]["target"] == 5.0
    assert experiments[1]["features"][0]["label"] == "Gas valve"
    assert steering.hazard_experiments({}, contrastive_map, None) == []


def test_contrast_experiments_use_discriminating_features(steering):
    base = {"water_heater_noise": [(1, 6.0), (2, 3.0), (3, 1.0)], "disposal_stuck": [(9, 2.0)]}
    variant = {"water_heater_noise": [(1, 1.0), (2, 3.5), (4, 2.0)]}
    labels = {"1": "Professional quote request", "3": "Gas water heater"}

    rows = steering.discriminating_features(
        base["water_heater_noise"], variant["water_heater_noise"], labels, k=5
    )
    assert [row["index"] for row in rows] == [1, 3]  # feature 2 is stronger on the variant
    assert rows[0]["gap"] == pytest.approx(5.0) and rows[0]["variantActivation"] == 1.0

    experiments = steering.contrast_experiments(
        base, variant, labels, k=5, reference_activations={"water_heater_noise": {1: 5.5}}
    )
    assert [e["id"] for e in experiments] == [
        "water_heater_noise_ablate_discriminating",
        "water_heater_noise_clamp_discriminating",
    ]
    ablate, clamp = experiments
    assert ablate["step"] == "water_heater_noise_InitialDecision" and ablate["mode"] == "ablate"
    assert clamp["step"] == "water_heater_noise_Contrast" and clamp["mode"] == "clamp"
    assert clamp["features"][0]["target"] == 6.0
    assert ablate["features"][0]["vllmActivation"] == 5.5
    assert "ProQuote" in ablate["description"]
    assert steering.distribution_deltas({"A": 0.2, "B": 0.8}, {"A": 0.5, "B": 0.5}) == {
        "A": 0.3,
        "B": -0.3,
    }


def test_matched_random_sets_respect_size_mass_and_exclusion(steering):
    active = [(i, float(i)) for i in range(1, 11)]  # activations 1..10
    sets = steering.matched_random_sets(
        active, exclude={10}, target_mass=12.0, target_size=2, draws=3, seed=1
    )
    assert len(sets) == 3
    for chosen in sets:
        assert 10 not in chosen and len(chosen) >= 2
        assert sum(chosen) >= 12.0  # activation == index here
    assert sets == steering.matched_random_sets(
        active, exclude={10}, target_mass=12.0, target_size=2, draws=3, seed=1
    )
    assert steering.matched_random_sets([], set(), 1.0, 1, 2, 0) == []


def test_attribution_plan_and_summary(steering):
    rows = [
        {"index": 1, "label": "Gasket search", "merged": [2]},
        {"index": 5, "label": "Silent under HF", "merged": []},
    ]
    hf_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0), (6, 0.5)]

    plan = steering.attribution_plan(rows, hf_active, draws=2, seed=0)

    assert plan["rows"][0]["family"] == [1, 2] and plan["rows"][0]["hfMass"] == 5.0
    assert plan["rows"][0]["hfActivation"] == 4.0 and not plan["rows"][0]["inactiveUnderHf"]
    assert plan["rows"][1]["inactiveUnderHf"] and plan["rows"][1]["hfActivation"] == 0.0
    assert plan["allFeatures"] == [1, 2, 5]
    assert len(plan["controls"]) == 4
    for control in plan["controls"]:
        assert not set(control["features"]) & {1, 2, 5}
        if control["matchesRow"] == 1:
            assert control["hfMass"] >= 5.0  # 3 + 4 (+0.5)

    baseline = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.8, "ManualCheck": 0.2}}
    row_readings = [
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.4, "ManualCheck": 0.6}},
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.8, "ManualCheck": 0.2}},
    ]
    all_reading = {
        "display": "ManualCheck",
        "distribution": {"PartsSearch": 0.3, "ManualCheck": 0.7},
    }
    control_readings = [
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.78, "ManualCheck": 0.22}}
    ] * 4

    summary = steering.summarize_attribution(
        plan, baseline, row_readings, all_reading, control_readings, "PartsSearch"
    )

    assert summary["rows"][0]["deltaTarget"] == pytest.approx(-0.4)
    assert summary["rows"][0]["argmaxChanged"] is True
    assert summary["rows"][1]["deltaTarget"] == pytest.approx(0.0)
    assert summary["controlThreshold"] == pytest.approx(0.02)
    assert summary["allRows"]["deltaTarget"] == pytest.approx(-0.5)
    assert summary["hfChoice"] == "PartsSearch"


def test_injection_plan_and_summary(steering):
    rows = [
        {"index": 1, "label": "Gasket search", "merged": [2]},
        {"index": 5, "label": "Absent on base", "merged": []},
    ]
    base_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0), (6, 0.5)]
    open_active = [(2, 0.5), (7, 2.0)]

    plan = steering.injection_plan(rows, base_active, open_active, draws=2, seed=0)

    assert plan["rows"][0]["targets"] == {"1": 4.0, "2": 1.0}
    assert plan["rows"][0]["baseMass"] == 5.0 and plan["rows"][0]["openMass"] == 0.5
    assert plan["rows"][1]["absentOnBase"] and plan["rows"][1]["targets"] == {}
    assert set(plan["allRowsTargets"]) == {"1", "2"}
    assert set(plan["allBaseTargets"]) == {"1", "2", "3", "4", "6"}
    assert len(plan["controls"]) == 4
    for control in plan["controls"]:
        assert not set(map(int, control["targets"])) & {1, 2, 5}
        if control["matchesRow"] == 1:
            assert control["baseMass"] >= 5.0

    baseline = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.1, "ManualCheck": 0.9}}
    row_readings = [
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.6, "ManualCheck": 0.4}},
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.1, "ManualCheck": 0.9}},
    ]
    all_rows = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.7, "ManualCheck": 0.3}}
    all_base = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.9, "ManualCheck": 0.1}}
    controls = [
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.12, "ManualCheck": 0.88}}
    ] * 4

    summary = steering.summarize_injection(
        plan, baseline, row_readings, all_rows, all_base, controls, "PartsSearch"
    )

    assert summary["rows"][0]["deltaTarget"] == pytest.approx(0.5)
    assert (
        summary["rows"][0]["argmaxChanged"] is True
        and summary["rows"][0]["choice"] == "PartsSearch"
    )
    assert summary["rows"][1]["deltaTarget"] == pytest.approx(0.0)
    assert summary["allRows"]["deltaTarget"] == pytest.approx(0.6)
    assert summary["allBase"]["deltaTarget"] == pytest.approx(0.8)
    assert summary["controlThreshold"] == pytest.approx(0.02)
    assert summary["hfChoice"] == "ManualCheck"


def test_residual_patch_hook_swaps_one_token_and_leaves_the_rest(steering):
    """The ceiling arm patches the model's own basis, no dictionary involved."""
    hidden = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    donor = torch.tensor([9.0, 9.0, 9.0])

    out = _run_hook(steering.make_residual_patch_hook(donor), hidden)
    assert torch.allclose(out[0, 0], hidden[0, 0])
    assert torch.allclose(out[0, 1], donor)

    # the second forward appends a token, so the decision token is at -2
    second = _run_hook(steering.make_residual_patch_hook(donor, position=-2), hidden, True)
    assert torch.allclose(second[0, 0], donor)
    assert torch.allclose(second[0, 1], hidden[0, 1])

    # a sequence too short for the requested position is left alone rather than
    # patched at the wrong place
    short = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert torch.allclose(
        _run_hook(steering.make_residual_patch_hook(donor, position=-2), short), short
    )

    with pytest.raises(ValueError):
        steering.make_residual_patch_hook(donor, position=0)


def test_contrast_controls_match_how_much_the_set_differs_across_the_pair(steering):
    """Mass-matching leaves the selection rule uncontrolled.

    Cue families are chosen *because* they differ across the pair, so a control
    matched only on activation mass answers "does removing this much matter?"
    and not "does removing the differing features matter?".  The contrast arm
    draws sets whose across-pair difference matches the cue set's.
    """
    rows = [{"index": 1, "label": "Gasket search", "merged": []}]
    # feature 1 is the cue: present here, absent on the other side.  Feature 3
    # is heavy but identical across the pair, so it is a poor contrast match;
    # features 4 and 6 differ and are the ones a contrast draw should reach for.
    hf_active = [(1, 4.0), (3, 9.0), (4, 3.0), (6, 2.0)]
    other_active = [(3, 9.0), (4, 0.5), (6, 0.0)]

    plan = steering.attribution_plan(rows, hf_active, draws=2, seed=0, other_active=other_active)

    assert plan["setContrastMass"] == 4.0  # |4.0 - 0.0| for the cue feature
    assert len(plan["contrastControls"]) == 2
    for control in plan["contrastControls"]:
        assert 1 not in control["features"]
        assert control["targetContrastMass"] == 4.0
        # a mass-matched draw could satisfy itself with feature 3 alone; a
        # contrast-matched one cannot, because feature 3 does not differ
        assert control["features"] != [3]
        assert control["contrastMass"] >= 4.0 - 1e-6 or not control["massMatched"]


def test_contrast_controls_report_a_ceiling_when_the_cue_set_is_the_difference(steering):
    """If nothing else differs, that is the finding, not a broken control."""
    rows = [{"index": 1, "label": "Cue", "merged": []}]
    hf_active = [(1, 6.0), (3, 9.0)]
    other_active = [(3, 9.0)]  # only feature 1 differs across the pair

    plan = steering.attribution_plan(rows, hf_active, draws=3, seed=0, other_active=other_active)

    controls = plan["contrastControls"]
    assert controls and not any(c["massMatched"] for c in controls)
    # the pool outside the cue set carries no across-pair difference at all, so
    # the arm records an explicit empty draw rather than going missing
    assert controls == [
        {
            "matchesRow": None,
            "draw": 0,
            "features": [],
            "size": 0,
            "contrastMass": 0.0,
            "targetContrastMass": 6.0,
            "massMatched": False,
            "poolEmpty": True,
        }
    ]


def test_attribution_plan_without_the_other_side_has_no_contrast_arm(steering):
    """Callers that cannot supply the pair's other side keep the old plan."""
    plan = steering.attribution_plan(
        [{"index": 1, "label": "Cue", "merged": []}], [(1, 4.0), (3, 2.0)], draws=2, seed=0
    )
    assert "contrastControls" not in plan and "setContrastMass" not in plan


def test_attribution_controls_record_the_tool_they_produced(steering):
    """Without the argmax, the ablation arm has no outcome partition."""
    rows = [{"index": 1, "label": "Cue", "merged": []}]
    hf_active = [(1, 4.0), (3, 2.0), (4, 2.0)]
    plan = steering.attribution_plan(rows, hf_active, draws=2, seed=0, other_active=[(3, 2.0)])
    baseline = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.7, "ManualCheck": 0.3}}
    moved = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.3, "ManualCheck": 0.7}}

    summary = steering.summarize_attribution(
        plan,
        baseline,
        [moved],
        moved,
        [baseline] * len(plan["controls"]),
        "PartsSearch",
        [baseline] * len(plan["setControls"]),
        [moved] * len(plan["contrastControls"]),
    )

    assert all(c["choice"] == "PartsSearch" for c in summary["controls"])
    assert all(c["choice"] == "PartsSearch" for c in summary["setControls"])
    assert all(c["choice"] == "ManualCheck" for c in summary["contrastControls"])
    assert summary["allRows"]["choice"] == "ManualCheck"
    assert summary["contrastControlThreshold"] == 0.4


def test_set_controls_match_the_whole_cue_set_not_one_family(steering):
    """The set-level arm needs its own draws: row-matched ones are far lighter.

    ``allFeatures`` / ``allRowsTargets`` ablate or clamp every family at once,
    so the only honest reference is a draw matched to the union's count and
    mass.  ``controls`` cannot serve: each is matched to a single row, and a
    row that is silent under HF gets a one-feature draw.
    """
    rows = [
        {"index": 1, "label": "Gasket search", "merged": [2]},
        {"index": 5, "label": "Silent under HF", "merged": []},
    ]
    hf_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0), (6, 0.5)]

    plan = steering.attribution_plan(rows, hf_active, draws=2, seed=0)

    assert plan["setMass"] == 5.0 and plan["setActiveSize"] == 2
    assert len(plan["setControls"]) == 2
    for control in plan["setControls"]:
        assert control["matchesRow"] is None
        assert not set(control["features"]) & {1, 2, 5}
        assert control["size"] >= plan["setActiveSize"]
        assert control["hfMass"] >= plan["setMass"]
        assert control["targetMass"] == plan["setMass"] and control["massMatched"]
    # the row matched to the silent family is a single feature — 0.5 of mass
    # against the set's 5.0, which is exactly why it cannot stand in for it
    silent = [c for c in plan["controls"] if c["matchesRow"] == 5]
    assert silent and min(c["hfMass"] for c in silent) < plan["setMass"]

    baseline = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.8, "ManualCheck": 0.2}}
    row_readings = [
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.4, "ManualCheck": 0.6}},
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.8, "ManualCheck": 0.2}},
    ]
    all_reading = {
        "display": "ManualCheck",
        "distribution": {"PartsSearch": 0.3, "ManualCheck": 0.7},
    }
    control_readings = [
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.78, "ManualCheck": 0.22}}
    ] * 4
    set_control_readings = [
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.7, "ManualCheck": 0.3}},
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.75, "ManualCheck": 0.25}},
    ]

    summary = steering.summarize_attribution(
        plan,
        baseline,
        row_readings,
        all_reading,
        control_readings,
        "PartsSearch",
        set_control_readings,
    )

    # the two bands are different quantities and both are reported
    assert summary["controlThreshold"] == pytest.approx(0.02)
    assert summary["setControlThreshold"] == pytest.approx(0.10)
    assert summary["setMass"] == 5.0 and len(summary["setControls"]) == 2
    assert summary["setControlMassMatched"] is True
    assert summary["setControlDistinctDraws"] == 2

    # ...and omitting the readings leaves the set band out rather than
    # silently reusing the per-family one
    without = steering.summarize_attribution(
        plan, baseline, row_readings, all_reading, control_readings, "PartsSearch"
    )
    assert "setControlThreshold" not in without


def test_set_controls_report_when_the_pool_cannot_reach_the_cue_mass(steering):
    """A cue set heavier than everything else active gets a ceiling, not a draw.

    ``matched_random_sets`` then returns the whole pool for every draw.  The
    plan has to say so, or the band reads as a matched control when it is
    really "every other active feature at once".
    """
    rows = [{"index": 1, "label": "Carries most of the mass", "merged": [2]}]
    hf_active = [(1, 40.0), (2, 30.0), (3, 2.0), (4, 1.0)]

    plan = steering.attribution_plan(rows, hf_active, draws=3, seed=0)

    assert plan["setMass"] == 70.0
    assert len({tuple(c["features"]) for c in plan["setControls"]}) == 1  # the whole pool
    assert all(c["features"] == [3, 4] for c in plan["setControls"])
    assert not any(c["massMatched"] for c in plan["setControls"])

    baseline = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.9, "ManualCheck": 0.1}}
    reading = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.2, "ManualCheck": 0.8}}
    flat = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.88, "ManualCheck": 0.12}}
    summary = steering.summarize_attribution(
        plan, baseline, [reading], reading, [flat] * 3, "PartsSearch", [flat] * 3
    )

    assert summary["setControlMassMatched"] is False
    assert summary["setControlDistinctDraws"] == 1
    assert summary["setControlThreshold"] == pytest.approx(0.02)


def test_injection_set_controls_match_the_clamped_set(steering):
    rows = [
        {"index": 1, "label": "Gasket search", "merged": [2]},
        {"index": 5, "label": "Absent on base", "merged": []},
    ]
    base_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0), (6, 0.5)]
    open_active = [(2, 0.5), (7, 2.0)]

    plan = steering.injection_plan(rows, base_active, open_active, draws=2, seed=0)

    assert plan["setMass"] == 5.0 and plan["setActiveSize"] == 2
    assert len(plan["setControls"]) == 2
    for control in plan["setControls"]:
        assert control["matchesRow"] is None
        assert not set(map(int, control["targets"])) & {1, 2, 5}
        assert control["baseMass"] >= plan["setMass"]

    baseline = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.1, "ManualCheck": 0.9}}
    row_readings = [
        {"display": "PartsSearch", "distribution": {"PartsSearch": 0.6, "ManualCheck": 0.4}},
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.1, "ManualCheck": 0.9}},
    ]
    all_rows = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.7, "ManualCheck": 0.3}}
    all_base = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.9, "ManualCheck": 0.1}}
    controls = [
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.12, "ManualCheck": 0.88}}
    ] * 4
    set_controls = [
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.2, "ManualCheck": 0.8}},
        {"display": "ManualCheck", "distribution": {"PartsSearch": 0.15, "ManualCheck": 0.85}},
    ]

    summary = steering.summarize_injection(
        plan, baseline, row_readings, all_rows, all_base, controls, "PartsSearch", set_controls
    )

    assert summary["controlThreshold"] == pytest.approx(0.02)
    assert summary["setControlThreshold"] == pytest.approx(0.10)
    assert summary["setControls"][0]["choice"] == "ManualCheck"


def test_delta_controls_match_the_change_the_clamp_makes(steering):
    """Donor mass is not the perturbation; sum |donor - recipient| is.

    A cross-patch clamps a donor activation onto a recipient that may already
    carry some of it, so a feature can be heavy on the donor and move nothing.
    Cue features are picked for differing across the pair, so the cue set moves
    most of its donor mass -- a donor-matched random draw carries no such
    guarantee, and would be a lighter intervention wearing a matched label.
    """
    rows = [
        {"index": 1, "label": "Gasket search", "merged": [2]},
        {"index": 5, "label": "Absent on base", "merged": []},
    ]
    base_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0), (6, 0.5)]
    # feature 1 is already mostly there on the recipient, and feature 3 sits at
    # exactly its donor value: clamping 3 is a no-op however heavy it is
    open_active = [(1, 3.0), (3, 2.0), (7, 2.0)]

    plan = steering.injection_plan(rows, base_active, open_active, draws=2, seed=0)

    # the cue row weighs 5.0 on the donor but moves 2.0 = |4-3| + |1-0|
    assert plan["rows"][0]["baseMass"] == 5.0
    assert plan["rows"][0]["deltaMass"] == 2.0
    assert plan["setMass"] == 5.0 and plan["setDeltaMass"] == 2.0

    # the delta-matched draws reach the realised change, and feature 3 is not
    # eligible at all -- zero movement means zero weight in that pool
    assert len(plan["deltaControls"]) == 2
    for control in plan["deltaControls"]:
        assert "3" not in control["targets"]
        assert control["deltaMass"] >= plan["setDeltaMass"]
        assert control["targetDeltaMass"] == plan["setDeltaMass"]
        assert control["deltaMassMatched"]

    # ...whereas a donor-matched draw is free to spend its mass on feature 3,
    # which is exactly the overstatement this arm exists to expose
    padded = [c for c in plan["setControls"] if "3" in c["targets"]]
    assert padded and all(c["deltaMass"] < c["baseMass"] for c in padded)

    # every stored control now carries its realised change, so the match can be
    # checked from the artefacts instead of taken on trust
    for control in plan["controls"] + plan["setControls"] + plan["deltaControls"]:
        assert "deltaMass" in control


def test_ablation_plan_needs_no_delta_arm(steering):
    """Ablation's target is zero, so its donor mass *is* its realised change."""
    rows = [{"index": 1, "label": "Gasket search", "merged": [2]}]
    hf_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0)]

    plan = steering.attribution_plan(rows, hf_active, draws=2, seed=0)

    assert "deltaControls" not in plan
    # switching off against a silent recipient: the change equals the mass
    assert steering._delta_mass({1: 4.0, 2: 1.0}, {}) == plan["rows"][0]["hfMass"] == 5.0


def test_injection_summary_reports_all_three_bands(steering):
    rows = [{"index": 1, "label": "Gasket search", "merged": [2]}]
    base_active = [(1, 4.0), (2, 1.0), (3, 2.0), (4, 3.0), (6, 0.5)]
    open_active = [(1, 3.0), (3, 2.0), (7, 2.0)]

    plan = steering.injection_plan(rows, base_active, open_active, draws=2, seed=0)

    baseline = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.1, "ManualCheck": 0.9}}
    hit = {"display": "PartsSearch", "distribution": {"PartsSearch": 0.7, "ManualCheck": 0.3}}
    quiet = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.12, "ManualCheck": 0.88}}
    louder = {"display": "ManualCheck", "distribution": {"PartsSearch": 0.25, "ManualCheck": 0.75}}

    summary = steering.summarize_injection(
        plan, baseline, [hit], hit, hit, [quiet] * 2, "PartsSearch", [quiet] * 2, [louder] * 2
    )

    assert summary["controlThreshold"] == pytest.approx(0.02)
    assert summary["setControlThreshold"] == pytest.approx(0.02)
    assert summary["deltaControlThreshold"] == pytest.approx(0.15)
    assert summary["setDeltaMass"] == 2.0
    # the headline of the finding: only 40% of the donor mass is a real change
    assert summary["setDeltaOverDonorMass"] == pytest.approx(0.4)
    assert summary["deltaControlMassMatched"] is True

    # omitting the readings leaves the band out rather than reusing a weaker one
    without = steering.summarize_injection(
        plan, baseline, [hit], hit, hit, [quiet] * 2, "PartsSearch", [quiet] * 2
    )
    assert "deltaControlThreshold" not in without


def test_delta_mode_edits_an_earlier_position_when_asked(steering):
    sae = FakeSAE()
    hidden = torch.tensor([[[9.0, 17.0, 3.0], [9.0, 17.0, 3.0], [0.0, 0.0, 0.0]]])

    out = _run_hook(steering.make_feature_edit_hook(sae, {1: None}, "delta", position=-2), hidden)

    assert torch.allclose(out[0, 0], hidden[0, 0]) and torch.allclose(out[0, 2], hidden[0, 2])
    assert torch.allclose(out[0, 1], torch.tensor([9.0, 1.0, 3.0]))
    with pytest.raises(ValueError):
        steering.make_feature_edit_hook(sae, {}, "delta", position=0)
