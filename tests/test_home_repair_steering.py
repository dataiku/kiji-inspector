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


def test_delta_mode_edits_an_earlier_position_when_asked(steering):
    sae = FakeSAE()
    hidden = torch.tensor([[[9.0, 17.0, 3.0], [9.0, 17.0, 3.0], [0.0, 0.0, 0.0]]])

    out = _run_hook(steering.make_feature_edit_hook(sae, {1: None}, "delta", position=-2), hidden)

    assert torch.allclose(out[0, 0], hidden[0, 0]) and torch.allclose(out[0, 2], hidden[0, 2])
    assert torch.allclose(out[0, 1], torch.tensor([9.0, 1.0, 3.0]))
    with pytest.raises(ValueError):
        steering.make_feature_edit_hook(sae, {}, "delta", position=0)
