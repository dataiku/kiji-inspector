from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str):
    directory = Path(__file__).parents[1] / "demo" / "spec_sheet"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def splits():
    return _load("build_splits")


def _pair_row(anchor, contrast, scenario="tool_selection", contrast_type="x_vs_y"):
    return {
        "anchor_prompt": anchor,
        "contrast_prompt": contrast,
        "scenario_name": scenario,
        "contrast_type": contrast_type,
    }


def test_duo_records_maps_consecutive_pairs(splits):
    prompts = ["a1", "c1", "a2", "c2"]
    rows = [_pair_row("a1", "c1"), _pair_row("a2", "c2", scenario="home_repair")]
    records = splits.duo_records(prompts, rows)
    assert [r["duo"] for r in records] == [0, 1]
    assert records[0]["scenario"] == "tool_selection"
    assert records[1]["scenario"] == "home_repair"


def test_duo_records_rejects_odd_and_unknown(splits):
    with pytest.raises(ValueError, match="odd"):
        splits.duo_records(["a"], [])
    with pytest.raises(ValueError, match="not found"):
        splits.duo_records(["a", "b"], [_pair_row("a", "z")])


def test_prompt_components_links_transitively(splits):
    # a-b and b-c share prompt b -> one component; d-e is separate
    records = splits.duo_records(
        ["a", "b", "b", "c", "d", "e"],
        [_pair_row("a", "b"), _pair_row("b", "c"), _pair_row("d", "e")],
    )
    components = splits.prompt_components(records)
    assert components["a"] == components["b"] == components["c"]
    assert components["d"] == components["e"]
    assert components["a"] != components["d"]


def test_choose_eval_components_is_deterministic_and_covers_fraction(splits):
    component_prompts = {f"root{i}": [f"p{i}a", f"p{i}b"] for i in range(50)}
    first = splits.choose_eval_components(component_prompts, 0.2, seed=7)
    second = splits.choose_eval_components(component_prompts, 0.2, seed=7)
    assert first == second
    covered = sum(len(component_prompts[r]) for r in first)
    assert covered >= 0.2 * 100
    assert covered <= 0.2 * 100 + 2  # stops after reaching the fraction
    assert splits.choose_eval_components(component_prompts, 0.2, seed=8) != first


def test_split_row_indices_is_leak_free(splits):
    prompts = ["a", "b", "b", "c", "d", "e", "f", "g"]
    rows = [
        _pair_row("a", "b"),
        _pair_row("b", "c"),
        _pair_row("d", "e", scenario="home_repair"),
        _pair_row("f", "g", scenario="home_repair"),
    ]
    records = splits.duo_records(prompts, rows)
    components = splits.prompt_components(records)
    eval_roots = {
        "tool_selection": {components["a"]},  # holds out a, b, c
        "home_repair": {components["f"]},  # holds out f, g
    }
    indices = splits.split_row_indices(records, components, eval_roots)
    eval_prompts = {prompts[r] for r in indices["eval_tool_selection"]}
    train_prompts = {prompts[r] for r in indices["train_tool_selection_only"]}
    assert eval_prompts == {"a", "b", "c"}
    assert not eval_prompts & train_prompts
    # eval is deduplicated: prompt "b" fills two rows but appears once
    assert len(indices["eval_tool_selection"]) == 3
    # joint = union of the two scenario training splits
    assert sorted(indices["train_joint"]) == sorted(
        indices["train_home_repair_only"] + indices["train_tool_selection_only"]
    )
    assert {prompts[r] for r in indices["train_home_repair_only"]} == {"d", "e"}


def test_component_prompts_rejects_scenario_mixing(splits):
    records = splits.duo_records(
        ["a", "b", "b", "c"],
        [
            _pair_row("a", "b", scenario="tool_selection"),
            _pair_row("b", "c", scenario="home_repair"),
        ],
    )
    components = splits.prompt_components(records)
    with pytest.raises(ValueError, match="spans"):
        splits.component_prompts_by_scenario(records, components)
