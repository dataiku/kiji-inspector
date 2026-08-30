"""The demo READMEs quote numbers that live in the generated report.

Nothing regenerates a README, so its numbers drift silently when a battery is
re-run --- which is exactly what happened to the depth grid, where the READMEs
kept an any-tool count after the paper moved to directed flips.  These tests
re-read the report and check the quoted figures, so the drift fails a test run
instead of surviving to a reader.

They skip when the report has not been generated, since it is a build product.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_REPORT = _ROOT / "paper" / "steering" / "results" / "steering_report.json"
_STEER = _ROOT / "demo" / "steering"


def _quotes_rate(text: str, value: float) -> bool:
    """Does the prose quote this rate, at the report's precision or rounded to 2dp?

    A README writes 0.20 where the report stores 0.2, and 0.26 where it stores
    0.262; both are the same claim, and neither should fail the check.
    """
    return any(f"**{form}**" in text for form in (value, f"{value:.2f}", f"{value:.3f}"))


@pytest.fixture(scope="module")
def report() -> dict:
    if not _REPORT.exists():
        pytest.skip("steering_report.json not generated (run paper/steering/extract_results.py)")
    return json.loads(_REPORT.read_text())


def test_depth_grid_in_the_demo_readme_matches_the_report(report):
    md = (_STEER / "README.md").read_text()
    table = re.search(
        r"\| layer \| tool_selection \| supply_chain \| customer_support \|\n\|[-|]+\|\n((?:\|.*\n)+)",
        md,
    )
    assert table, "depth grid table not found in demo/steering/README.md"
    rows = table.group(1).strip().splitlines()
    assert len(rows) == 6, "expected one row per SAE layer"
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        layer = cells[0]
        for name, cell in zip(
            ("tool_selection", "supply_chain", "customer_support"), cells[1:], strict=True
        ):
            cross, ablation = (x.strip() for x in cell.replace("**", "").split("·"))
            block = report["scenarios"][name]["layers"][layer]
            assert cross.split()[0] == (
                f"{block['crossPatchFlips']}/{block['crossPatchDirections']}"
            ), f"cross-patch cell for {name} at layer {layer}"
            assert ablation.split()[0] == (
                f"{block['ablationFlips']}/{block['ablationSides']}"
            ), f"ablation cell for {name} at layer {layer}"


@pytest.mark.parametrize(
    ("scenario", "layer"),
    [("supply_chain_expanded", "43"), ("customer_support_expanded", "34")],
)
def test_expanded_readme_headline_matches_the_report(report, scenario, layer):
    md = (_STEER / scenario / "README.md").read_text()
    block = report["scenarios"][scenario]["layers"][layer]
    assert f"**{block['ablationFlips']} / {block['ablationSides']}**" in md
    assert f"**{block['crossPatchFlips']} / {block['crossPatchDirections']}**" in md


@pytest.mark.parametrize(
    "scenario", ["tool_selection", "supply_chain", "customer_support"]
)
def test_ceiling_block_matches_the_report(report, scenario):
    recovery = (report["stats"].get("recovery") or {}).get("perScenario", {}).get(scenario)
    if not recovery:
        pytest.skip(f"no ceiling run for {scenario}")
    md = (_STEER / scenario / "README.md").read_text()
    assert "### Against the ceiling" in md, "ceiling section missing"
    n = recovery["directions"]
    assert f"**{recovery['ceilingFlips']} / {n}**" in md
    assert f"**{recovery['cueFlips']} / {n}**" in md
    assert f"**{recovery['bulkFlips']} / {n}**" in md


@pytest.mark.parametrize(
    "scenario", ["supply_chain_heldout", "customer_support_heldout"]
)
def test_heldout_ceiling_block_matches_the_report(report, scenario):
    """The held-out READMEs quote their own recovery against the in-distribution one."""
    held = (report["stats"].get("recoveryHeldout") or {}).get("perScenario", {}).get(scenario)
    if not held:
        pytest.skip(f"no ceiling run for {scenario}")
    md = (_STEER / scenario / "README.md").read_text()
    assert "### Against the ceiling" in md
    assert f"**{held['ceilingFlips']} / {held['directions']}**" in md
    assert _quotes_rate(md, held["cueOverCeiling"]), "held-out recovery not quoted"
    # and against the matching in-distribution scenario, not the pooled figure
    indist = scenario.replace("_heldout", "_expanded")
    other = (report["stats"]["recoveryExpanded"]["perScenario"] or {}).get(indist)
    assert other and _quotes_rate(md, other["cueOverCeiling"]), (
        "held-out recovery must be compared with its own scenario in distribution"
    )


def test_readmes_do_not_claim_the_paper_and_demos_differ_on_flip_definition():
    """They used to; both now quote directed flips and the claim must not return."""
    for path in [_STEER / "README.md", *sorted(_STEER.glob("*/README.md"))]:
        text = path.read_text()
        assert "any-tool count the demo READMEs quote" not in text, path
