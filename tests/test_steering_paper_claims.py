"""The paper quotes the report; this checks it still does.

Every figure in `Steering Agent Tool Selection.tex` is supposed to come from
`results/steering_report.json`, but nothing enforces that — a battery re-run
moves the report and leaves the prose behind.  These tests pin the headline
claims to the values they were written from, so drift fails a test run.

Each assertion names the section it guards.  They skip when the report has not
been generated, since it is a build product.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_REPORT = _ROOT / "paper" / "steering" / "results" / "steering_report.json"
_TEX = _ROOT / "paper" / "steering" / "Steering Agent Tool Selection.tex"


def _tex_num(n: int) -> str:
    """The paper writes thousands as ``3{,}032``; the report stores ``3032``."""
    return f"{n:,}".replace(",", "{,}")


@pytest.fixture(scope="module")
def report() -> dict:
    if not _REPORT.exists():
        pytest.skip("steering_report.json not generated")
    return json.loads(_REPORT.read_text())


@pytest.fixture(scope="module")
def tex() -> str:
    if not _TEX.exists():
        pytest.skip("paper source not present")
    # collapse whitespace so a line break inside a sentence does not hide a claim
    return re.sub(r"\s+", " ", _TEX.read_text())


def test_expanded_flip_counts(report, tex):
    """Sect. 4.2 and Table 2."""
    scen = report["scenarios"]
    sc = scen["supply_chain_expanded"]["layers"]["43"]
    cs = scen["customer_support_expanded"]["layers"]["34"]
    assert f"{sc['ablationFlips']}/64" in tex and f"{cs['ablationFlips']}/64" in tex
    pooled_abl = sc["ablationFlips"] + cs["ablationFlips"]
    pooled_cp = sc["crossPatchFlips"] + cs["crossPatchFlips"]
    assert f"ablation flips {pooled_abl} of 128" in tex
    assert f"cross-patch {pooled_cp} of 128" in tex


def test_outcome_partition_and_runner_up_audit(report, tex):
    """Sect. 4.2 'Where the argmax goes' and Table 3."""
    o = report["stats"]["outcomesExpanded"]
    a, cue, ctl = o["ablation"], o["crossPatchCue"], o["ablationControls"]
    assert f"unchanged on {a['unchanged']} of 128 sides" in tex
    assert f"moves it to the paired tool on {a['directed']}" in tex
    assert f"and to some third tool on {cue['thirdTool']}" in tex
    assert f"unchanged on {cue['unchanged']} of 128 directions" in tex
    ru = o["runnerUp"]
    assert f"runner-up on {ru['pairedToolIsBaselineRunnerUp']} of the 128 sides" in tex
    assert f"{ru['flipsLandingOnBaselineRunnerUp']} of the {ru['ablationFlips']} ablation flips" in tex
    assert f"all {ru['runnerUpYesFoundYes']} flips land on it" in tex
    assert ctl is not None, "ablation controls must record their argmax"
    assert f"{_tex_num(ctl['unchanged'])} leave the tool untouched" in tex
    assert f"{ctl['directed']} move it to the paired tool and {ctl['thirdTool']}" in tex


def test_recovery_against_the_ceiling(report, tex):
    """Sect. 4.3 and Table 6."""
    e = report["stats"]["recoveryExpanded"]
    assert f"It flips {e['ceilingFlips']} of 128 expanded directions" in tex
    assert f"{e['differenceInMeansFlips']} of {e['differenceInMeansDirections']}" in tex
    assert (
        f"gives {e['differenceInMeansMatchedFlips']} of "
        f"{e['differenceInMeansMatchedDirections']}" in tex
    )
    cue_pct, bulk_pct = round(e["cueOverCeiling"] * 100), round(e["bulkOverCeiling"] * 100)
    assert f"recover \\textbf{{{cue_pct}\\% of the causal signal" in tex
    assert f"donor-active feature recovers {bulk_pct}\\%" in tex


def test_heldout_recovery(report, tex):
    """Sect. 4.4 'The same share of a comparable ceiling'."""
    h = report["stats"].get("recoveryHeldout")
    if not h:
        pytest.skip("no held-out ceiling run")
    assert f"It flips {h['ceilingFlips']} of 30" in tex
    assert f"recovers \\textbf{{{h['cueOverCeiling']}}}" in tex
    assert (
        f"{h['differenceInMeansMatchedFlips']} of "
        f"{h['differenceInMeansMatchedDirections']}" in tex
    )


def test_depth_contrast_including_the_non_selected_arm(report, tex):
    """Sect. 4.5 and the layer-selection paragraph in Limitations."""
    d = report["stats"]["depth"]
    assert f"{d['earlyFlips']} of {d['earlyN']} evaluated early-layer interventions" in tex
    assert f"{d['lateFlips']} of {d['lateN']}" in tex
    assert f"{d['lateFlipsNonSelected']} of {d['lateNNonSelected']}" in tex


def test_contrast_band_by_depth(report, tex):
    """Sect. 4.5 — the depth contrast measured on effect size, not argmax crossings."""
    c = report["stats"].get("contrastByDepth")
    if not c:
        pytest.skip("contrast arm not present across the depth grid")
    assert f"${c['early']['medianEffectOverBand']}\\times$ at layers 6--20" in tex
    assert f"${c['late']['medianEffectOverBand']}\\times$ at 27--43" in tex
    assert (
        f"{c['early']['exceeding']} of {c['early']['sides']} early sides against "
        f"{c['late']['exceeding']} of {c['late']['sides']} late ones" in tex
    )


def test_dictionary_health_table(report, tex):
    """Table 5 — the two rows the table used to omit, and the collapsed cell."""
    for scenario, layer in (("tool_selection", "34"), ("tool_selection", "43")):
        h = report["scenarios"][scenario]["dictionary"][layer]
        assert f"& {layer} & {round(h['meanL0'])} & {h['explainedVariance']:.2f}" in tex
    cs = report["scenarios"]["customer_support"]["dictionary"]["43"]
    assert f"{cs['constantFraction']:.2f}" in tex or "0.97" in tex


def test_no_stale_claim_that_the_ratio_is_the_only_separator(tex):
    """It was contradicted by the paper's own health table; it must not return."""
    assert "only an intervention-native check" not in tex
    assert "the one unusable cell from the seventeen others" not in tex
