"""The workshop paper quotes the report; this checks it still does.

``From Readable to Causal.tex`` leads with a paired cue-versus-dense
comparison and two nesting audits.  Those numbers were computed by hand while
the manuscript was being written and had no regeneration path, so a re-run of
the battery would have moved them silently.  ``paper/steering/extract_results``
now derives them, and these tests pin them two ways:

* against the values the submitted manuscript carries, which runs today; and
* against the ``.tex`` itself, which runs once the workshop paper and the
  report share a checkout.

The second set skips while the workshop paper lives on its own branch --- the
same convention as the sibling module, which skips when the report has not
been generated.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

# the manuscript spells small numbers out in prose and uses digits in tables,
# so assertions against running text have to spell them out too
_WORDS = {2: "two", 5: "five", 6: "six", 7: "seven", 13: "thirteen"}

_ROOT = Path(__file__).parents[1]
_REPORT = _ROOT / "paper" / "steering" / "results" / "steering_report.json"
_TEX = _ROOT / "paper" / "steering_workshop" / "From Readable to Causal.tex"
_ARTIFACTS = _ROOT / "paper" / "steering_workshop" / "artifacts"
_EXTRACTOR = _ROOT / "paper" / "steering" / "extract_results.py"


@pytest.fixture(scope="module")
def stats() -> dict:
    if not _REPORT.exists():
        pytest.skip("steering_report.json not generated")
    return json.loads(_REPORT.read_text())["stats"]


@pytest.fixture(scope="module")
def tex() -> str:
    if not _TEX.exists():
        pytest.skip("workshop paper not in this checkout")
    return re.sub(r"\s+", " ", _TEX.read_text())


def test_paired_two_by_two(stats):
    """Sect. 3.2 and Appendix Table 4 --- the cells, not just the margins."""
    p = stats["pairedCueDense"]
    assert (p["bothRedirect"], p["cueOnly"], p["denseOnly"], p["neither"]) == (22, 6, 8, 88)
    assert (p["cueRedirects"], p["denseRedirects"]) == (28, 30)
    assert p["directions"] == 124 and p["denseUndefinedDirections"] == 4
    assert p["discordant"] == 14
    assert p["differencePp"] == -1.61


def test_paired_cluster_structure(stats):
    """Eight clusters, four per scenario --- the enumeration sizes follow from this."""
    p = stats["pairedCueDense"]
    assert p["clusters"] == 8
    assert sorted(p["clustersPerScenario"].values()) == [4, 4]


def test_paired_exact_brackets(stats):
    """Appendix Table 4 --- enumerated, so the endpoints carry no seed dependence."""
    p = stats["pairedCueDense"]
    assert (p["stratified"]["vectors"], p["stratified"]["orderedDraws"]) == (1225, 65_536)
    assert (p["stratified"]["lo"], p["stratified"]["hi"]) == (-7.03, 7.63)
    assert (p["pooled"]["vectors"], p["pooled"]["orderedDraws"]) == (6435, 16_777_216)
    assert (p["pooled"]["lo"], p["pooled"]["hi"]) == (-7.55, 7.14)


def test_heldout_nesting(stats):
    """Sect. 3.3 --- the cue redirects are a subset, across five of six axes."""
    h = stats["heldoutOverlap"]["heldout"]
    assert (h["directions"], h["referenceRedirects"], h["cueRedirects"]) == (30, 25, 7)
    assert h["cueRedirectsInsideReference"] == h["cueRedirects"]
    assert (h["axes"], h["axesWithCueRedirect"]) == (6, 5)
    assert h["flipRatio"] == 0.28


def test_tool_selection_split(stats):
    """Sect. 3.3 --- the separate split, nested too, but from only two pairs."""
    t = stats["heldoutOverlap"]["toolSelection"]
    assert (t["directions"], t["referenceRedirects"], t["cueRedirects"]) == (14, 13, 2)
    assert t["cueRedirectsInsideReference"] == t["cueRedirects"]
    assert (t["pairs"], t["pairsWithCueRedirect"]) == (7, 2)


def test_ratio_brackets(stats):
    """Appendix --- all three sensitivity ranges, and no empty denominator."""
    h = stats["heldoutOverlap"]["heldout"]
    t = stats["heldoutOverlap"]["toolSelection"]
    assert (h["bracketPooled"]["lo"], h["bracketPooled"]["hi"]) == (0.1481, 0.4375)
    assert (h["bracketStratified"]["lo"], h["bracketStratified"]["hi"]) == (0.16, 0.4375)
    assert (t["bracketPooled"]["lo"], t["bracketPooled"]["hi"]) == (0.0, 0.3846)
    for bracket in (h["bracketPooled"], h["bracketStratified"], t["bracketPooled"]):
        assert bracket["zeroDenominatorVectors"] == 0, "a zero denominator would void the ratio"


def test_heldout_comparator_arms(stats):
    """Sect. 3.3 --- the dense and random arms the flip ratio is read against."""
    r = stats["recoveryHeldout"]
    assert (r["differenceInMeansMatchedFlips"], r["differenceInMeansMatchedDirections"]) == (7, 28)
    assert (r["randomMatchedFlips"], r["randomMatchedDraws"]) == (0, 84)
    assert (r["cueFlips"], r["ceilingFlips"]) == (7, 25)


def test_nesting_is_not_asserted_by_construction(stats):
    """The audit would be vacuous if a cue redirect implied a reference one.

    Nothing in the extractor forces the subset relation --- the two arms read
    different records --- so the equality in the tests above is a finding.  It
    stops being one if the reference arm ever starts flipping everywhere, which
    this bounds.
    """
    for split in stats["heldoutOverlap"].values():
        assert split["referenceRedirects"] < split["directions"], (
            "the reference arm must leave some direction unflipped, or nesting is trivial"
        )


def test_workshop_tex_quotes_the_report(stats, tex):
    """Every figure above, as it appears in the manuscript."""
    p = stats["pairedCueDense"]
    h = stats["heldoutOverlap"]["heldout"]
    t = stats["heldoutOverlap"]["toolSelection"]

    assert f"$[{p['stratified']['lo']},+{p['stratified']['hi']}]$" in tex
    assert f"$[{p['pooled']['lo']},+{p['pooled']['hi']}]$" in tex
    assert f"{p['pooled']['vectors']:,} vectors" in tex
    assert f"$35^2=1{{,}}{p['stratified']['vectors'] % 1000}$" in tex

    row = f"{p['bothRedirect']} & {p['cueOnly']} & {p['cueRedirects']}"
    assert row in re.sub(r" +", " ", tex)
    assert f"{p['denseOnly']} & {p['neither']} & 96" in re.sub(r" +", " ", tex)

    assert (
        f"cue redirects fall among the {h['referenceRedirects']} "
        f"full-state-reference redirects of {h['directions']}" in tex
    )
    assert (
        f"span {_WORDS[h['axesWithCueRedirect']]} of the {_WORDS[h['axes']]} "
        "represented axes" in tex
    )
    # the same split stated twice: as a ratio in the appendix, in words in Sect. 3.3
    assert f"gives {t['cueRedirects']}/{t['referenceRedirects']}" in tex
    # "both cue redirects", because there are two of them --- so pin the denominator
    assert f"cue redirects fall among {_WORDS[t['referenceRedirects']]} reference" in tex
    assert f"of its {_WORDS[t['pairs']]} pairs" in tex
    assert f"from {_WORDS[t['pairsWithCueRedirect']]} of {_WORDS[t['pairs']]} pairs" in tex

    for bracket in (h["bracketPooled"], h["bracketStratified"], t["bracketPooled"]):
        assert f"{bracket['lo']:.3f}--{bracket['hi']:.3f}" in tex


@pytest.fixture(scope="module")
def extractor():
    if not _EXTRACTOR.exists():
        pytest.skip("extractor not present")
    spec = importlib.util.spec_from_file_location("_extract_results", _EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_artifacts_are_complete(extractor):
    """Every battery the extractor reads for the workshop paper is published."""
    if not _ARTIFACTS.exists():
        pytest.skip("artifacts not in this checkout")
    expected = set()
    for scenario, layer in {**extractor.EXPANDED, **extractor.HELDOUT}.items():
        expected.add(_ARTIFACTS / scenario / f"steering_layer{layer}" / "steering_results.json")
        expected.add(_ARTIFACTS / scenario / f"ceiling_layer{layer}" / "ceiling_results.json")
    layer = extractor.PRIMARY["tool_selection"]
    expected.add(_ARTIFACTS / "tool_selection" / f"steering_layer{layer}" / "steering_results.json")
    expected.add(_ARTIFACTS / "tool_selection" / f"ceiling_layer{layer}" / "ceiling_results.json")
    missing = sorted(str(p.relative_to(_ROOT)) for p in expected if not p.exists())
    assert not missing, f"published artifact set is short: {missing}"
    assert len(list(_ARTIFACTS.rglob("*.json"))) == len(expected), "unreferenced artifact present"


def test_artifacts_alone_regenerate_the_statistics(stats, extractor, monkeypatch):
    """The published subset must be *sufficient*, not merely present.

    This is the claim the artifacts exist to support: someone with the
    repository and none of the 3.4 GB of ignored run output can recompute the
    paper's headline comparison and get the same numbers.  Reading them back
    through the extractor is the only check that actually establishes it.
    """
    if not _ARTIFACTS.exists():
        pytest.skip("artifacts not in this checkout")
    monkeypatch.setenv("KIJI_ARTIFACTS", "1")
    assert extractor.paired_cue_dense(extractor.EXPANDED) == stats["pairedCueDense"]
    assert (
        extractor.heldout_overlap(
            {
                "heldout": extractor.HELDOUT,
                "toolSelection": {"tool_selection": extractor.PRIMARY["tool_selection"]},
            }
        )
        == stats["heldoutOverlap"]
    )


def test_artifacts_match_the_runs_they_were_copied_from():
    """A stale copy would publish numbers the report was not built from."""
    if not _ARTIFACTS.exists():
        pytest.skip("artifacts not in this checkout")
    checked = 0
    for published in sorted(_ARTIFACTS.rglob("*.json")):
        scenario, battery, name = published.relative_to(_ARTIFACTS).parts
        source = _ROOT / "demo" / "steering" / scenario / "output" / battery / name
        if not source.exists():  # the ignored tree is absent on a fresh clone
            continue
        checked += 1
        assert hashlib.sha256(published.read_bytes()).digest() == hashlib.sha256(
            source.read_bytes()
        ).digest(), f"{published.relative_to(_ROOT)} has drifted from its run"
    if not checked:
        pytest.skip("no run output present to compare against")
