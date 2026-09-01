#!/usr/bin/env python3
"""Collect every number the steering paper cites into one JSON.

Reads the run artefacts under ``demo/steering/<scenario>/output/`` and writes
``paper/steering/results/steering_report.json``.  Nothing in the paper should be
a hand-copied figure: if a number appears in the .tex it should appear here.

Run from the repository root::

    python paper/steering/extract_results.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import statistics as st
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEER = ROOT / "demo" / "steering"
OUT = Path(__file__).resolve().parent / "results" / "steering_report.json"
GATE_POP = Path(__file__).resolve().parent / "results" / "gate_population.json"
# The ten batteries the workshop paper reads, copied out of the ignored
# ``output/`` trees so its numbers can be regenerated from the repository alone.
# ``KIJI_ARTIFACTS=1`` reads those in place of the full runs, which is what
# proves the published subset is sufficient rather than merely present.
ARTIFACTS = Path(__file__).resolve().parents[1] / "steering_workshop" / "artifacts"

LAYERS = [6, 13, 20, 27, 34, 43]
# scenario -> layer the demo is read at (None = layer study only, no page)
SCENARIOS = {"tool_selection": None, "supply_chain": 43, "customer_support": 34}
# scenario -> the layer its causal claims are made at (tool_selection has no
# page but its battery is read at 43 like supply_chain)
PRIMARY = {"tool_selection": 43, "supply_chain": 43, "customer_support": 34}
# rate-estimation sets: 32 pairs sampled from the same gate the demo pairs were
# selected from (rank_flips.py --sample 32 --theme-cap 10), run at the primary
# layer and one early layer only
EXPANDED = {"supply_chain_expanded": 43, "customer_support_expanded": 34}
# held-out probe: pairs written on contrast axes absent from the corpus the
# SAEs were trained on, run through the same gate and battery at the primary
# layer. Kept out of the depth aggregate, which describes the in-distribution
# grid only.
HELDOUT = {"supply_chain_heldout": 43, "customer_support_heldout": 34}
# the same expanded pairs run at a layer that was NOT selected, so the primary
# layer's rates can be read against a non-selected comparison point.  Layer 27
# is the deepest layer below the two primaries and the shallowest at which any
# flip occurs at all.
NON_SELECTED = {"supply_chain_l27": 27, "customer_support_l27": 27}
EARLY = [6, 13, 20]


# Re-runs that add a control family are written to ``*_setctl`` directories so
# the canonical artefacts are never overwritten mid-batch.  Setting
# ``KIJI_SUFFIX=_setctl`` reads those in preference, falling back to the
# canonical directory for any battery that has not been re-run yet.
SUFFIX = os.environ.get("KIJI_SUFFIX", "")


def _scenario_dir(scenario: str) -> Path:
    """Where this scenario's battery directories live.

    Read at call time, not import time, so a test can point the extractor at
    the published artifacts without reimporting the module.
    """
    if os.environ.get("KIJI_ARTIFACTS"):
        return ARTIFACTS / scenario
    return STEER / scenario / "output"


def _ceiling_path(scenario: str, layer: int) -> Path:
    """The full-residual-patch run for one scenario and layer."""
    return _scenario_dir(scenario) / f"ceiling_layer{layer}" / "ceiling_results.json"


def _battery_dir(scenario: str, name: str) -> Path:
    base = _scenario_dir(scenario)
    alt = base / f"{name}{SUFFIX}"
    # A re-run directory that exists but holds no results yet is a battery
    # still on the GPU; fall back rather than silently drop it from the report.
    if SUFFIX and alt.is_dir() and any(alt.glob("*.json")):
        return alt
    return base / name


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _wilson(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson 95% score interval — the standard interval for small-n counts."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "rate": round(p, 4),
            "lo": round(max(0.0, centre - half), 4),
            "hi": round(min(1.0, centre + half), 4)}


def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _exact_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact (Clopper--Pearson) upper bound on a binomial rate.

    The smallest p with P(Bin(n, p) <= k) equal to 1 - conf, found by
    bisection; for k = 0 this reduces to 1 - (1-conf)^(1/n) (the rule of
    three, exactly).
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(0, k + 1))
        if tail > 1 - conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(first row's success count this extreme or more), hypergeometric.

    Rows are (successes, failures); tests whether the first row's rate exceeds
    the second's.
    """
    n = a + b + c + d
    return sum(
        math.comb(a + b, a + c - k) * math.comb(c + d, k) for k in range(0, c + 1)
    ) / math.comb(n, a + c)


def _percentile(values: list[float], q: float) -> float:
    """Linearly interpolated sample percentile."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _multinomial(counts: list[int]) -> int:
    """How many ordered draws collapse onto one cluster-count vector."""
    total = math.factorial(sum(counts))
    for c in counts:
        total //= math.factorial(c)
    return total


def _weighted_percentile(atoms: list[tuple[int, float]], q: float) -> float:
    """Percentile of a discrete distribution given as ``(weight, value)``."""
    ordered = sorted(atoms, key=lambda t: t[1])
    total = sum(w for w, _ in ordered)
    seen = 0
    for weight, value in ordered:
        seen += weight
        if seen >= q * total:
            return value
    return ordered[-1][1]


def _stratum_atoms(clusters: list[tuple[int, ...]]) -> list[tuple[int, tuple[int, ...]]]:
    """Every way to draw ``len(clusters)`` clusters from ``clusters``, with weights.

    Order does not matter to a sum, so the distinct outcomes are the
    count vectors and each carries the number of ordered draws that produce
    it.  Four clusters give ``C(7, 4) = 35`` vectors over ``4**4 = 256``
    ordered draws.
    """
    m = len(clusters)
    width = len(clusters[0])
    atoms = []
    for combo in itertools.combinations_with_replacement(range(m), m):
        counts = [combo.count(i) for i in range(m)]
        totals = tuple(
            sum(counts[i] * clusters[i][j] for i in range(m)) for j in range(width)
        )
        atoms.append((_multinomial(counts), totals))
    return atoms


def _enumerate_resamples(
    strata: list[list[tuple[int, ...]]]
) -> list[tuple[int, tuple[int, ...]]]:
    """Exact whole-cluster bootstrap distribution over one or more strata.

    Each stratum is resampled to its own size with replacement, so a single
    stratum is the pooled bootstrap and one stratum per scenario is the
    stratified one.  Enumerating the count vectors and weighting each by its
    multiplicity *is* the ordinary bootstrap's sampling distribution, so the
    percentile endpoints carry no Monte-Carlo error --- worth having where the
    cluster count is small enough that a sampled endpoint wobbles between
    adjacent atoms.
    """
    atoms: list[tuple[int, tuple[int, ...] | None]] = [(1, None)]
    for clusters in strata:
        atoms = [
            (w_a * w_b, b if a is None else tuple(x + y for x, y in zip(a, b)))
            for w_a, a in atoms
            for w_b, b in _stratum_atoms(clusters)
        ]
    return atoms


def _ratio_bracket(strata: list[list[tuple[int, int]]], num: int = 0, den: int = 1) -> dict:
    """2.5--97.5 percentile endpoints of a ratio under the exact resampling."""
    atoms = _enumerate_resamples(strata)
    ratios = [(w, v[num] / v[den]) for w, v in atoms if v[den]]
    return {
        "vectors": len(atoms),
        "orderedDraws": sum(w for w, _ in atoms),
        "zeroDenominatorVectors": len(atoms) - len(ratios),
        "lo": round(_weighted_percentile(ratios, 0.025), 4),
        "hi": round(_weighted_percentile(ratios, 0.975), 4),
    }


def _argmax(dist: dict | None) -> str | None:
    return max(dist, key=dist.get) if dist else None


def _flip(entry: dict | None, target: str) -> bool:
    """A flip is an argmax change that lands on the *donor's* tool.

    ``argmaxChanged`` alone is not enough — an intervention that knocks the
    recipient onto some third tool is a disruption, not a steer.
    """
    if not entry:
        return False
    return bool(entry.get("argmaxChanged")) and entry.get("choice") == target


def _bands(v: dict) -> tuple[float | None, float | None]:
    """This side's two control bands: (per-family, set-matched).

    ``controlThreshold`` is the max over draws matched to *one cue family each*
    -- the right yardstick for a single family, and the wrong one for the
    whole cue set, which typically carries several times more activation mass.
    ``setControlThreshold`` is the max over draws matched to the union's count
    and mass.  It is absent from batteries run before the set-matched arm was
    added, so every consumer here has to handle ``None``.
    """
    per_family = v.get("controlThreshold")
    return (
        float(per_family) if per_family else None,
        float(v["setControlThreshold"]) if v.get("setControlThreshold") else None,
    )


def _delta_band(v: dict) -> float | None:
    """The band matched to what a cross-patch clamp actually moves.

    ``controlThreshold`` and ``setControlThreshold`` both match the *donor's*
    activation.  For an ablation that is the same thing -- the target is zero,
    so the realised change is the activation itself -- but a cross-patch clamps
    a donor value onto a recipient that may already carry some of it, and cue
    features are chosen for differing across the pair, so the cue set usually
    moves most of its donor mass while a donor-matched random draw need not.
    ``deltaControlThreshold`` is matched on sum |donor - recipient| instead.
    Absent from batteries run before that arm existed, hence ``None``.
    """
    band = v.get("deltaControlThreshold")
    return float(band) if band else None


def _delta_over_donor(v: dict) -> float | None:
    """Fraction of the cue set's donor mass the clamp actually moves."""
    donor, moved = v.get("setMass"), v.get("setDeltaMass")
    if not donor or moved is None:
        return None
    return float(moved) / float(donor)


def _mass_ratio(v: dict, mass_key: str) -> float | None:
    """How much heavier the cue set is than the heaviest row-matched draw."""
    controls = [c for c in v.get("controls") or [] if c.get(mass_key) is not None]
    if not controls:
        return None
    heaviest = max(float(c[mass_key]) for c in controls)
    cue_mass = v.get("setMass")
    if cue_mass is None:
        cue_mass = sum(float(r.get(mass_key) or 0.0) for r in v.get("rows") or [])
    return float(cue_mass) / heaviest if heaviest else None


def layer_battery(
    scenario: str, layer: int, exclude: frozenset[str] | set[str] = frozenset()
) -> dict | None:
    """Every count and effect size for one battery.

    ``exclude`` drops pairs by id before counting; it exists so the expanded
    rates can be recomputed without the pairs that also appear in the
    demonstration set the primary layer was chosen on.  Empty by default, so
    every existing caller gets exactly the numbers it got before.
    """
    res = _load(_battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json")
    if not res:
        return None

    ab_sides = ab_flips = ab_any = 0
    ab_best = 0.0
    ab_controls: list[float] = []
    ratios: list[float] = []
    set_ratios: list[float] = []
    delta_ratios: list[float] = []
    contrast_ratios: list[float] = []
    contrast_sides = contrast_ceilings = 0
    mass_ratios: list[float] = []
    delta_over_donor: list[float] = []
    for pair_id, sides in res["attribution"].items():
        if pair_id in exclude:
            continue
        for side in ("a", "b"):
            v = sides.get(side)
            if not v:
                continue
            ab_sides += 1
            rows = v.get("allRows") or {}
            # ablation "flips" to the side's runner-up, i.e. the other side's
            # tool.  Attribution rows carry no ``choice``, so read the argmax
            # of the post-intervention distribution.
            if rows.get("argmaxChanged"):
                ab_any += 1
                if _argmax(rows.get("intervened")) == v.get("otherTool"):
                    ab_flips += 1
            ab_best = max(ab_best, abs(rows.get("deltaTarget") or 0.0))
            if v.get("controlThreshold") is not None:
                ab_controls.append(v["controlThreshold"])
            # effect measured against this side's own control band, so it is
            # comparable across layers and across pair sets of different scale
            per_family, set_band = _bands(v)
            effect = abs(rows.get("deltaTarget") or 0.0)
            if per_family:
                ratios.append(effect / per_family)
            if set_band:
                set_ratios.append(effect / set_band)
            # ablation's answer to "cue-ness or mass?": a band drawn to how much
            # the set differs across the pair, not to how much of it is there
            contrast_band = v.get("contrastControlThreshold")
            if contrast_band:
                contrast_ratios.append(effect / float(contrast_band))
            if v.get("contrastControls") is not None:
                contrast_sides += 1
                contrast_ceilings += not v.get("contrastControlMassMatched")
            ratio = _mass_ratio(v, "hfMass")
            if ratio:
                mass_ratios.append(ratio)

    cp_dirs = cp_flips = cp_cue = cp_bulk = cp_any = 0
    cp_best = 0.0
    base_sizes: list[int] = []
    for pair_id, dirs in res["crossPatch"].items():
        if pair_id in exclude:
            continue
        for v in dirs.values():
            if not v:
                continue
            cp_dirs += 1
            target = v.get("targetTool")
            cue = _flip(v.get("allRows"), target)
            bulk = _flip(v.get("allBase"), target)
            cp_cue += cue
            cp_bulk += bulk and not cue
            cp_flips += cue or bulk
            cp_any += bool((v.get("allRows") or {}).get("argmaxChanged")) or bool(
                (v.get("allBase") or {}).get("argmaxChanged")
            )
            for key in ("allRows", "allBase"):
                cp_best = max(cp_best, abs((v.get(key) or {}).get("deltaTarget") or 0.0))
            per_family, set_band = _bands(v)
            effect = abs((v.get("allRows") or {}).get("deltaTarget") or 0.0)
            if per_family:
                ratios.append(effect / per_family)
            if set_band:
                set_ratios.append(effect / set_band)
            delta_band = _delta_band(v)
            if delta_band:
                delta_ratios.append(effect / delta_band)
            moved = _delta_over_donor(v)
            if moved is not None:
                delta_over_donor.append(moved)
            ratio = _mass_ratio(v, "baseMass")
            if ratio:
                mass_ratios.append(ratio)
            if (v.get("allBase") or {}).get("size"):
                base_sizes.append(v["allBase"]["size"])

    return {
        "layer": layer,
        "ablationFlips": ab_flips,
        "ablationFlipsAnyTool": ab_any,
        "ablationSides": ab_sides,
        "ablationBestDelta": round(ab_best, 4),
        "ablationControlMedian": round(st.median(ab_controls), 4) if ab_controls else None,
        "crossPatchFlips": cp_flips,
        "crossPatchFlipsAnyTool": cp_any,
        "crossPatchDirections": cp_dirs,
        "crossPatchCueDriven": cp_cue,
        "crossPatchBulkOnly": cp_bulk,
        "crossPatchBestDelta": round(cp_best, 4),
        "medianBaseActive": int(st.median(base_sizes)) if base_sizes else None,
        # cue-set |dp| divided by that side's own control band, pooled over
        # ablation and cross-patch: an effect size that does not depend on
        # where the argmax boundary happens to sit.  Two versions, because the
        # two bands answer different questions -- the ...SetControl pair is the
        # matched one (draws carrying the whole cue set's mass), the plain pair
        # divides by the per-family band and is reported only so the older
        # batteries remain comparable.
        "medianEffectOverControl": round(st.median(ratios), 3) if ratios else None,
        "fractionExceedingControl": round(sum(r > 1 for r in ratios) / len(ratios), 3)
        if ratios
        else None,
        "medianEffectOverSetControl": round(st.median(set_ratios), 3) if set_ratios else None,
        "fractionExceedingSetControl": round(sum(r > 1 for r in set_ratios) / len(set_ratios), 3)
        if set_ratios
        else None,
        "setControlSides": len(set_ratios),
        # ablation only: the band matched on how much the set differs across the
        # pair.  ``contrastCeilingSides`` counts sides where nothing else differs
        # that much, which is the strong form of the answer rather than a gap.
        "medianEffectOverContrastControl": round(st.median(contrast_ratios), 3)
        if contrast_ratios
        else None,
        "fractionExceedingContrastControl": round(
            sum(r > 1 for r in contrast_ratios) / len(contrast_ratios), 3
        )
        if contrast_ratios
        else None,
        "contrastControlSides": contrast_sides or None,
        "contrastCeilingSides": contrast_ceilings if contrast_sides else None,
        # cross-patch only: the band matched to what the clamp actually moves,
        # rather than to the donor activation behind it
        "medianEffectOverDeltaControl": round(st.median(delta_ratios), 3) if delta_ratios else None,
        "fractionExceedingDeltaControl": round(
            sum(r > 1 for r in delta_ratios) / len(delta_ratios), 3
        )
        if delta_ratios
        else None,
        "deltaControlDirections": len(delta_ratios),
        # how much of the cue set's donor mass the clamp really moves: 1.0 means
        # the recipient carried none of it, and the two matchings coincide
        "medianCueDeltaOverDonorMass": round(st.median(delta_over_donor), 3)
        if delta_over_donor
        else None,
        # how far the per-family band was from matching the cue set it was
        # being compared with: the reason the set-matched arm exists
        "medianCueMassOverHeaviestRowDraw": round(st.median(mass_ratios), 2)
        if mass_ratios
        else None,
        "sidesWhereRowDrawReachesCueMass": sum(r <= 1 for r in mass_ratios) if mass_ratios else None,
        "massAuditSides": len(mass_ratios),
    }


def clustered_flip_intervals(scenarios: dict[str, int]) -> dict:
    """Percentile brackets from an exact whole-cluster resampling.

    A cluster is the canonical pair title in pairs.json. Resampling whole
    clusters preserves dependence between the two directions of a pair and
    among near-duplicate pairs of the same contrast type. Scenario names are
    included in cluster IDs so similarly named contrasts stay distinct.

    Resampling is stratified by scenario, and enumerated rather than sampled.
    Stratifying matters because the design fixed 32 pairs per scenario: pooling
    the ten clusters lets a resample draw eight supply-chain clusters and two
    customer-support ones, varying a scenario mix the study held constant, and
    that shows up as extra spread the design does not have. Enumerating removes
    the seed dependence --- 35 count vectors for one scenario's four clusters
    and 462 for the other's six, 16,170 together, each weighted by the ordered
    draws that produce it.

    ``pooled`` keeps the unstratified version as a check that the conclusion
    does not rest on how the strata were drawn.
    """
    clusters: dict[str, dict[str, int]] = {}
    strata: dict[str, list[str]] = {}
    for scenario, layer in scenarios.items():
        pairs = _load(STEER / scenario / "pairs.json") or {}
        res = _load(
            _battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json"
        )
        if not res:
            continue
        titles = {p["id"]: p["title"] for p in pairs.get("pairs", [])}
        for pair_id, title in titles.items():
            key = f"{scenario}:{title}"
            cluster = clusters.setdefault(
                key, {"ab_k": 0, "ab_n": 0, "cp_k": 0, "cp_n": 0}
            )
            if key not in strata.setdefault(scenario, []):
                strata[scenario].append(key)
            for v in (res.get("attribution", {}).get(pair_id) or {}).values():
                if not v:
                    continue
                cluster["ab_n"] += 1
                rows = v.get("allRows") or {}
                if rows.get("argmaxChanged") and _argmax(rows.get("intervened")) == v.get(
                    "otherTool"
                ):
                    cluster["ab_k"] += 1
            for v in (res.get("crossPatch", {}).get(pair_id) or {}).values():
                if not v:
                    continue
                cluster["cp_n"] += 1
                target = v.get("targetTool")
                cluster["cp_k"] += int(
                    _flip(v.get("allRows"), target) or _flip(v.get("allBase"), target)
                )

    order = ("ab_k", "ab_n", "cp_k", "cp_n")
    arms = {"ablation": (0, 1), "crossPatch": (2, 3)}

    def brackets(groups: list[list[str]]) -> dict:
        atoms = _enumerate_resamples(
            [[tuple(clusters[k][f] for f in order) for k in g] for g in groups]
        )
        out = {"vectors": len(atoms), "orderedDraws": sum(w for w, _ in atoms)}
        for arm, (num, den) in arms.items():
            vals = [(w, v[num] / v[den]) for w, v in atoms if v[den]]
            out[arm] = {
                "lo": round(_weighted_percentile(vals, 0.025), 4),
                "hi": round(_weighted_percentile(vals, 0.975), 4),
            }
        return out

    stratified = brackets([g for _, g in sorted(strata.items())])
    pooled = brackets([sorted(clusters)])
    totals = {
        arm: (
            sum(c[order[num]] for c in clusters.values()),
            sum(c[order[den]] for c in clusters.values()),
        )
        for arm, (num, den) in arms.items()
    }
    return {
        "unit": "scenario and contrast title",
        "method": "exact multinomially-weighted enumeration, scenario-stratified",
        "clusters": len(clusters),
        "clustersPerScenario": {s: len(g) for s, g in sorted(strata.items())},
        "vectors": stratified["vectors"],
        "orderedDraws": stratified["orderedDraws"],
        **{
            arm: {
                "k": totals[arm][0],
                "n": totals[arm][1],
                "rate": round(totals[arm][0] / totals[arm][1], 4),
                **stratified[arm],
            }
            for arm in arms
        },
        "pooled": pooled,
    }


def _theme_of(pair_id: str, themes) -> str | None:
    """Recover a sampled pair's contrast type from its id.

    ``rank_flips.to_pair_records`` writes the bare theme when a theme
    contributes one pair and ``<theme>_<n>`` when it contributes several, so
    the id carries the stratum and nothing else has to be stored.
    """
    if pair_id in themes:
        return pair_id
    stem = re.match(r"^(.*)_\d+$", pair_id)
    return stem.group(1) if stem and stem.group(1) in themes else None


def design_weighted_rates(scenarios: dict[str, int]) -> dict | None:
    """Flip rates under the three weightings the expanded sample admits.

    The sample was drawn uniformly from the gate-passing pairs but capped at ten
    pairs per contrast type, and the populations are extremely skewed -- 92% of
    the supply-chain pairs and 97% of the customer-support pairs sit in one type.
    Capped-uniform is not uniform, so the three quantities below differ, and
    which one a sentence means has to be said rather than assumed:

    ``quota``
        every sampled side weighted equally.  This is what every count in the
        report already is: a rate in the sample as drawn, not in the population.
    ``typeAveraged``
        the unweighted mean of the per-type rates.  The cap pushes ``quota``
        toward this but does not reach it, because the realised quota sizes are
        not equal (a type with three pairs in the population contributes three).
    ``populationWeighted``
        the stratified (Horvitz-Thompson) estimator, sum_h (N_h / N) * (k_h /
        n_h), which weights each type by its share of the gate population.  This
        is the only one of the three that estimates a rate over that population.

    Returns ``None`` when the cached population sizes are absent; run
    ``gate_population.py`` to build them.
    """
    cached = _load(GATE_POP)
    if not cached:
        return None
    out: dict = {"populationSource": str(GATE_POP.name), "scenarios": {}}
    for scenario, layer in scenarios.items():
        sweep_scenario = scenario.replace("_expanded", "")
        pop = (cached["scenarios"].get(sweep_scenario) or {}).get("byTheme")
        res = _load(_battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json")
        pairs = _load(STEER / scenario / "pairs.json")
        if not (pop and res and pairs):
            continue
        total = sum(pop.values())
        themes = {p["id"]: _theme_of(p["id"], pop) for p in pairs.get("pairs", [])}
        if not all(themes.values()):
            continue  # an id that does not resolve would silently drop a stratum

        strata: dict[str, dict[str, int]] = {}
        for pair_id, sides in (res.get("attribution") or {}).items():
            cell = strata.setdefault(themes[pair_id], {"ab_k": 0, "ab_n": 0, "cp_k": 0, "cp_n": 0})
            for v in sides.values():
                if not v:
                    continue
                cell["ab_n"] += 1
                rows = v.get("allRows") or {}
                if rows.get("argmaxChanged") and _argmax(rows.get("intervened")) == v.get("otherTool"):
                    cell["ab_k"] += 1
        for pair_id, dirs in (res.get("crossPatch") or {}).items():
            cell = strata.setdefault(themes[pair_id], {"ab_k": 0, "ab_n": 0, "cp_k": 0, "cp_n": 0})
            for v in dirs.values():
                if not v:
                    continue
                cell["cp_n"] += 1
                target = v.get("targetTool")
                cell["cp_k"] += int(_flip(v.get("allRows"), target) or _flip(v.get("allBase"), target))

        def _arm(k_key: str, n_key: str, strata=strata, pop=pop, total=total) -> dict:
            present = {t: c for t, c in strata.items() if c[n_key]}
            k = sum(c[k_key] for c in present.values())
            n = sum(c[n_key] for c in present.values())
            per_type = [c[k_key] / c[n_key] for c in present.values()]
            # renormalise over the sampled strata: a type the draw missed
            # entirely carries no information about its own rate
            covered = sum(pop[t] for t in present) / total
            weighted = sum((pop[t] / total) * (c[k_key] / c[n_key]) for t, c in present.items())
            # Precision, not just the point estimate: the stratified variance
            # sum_h W_h^2 p_h(1-p_h)/n_h is dominated by the one type that holds
            # ~92-97% of the weight, so the population estimate rests on that
            # type's ten pairs rather than on all thirty-two.  Treats the draw
            # within a type as simple random sampling and ignores the finite
            # population correction, which only matters for the tiny strata that
            # carry almost no weight.
            estimate = weighted / covered if covered else None
            variance = sum(
                ((pop[t] / total / covered) ** 2)
                * (c[k_key] / c[n_key])
                * (1 - c[k_key] / c[n_key])
                / c[n_key]
                for t, c in present.items()
            )
            effective = (
                estimate * (1 - estimate) / variance
                if variance and estimate not in (None, 0.0, 1.0)
                else None
            )
            return {
                "k": k,
                "n": n,
                "quota": round(k / n, 4) if n else None,
                "typeAveraged": round(sum(per_type) / len(per_type), 4) if per_type else None,
                "populationWeighted": round(estimate, 4) if estimate is not None else None,
                "populationWeightedSE": round(variance**0.5, 4),
                "populationEffectiveN": round(effective, 1) if effective else None,
                "populationCovered": round(covered, 4),
            }

        out["scenarios"][scenario] = {
            "layer": layer,
            "population": total,
            "strata": {
                t: {
                    "population": pop[t],
                    "populationShare": round(pop[t] / total, 4),
                    "ablation": [c["ab_k"], c["ab_n"]],
                    "crossPatch": [c["cp_k"], c["cp_n"]],
                }
                for t, c in sorted(strata.items(), key=lambda kv: -pop[kv[0]])
            },
            "ablation": _arm("ab_k", "ab_n"),
            "crossPatch": _arm("cp_k", "cp_n"),
        }

    if not out["scenarios"]:
        return None
    # Pooled: the two scenarios are separate populations, so the population
    # estimate is their mean, not a rate over their union.  The quota figure
    # pools sides directly, matching the pooled counts elsewhere in the report.
    for arm in ("ablation", "crossPatch"):
        entries = [e[arm] for e in out["scenarios"].values()]
        k = sum(e["k"] for e in entries)
        n = sum(e["n"] for e in entries)
        weighted = [e["populationWeighted"] for e in entries if e["populationWeighted"] is not None]
        out.setdefault("pooled", {})[arm] = {
            "k": k,
            "n": n,
            "quota": round(k / n, 4) if n else None,
            "populationWeightedMean": round(sum(weighted) / len(weighted), 4) if weighted else None,
        }
    return out


def layer_selection_audit() -> dict:
    """How much of the reported rate is the layer having been chosen.

    The primary layer of each scenario is the argmax of the very depth grid the
    paper reports, picked after seeing it, and the expanded 32-pair samples were
    then run only there.  Table~2's rates are therefore rates at the most
    responsive of six evaluated layers.  This function makes the size of that
    conditioning reproducible rather than a matter of assertion:

    * ``argmax`` records, per scenario, which layer maximises each intervention
      type and by what margin.  A one-flip margin is a much weaker claim to
      "best layer" than a six-flip one, so the margin travels with the claim.
    * ``nonSelected`` runs the identical 32 pairs through the identical battery
      at layer 27, which was not selected.
    * ``demoOverlap`` counts the demonstration pairs -- the ones the layer was
      chosen on -- that reappear inside the expanded sample, and recomputes the
      expanded rates without them.
    * ``depthAtNonSelected`` rebuilds the early-vs-late contrast with layer 27
      as the late arm, to separate the depth *conclusion* from the depth
      *effect size* (only the latter is inflated by the selection).
    """
    argmax = {}
    for scenario in PRIMARY:
        grid = {}
        for layer in LAYERS:
            b = layer_battery(scenario, layer)
            if b:
                grid[layer] = {
                    "ablationFlips": b["ablationFlips"],
                    "ablationSides": b["ablationSides"],
                    "crossPatchFlips": b["crossPatchFlips"],
                    "crossPatchDirections": b["crossPatchDirections"],
                    "combinedFlips": b["ablationFlips"] + b["crossPatchFlips"],
                }
        if not grid:
            continue

        def _best(key: str, grid=grid) -> dict:
            ranked = sorted(grid.items(), key=lambda kv: -kv[1][key])
            top, runner = ranked[0], ranked[1] if len(ranked) > 1 else None
            return {
                "layer": top[0],
                "flips": top[1][key],
                "runnerUpLayer": runner[0] if runner else None,
                "runnerUpFlips": runner[1][key] if runner else None,
                # a margin of 0 or 1 flip does not really single out a layer
                "margin": top[1][key] - runner[1][key] if runner else None,
            }

        argmax[scenario] = {
            "primary": PRIMARY[scenario],
            "grid": grid,
            "ablation": _best("ablationFlips"),
            "crossPatch": _best("crossPatchFlips"),
            "combined": _best("combinedFlips"),
            "primaryIsAblationArgmax": _best("ablationFlips")["layer"] == PRIMARY[scenario],
            "primaryIsCrossPatchArgmax": _best("crossPatchFlips")["layer"] == PRIMARY[scenario],
        }

    available = {k: v for k, v in NON_SELECTED.items() if (STEER / k).exists()}
    non_selected = None
    if available:
        non_selected = {
            "layer": sorted(set(available.values())),
            "clusterBootstrap": clustered_flip_intervals(available),
            "perScenario": {
                name: {
                    "ablation": _wilson(b["ablationFlips"], b["ablationSides"]),
                    "crossPatch": _wilson(b["crossPatchFlips"], b["crossPatchDirections"]),
                }
                for name, layer in available.items()
                if (b := layer_battery(name, layer))
            },
        }

    # Which demonstration pairs -- the ones the layer was chosen on -- are
    # inside the expanded sample?  Ids differ between the two selections, so
    # match on the request text, which is what the model actually sees.
    demo_overlap = {}
    for demo_name, exp_name in (
        ("supply_chain", "supply_chain_expanded"),
        ("customer_support", "customer_support_expanded"),
    ):
        demo_pairs = (_load(STEER / demo_name / "pairs.json") or {}).get("pairs") or []
        exp_pairs = (_load(STEER / exp_name / "pairs.json") or {}).get("pairs") or []
        if not demo_pairs or not exp_pairs:
            continue

        def _requests(pairs) -> dict[str, str]:
            out = {}
            for pair in pairs:
                for side in ("a", "b"):
                    text = (pair.get(side) or {}).get("request")
                    if text:
                        out[text.strip()] = pair["id"]
            return out

        demo_req, exp_req = _requests(demo_pairs), _requests(exp_pairs)
        shared = set(demo_req) & set(exp_req)
        contaminated = {exp_req[t] for t in shared}
        layer = EXPANDED[exp_name]
        full = layer_battery(exp_name, layer)
        clean = layer_battery(exp_name, layer, exclude=contaminated)
        demo_overlap[exp_name] = {
            "demoPairs": len(demo_pairs),
            "expandedPairs": len(exp_pairs),
            "sharedRequests": len(shared),
            "expandedPairsContainingADemoRequest": sorted(contaminated),
            "withOverlap": {
                "ablation": _wilson(full["ablationFlips"], full["ablationSides"]),
                "crossPatch": _wilson(full["crossPatchFlips"], full["crossPatchDirections"]),
            },
            "overlapRemoved": {
                "ablation": _wilson(clean["ablationFlips"], clean["ablationSides"]),
                "crossPatch": _wilson(clean["crossPatchFlips"], clean["crossPatchDirections"]),
            },
        }

    return {"argmax": argmax, "nonSelected": non_selected, "demoOverlap": demo_overlap}


def _health_inputs(scenario: str) -> tuple[list[str], list[dict]] | None:
    """What the health screen reads, from the capture or the published subset.

    The full capture carries every activation and is far too large to publish
    (478 MB across the grid).  ``health_inputs.json`` holds the same screen's
    inputs at a thousandth of the size --- positive-activation feature ids, L0
    over the pair prompts, and per-prompt explained variance --- so the screen
    reproduces from the repository alone.  Either source yields the same
    numbers; the compact one is preferred when both are present because it is
    what a reader outside this machine will have.
    """
    root = _scenario_dir(scenario) / "capture"
    compact = _load(root / "health_inputs.json")
    if compact:
        return compact["pairPrompts"], [
            {
                "layer": blk["layer"],
                "l0": blk["l0"],
                "explainedVariance": blk["explainedVariance"],
                "active": [set(ids) for ids in blk["activeFeatures"]],
            }
            for blk in compact["layers"]
        ]
    ev = _load(root / "evaluation.json")
    if not ev:
        return None
    names = [p.get("step") if isinstance(p, dict) else p for p in ev["prompts"]]
    pair_idx = [i for i, n in enumerate(names) if str(n).endswith(("_A", "_B"))]
    return [names[i] for i in pair_idx], [
        {
            "layer": blk["layer"],
            "l0": [blk["l0"][i] for i in pair_idx],
            "explainedVariance": blk["explained_variance"],
            "active": [
                {f["index"] for f in blk["active_features"][i] if f["activation"] > 0}
                for i in pair_idx
            ],
        }
        for blk in ev["layers"]
    ]


def dictionary_health(scenario: str) -> dict[str, dict]:
    """Per-layer L0, explained variance, and how much of the code varies.

    ``constant`` = features active on *every* pair prompt; ``varying`` = active
    on some but not all.  A layer whose code is almost entirely constant has
    nothing left for a cue analysis to work with, however well it reconstructs.
    """
    inputs = _health_inputs(scenario)
    if not inputs:
        return {}
    names, blocks = inputs

    out = {}
    for blk in blocks:
        by_step = dict(zip(names, blk["active"]))
        sets = list(by_step.values())
        const = set.intersection(*sets)
        union = set.union(*sets)

        # How much of the code is left once the two sides of a pair are
        # compared?  On a collapsed layer this drops to single digits.
        side_specific = []
        for step in by_step:
            if not step.endswith("_A"):
                continue
            other = step[:-2] + "_B"
            if other in by_step:
                side_specific.append(len(by_step[step] - by_step[other]))
                side_specific.append(len(by_step[other] - by_step[step]))

        out[str(blk["layer"])] = {
            "meanL0": round(st.mean(blk["l0"]), 1),
            "explainedVariance": round(st.mean(blk["explainedVariance"]), 4),
            "constant": len(const),
            "varying": len(union - const),
            # the share of the code that never varies across these prompts.
            # Unlike explained variance this is measured on the analysis's own
            # inputs, and unlike the raw count it does not penalise a dictionary
            # that is merely dense (see Table "health" in the paper).
            "constantFraction": round(len(const) / len(union), 4) if union else None,
            "sideSpecificMin": min(side_specific) if side_specific else None,
            "sideSpecificMedian": int(st.median(side_specific)) if side_specific else None,
        }
    return out


def dose(scenario: str, layer: int) -> list[dict]:
    tr = _load(_battery_dir(scenario, f"trace_layer{layer}") / "trace_results.json")
    if not tr:
        return []
    rows = []
    for pid, dirs in tr["dose"].items():
        for dk, v in dirs.items():
            if not v:
                continue
            curve = {r["scale"]: r["p"] for r in v["allRows"]}
            crossing = next(
                (r["scale"] for r in v["allRows"] if r["choice"] == v["targetTool"]), None
            )
            best = v.get("bestRow") or {}
            best_curve = {r["scale"]: r["p"] for r in (best.get("curve") or [])}
            rows.append(
                {
                    "pair": pid,
                    "direction": dk,
                    "targetTool": v["targetTool"],
                    "baselineP": v["baselineP"],
                    "numFeatures": v.get("numFeatures"),
                    "curve": curve,
                    "crossesAt": crossing,
                    # band for the all-families curve: draws matched to the
                    # whole clamped set.  ``controlBand`` matches one family
                    # each and is the reference for ``bestSingleFeature``.
                    "setControlBand": v.get("setControlBand"),
                    # matched to the realised change, the strictest of the three
                    "deltaControlBand": v.get("deltaControlBand"),
                    "controlBand": v.get("controlBand"),
                    "bestSingleFeature": (
                        {
                            "index": best.get("index"),
                            "label": best.get("label"),
                            "curve": best_curve,
                        }
                        if best
                        else None
                    ),
                }
            )
    return rows


def _first_tool(text: str, tools: list[str]) -> str | None:
    """First tool named in a free-form continuation.

    The model writes tool names both as identifiers (``supplier_database``) and
    as prose (``supplier database``), so match on both spellings and take
    whichever appears earliest.
    """
    lowered = text.lower()
    best: tuple[int, str] | None = None
    for tool in tools:
        for spelling in (tool, tool.replace("_", " ")):
            i = lowered.find(spelling)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, tool)
    return best[1] if best else None


def generations(scenario: str, layer: int) -> list[dict]:
    tr = _load(_battery_dir(scenario, f"trace_layer{layer}") / "trace_results.json")
    if not tr:
        return []
    ui = _load(_scenario_dir(scenario) / "ui_data.json") or {}
    tools = [t["id"] if isinstance(t, dict) else t for t in (ui.get("scenario") or {}).get("tools", [])]
    rows = []
    for pid, dirs in tr["generations"].items():
        for dk, v in dirs.items():
            if not v:
                continue
            rows.append(
                {
                    "pair": pid,
                    "direction": dk,
                    "targetTool": v["targetTool"],
                    "numFeatures": v.get("numFeatures"),
                    "baseline": v["baseline"],
                    "steered": v["steered"],
                    "control": v["control"],
                    "controlSize": v.get("controlSize"),
                    "controlFamily": v.get("controlFamily"),
                    "controlDeltaMass": v.get("controlDeltaMass"),
                    "steeredDeltaMass": v.get("steeredDeltaMass"),
                    "controlIdenticalToBaseline": v["control"] == v["baseline"],
                    "baselineTool": _first_tool(v["baseline"], tools),
                    "steeredTool": _first_tool(v["steered"], tools),
                    "controlTool": _first_tool(v["control"], tools),
                    # a generation flip: the first tool named in the free
                    # continuation moves to the donor's tool, and the
                    # mass-matched control does not follow it
                    "flipped": (
                        _first_tool(v["steered"], tools) == v["targetTool"]
                        and _first_tool(v["baseline"], tools) != v["targetTool"]
                    ),
                    "controlFollowed": _first_tool(v["control"], tools) == v["targetTool"]
                    and _first_tool(v["baseline"], tools) != v["targetTool"],
                    # The arm above clamps every position on prefill and every
                    # decoded token, which is a larger intervention than the
                    # decision-token clamp the rest of the paper reports, so it
                    # bounds the effect rather than confirming it.  These fields
                    # carry the decision-token-only arm, which is the one that
                    # speaks to the paper's claim.
                    "decisionPosition": v.get("decisionPosition"),
                    "steeredDecisionToken": v.get("steeredDecisionToken"),
                    "controlDecisionToken": v.get("controlDecisionToken"),
                    "steeredDecisionTool": _first_tool(v.get("steeredDecisionToken") or "", tools),
                    "controlDecisionTool": _first_tool(v.get("controlDecisionToken") or "", tools),
                    "flippedDecisionToken": (
                        v.get("steeredDecisionToken") is not None
                        and _first_tool(v["steeredDecisionToken"], tools) == v["targetTool"]
                        and _first_tool(v["baseline"], tools) != v["targetTool"]
                    ),
                    "controlFollowedDecisionToken": (
                        v.get("controlDecisionToken") is not None
                        and _first_tool(v["controlDecisionToken"], tools) == v["targetTool"]
                        and _first_tool(v["baseline"], tools) != v["targetTool"]
                    ),
                    "decisionTokenIdenticalToBaseline": (
                        v.get("steeredDecisionToken") is not None
                        and v["steeredDecisionToken"] == v["baseline"]
                    ),
                }
            )
    return rows


def position_ablation(scenario: str, layer: int) -> dict:
    """Where along the prompt do the cue features actually matter?

    ``trace_pairs.py`` ablates each side's cue families at four position sets:
    the request tokens only, the decision token only, everything *but* the
    decision token, and everywhere.  If the cue were doing its work while the
    request is being read, the request-only condition would move the decision.
    """
    tr = _load(_battery_dir(scenario, f"trace_layer{layer}") / "trace_results.json")
    if not tr:
        return {}
    conditions = ("request", "decision", "allButDecision", "all")
    rows = []
    for step, pos in tr["positions"].items():
        blk = (pos.get("layers") or {}).get(str(layer)) or {}
        ab = blk.get("ablate") or {}
        base_tool = pos["targetTool"]
        base_p = pos["baseline"][base_tool]
        row = {"step": step, "baselineP": round(base_p, 4)}
        for cond in conditions:
            v = ab.get(cond)
            if not v:
                continue
            row[cond] = {
                "p": round(v["p"], 4),
                "deltaP": round(v["p"] - base_p, 4),
                "flipped": v["choice"] != base_tool,
                # the arm switches off every cue family at once, so the band
                # that applies is the set-matched one where it was recorded
                "controlBand": v.get("controlBand"),
                "setControlBand": v.get("setControlBand"),
            }
        rows.append(row)
    summary = {}
    for cond in conditions:
        present = [r[cond] for r in rows if cond in r]
        if present:
            summary[cond] = {
                "flips": sum(x["flipped"] for x in present),
                "sides": len(present),
                "maxAbsDelta": round(max(abs(x["deltaP"]) for x in present), 4),
            }
    return {"summary": summary, "rows": rows}


def probes(scenario: str) -> dict:
    """Keyword controls and paraphrases.

    A *keyword control* takes the other side's request and slips this side's cue
    word in without changing what is asked; it is correct when the model keeps
    the other side's tool.  A *paraphrase* restates this side's request without
    its cue words; it is correct when the model keeps this side's tool.
    """
    ui = _load(_scenario_dir(scenario) / "ui_data.json")
    if not ui:
        return {}
    para = []
    kw = []
    for pair in ui["pairs"]:
        tools = {s: pair[s]["modelChoice"]["display"] for s in ("a", "b")}
        for side in ("a", "b"):
            other = tools["b" if side == "a" else "a"]
            pb = pair[side].get("probes") or {}
            for p in pb.get("paraphrases") or []:
                para.append(
                    {
                        "pair": pair["id"],
                        "side": side,
                        "expected": tools[side],
                        "got": p["modelChoice"]["display"],
                        "held": p["modelChoice"]["display"] == tools[side],
                        "familiesFiring": p.get("familiesFiring"),
                        "familiesTotal": p.get("familiesTotal"),
                    }
                )
            k = pb.get("keyword")
            if k:
                kw.append(
                    {
                        "pair": pair["id"],
                        "side": side,
                        "cue": k.get("cue"),
                        "expected": other,
                        "got": k["modelChoice"]["display"],
                        "held": k["modelChoice"]["display"] == other,
                    }
                )
    return {
        "paraphrases": {"held": sum(p["held"] for p in para), "total": len(para), "rows": para},
        "keywordControls": {"held": sum(k["held"] for k in kw), "total": len(kw), "rows": kw},
    }


def parity(scenario: str, layer: int) -> dict | None:
    res = _load(_battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json")
    if not res or not res.get("parity"):
        return None
    cos = [v["mean_cosine_similarity"] for v in res["parity"].values() if v]
    return {"meanCosine": round(st.mean(cos), 4), "minCosine": round(min(cos), 4), "n": len(cos)}


def _set_exceedance(v: dict, signed: bool, family: str = "setControls") -> tuple[str, bool] | None:
    """Compare one side's cue-set effect with its *set-matched* draws.

    Returns ``(kind, exceeded)`` where kind is ``"matched"`` when the three
    draws really are random sets of the cue set's count and mass -- the case
    the 1/4 exchangeability null describes -- and ``"ceiling"`` when the rest
    of the active set is too light to reach that mass, so the draws collapse
    to the one set containing every other active feature.  A ceiling side
    still says something (no other feature can do this), but it is not a
    random draw and must not be counted in the exact test.
    """
    controls = v.get(family)
    if not controls:
        return None
    effect = (v.get("allRows") or {}).get("deltaTarget")
    if effect is None:
        return None
    effect = float(effect) if signed else abs(float(effect))
    band = max(abs(float(c["deltaTarget"])) for c in controls)
    flag, draws_key = {
        "deltaControls": ("deltaMassMatched", "deltaControlDistinctDraws"),
        "contrastControls": ("massMatched", "contrastControlDistinctDraws"),
    }.get(family, ("massMatched", "setControlDistinctDraws"))
    matched = all(c.get(flag) for c in controls) and v.get(draws_key) == len(controls)
    return ("matched" if matched else "ceiling", effect > band)


def sparse_recovery(scenarios: dict[str, int]) -> dict | None:
    """What fraction of the available causal signal the SAE arms recover.

    ``ceiling_pairs.py`` patches the donor's whole residual into the recipient's
    decision token, with no dictionary in the path, so its flip count is an
    upper bound for any decomposition read at that position.  Dividing the
    cue-set and bulk flip counts by it turns them from bare numbers into a
    sparse causal recovery rate.  The difference-in-means arm is the other
    reference a reader wants: a direction that needed no per-pair activations
    and no dictionary at all.

    Denominators are aligned by construction --- both runs enumerate the same
    pairs and directions --- and directions missing from either side are
    dropped rather than counted as failures.
    """
    ceiling_flips = cue_flips = bulk_flips = n = 0
    dim_flips = dim_n = 0
    dim_matched_flips = dim_matched_n = 0
    random_flips = random_draws = 0
    random_matched_flips = random_matched_draws = 0
    per_scenario: dict[str, dict] = {}
    for scenario, layer in scenarios.items():
        ceil = _load(_ceiling_path(scenario, layer))
        res = _load(_battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json")
        if not ceil or not res:
            continue
        s_ceiling = s_cue = s_bulk = s_n = 0
        for pid, dirs in ceil["directions"].items():
            for dk, rec in dirs.items():
                patch = (res.get("crossPatch") or {}).get(pid, {}).get(dk)
                if not patch:
                    continue
                s_n += 1
                s_ceiling += bool(rec["ceiling"]["flipped"])
                s_cue += _flip(patch.get("allRows"), patch["targetTool"])
                s_bulk += _flip(patch.get("allBase"), patch["targetTool"])
                dim = rec.get("differenceInMeans")
                if dim:
                    dim_n += 1
                    dim_flips += bool(dim["flipped"])
                random_flips += rec.get("randomFlips", 0)
                random_draws += len(rec.get("randomControls") or [])
                dim_m = rec.get("differenceInMeansMatched")
                if dim_m:
                    dim_matched_n += 1
                    dim_matched_flips += bool(dim_m["flipped"])
                random_matched_flips += rec.get("randomMatchedFlips", 0)
                random_matched_draws += len(rec.get("randomMatched") or [])
        if not s_n:
            continue
        per_scenario[scenario] = {
            "layer": layer,
            "directions": s_n,
            "ceilingFlips": s_ceiling,
            "cueFlips": s_cue,
            "bulkFlips": s_bulk,
            "cueOverCeiling": round(s_cue / s_ceiling, 3) if s_ceiling else None,
            "bulkOverCeiling": round(s_bulk / s_ceiling, 3) if s_ceiling else None,
        }
        ceiling_flips += s_ceiling
        cue_flips += s_cue
        bulk_flips += s_bulk
        n += s_n
    if not n:
        return None
    return {
        "directions": n,
        "ceilingFlips": ceiling_flips,
        "cueFlips": cue_flips,
        "bulkFlips": bulk_flips,
        "cueOverCeiling": round(cue_flips / ceiling_flips, 3) if ceiling_flips else None,
        "bulkOverCeiling": round(bulk_flips / ceiling_flips, 3) if ceiling_flips else None,
        "differenceInMeansFlips": dim_flips,
        "differenceInMeansDirections": dim_n,
        "randomDirectionFlips": random_flips,
        "randomDirectionDraws": random_draws,
        # The three arms above act at the donor-minus-recipient norm, which is
        # larger than the change a cue-set clamp makes, so they bound rather
        # than match it.  These two act at the clamp's own norm and are the
        # like-for-like comparison.
        "differenceInMeansMatchedFlips": dim_matched_flips,
        "differenceInMeansMatchedDirections": dim_matched_n,
        "randomMatchedFlips": random_matched_flips,
        "randomMatchedDraws": random_matched_draws,
        "perScenario": per_scenario,
    }


def contrast_band_by_depth() -> dict | None:
    """Does the cue set beat a *difference-matched* control, early versus late?

    The depth result rests on flip counts, which are argmax crossings and so
    depend on where each baseline happens to sit.  This is the same contrast
    measured as effect size against the strictest control available --- a random
    set matched on how much it differs across the pair --- and it needs no
    argmax to cross.

    A *ceiling* draw is the whole eligible pool rather than a matched sample, so
    its band is not a match and its ratio flatters the cue set (late ceiling
    sides exceed on every one).  The headline fields are therefore the genuinely
    matched sides only; ceiling sides are reported beside them under ``ceiling``
    rather than pooled in, and sides whose band is zero --- where no draw moved
    anything, so the ratio is undefined --- are counted in ``zeroBandSides``.
    """
    matched: dict[str, list[float]] = {"early": [], "late": []}
    ceiling: dict[str, list[float]] = {"early": [], "late": []}
    zero_band = {"early": 0, "late": 0}
    with_arm = {"early": 0, "late": 0}
    for scenario in SCENARIOS:
        for layer in LAYERS:
            res = _load(
                _battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json"
            )
            if not res:
                continue
            key = "early" if layer in EARLY else "late"
            for sides in res["attribution"].values():
                for v in sides.values():
                    if not isinstance(v, dict) or v.get("contrastControls") is None:
                        continue
                    with_arm[key] += 1
                    band = v.get("contrastControlThreshold")
                    if not band:
                        zero_band[key] += 1
                        continue
                    ratio = abs((v.get("allRows") or {}).get("deltaTarget") or 0.0) / float(band)
                    bucket = matched if v.get("contrastControlMassMatched") else ceiling
                    bucket[key].append(ratio)
    if not matched["early"] or not matched["late"]:
        return None
    out = {}
    for key, values in matched.items():
        exceed = sum(r > 1 for r in values)
        pool = ceiling[key]
        out[key] = {
            "sides": len(values),
            "medianEffectOverBand": round(st.median(values), 2),
            "exceeding": exceed,
            "fractionExceeding": round(exceed / len(values), 3),
            "ceiling": {
                "sides": len(pool),
                "exceeding": sum(r > 1 for r in pool),
                "medianEffectOverBand": round(st.median(pool), 2) if pool else None,
            },
            "zeroBandSides": zero_band[key],
            "sidesWithArm": with_arm[key],
        }
    out["fisherP"] = _fisher_one_sided(
        out["late"]["exceeding"], out["late"]["sides"] - out["late"]["exceeding"],
        out["early"]["exceeding"], out["early"]["sides"] - out["early"]["exceeding"],
    )
    return out


def _paired_arms(scenario: str, layer: int):
    """Per-direction (cluster, cue redirect, dense redirect, reference redirect).

    Yields one record per direction that both the battery and the ceiling run
    enumerate.  A *redirect* here carries the full directed-flip requirement:
    the baseline must not already be the target, so a direction the model
    already gets "right" cannot be scored as a success for any arm.
    """
    res = _load(_battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json")
    ceil = _load(_ceiling_path(scenario, layer))
    pairs = _load(STEER / scenario / "pairs.json") or {}
    if not res or not ceil:
        return
    titles = {p["id"]: p.get("title") for p in pairs.get("pairs", [])}
    for pair_id, dirs in ceil["directions"].items():
        for key, rec in dirs.items():
            patch = (res.get("crossPatch") or {}).get(pair_id, {}).get(key)
            if not patch:
                continue
            target = rec["targetTool"]
            live = rec.get("baselineChoice") != target
            dense = rec.get("differenceInMeansMatched")
            yield {
                "cluster": f"{scenario}:{titles.get(pair_id)}",
                "pair": pair_id,
                "axis": titles.get(pair_id),
                "cue": bool(live and (patch.get("allRows") or {}).get("choice") == target),
                "dense": bool(live and dense and dense.get("choice") == target),
                "reference": bool(live and rec["ceiling"].get("choice") == target),
                "denseDefined": dense is not None,
            }


def paired_cue_dense(scenarios: dict[str, int]) -> dict | None:
    """Cue clamp against an equal-norm dense direction, direction by direction.

    Both arms act on the same recipient at the same token at the same norm, so
    the comparison is paired: the two marginal counts are computed over the
    same directions and share every concordant one.  Only the discordant cells
    say which arm is stronger, so the whole 2x2 is reported rather than the
    margins alone.

    Uncertainty is an exact enumeration rather than a sampled bootstrap.  With
    four clusters per scenario there are only ``35**2 = 1,225`` distinct
    cluster-count vectors, few enough that a 100,000-draw Monte-Carlo endpoint
    lands on either side of a boundary atom depending on the seed --- the
    97.5th percentile here sits at cumulative weight 0.9752.  Enumerating
    removes that wobble.  With so few clusters the brackets remain uncalibrated
    sensitivity ranges, not confidence intervals, and the paper says so.
    """
    per_cluster: dict[str, list[int]] = {}
    cells = {"bothRedirect": 0, "cueOnly": 0, "denseOnly": 0, "neither": 0}
    strata: dict[str, list[str]] = {}
    directions = dense_undefined = 0
    for scenario, layer in scenarios.items():
        for row in _paired_arms(scenario, layer):
            if not row["denseDefined"]:
                dense_undefined += 1
                continue
            directions += 1
            entry = per_cluster.setdefault(row["cluster"], [0, 0, 0])
            entry[0] += row["cue"]
            entry[1] += row["dense"]
            entry[2] += 1
            strata.setdefault(scenario, [])
            if row["cluster"] not in strata[scenario]:
                strata[scenario].append(row["cluster"])
            key = (
                "bothRedirect" if row["cue"] and row["dense"]
                else "cueOnly" if row["cue"]
                else "denseOnly" if row["dense"]
                else "neither"
            )
            cells[key] += 1
    if not directions:
        return None

    def brackets(groups: list[list[str]]) -> dict:
        atoms = _enumerate_resamples(
            [[tuple(per_cluster[k]) for k in sorted(g)] for g in groups]
        )
        diffs = [(w, (v[0] - v[1]) / v[2] * 100) for w, v in atoms]
        return {
            "vectors": len(atoms),
            "orderedDraws": sum(w for w, _ in atoms),
            "lo": round(_weighted_percentile(diffs, 0.025), 2),
            "hi": round(_weighted_percentile(diffs, 0.975), 2),
        }

    cue_flips = cells["bothRedirect"] + cells["cueOnly"]
    dense_flips = cells["bothRedirect"] + cells["denseOnly"]
    return {
        "unit": "scenario and contrast title",
        "directions": directions,
        "denseUndefinedDirections": dense_undefined,
        "clusters": len(per_cluster),
        "clustersPerScenario": {s: len(g) for s, g in sorted(strata.items())},
        **cells,
        "discordant": cells["cueOnly"] + cells["denseOnly"],
        "cueRedirects": cue_flips,
        "denseRedirects": dense_flips,
        "differencePp": round((cue_flips - dense_flips) / directions * 100, 2),
        # the reported bracket: each scenario resamples its own clusters
        "stratified": brackets([g for _, g in sorted(strata.items())]),
        # and the same enumeration ignoring the scenario split, as a check that
        # the conclusion is not an artefact of how the strata were drawn
        "pooled": brackets([sorted(per_cluster)]),
    }


def heldout_overlap(splits: dict[str, dict[str, int]]) -> dict:
    """Are the cue redirects a subset of what the full residual patch reaches?

    A flip ratio says how many reference redirects the cue set also finds; it
    does not say they are the *same* directions, nor that they come from more
    than one contrast.  Both would let a ratio read as breadth when it is one
    lucky axis, so the nesting and the spread across axes and pairs are
    recorded next to the ratio.

    These probes are curated, not sampled, so the brackets are exact
    sensitivity ranges over whole-axis resampling and the paper quotes no
    nominal interval for them.
    """
    out: dict[str, dict] = {}
    for name, scenarios in splits.items():
        per_cluster: dict[str, list[int]] = {}
        strata: dict[str, list[str]] = {}
        directions = reference = cue = nested = 0
        axes: set[str | None] = set()
        cue_axes: set[str | None] = set()
        pairs: set[str] = set()
        cue_pairs: set[str] = set()
        for scenario, layer in scenarios.items():
            for row in _paired_arms(scenario, layer):
                directions += 1
                reference += row["reference"]
                cue += row["cue"]
                nested += row["cue"] and row["reference"]
                axes.add(row["cluster"])
                pairs.add(f"{scenario}:{row['pair']}")
                entry = per_cluster.setdefault(row["cluster"], [0, 0])
                entry[0] += row["cue"]
                entry[1] += row["reference"]
                strata.setdefault(scenario, [])
                if row["cluster"] not in strata[scenario]:
                    strata[scenario].append(row["cluster"])
                if row["cue"]:
                    cue_axes.add(row["cluster"])
                    cue_pairs.add(f"{scenario}:{row['pair']}")
        if not directions:
            continue
        groups = [[tuple(per_cluster[k]) for k in sorted(g)] for _, g in sorted(strata.items())]
        out[name] = {
            "scenarios": dict(sorted(scenarios.items())),
            "directions": directions,
            "referenceRedirects": reference,
            "cueRedirects": cue,
            "cueRedirectsInsideReference": nested,
            "axes": len(axes),
            "axesWithCueRedirect": len(cue_axes),
            "pairs": len(pairs),
            "pairsWithCueRedirect": len(cue_pairs),
            "flipRatio": round(cue / reference, 3) if reference else None,
            "bracketPooled": _ratio_bracket([[tuple(per_cluster[k]) for k in sorted(per_cluster)]]),
            **({"bracketStratified": _ratio_bracket(groups)} if len(groups) > 1 else {}),
        }
    return out


def outcome_partition(scenarios: dict[str, int]) -> dict:
    """Where the argmax actually goes, not just whether it moved.

    A "directed flip" requires the new argmax to be the paired tool.  On a
    minimal pair that tool is usually the baseline *runner-up* --- the gate
    demands two near-identical prompts that pick different tools --- so an
    intervention that merely degrades the winner will produce a directed flip
    by default.  This function reports the full partition (unchanged /
    directed / third tool) for both arms and audits that confound directly:
    how often the paired tool is the runner-up, how often flips land on the
    runner-up whatever it is, and whether the intervention still finds the
    paired tool when it is *not* second.

    Cross-patch controls carry their resulting ``choice`` and get the same
    partition.  Attribution controls store only ``deltaTarget``, so the
    ablation arm has no control partition until those batteries are re-run;
    ``ablationControls`` is ``None`` rather than zero to say so.
    """
    abl = {"unchanged": 0, "directed": 0, "thirdTool": 0}
    cue = {"unchanged": 0, "directed": 0, "thirdTool": 0}
    bulk = {"unchanged": 0, "directed": 0, "thirdTool": 0}
    ctl = {"unchanged": 0, "directed": 0, "thirdTool": 0}
    abl_ctl = {"unchanged": 0, "directed": 0, "thirdTool": 0}
    paired_is_runner_up = 0
    flips = flips_on_runner_up = 0
    split = {"runnerUpYesFoundYes": 0, "runnerUpYesFoundNo": 0,
             "runnerUpNoFoundYes": 0, "runnerUpNoFoundNo": 0}

    def bump(bucket: dict, new: str | None, base: str | None, target: str | None) -> None:
        if new is None or new == base:
            bucket["unchanged"] += 1
        elif new == target:
            bucket["directed"] += 1
        else:
            bucket["thirdTool"] += 1

    for name, layer in scenarios.items():
        res = _load(_battery_dir(name, f"steering_layer{layer}") / "steering_results.json")
        if not res:
            continue
        for sides in res["attribution"].values():
            for side in ("a", "b"):
                v = sides.get(side)
                if not v:
                    continue
                base = v.get("baseline") or {}
                if not base:
                    continue
                order = sorted(base, key=base.get, reverse=True)
                winner = order[0]
                runner_up = order[1] if len(order) > 1 else None
                other = v.get("otherTool")
                new = _argmax((v.get("allRows") or {}).get("intervened"))
                bump(abl, new, winner, other)
                is_ru = runner_up == other
                paired_is_runner_up += is_ru
                if new is not None and new != winner:
                    flips += 1
                    flips_on_runner_up += new == runner_up
                    key = f"runnerUp{'Yes' if is_ru else 'No'}Found{'Yes' if new == other else 'No'}"
                    split[key] += 1
                for family in ("controls", "setControls", "contrastControls"):
                    for c in v.get(family) or []:
                        if c.get("choice") is not None:
                            bump(abl_ctl, c["choice"], winner, other)
        for dirs in res["crossPatch"].values():
            for v in dirs.values():
                if not isinstance(v, dict) or "allRows" not in v:
                    continue
                base = v.get("intoBaselineChoice")
                target = v.get("targetTool")
                bump(cue, (v.get("allRows") or {}).get("choice"), base, target)
                bump(bulk, (v.get("allBase") or {}).get("choice"), base, target)
                for family in ("controls", "setControls", "deltaControls"):
                    for c in v.get(family) or []:
                        if c.get("choice") is not None:
                            bump(ctl, c["choice"], base, target)

    n_abl = sum(abl.values())
    return {
        "ablation": {**abl, "n": n_abl},
        "crossPatchCue": {**cue, "n": sum(cue.values())},
        "crossPatchBulk": {**bulk, "n": sum(bulk.values())},
        "crossPatchControls": {**ctl, "n": sum(ctl.values())},
        # ``None`` where the batteries predate the control-argmax field, so a
        # missing partition never reads as "no control ever moved the tool"
        "ablationControls": ({**abl_ctl, "n": sum(abl_ctl.values())} if sum(abl_ctl.values()) else None),
        "runnerUp": {
            "pairedToolIsBaselineRunnerUp": paired_is_runner_up,
            "sides": n_abl,
            "ablationFlips": flips,
            "flipsLandingOnBaselineRunnerUp": flips_on_runner_up,
            **split,
        },
    }


def paired_stats(scenarios: dict[str, int]) -> dict:
    """Exact tests pairing each cue-set intervention with its stored controls.

    Each intervention carries two families of random draws, and they answer
    different questions.  Three draws are matched to *each cue family* and
    three to the *whole cue set*; the set-level test must use the latter,
    because a family-matched draw carries only a fraction of the set's
    activation mass (``massAudit`` below measures how large that gap is).

    Under the null that the cue set is exchangeable with a random set of the
    same count and mass, it beats the maximum of its three set-matched draws
    with probability 1/4 — so counting exceedances over sides gives an exact
    binomial test that does not depend on the flip counts at all.  Sides where
    every other active feature together weighs less than the cue set are held
    out of that test and reported separately: there the "draw" is the entire
    rest of the active set, which is a ceiling, not a sample.

    The row level compares like with like without any of this: single cue
    families against the draws matched to them, on whether the tool changes.
    """
    ab_n = ab_exceed = cp_n = cp_exceed = 0
    ab_draws = cp_draws = cp_ctrl_flips = 0
    row_n = row_flips = 0
    set_counts = {
        arm: {"matched": 0, "matchedExceed": 0, "ceiling": 0, "ceilingExceed": 0}
        for arm in ("ablation", "crossPatch")
    }
    delta_counts = {"matched": 0, "matchedExceed": 0, "ceiling": 0, "ceilingExceed": 0}
    contrast_counts = {
        "matched": 0, "matchedExceed": 0, "ceiling": 0, "ceilingExceed": 0,
        "sides": 0, "poolEmpty": 0,
    }
    mass_ratios: list[float] = []
    delta_over_donor: list[float] = []
    for scenario, layer in scenarios.items():
        res = _load(
            _battery_dir(scenario, f"steering_layer{layer}") / "steering_results.json"
        )
        if not res:
            continue
        for sides in res["attribution"].values():
            for side in ("a", "b"):
                v = sides.get(side)
                if not v:
                    continue
                ab_n += 1
                ab_draws += len(v["controls"])
                thr = max(abs(c["deltaTarget"]) for c in v["controls"])
                if abs(v["allRows"]["deltaTarget"]) > thr:
                    ab_exceed += 1
                verdict = _set_exceedance(v, signed=False)
                if verdict:
                    kind, exceeded = verdict
                    set_counts["ablation"][kind] += 1
                    set_counts["ablation"][f"{kind}Exceed"] += exceeded
                verdict = _set_exceedance(v, signed=False, family="contrastControls")
                if verdict:
                    kind, exceeded = verdict
                    contrast_counts[kind] += 1
                    contrast_counts[f"{kind}Exceed"] += exceeded
                    contrast_counts["sides"] += 1
                    contrast_counts["poolEmpty"] += bool(v.get("contrastPoolEmpty"))
                ratio = _mass_ratio(v, "hfMass")
                if ratio:
                    mass_ratios.append(ratio)
        for dirs in res["crossPatch"].values():
            for v in dirs.values():
                if not v:
                    continue
                cp_n += 1
                cp_draws += len(v["controls"])
                thr = max(abs(c["deltaTarget"]) for c in v["controls"])
                if v["allRows"]["deltaTarget"] > thr:
                    cp_exceed += 1
                verdict = _set_exceedance(v, signed=True)
                if verdict:
                    kind, exceeded = verdict
                    set_counts["crossPatch"][kind] += 1
                    set_counts["crossPatch"][f"{kind}Exceed"] += exceeded
                verdict = _set_exceedance(v, signed=True, family="deltaControls")
                if verdict:
                    kind, exceeded = verdict
                    delta_counts[kind] += 1
                    delta_counts[f"{kind}Exceed"] += exceeded
                moved = _delta_over_donor(v)
                if moved is not None:
                    delta_over_donor.append(moved)
                ratio = _mass_ratio(v, "baseMass")
                if ratio:
                    mass_ratios.append(ratio)
                base = v["intoBaselineChoice"]
                for r in v["rows"]:
                    row_n += 1
                    row_flips += r.get("choice", base) != base
                for c in v["controls"]:
                    cp_ctrl_flips += c.get("choice", base) != base

    def _set_block(arm: str) -> dict:
        c = set_counts[arm]
        return {
            "matchedSides": c["matched"],
            "matchedExceed": c["matchedExceed"],
            # exchangeability null: the cue set is the max of 4 exchangeable
            # values with probability 1/4.  Only the matched sides qualify.
            "pExchangeable": _binom_tail(c["matchedExceed"], c["matched"], 0.25)
            if c["matched"]
            else None,
            "pSign": _binom_tail(c["matchedExceed"], c["matched"], 0.5) if c["matched"] else None,
            # sides where the rest of the active set cannot reach the cue
            # set's mass: the comparison is against every other active
            # feature, reported but not tested
            "ceilingSides": c["ceiling"],
            "ceilingExceed": c["ceilingExceed"],
        }

    return {
        "ablation": {
            "exceed": ab_exceed,
            "n": ab_n,
            "controlDraws": ab_draws,
            # against the per-family band, which does not match the cue set's
            # mass -- kept for continuity with the pre-set-control batteries
            "pExchangeable": _binom_tail(ab_exceed, ab_n, 0.25),
            "pSign": _binom_tail(ab_exceed, ab_n, 0.5),
            "perFamilyBand": True,
        },
        "crossPatch": {
            "exceed": cp_exceed,
            "n": cp_n,
            "controlDraws": cp_draws,
            "controlFlips": cp_ctrl_flips,
            "pExchangeable": _binom_tail(cp_exceed, cp_n, 0.25),
            "pSign": _binom_tail(cp_exceed, cp_n, 0.5),
            "perFamilyBand": True,
        },
        "setMatched": {
            "ablation": _set_block("ablation"),
            "crossPatch": _set_block("crossPatch"),
        },
        # cross-patch only.  Ablation needs no entry: its target is zero, so
        # the donor mass it matches on *is* the change it applies.
        "deltaMatched": {
            "crossPatch": {
                "matchedSides": delta_counts["matched"],
                "matchedExceed": delta_counts["matchedExceed"],
                "pExchangeable": _binom_tail(
                    delta_counts["matchedExceed"], delta_counts["matched"], 0.25
                )
                if delta_counts["matched"]
                else None,
                "pSign": _binom_tail(delta_counts["matchedExceed"], delta_counts["matched"], 0.5)
                if delta_counts["matched"]
                else None,
                "ceilingSides": delta_counts["ceiling"],
                "ceilingExceed": delta_counts["ceilingExceed"],
            },
            "medianCueDeltaOverDonorMass": round(st.median(delta_over_donor), 3)
            if delta_over_donor
            else None,
            "directions": len(delta_over_donor),
        },
        # Ablation's answer to "is it the cue-ness or the mass?".  Draws are
        # matched on how much the set differs across the pair, not on how much
        # of it is there, and exclude the cue set.  A ceiling means no other
        # differing set that heavy exists; ``poolEmpty`` means nothing else
        # differs at all, which is the strongest form of the same answer.
        "contrastMatched": (
            {
                "ablation": {
                    "matchedSides": contrast_counts["matched"],
                    "matchedExceed": contrast_counts["matchedExceed"],
                    "pExchangeable": _binom_tail(
                        contrast_counts["matchedExceed"], contrast_counts["matched"], 0.25
                    )
                    if contrast_counts["matched"]
                    else None,
                    "pSign": _binom_tail(
                        contrast_counts["matchedExceed"], contrast_counts["matched"], 0.5
                    )
                    if contrast_counts["matched"]
                    else None,
                    "ceilingSides": contrast_counts["ceiling"],
                    "ceilingExceed": contrast_counts["ceilingExceed"],
                    "poolEmptySides": contrast_counts["poolEmpty"],
                },
                "sides": contrast_counts["sides"],
            }
            if contrast_counts["sides"]
            else None
        ),
        # how much heavier the cue set is than the heaviest draw matched to one
        # of its families: the size of the mismatch the set-matched arm fixes
        "massAudit": {
            "sides": len(mass_ratios),
            "medianCueMassOverHeaviestRowDraw": round(st.median(mass_ratios), 2)
            if mass_ratios
            else None,
            "sidesWhereRowDrawReachesCueMass": sum(r <= 1 for r in mass_ratios),
        },
        "rowLevel": {
            "cueFamilyFlips": row_flips,
            "cueFamilies": row_n,
            "controlFlips": cp_ctrl_flips,
            "controlDraws": cp_draws,
            "fisherP": _fisher_one_sided(
                row_flips, row_n - row_flips, cp_ctrl_flips, cp_draws - cp_ctrl_flips
            ),
        },
    }


def build_stats(report: dict) -> dict:
    """Interval estimates and exact tests over the batteries already collected.

    Everything here is a restatement of counts elsewhere in the report — no new
    measurements — so the paper can cite uncertainty without hand-derived
    arithmetic.
    """
    scen = report["scenarios"]

    def battery(name: str, layer: int) -> dict | None:
        return scen.get(name, {}).get("layers", {}).get(str(layer))

    wilson: dict = {}
    pools = {
        "demo": list(PRIMARY),
        "expanded": [e for e in EXPANDED if e in scen],
        "heldout": [h for h in HELDOUT if h in scen],
    }
    for pool, names in pools.items():
        ab_k = ab_n = cp_k = cp_n = 0
        for name in names:
            layer = PRIMARY.get(name) or EXPANDED.get(name) or HELDOUT[name]
            b = battery(name, layer)
            if not b:
                continue
            wilson[name] = {
                "layer": layer,
                "ablation": _wilson(b["ablationFlips"], b["ablationSides"]),
                "crossPatch": _wilson(b["crossPatchFlips"], b["crossPatchDirections"]),
            }
            ab_k += b["ablationFlips"]
            ab_n += b["ablationSides"]
            cp_k += b["crossPatchFlips"]
            cp_n += b["crossPatchDirections"]
        if ab_n:
            wilson[f"pooled_{pool}"] = {
                "ablation": _wilson(ab_k, ab_n),
                "crossPatch": _wilson(cp_k, cp_n),
            }

    for name in PRIMARY:
        pr = scen.get(name, {}).get("probes") or {}
        for key in ("keywordControls", "paraphrases"):
            if pr.get(key, {}).get("total"):
                wilson[name][key] = _wilson(pr[key]["held"], pr[key]["total"])

    # decision-token positional ablation, pooled over the two demo pages
    pos_k = pos_n = 0
    for name in PRIMARY:
        sm = (scen.get(name, {}).get("positionAblation") or {}).get("summary") or {}
        if sm.get("decision"):
            pos_k += sm["decision"]["flips"]
            pos_n += sm["decision"]["sides"]
    if pos_n:
        wilson["decisionTokenPooled"] = _wilson(pos_k, pos_n)

    # Depth: everything below the last quarter of the network vs the primary
    # layers, over every battery in the report (demo and expanded).
    early_k = early_n = late_k = late_n = 0
    for name, entry in scen.items():
        if name in HELDOUT:  # probe of unseen prompts, not part of the depth grid
            continue
        primary = PRIMARY.get(name) or EXPANDED.get(name)
        for layer_str, b in entry.get("layers", {}).items():
            flips = b["ablationFlips"] + b["crossPatchFlips"]
            n = b["ablationSides"] + b["crossPatchDirections"]
            if int(layer_str) in EARLY:
                early_k += flips
                early_n += n
            elif int(layer_str) == primary:
                late_k += flips
                late_n += n

    # the same "late" quantity read at layer 27, which no scenario selected
    late27_k = late27_n = 0
    for name, layer in list(PRIMARY.items()) + list(NON_SELECTED.items()):
        b = layer_battery(name, 27 if name in PRIMARY else layer)
        if not b:
            continue
        late27_k += b["ablationFlips"] + b["crossPatchFlips"]
        late27_n += b["ablationSides"] + b["crossPatchDirections"]

    return {
        "primaryLayers": {**PRIMARY, **{k: v for k, v in EXPANDED.items() if k in scen}},
        "wilson": wilson,
        "clusterBootstrapExpanded": clustered_flip_intervals(
            {k: v for k, v in EXPANDED.items() if k in scen}
        ),
        "designWeighted": design_weighted_rates(
            {k: v for k, v in EXPANDED.items() if k in scen}
        ),
        "paired": paired_stats(PRIMARY),
        "pairedExpanded": paired_stats({k: v for k, v in EXPANDED.items() if k in scen}),
        "outcomes": outcome_partition(PRIMARY),
        "outcomesExpanded": outcome_partition({k: v for k, v in EXPANDED.items() if k in scen}),
        "recovery": sparse_recovery(PRIMARY),
        "recoveryExpanded": sparse_recovery({k: v for k, v in EXPANDED.items() if k in scen}),
        # the held-out probe needs a denominator too, and more than the others:
        # its lower flip rates could mean weaker features or less available
        # signal at the decision token, and only the ceiling separates those
        "recoveryHeldout": sparse_recovery({k: v for k, v in HELDOUT.items() if k in scen}),
        "depth": {
            "earlyLayers": EARLY,
            "earlyFlips": early_k,
            "earlyN": early_n,
            "earlyUpper95": round(_exact_upper(early_k, early_n), 4) if early_n else None,
            "lateFlips": late_k,
            "lateN": late_n,
            "fisherP": _fisher_one_sided(late_k, late_n - late_k, early_k, early_n - early_k),
            # the late arm sits at the selected layer, so rebuild it at a layer
            # that was not selected: the contrast survives, the effect size
            # does not, and the paper should claim only the former
            **(
                {
                    "lateFlipsNonSelected": late27_k,
                    "lateNNonSelected": late27_n,
                    "fisherPNonSelected": _fisher_one_sided(
                        late27_k, late27_n - late27_k, early_k, early_n - early_k
                    ),
                }
                if late27_n
                else {}
            ),
        },
        "layerSelection": layer_selection_audit(),
        "contrastByDepth": contrast_band_by_depth(),
        # the paired cue-versus-dense comparison the workshop paper leads with,
        # and the nesting audit behind its held-out probes
        "pairedCueDense": paired_cue_dense({k: v for k, v in EXPANDED.items() if k in scen}),
        "heldoutOverlap": heldout_overlap(
            {
                "heldout": {k: v for k, v in HELDOUT.items() if k in scen},
                "toolSelection": {"tool_selection": PRIMARY["tool_selection"]},
            }
        ),
    }


def _resolve_out(argv=None) -> Path:
    """Where to write, refusing to clobber the committed report from artifacts.

    Reading the published subset produces a report the canonical one is not
    supposed to equal --- it is built from a deliberately smaller input set ---
    so writing it over ``results/steering_report.json`` corrupts the checkout
    and fails the claim tests.  Artifact mode therefore writes beside the
    system temp directory unless ``--out`` says otherwise, and refuses the
    canonical path outright.
    """
    ap = argparse.ArgumentParser(description="Collect the steering paper's numbers.")
    ap.add_argument("--out", type=Path, default=None, help=f"output path (default {OUT})")
    args = ap.parse_args(argv)
    artifacts = bool(os.environ.get("KIJI_ARTIFACTS"))
    if args.out is None:
        if not artifacts:
            return OUT
        return Path(tempfile.gettempdir()) / "steering_report.artifacts.json"
    if artifacts and args.out.resolve() == OUT.resolve():
        raise SystemExit(
            "Refusing to overwrite the committed report from KIJI_ARTIFACTS mode.\n"
            "Pass a different --out, or unset KIJI_ARTIFACTS to rebuild from the run tree."
        )
    return args.out


def main(argv=None) -> None:
    out = _resolve_out(argv)
    report: dict = {"scenarios": {}}
    for scenario, demo_layer in SCENARIOS.items():
        pairs = _load(STEER / scenario / "pairs.json") or {}
        entry = {
            "demoLayer": demo_layer,
            "numPairs": len(pairs.get("pairs", [])),
            "pairsScored": (pairs.get("source") or {}).get("pairsScored"),
            "pairsFlipping": (pairs.get("source") or {}).get("pairsFlipping"),
            "layers": {str(l): layer_battery(scenario, l) for l in LAYERS},
            "dictionary": dictionary_health(scenario),
        }
        entry["layers"] = {k: v for k, v in entry["layers"].items() if v}
        if demo_layer is not None:
            entry["dose"] = dose(scenario, demo_layer)
            entry["generations"] = generations(scenario, demo_layer)
            entry["probes"] = probes(scenario)
            entry["positionAblation"] = position_ablation(scenario, demo_layer)
            # Depth and position could trade off: a cue that is inert at the
            # decision token early might still act at the request tokens.  The
            # same battery at an early layer is what rules that out, so keep it
            # beside the primary one rather than in a separate report.
            early = position_ablation(scenario, EARLY[-1])
            if early:
                entry["positionAblationEarly"] = {"layer": EARLY[-1], **early}
            entry["parity"] = parity(scenario, demo_layer)
        report["scenarios"][scenario] = entry

    # Rate-estimation sets: same gate, same battery, 32 sampled pairs each.
    for scenario, layer in EXPANDED.items():
        pairs = _load(STEER / scenario / "pairs.json")
        if not pairs:
            continue
        entry = {
            "primaryLayer": layer,
            "numPairs": len(pairs.get("pairs", [])),
            "sample": pairs.get("sample"),
            "pairsPassingGate": (pairs.get("source") or {}).get("pairsPassingGate"),
            "layers": {str(l): layer_battery(scenario, l) for l in LAYERS},
            "dictionary": dictionary_health(scenario),
            "parity": parity(scenario, layer),
        }
        entry["layers"] = {k: v for k, v in entry["layers"].items() if v}
        report["scenarios"][scenario] = entry

    # Held-out probe: same gate, same battery, on contrast axes the dictionaries
    # were never trained on.
    for scenario, layer in HELDOUT.items():
        pairs = _load(STEER / scenario / "pairs.json")
        if not pairs:
            continue
        entry = {
            "primaryLayer": layer,
            "numPairs": len(pairs.get("pairs", [])),
            "heldOut": True,
            "candidatesAuthored": (pairs.get("source") or {}).get("pairsScored"),
            "layers": {str(layer): layer_battery(scenario, layer)},
            "dictionary": dictionary_health(scenario),
            "parity": parity(scenario, layer),
        }
        entry["layers"] = {k: v for k, v in entry["layers"].items() if v}
        report["scenarios"][scenario] = entry

    # Aggregate: nothing moves in the first three quarters of the network.
    early = EARLY
    ab = cp = 0
    for s in SCENARIOS:
        for l in early:
            b = report["scenarios"][s]["layers"].get(str(l))
            if b:
                ab += b["ablationSides"]
                cp += b["crossPatchDirections"]
    report["earlyLayerTotals"] = {"layers": early, "ablationSides": ab, "crossPatchDirections": cp}

    report["stats"] = build_stats(report)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    try:
        print(f"wrote {out.relative_to(ROOT)}")
    except ValueError:  # written outside the repo, e.g. artifact mode's default
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
