# Steering paper

LaTeX source for the causal-steering follow-up to [`../`](../)
("Opening the Black Box"), formatted as a NeurIPS 2026 submission — the style file and
checklist live in `neurips_2026_formatting/`, which `build.sh` puts on `TEXINPUTS` along with
`..`, so `\bibliography{references,steering_refs}` picks up the shared bibliography plus the
entries added here.

| File | What |
|---|---|
| `Steering Agent Tool Selection.tex` | The paper. |
| `neurips_2026_formatting/` | `neurips_2026.sty` and the filled-in `checklist.tex`. |
| `steering_refs.bib` | New references only; shared ones stay in `../references.bib`. |
| `extract_results.py` | Reads `demo/steering/*/output/` → `results/steering_report.json`. |
| `gate_population.py` | Streams the sweep corpora to rebuild the per-contrast-type gate populations → `results/gate_population.json`. Hard-fails if the total stops matching `pairs.json`. |
| `generate_charts.py` | Reads that JSON → `images/*.png`. |
| `build.sh` | Runs the extractor and charts, then compiles. |

## No number is hand-copied

Every figure quoted in the paper comes from `results/steering_report.json`, which
`extract_results.py` regenerates from the run artefacts under
[`../../demo/steering/`](../../demo/steering/). If a run is repeated, re-run the extractor and the
numbers in the report move with it. The report holds, per scenario:

- `layers.<L>` — directed and any-tool flip counts for ablation and cross-patch, best |Δp|,
  median control threshold, median base-active set size;
- `dictionary.<L>` — mean L0, explained variance, features constant across all prompts, and the
  median number of features that are side-specific within a pair;
- `dose`, `generations`, `probes`, `positionAblation`, `parity` at the scenario's demo layer;
- `positionAblationEarly` — the same position sweep at layer 20, which is what rules out the
  possibility that early layers are merely being intervened on in the wrong place.

The report also carries blocks the demo READMEs do not:

- `scenarios.<name>_expanded` — the rate-estimation runs: 32 pairs per scenario sampled from the
  full gate-passing population (`rank_flips.py --sample 32 --theme-cap 10 --seed 0`), run through
  the same battery at the primary layer and layer 13. The sample provenance (seed, population
  size, theme cap) is recorded in each `pairs.json` and echoed here.
- `scenarios.<name>_l27` / `_seed1` / `_early` are not separate report blocks — they are read
  through `stats.layerSelection` and the prose, since they exist to bound the primary numbers
  rather than to stand on their own.
- `stats` — every interval and test the paper quotes. All closed-form, no libraries:
  - `wilson`, `clusterBootstrapExpanded` — headline proportions, the latter resampling whole
    contrast-type clusters rather than sides. Stratified within scenario, because the design fixed
    32 pairs per scenario and pooling the ten clusters lets a resample vary that mix; `pooled`
    keeps the unstratified version as a check (it widens ablation to 0.40–0.66). Clusters are
    unequal in size, so stratifying the draws fixes how many *clusters* each scenario contributes,
    not how many pairs — `equalScenarioWeights` averages the two scenario rates instead, which
    does fix it, and is reported as a sensitivity rather than the headline because the point
    estimate is the pooled rate. Enumerated
    exactly — 35 count vectors for one scenario's four clusters, 462 for the other's six — so the
    endpoints carry no seed dependence. With ten clusters a percentile bracket still under-covers,
    so the paper treats these as uncalibrated ranges and the counts as the result.
  - `paired` / `pairedExpanded` — exceedance against each control family (`setMatched`,
    `deltaMatched`, `contrastMatched`), each split into genuinely matched sides and *ceiling* sides
    where the pool could not reach the target; `massAudit` records how much heavier a cue set is
    than the heaviest draw matched to one of its families; `rowLevel` holds the cue-vs-control
    Fisher test.
  - `outcomes` / `outcomesExpanded` — the full argmax partition (unchanged / directed / third tool)
    for both arms and for every stored control draw, plus the **runner-up audit**: on a minimal pair
    the other side's tool is usually the baseline runner-up, so this records how often that is true
    and whether flips still find it when it is not.
  - `recovery` / `recoveryExpanded` — flip counts divided by the dictionary-free ceiling from
    `ceiling_pairs.py`, with the difference-in-means and random arms at both norms.
  - `designWeighted` — the same interventions re-weighted to the gate population by stratum share,
    so the quota rate can be read against a population one.
  - `layerSelection` — which layer the grid's argmax picks per scenario and by how much, plus the
    layer-27 rerun that gives a non-selected comparison.
  - `depth` — the early-vs-late Fisher test with the exact upper bound on the early-layer flip rate,
    computed twice: once against the selected layers and once against layer 27.
  - `pairedCueDense` — the full cue × equal-norm-dense 2×2 over the expanded directions where the
    dense direction is defined, with exact bootstrap brackets. Enumerated rather than sampled: eight
    clusters give only 1,225 stratified count vectors, and the 97.5th percentile sits at cumulative
    weight 0.9752, so a Monte-Carlo endpoint lands either side of a boundary atom depending on the
    seed. Each count vector is weighted by the number of ordered draws producing it.
  - `heldoutOverlap` — whether the cue redirects are a *subset* of the directions the full residual
    patch reaches, and how many axes and pairs they come from, so a flip ratio carried by one axis
    cannot read as breadth. Reported for the held-out pair and for the separate tool-selection split.

Five definitions are worth knowing when reading it:

- A **directed flip** is an argmax change that lands on the donor's tool (for ablation, the other
  side's tool). The looser any-tool count is kept alongside as `*FlipsAnyTool`; the two disagree in
  5 of the grid's 36 cells. The paper and the demo READMEs both quote the directed count.
- A **keyword control** is the *other* side's request with this side's cue word slipped in
  inertly, so it is correct when the model keeps the *other* side's tool.
- A **ceiling** appears in two unrelated senses, both deliberate. A *control* ceiling is a draw that
  could not reach its target mass, so it is the whole eligible pool rather than a matched sample
  (`massMatched: false`). The *ceiling arm* is the full residual patch of `ceiling_pairs.py`, which
  bounds what any decomposition read at that token could do.
- A **recovery fraction** is a flip count over that ceiling. It is the form the paper reports its
  headline in, because a flip rate on its own has no denominator.
- A **redirect** is a directed flip with the baseline requirement made explicit: the recipient must
  not already choose the target, so a direction the model gets right unaided cannot be scored as a
  success for any arm. `pairedCueDense` and `heldoutOverlap` apply it to every arm they compare.

## Regenerating from a partial re-run

A battery re-run that adds a field is written to a suffixed directory (`steering_layer43_setctl`,
`_ctl2`, …) so the canonical artefacts are never overwritten mid-batch. Set `KIJI_SUFFIX` to read
those in preference, falling back per battery to the canonical directory for anything not yet
re-run:

```bash
KIJI_SUFFIX=_setctl python3 paper/steering/extract_results.py
```

A suffixed directory that exists but holds no JSON yet is treated as still-running and skipped, so a
report generated mid-batch is complete rather than silently missing a battery. Once the re-run is
verified, promote it over the canonical directory and drop the variable.

## Build

```bash
./build.sh            # extract → charts → pdflatex ×3 (color)
./build.sh bw         # same, B&W figures
```

`build.sh` uses a local `pdflatex` if there is one and otherwise falls back to the
`texlive/texlive` Docker image (override with `TEX_IMAGE=`), so no TeX installation is required —
only Docker. If neither is present it still refreshes the results and figures, then stops with a
note. Chart generation uses `uv run --no-project --with matplotlib`, so there is no project
dependency either.

After the three passes it greps the log for errors, overfull boxes and undefined
references/citations before deleting it, and exits non-zero on a LaTeX error. A clean run prints
only the `✓` line.

Charts are authored at roughly the print text width with font sizes set once via
`rcParams`. Authoring them wider and letting `\includegraphics[width=\textwidth]` shrink them is
what makes axis labels come out at 4 pt — if you change a `figsize`, check the rendered PDF, not
the PNG.
