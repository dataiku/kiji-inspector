# supply_chain_heldout — held-out probe (no page)

Pairs written for this probe on **contrast axes that do not occur in the corpus the SAEs were
trained on**. The dictionaries are fitted on decision-token activations from
`575-lab/kiji-inspector-pairs` (2,011,672 rows over five scenarios and 37 contrast types), and the
paper's other evaluation sets are sampled from that same corpus — 136 of the 150 requests they use
appear in it verbatim. This directory exists to check what happens on prompts the dictionary has
never seen.

## What makes it held out

- Four new contrast axes, none among the corpus's 37: **warehouse vs transit capacity**,
  **vendor defect record vs pricing record**, **recorded vs projected consumption**,
  **carrier distance vs carrier price**.
- 48 authored pairs → 96 distinct requests, verified to have **zero exact matches** against all
  2.0M rows of the training corpus.
- Each pair is a single content-word swap in a long frame, so the selection gate's Jaccard
  requirement is met without loosening it (authored J: 0.83–0.88).

## Selection

The pairs go through the **same** stage-2 sweep and the **same** gate as every other set in the
paper — no threshold was relaxed for them:

    flip ≥ 0.6 on both sides · weaker side < 0.8 · content-word Jaccard ≥ 0.7 · no tool named

48 authored pairs → 20 flip the tool → **8 pass the full gate**, spread over three of the four axes.

## Results

| layer | ablation flips | cross-patch flips |
|---|---:|---:|
| **43** | **4 / 16** (0.25, Wilson 95% [0.10, 0.49]) | **3 / 16** (3 cue + 0 bulk; 0.19 [0.07, 0.43]) |

Median cue-set |Δp| against that side's own set-matched control band: **4.597×** (71% of sides exceed their band). In-distribution at the same layer: 2.720×.

### Against the ceiling

`ceiling_pairs.py` patches the donor's whole residual into the recipient's decision token, with no
dictionary in the path, which bounds what any decomposition read there could do:

| | held out | in distribution (supply_chain_expanded) |
|---|---:|---:|
| ceiling | **15 / 16** (0.94) | 61 / 64 (0.95) |
| cue set | 3 / 16 | 16 / 64 |
| **recovery** (cue / ceiling) | **0.20** | **0.26** |
| difference-in-means, clamp norm | 4 / 14 | — |
| random, either norm | 0 of 48, 0 of 42 | — |

The ceiling is what makes this comparison mean anything: a lower flip rate off-distribution could be
weaker features or simply less decision left to move at that token, and only the denominator
separates them. Here the ceiling is essentially the same as in distribution (0.94 against 0.95), so the
lower recovery — 0.20 against 0.26 — is a real if small shortfall on 16 directions, not an artefact
of how much signal was available.

Pooled over both held-out scenarios the recovery is **0.28**, against 0.29 in distribution — the
statement the paper makes, and a stronger one than the raw flip rates support on their own.

## Caveat

We chose the axes and wrote the sentences, so this is a **probe, not a second rate estimate**: its
denominator reflects our drafting rather than a population. What it addresses is the specific worry
that the reported effects depend on the dictionary having been fitted on these exact prompts.

## Files

| Path | What |
|---|---|
| `pairs.json` | The 8 gated pairs, emitted by `rank_flips.py` with the flags above. |
| `output/capture/` | Decisions + residuals at layer 43. |
| `output/steering_layer43/` | The battery. |

Candidates and sweep live in `demo/steering/sweep/output/sweep_candidates/supply_chain_heldout/`.
`scenarios/supply_chain_heldout.json` is a copy of `scenarios/supply_chain.json`, so the tools and
system prompt are identical to the in-distribution runs.

> Note: both sweep scripts compute `_REPO_ROOT = _DEMO_DIR.parents[1]`, which resolves to `demo/`
> rather than the repository root, so they look for `demo/scenarios/<name>.json`. A copy of the
> scenario config is kept there until that is fixed.
