# customer_support_heldout — held-out probe (no page)

Pairs written for this probe on **contrast axes that do not occur in the corpus the SAEs were
trained on**. The dictionaries are fitted on decision-token activations from
`575-lab/kiji-inspector-pairs` (2,011,672 rows over five scenarios and 37 contrast types), and the
paper's other evaluation sets are sampled from that same corpus — 136 of the 150 requests they use
appear in it verbatim. This directory exists to check what happens on prompts the dictionary has
never seen.

## What makes it held out

- Four new contrast axes, none among the corpus's 37: **published spec vs how-to guidance**,
  **prior orders vs open invoice**, **many reporters vs one reporter**,
  **warranty coverage vs repair guidance**.
- 48 authored pairs → 96 distinct requests, verified to have **zero exact matches** against all
  2.0M rows of the training corpus.
- 0.55 is the customer-support Jaccard threshold used throughout the paper, because its contrasts
  are rephrasings rather than word swaps; it was not relaxed for this probe.

## Selection

The pairs go through the **same** stage-2 sweep and the **same** gate as every other set in the
paper — no threshold was relaxed for them:

    flip ≥ 0.6 on both sides · weaker side < 0.8 · content-word Jaccard ≥ 0.55 · no tool named

48 authored pairs → 25 flip the tool → **7 pass the full gate**, spread over three of the four axes.

## Results

| layer | ablation flips | cross-patch flips |
|---|---:|---:|
| **34** | **5 / 14** (0.36, Wilson 95% [0.16, 0.61]) | **4 / 14** (4 cue + 0 bulk; 0.29 [0.12, 0.55]) |

Median cue-set |Δp| against that side's own set-matched control band: **5.302×** (96% of sides exceed their band). In-distribution at the same layer: 7.236×.

### Against the ceiling

`ceiling_pairs.py` patches the donor's whole residual into the recipient's decision token, with no
dictionary in the path, which bounds what any decomposition read there could do:

| | held out | in distribution (customer_support_expanded) |
|---|---:|---:|
| ceiling | **10 / 14** (0.71) | 39 / 64 (0.61) |
| cue set | 4 / 14 | 13 / 64 |
| **recovery** (cue / ceiling) | **0.40** | **0.33** |
| difference-in-means, clamp norm | 3 / 14 | — |
| random, either norm | 0 of 42, 0 of 42 | — |

The ceiling is what makes this comparison mean anything: a lower flip rate off-distribution could be
weaker features or simply less decision left to move at that token, and only the denominator
separates them. Here more of the decision is available off-distribution than in it (0.71 against 0.61), and
the cue features recover a larger share of it — 0.40 against 0.33.

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
| `output/capture/` | Decisions + residuals at layer 34. |
| `output/steering_layer34/` | The battery. |

Candidates and sweep live in `demo/steering/sweep/output/sweep_candidates/customer_support_heldout/`.
`scenarios/customer_support_heldout.json` is a copy of `scenarios/customer_support.json`, so the tools and
system prompt are identical to the in-distribution runs.

> Note: both sweep scripts compute `_REPO_ROOT = _DEMO_DIR.parents[1]`, which resolves to `demo/`
> rather than the repository root, so they look for `demo/scenarios/<name>.json`. A copy of the
> scenario config is kept there until that is fixed.
