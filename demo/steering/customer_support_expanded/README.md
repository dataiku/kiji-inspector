# customer_support_expanded — rate estimation (no page)

Thirty-two pairs sampled from the **full gate-passing population** of the customer-support sweep
(1,616 pairs pass flip ≥ 0.6, weaker side < 0.8, J ≥ 0.55, no tool named), run through the same
battery as [`../customer_support/`](../customer_support/) at its primary layer 34 and at layer 13.
Sampled uniformly at random (`--sample 32 --seed 0`) with at most ten pairs per contrast type —
uncapped, 28 of 32 draws land on `billing_vs_technical`, whose pairs are template near-duplicates.

No `index.html`, no probes, no trace: a measurement for `paper/steering/`, not a demo.

## Results (64 sides, 64 directions per layer)

| layer | ablation flips | cross-patch flips (cue + bulk) |
|---|---:|---:|
| 13 | 1 / 64 † | 0 / 64 |
| **34** | **28 / 64** (0.44, Wilson 95% [0.32, 0.56]) | **18 / 64** (13 + 5; 0.28 [0.19, 0.40]) |

† The one early flip is a near-tied baseline (0.55 vs 0.43) tipped by Δp = −0.18 against a
control threshold of 0.12 — reported in the paper as 1/460 early interventions overall.

Counts are **directed** flips — the definition used everywhere in this repo and in the paper.
The looser any-tool count at layer 34 is 41/64 ablation and 22/64 cross-patch, so 13 of the
ablation argmax changes here land on a third tool rather than the paired one. Baseline capture
agrees with the sweep on 64/64 sides; parity mean cosine 0.998 (min 0.987).

### Against the ceiling

`ceiling_pairs.py` bounds what any decomposition read at the decision token could do, by patching
the donor's whole residual there with no dictionary in the path:

| arm | flips | of the ceiling |
|---|---:|---:|
| full residual patch (ceiling) | 39 / 64 | — |
| difference-in-means, difference norm | 29 / 60 | 0.74 |
| all donor-active features (bulk) | 18 / 64 | 0.46 |
| difference-in-means, clamp norm | 14 / 60 | 0.36 |
| cue set | 13 / 64 | **0.33** |
| random direction, either norm | 2 / 192, 0 / 180 | 0.05, 0.00 |

The ceiling is much lower here than in supply chain (39/64 against 61/64): on 25 directions a full
transplant moves p(donor's tool) a long way without crossing, so the decision is partly re-derived
downstream rather than settled where we intervene. Normalising by it reverses which scenario looks
stronger — the cue set flips fewer directions here than in supply chain (0.20 against 0.25) but
recovers a larger share of what is available (0.33 against 0.26).

Intervals, paired tests and pooled rates: `paper/steering/results/steering_report.json`
(`scenarios.customer_support_expanded`, `stats`).

## Files

| Path | What |
|---|---|
| `pairs.json` | The 32 pairs; `sample` records seed, theme cap and population size. |
| `output/capture/` | Decisions + residuals at layers 13 and 34. |
| `output/steering_layer{13,34}/` | The batteries. |

`scenarios/customer_support_expanded.json` is a copy of `scenarios/customer_support.json` so the
shared drivers resolve this directory via `--scenario customer_support_expanded`.

## Run

As in [`../supply_chain_expanded/README.md`](../supply_chain_expanded/README.md), with
`--scenario customer_support_expanded`, `--min-jaccard 0.55` and layers `13 34`.
