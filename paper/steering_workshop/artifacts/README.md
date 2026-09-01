# Published run artifacts

Every canonical battery output the paper's numbers are computed from, copied
out of `demo/steering/<scenario>/output/`, which `.gitignore` excludes because
the full tree is 3.4 GB. These are the raw files, unmodified — same bytes, same
schema, one `output/` path segment shorter so no ignore rule applies.

45 files, 44 MB, across all thirteen scenarios:

| Scenario group | Layers | Backs |
|---|---|---|
| `supply_chain`, `customer_support`, `tool_selection` | 6, 13, 20, 27, 34, 43 | the depth grid, layer selection, the demo pages |
| `*_expanded` | 13, 34/43 | the paired cue-vs-dense 2×2, the rate estimates |
| `*_heldout` | 34/43 | the held-out nesting audit |
| `*_l27` | 27 | the non-selected late-layer comparison |
| `*_early` | 6, 20 | the early-layer arm of the depth contrast |
| `*_seed1` | 34/43 | the second draw |

Each battery directory holds a `steering_results.json` (per-direction ablation
and cross-patch outcomes, with every control draw), and where the arm was run,
a `ceiling_results.json` (full residual patch, equal-norm difference-in-means,
random directions) or a `trace_results.json`. Pair manifests are already
tracked at `demo/steering/<scenario>/pairs.json`.

All come from the same configuration: NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16
with the MTP head stripped, HF backend, JumpReLU threshold offset 1.12890625,
and each layer's `sae_final.pt` from the released checkpoint (see
`src/kiji_inspector/core/registry.py` for the Hugging Face repo).

## They are sufficient, not just present

`KIJI_ARTIFACTS=1` makes the extractor read this directory instead of the run
tree, so the claim is checkable:

```bash
KIJI_ARTIFACTS=1 python paper/steering/extract_results.py
```

That regenerates `paper/steering/results/steering_report.json` identically to
the committed copy, with one documented exception below.
`tests/test_steering_workshop_claims.py` runs it and asserts the equality, and
separately hashes each published file against the run it was copied from
whenever the full tree is present, so a stale copy fails rather than quietly
publishing numbers the report was not built from.

## What is not here

**The activation captures.** `capture/evaluation.json` is 478 MB across the
grid — 237 MB for `tool_selection` alone — so it stays out. It feeds only the
dictionary-health block (mean L0, explained variance, constant fraction); with
it absent that block is empty and every intervention claim still reproduces.
The health figures themselves remain published in the committed report.

**The gate sweep.** `demo/steering/sweep/output/` is 2.8 GB of candidate
scoring. The gate populations derived from it are in
`paper/steering/results/gate_population.json`.

**Re-run variants.** Directories suffixed `_setctl`, `_ctl2`, `_ctl3` and the
alternate `ceiling_layer43_matched` / `_v2` arms are working copies from
batches that added a control family. The report reads the canonical directory
unless `KIJI_SUFFIX` says otherwise, so publishing them would ship files no
quoted number comes from.
