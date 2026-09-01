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
# wrote /tmp/steering_report.artifacts.json
```

It regenerates the whole of `paper/steering/results/steering_report.json`,
byte for byte, from the published files alone. Artifact mode writes to a temp
path rather than the committed report — that report is built from the full run
tree and overwriting it from a smaller input set would corrupt the checkout —
and refuses `--out` pointing at it. Pass `--out` to put the result elsewhere.
`tests/test_steering_workshop_claims.py` runs it and asserts the equality, and
separately hashes each published file against the run it was copied from
whenever the full tree is present, so a stale copy fails rather than quietly
publishing numbers the report was not built from.

## Which weights

`provenance.json` records the exact revisions and checksums behind these runs:
the SAE repository at commit `2380c95c`, the base checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16` at `d468880b`, SHA-256 for all
six SAE checkpoints and all 27 files of the stripped base checkpoint. The
interventions ran against that base commit, verified by inode against the
Hugging Face cache. The dictionaries were fitted on the same weights, which the
SAE model card records under their pre-release name
`ga_nvidia_nemotron_3_5_nano_bf16_07292026_vv0.1` at code revision `65ae86b`.

`hf_hub_download` resolves `main` unless told otherwise, so naming a repository
does not identify what a rerun loads. That is not a hypothetical risk here:
upstream `main` for the base model has moved past the commit these results were
produced against. `src/kiji_inspector/core/registry.py` now pins every
registered repo and the loader sends the pin by default; pass
`revision="main"` to opt back into the head.

The stripped base checkpoint is a thin derivation: `strip_mtp` rewrites only
`config.json` and `model.safetensors.index.json` and hardlinks every weight
shard, so the shards carry upstream's own checksums and the record says which
two files are ours. Regenerate with:

```bash
python paper/steering_workshop/provenance.py \
    --model-dir ~/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp \
    --hf-cache ~/.cache/huggingface/hub
```

## Dictionary health

The health screen — mean L0, explained variance, and how much of the code never
varies across the prompts — is a prerequisite for the paper's reading, not a
detail: a layer whose code is almost entirely constant has nothing for a cue
analysis to work with, however well it reconstructs. So it has to be
recomputable here, not taken on trust.

The captures it reads are 478 MB across the grid and stay out (below), but the
screen needs very little of them. `<scenario>/capture/health_inputs.json`
carries exactly what it reads — per pair prompt, the positive-activation
feature ids and L0, plus per-prompt explained variance — in 0.9 MB for all
thirteen scenarios, against 500 MB of captures. Feature ids only, no
activations, so no capture can be reconstructed from them. Regenerate with
`python paper/steering_workshop/health_inputs.py`.

## What is not here

**The activation captures.** `capture/evaluation.json` is 478 MB across the
grid — 237 MB for `tool_selection` alone — so it stays out; `health_inputs.json`
stands in for the one analysis that reads it. Nothing else in the report
touches the captures.

**The gate sweep.** `demo/steering/sweep/output/` is 2.8 GB of candidate
scoring. The gate populations derived from it are in
`paper/steering/results/gate_population.json`.

**Re-run variants.** Directories suffixed `_setctl`, `_ctl2`, `_ctl3` and the
alternate `ceiling_layer43_matched` / `_v2` arms are working copies from
batches that added a control family. The report reads the canonical directory
unless `KIJI_SUFFIX` says otherwise, so publishing them would ship files no
quoted number comes from.
