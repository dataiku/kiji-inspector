# Published run artifacts

The ten battery outputs the workshop paper's numbers are computed from, copied
out of `demo/steering/<scenario>/output/`, which `.gitignore` excludes because
the full tree is 3.4 GB. These are the raw files, unmodified — same bytes, same
schema, one `output/` path segment shorter so no ignore rule applies.

| Scenario | Layer | Backs |
|---|---|---|
| `supply_chain_expanded` | 43 | the paired cue-vs-dense 2×2, the depth grid |
| `customer_support_expanded` | 34 | the same |
| `supply_chain_heldout` | 43 | the held-out nesting audit |
| `customer_support_heldout` | 34 | the same |
| `tool_selection` | 43 | the separate released split |

Each holds a `steering_layer<L>/steering_results.json` (per-direction ablation
and cross-patch outcomes, with every control draw) and a
`ceiling_layer<L>/ceiling_results.json` (the full residual patch, the
equal-norm difference-in-means arm and the random directions).

All ten come from the same configuration: NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16
with the MTP head stripped, HF backend, JumpReLU threshold offset 1.12890625,
and the layer's `sae_final.pt` from the released checkpoint (see
`src/kiji_inspector/core/registry.py` for the Hugging Face repo).

## They are sufficient, not just present

Every figure the workshop paper quotes for the paired comparison and the
held-out probes can be recomputed from this directory alone, with none of the
ignored run output:

```bash
KIJI_ARTIFACTS=1 python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('ex', 'paper/steering/extract_results.py')
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
print(ex.paired_cue_dense(ex.EXPANDED))
print(ex.heldout_overlap({'heldout': ex.HELDOUT, 'toolSelection': {'tool_selection': 43}}))
"
```

`tests/test_steering_workshop_claims.py` runs exactly that and asserts the
result equals `paper/steering/results/steering_report.json`, so the claim is
checked rather than asserted. It also hashes each file against the run it was
copied from whenever the full tree is present, so a stale copy fails.

## What is not here

The depth grid's early layers, the layer-27 comparison, the second-draw sample,
the dose and position sweeps, and the 2.8 GB gate sweep. The paper's dose,
position-ablation, paraphrase and generation claims are demo-scenario results
and are published in full — including the generated text — under `scenarios`
in `paper/steering/results/steering_report.json`.
