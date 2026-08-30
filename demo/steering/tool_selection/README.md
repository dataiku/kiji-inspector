# tool_selection — layer study (no page)

The seven pairs published in `../../tool_selection/` re-run against the **shipped** SAE
(`output/layer_<N>/`), so that scenario appears in this folder's cross-layer comparison on the same
footing as `../supply_chain/` and `../customer_support/`.

There is deliberately no `index.html` here. This is a measurement, not a demo: the published page
in `../../tool_selection/` is built from a different dictionary and is left untouched.

`pairs.json` and `probes.json` are copies of the published demo's, so the prompts are identical —
which is what makes the comparison meaningful. The model agrees with that page on the tool
**14 / 14**, with probabilities within 0.05; what differs is the dictionary.

| | published page | shipped SAE (here) |
|---|---:|---:|
| mean L0 at layer 43 | 84 | 374 |
| its 5 displayed features for `internal_vs_external_A`, active? | — | 0 / 5 |

## Results

| layer | cross-patch flips | ablation flips |
|---|---:|---:|
| 6 | 0 / 14 | 0 / 14 |
| 13 | 0 / 14 | 0 / 14 |
| 20 | 0 / 14 | 0 / 14 |
| 27 | 1 / 14 | 0 / 14 |
| 34 | 2 / 14 | 0 / 14 |
| **43** | **5 / 14** | **3 / 14** |

Counts are **directed** flips, the definition used throughout this repo and the paper: the argmax
moved *and* landed on the other side's tool. The looser any-tool count is 1/14 at layer 34 and 5/14
at layer 43, so two of the layer-43 ablation argmax changes land on a third tool.

Cross-patch and ablation stay within a factor of 1.7 at layer 43 (5 and 3), which is the healthy
signature — see [`../README.md`](../README.md). The zero-ablation cells above are *uninformative*
for that check rather than failures: the ratio is undefined where the ablation arm never flips.

Note the dictionary here is dense — mean L0 374 at layer 43 and 633 at layer 34, against a target of
75 — and it works anyway, retaining 22 and 69 side-specific features per pair. Density is not
collapse; the disappearance of the side-specific column is.

### Against the ceiling

`ceiling_pairs.py` patches the donor's whole residual into the recipient's decision token —
activation patching in the model's own basis, no dictionary in the path — which bounds what any
decomposition read there could do. On these 14 directions it flips **13 / 14**, against **2 / 14** for
the cue set and **5 / 14** for every donor-active feature: a recovery of **0.15** and **0.38**.
Random directions at the same norm flip 0 of 42. Difference-in-means needs several pairs per contrast
type, so it is undefined here and reported on the `*_expanded` sets instead.

## Files

| Path | What |
|---|---|
| `pairs.json`, `probes.json` | Copies of the published demo's, unmodified. |
| `output/capture/` | Decisions + residuals at all six layers. |
| `output/steering_layer<L>/` | Ablation + cross-patch battery, one per layer. |

## Run

```bash
$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL --scenario tool_selection
for L in 6 13 20 27 34 43; do
  $DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
    python demo/tool_selection/attribute_pairs.py --model-name '"$MODEL"' --scenario tool_selection \
      --layer '"$L"' --activations demo/steering/tool_selection/output/capture/activations.npz'
done
```

`--scenario tool_selection` resolves here, not to the published demo — see `../README.md`.
