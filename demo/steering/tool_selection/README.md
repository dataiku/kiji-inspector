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
| 34 | 2 / 14 | 1 / 14 |
| **43** | **5 / 14** | **5 / 14** |

Cross-patch and ablation agree at layer 43 (5 and 5), which is the healthy signature — see
[`../README.md`](../README.md).

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
