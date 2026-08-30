# supply_chain_expanded — rate estimation (no page)

Thirty-two pairs sampled from the **full gate-passing population** of the supply-chain sweep
(1,094 pairs pass flip ≥ 0.6, weaker side < 0.8, J ≥ 0.7, no tool named), run through the same
capture + ablation + cross-patch battery as [`../supply_chain/`](../supply_chain/). The demo's four
pairs are top-scored — right for a page, biased upward for a rate — so this set is drawn
uniformly at random instead (`--sample 32 --seed 0`), capped at ten pairs per contrast type
because within-type pairs are template near-duplicates and the population is 92% one type.

There is deliberately no `index.html`, no probes and no trace stage: this is a measurement for
the paper's rate claims (`paper/steering/`), not a demo.

## Results (64 sides, 64 directions per layer)

| layer | ablation flips | cross-patch flips (cue + bulk) |
|---|---:|---:|
| 13 | 0 / 64 | 0 / 64 |
| **43** | **40 / 64** (0.63, Wilson 95% [0.50, 0.73]) | **23 / 64** (16 + 7; 0.36 [0.25, 0.48]) |

Counts are **directed** flips (argmax lands on the other side's / donor's tool), the definition
used everywhere in this repo and in the paper. Here the two definitions coincide: no intervention on
these pairs moved the argmax onto a third tool. Baseline capture agrees with the sweep's tool on
64/64 sides; HF/vLLM parity mean cosine 0.999 (min 0.997).

### Against the ceiling

Flip counts have no denominator on their own, so `ceiling_pairs.py` patches the donor's whole
residual into the recipient's decision token — activation patching in the model's own basis, no
dictionary in the path — to bound what any decomposition read there could do:

| arm | flips | of the ceiling |
|---|---:|---:|
| full residual patch (ceiling) | 61 / 64 | — |
| difference-in-means, difference norm | 50 / 64 | 0.82 |
| all donor-active features (bulk) | 23 / 64 | 0.38 |
| difference-in-means, clamp norm | 16 / 64 | 0.26 |
| cue set | 16 / 64 | **0.26** |
| random direction, either norm | 0 / 192, 0 / 192 | 0.00 |

At the norm of the cue clamp's own residual change the dense difference-in-means direction and the
sparse cue set are indistinguishable, so what the SAE basis costs is nothing and what it buys is an
intervention that can be named.

All intervals, paired control tests and the pooled rates live in
`paper/steering/results/steering_report.json` under `scenarios.supply_chain_expanded` and `stats`
(regenerate with `python paper/steering/extract_results.py`).

## Files

| Path | What |
|---|---|
| `pairs.json` | The 32 pairs; `sample` records seed, theme cap and population size. |
| `output/capture/` | Decisions + residuals at layers 13 and 43. |
| `output/steering_layer{13,43}/` | The batteries. |

`scenarios/supply_chain_expanded.json` is a copy of `scenarios/supply_chain.json` so the shared
drivers resolve this directory via `--scenario supply_chain_expanded`.

## Run

```bash
# select (CPU)
uv run python demo/steering/sweep/rank_flips.py \
  --meta  demo/steering/sweep/output/sweep_candidates/supply_chain/meta.json \
  --sweep demo/steering/sweep/output/sweep_candidates/supply_chain/sweep.jsonl \
  --exclude-tool-named --max-side-prob 0.8 --min-jaccard 0.7 \
  --sample 32 --seed 0 --theme-cap 10 --top 0 \
  --emit-pairs demo/steering/supply_chain_expanded/pairs.json

# capture + battery (one H100; ~2 min + ~3.5 min per layer)
$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL \
  --scenario supply_chain_expanded --no-probes --layers 13 43
for L in 43 13; do
  $DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
    python demo/tool_selection/attribute_pairs.py --model-name '"$MODEL"' \
      --scenario supply_chain_expanded --layer '"$L"' \
      --activations demo/steering/supply_chain_expanded/output/capture/activations.npz'
done
```

`$DOCKER` and `$MODEL` as in [`../supply_chain/README.md`](../supply_chain/README.md).
