# Supply chain — one word, one flip

Four pairs of supply-chain requests that differ by a **single word**, where that word changes which
tool the agent reaches for — and where clamping the other side's SAE features into the request,
with the prompt untouched, moves the decision across.

This is the decision-level companion of [`../../tool_selection/`](../../tool_selection/). That demo found
most of its pairs *descriptive*: the two top-ranked pairs were saturated (p = 1.00 both sides),
switching all cue families off moved them ≤ 1 pp, and the dose curve needed 3× before p(other tool)
reached 0.5. Here the pairs were selected to avoid that failure mode — **only pairs whose weaker
side sits below p = 0.8** qualify — and the causal picture is correspondingly stronger:
**7 of 8 sides flip under ablation, 3 of 8 cross-patch directions flip the recipient's tool, and two
dose curves cross 0.5 at 1×.**

## The pairs

Selected by rule from a sweep of all 378,490 clean `supply_chain` training pairs (`rank_flips.py
--select-demo`), not by hand: one pair per contrast type, at most two per tool combination, no tool
named in the request, flip ≥ 0.6, weaker side < 0.8, and lexical overlap ≥ 0.7 so the pair really is
a one-cue edit.

| theme | J | cue | A | B |
|---|---|---|---|---|
| `demand_pull_vs_supply_push` | 0.88 | *(nothing)* vs **forecasted** | inventory_manager 0.98 | demand_forecaster 0.80 |
| `local_vs_global_sourcing` | 0.82 | **local** vs **global** | supplier_database 0.95 | shipment_tracker 0.80 |
| `cost_vs_speed_optimization` | 0.78 | **economical** vs **express** | route_optimizer 0.83 | shipment_tracker 0.79 |
| `single_source_vs_multi_source` | 0.70 | **our main supplier** vs **all suppliers** | supplier_database 0.71 | inventory_manager 0.83 |

The first is the sharpest: side A has **no cue token at all**. B inserts one word into an otherwise
identical sentence, so there is no lexical anchor on the A side for a feature to latch onto.

**Two contrast types are deliberately absent.** Across all ~10 k qualifying pairs each,
`spot_buy_vs_contract` peaks at J = 0.47 and `just_in_time_vs_safety_stock` at J = 0.36 — their
contrast type is written as structurally different sentences, so no one-cue pair exists at any
threshold. Including them would have broken the page's premise; `--min-jaccard 0.7` drops them and
says so.

## What the run shows (layer 43)

**Ablation — switching a side's cue families off at the decision token.** 7 of 8 sides flip:

| pair · side | Δp(target) | control band |
|---|---:|---:|
| demand_pull · b | **−0.847** | 0.034 |
| single_source · b | **−0.666** | 0.035 |
| local_vs_global · b | **−0.633** | 0.089 |
| local_vs_global · a | **−0.585** | 0.037 |
| demand_pull · a | **−0.498** | 0.032 |
| cost_vs_speed · b | −0.405 | 0.205 |
| single_source · a | −0.281 | 0.183 |
| cost_vs_speed · a | −0.259 (no flip) | 0.123 |

The top five beat mass-matched random ablations by 10–25×.

**Cross-patch — clamping one side's cue families into the other side's request.** The prompt is not
changed; only feature activations at the decision token are. Three directions flip:

```
demand_pull     a→b   +0.583 (all families) / +0.745 (all base-active)   control 0.054   FLIPS
local_vs_global a→b   +0.532 / +0.575                                    control 0.021   FLIPS
single_source   b→a   +0.199 / +0.226                                    control 0.025   FLIPS
```

**Dose-response.** Two curves cross 0.5 at exactly **1×** — clamping to the value the feature takes
naturally on the donor request is sufficient, with no amplification:

```
demand_pull     a→b   0.16 → 0.73 (1×) → 1.00 (3×)     crosses at 1.0×
local_vs_global a→b   0.12 → 0.75 (1×) → 1.00 (3×)     crosses at 1.0×
single_source   b→a   0.03 → 0.49 (1×) → 0.70          crosses at 1.5×
```

**Depth.** The full ablation + cross-patch battery was run at all six SAE layers. Layers 6–20 are
inert — not weak, *zero* flips, best |Δp| below 0.06 against control bands of ~0.02:

| layer | cross-patch flips | ablation flips | best Δp |
|---|---:|---:|---:|
| 6 | 0 / 8 | 0 / 8 | 0.025 |
| 13 | 0 / 8 | 0 / 8 | 0.025 |
| 20 | 0 / 8 | 0 / 8 | 0.053 |
| 27 | 0 / 8 | 0 / 8 | 0.230 |
| 34 | 2 / 8 | 1 / 8 | 0.503 |
| **43** | **3 / 8** | **7 / 8** | **0.745** |

All three cross-patch flips here come from the six cue families; clamping every base-active feature
instead adds nothing. That is what a healthy layer looks like — compare `../customer_support/`,
where layer 43 flips 9 / 12 but 8 of those need all 1,162 active features clamped.

**Where the cue lives.** Ablating at the *request* tokens changes almost nothing (0.85 → 0.85);
ablating at the *decision token alone* does the whole job. Layer 27 barely moves (0.85 → 0.41 at
best, usually unchanged) — description early, causal leverage late, matching the spec sheet's depth
curve.

**What the model writes.** Two directions flip the tool name in the generated text. The cleanest,
from an identical prompt with 7 activations clamped:

```
baseline: "shipment_tracker tool to analyze the delivery performance of your global
           electronic components supplier over the last quarter…"
steered : "supplier database to look up the supplier information and then use the
           shipment tracker to get delivery performance data…"
control : byte-identical to baseline
```

The model names a different first tool and restructures its plan. `demand_pull a→b` also flips
(`demand forecasting tool` → `inventory_manager`), though its control drifted slightly in wording
while keeping the same tool.

**Is it the word or the meaning? The meaning — narrowly.** The keyword controls are 3/3 correct:
the cue word present but semantically inert never moved the decision (`"…rising customer order
frequency, using the cost codes from the forecasted budget approved last week"` stays
`inventory_manager`). So it is not token matching. But paraphrases held only 4/6 —
`"projected to rise"` held, `"expected to be ordered more often"` fell back — so the feature tracks
a fairly narrow sense of *forecast framing of the demand signal* rather than forecast-in-general.

## Caveats

- **`cost_vs_speed` is the weak pair.** Wide control bands (0.12–0.21), side A does not flip under
  ablation, and neither cross-patch direction flips. Three independent signals agree; treat it as
  inconclusive rather than load-bearing.
- **Patching is directional.** `a→b` succeeds where `b→a` fails on both top pairs, and
  `local_vs_global b→a` never crosses 0.5 even at 3×. Adding the *forecasted* / *global* reading is
  easy; removing it is hard.
- **The text flips are confirmation, not independent evidence.** The two directions that flip in
  generation are exactly the two that already crossed 0.5 at 1× in the dose curve.
- **Two theme maps in the shipped SAE are nearly empty** — `cost_vs_speed_optimization` has
  `num_pairs: 3` and `spot_buy_vs_contract` 1,281 in `contrastive_features.json`, versus ~66 k for
  the healthy ones — so theme-evidence bars for those are not meaningful.
- HF/vLLM parity: residual cosine 0.999 (min 0.998), same first tool on 8/8 requests.

## Note: the published `tool_selection` demo used a different SAE

`../../tool_selection/index.html` embeds feature data that does not correspond to the SAE release
used here, so its indices, labels and L0 figures **must not be compared with the numbers below**.
See [`../README.md`](../README.md) for the measurements. Everything in this demo comes from the
shipped SAE (`output/layer_<N>/`) and the current sweep, so it is internally consistent.

## Files

| File | Purpose |
|---|---|
| `pairs.json` | The four pairs, emitted by `rank_flips.py --select-demo` from the sweep. |
| `probes.json` | Hand-written paraphrases and keyword controls (23 prompts), validated against the same assertions `tests/test_tool_selection_demo.py` applies. |
| `index.html` | The page (reads `output/ui_data.json`, embedded fallback). |
| `output/` | `capture/`, `steering_layer43/`, `trace_layer43/`, `ui_data.json`. |

The drivers live in `../../tool_selection/` and are scenario-general — pass `--scenario supply_chain`.

## Run

```bash
DOCKER="docker run --rm --gpus all -v $PWD:/workspace -v /path/to/models:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace/demo/tool_selection -w /workspace 575lab/kiji-inspector:dev"
MODEL=/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp

# 0) (re)select the pairs from an existing sweep — CPU
uv run python demo/steering/sweep/rank_flips.py \
  --meta  demo/steering/sweep/output/sweep_candidates/supply_chain/meta.json \
  --sweep demo/steering/sweep/output/sweep_candidates/supply_chain/sweep.jsonl \
  --exclude-tool-named --max-side-prob 0.8 --min-jaccard 0.7 --select-demo --top 0 \
  --emit-pairs demo/steering/supply_chain/pairs.json

# 1) capture (vLLM, ~2 min)
$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL --scenario supply_chain

# 2) ablation + cross-patch (HF, ~2 min)
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/attribute_pairs.py --model-name '"$MODEL"' --scenario supply_chain \
    --layer 43 --activations demo/steering/supply_chain/output/capture/activations.npz'

# 3) cue map, dose-response, steered generations (HF, ~2.5 min)
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/trace_pairs.py --model-name '"$MODEL"' --scenario supply_chain \
    --layer 43 --compare-layers 27'

# 4) build the page payload
uv run python demo/tool_selection/tool_selection_demo.py --scenario supply_chain --layer 43 \
  --steering demo/steering/supply_chain/output/steering_layer43/steering_results.json \
  --trace    demo/steering/supply_chain/output/trace_layer43/trace_results.json
uv run python demo/home_repair/embed_ui_data.py \
  --ui-data demo/steering/supply_chain/output/ui_data.json --html demo/steering/supply_chain/index.html

python -m http.server 8001   # open http://localhost:8001/demo/steering/supply_chain/index.html
```

Needs the trained SAE under `output/layer_<N>/` (checkpoint + `feature_descriptions.json` +
`contrastive_features.json`). Note these must be **real files, not symlinks into the HF cache** —
the container mounts the cache elsewhere, so an absolute symlink dangles inside it.
