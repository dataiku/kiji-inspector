# Customer support — same words, different framing

Six pairs of customer-support requests where the tool the agent picks turns on **how the request is
framed**, not on which words it uses — and where clamping the other side's SAE features into the
request, prompt untouched, moves the decision across.

This is the counterpart to [`../supply_chain/`](../supply_chain/). There the cue is a single word
(*forecasted*, *local*/*global*). Here the sharpest pair contains the **same content words in a
different order**:

```
I was billed for a feature I can't find in my account   ->  billing_system    0.71
I can't find the feature I was billed for in my account ->  customer_history  0.79
```

Jaccard = 1.00. There is no word present on one side and absent on the other; only the order, and
therefore what the sentence is *about*, differs. Three of the six themes have no distinguishing
word at all.

## Read at layer 34, not 43

**Layer 43 is degenerate for this scenario.** 1,162 features fire on *every* prompt and only 32
vary across all twelve — pair 0 has exactly one side-specific feature per side, leaving nothing for
a cue analysis to work with. This does not happen in `supply_chain` (16 constant, 71 varying) and
is worth knowing before reusing the other demos' layer:

| layer | mean L0 | active in all prompts | varying |
|---|---:|---:|---:|
| 20 | 84 | 68 | 107 |
| 27 | 53 | 16 | 297 |
| **34** | **73** | **42** | **181** |
| 43 | 1167 | 1162 | **32** |

All six layers were run through the full ablation + cross-patch battery and compared on measured
effect, not chosen by assumption. Layers 6–20 are inert — zero flips, best |Δp| below 0.08:

| layer | cross-patch flips | ablation flips | best Δp | clamped for cross-patch |
|---|---:|---:|---:|---:|
| 6 | 0 / 12 | 0 / 12 | 0.073 | 17 |
| 13 | 0 / 12 | 0 / 12 | 0.050 | 9 |
| 20 | 0 / 12 | 0 / 12 | 0.037 | 68 |
| 27 | 1 / 12 | 1 / 12 | 0.403 | 23 |
| **34** | **4 / 12** | **8 / 12** | 0.538 | 49 |
| 43 | 9 / 12 (see below) | 2 / 12 | 0.817 | **1,162** |

**Layer 43's 9 / 12 is an artifact, not a better result.** Only **one** of those flips comes from
clamping the six cue families; the other eight require all 1,162 base-active features — which at
this layer is the entire representation. That transplants the whole prompt rather than steering a
cue, and the inverted ablation score (2 / 12, because removing six families out of 1,169 changes
nothing) is the tell. Discounting the bulk transplant, layer 34 wins on cue-driven cross-patching
(2 / 12 vs 1 / 12) as well as on ablation.

That divergence between cross-patch and ablation is a cheap degeneracy check: on a healthy layer the
two agree (34 here: 4 and 8; `../supply_chain/` at 43: 3 and 7), and they invert when the dictionary
has collapsed.

## The pairs

Selected by rule from a sweep of all 335,799 clean `customer_support` training pairs
(`rank_flips.py --select-demo`): one per contrast type, ≤2 per tool combination, no tool named,
flip ≥ 0.6, weaker side < 0.8. The lexical-overlap floor is **0.55**, not the 0.7 used for
supply_chain — these requests are rephrased rather than word-swapped, so a higher gate would drop
half the themes for the wrong reason.

| theme | J | A | B |
|---|---|---|---|
| `billing_vs_technical` | 1.00 | billing_system 0.71 | customer_history 0.79 |
| `self_service_vs_agent_assist` | 0.80 | knowledge_base 0.94 | ticket_lookup 0.75 |
| `refund_vs_troubleshoot` | 0.75 | billing_system 0.72 | knowledge_base 0.95 |
| `urgent_vs_routine` | 0.60 | customer_history 0.63 | knowledge_base 0.96 |
| `new_vs_returning_customer` | 0.57 | customer_history 0.75 | ticket_lookup 0.69 |
| `single_issue_vs_complex_case` | 0.56 | billing_system 0.74 | customer_history 0.70 |

## What the run shows (layer 34)

**Ablation** — switching a side's cue families off at the decision token changes the tool on 8 of
12 sides, 6 of them onto the other side's tool (the *directed* count `paper/steering/` reports),
with set effects up to −0.64 against set-matched control bands of 0.019–0.121.

**Cross-patch** — clamping one side's cue families into the other side's request flips 4 of 12
directions. The prompt is never modified.

**Dose-response** — two curves cross 0.5 at **1×**, i.e. the donor's natural activation suffices:

```
single_issue a→b  p(billing_system)   0.12 → 0.63 (1×) → 0.94 (3×)   crosses 1.0×
urgent_routine b→a p(knowledge_base)  0.14 → 0.55 (1×) → 0.94 (3×)   crosses 1.0×
new_returning  a→b p(customer_history) 0.04 → 0.36 → 0.55 (1.5×)
new_returning  b→a p(ticket_lookup)    0.07 → 0.28 → 0.54 (2×)
```

**What the model writes** — three directions flip the tool name in the generated text, two with
byte-identical matched-random controls:

```
new_vs_returning b→a
  baseline: "customer_history tool to understand your account context first…"
  steered : "ticket_lookup tool to check if there are any existing tickets…"
  control : identical to baseline

single_issue a→b
  baseline: "customer_history tool to understand your current subscription status…"
  steered : "billing_system to look up your recent charges and subscription details…"
  control : identical to baseline

urgent_vs_routine b→a
  baseline: "customer_history tool to check your subscription details…"
  steered : "knowledge base to check our documentation about SSO…"
  control : customer_history (reworded, same tool)
```

### Against the ceiling

`ceiling_pairs.py` patches the donor's whole residual into the recipient's decision token —
activation patching in the model's own basis, no dictionary in the path — which bounds what any
decomposition read there could do. On these 12 directions it flips **9 / 12**, against **2 / 12** for
the cue set and **4 / 12** for every donor-active feature: a recovery of **0.22** and **0.44**.
Random directions at the same norm flip 1 of 36. Difference-in-means needs several pairs per contrast
type, so it is undefined here and reported on the `*_expanded` sets instead.


## Caveats

- **The probe panel is thinner here by construction.** Half the sides have no lexical cue, so they
  carry paraphrases only and no keyword control — there is no word to slip into the other request.
  `probes.json` says which, and the page simply omits the control where none exists.
- **`urgent_vs_routine` is directional.** `b→a` crosses 0.5 at 1×, but `a→b` never moves
  (p(customer_history) 0.01 → 0.05 at 3×). Adding the routine/documentation reading is easy;
  adding urgency is not.
- **`single_issue_vs_complex_case` is not a minimal pair.** Side B is side A plus an appended list
  of extra problems, so the contrast is length and complexity as much as framing.
- **Text flips are confirmation, not independent evidence** — each one is a direction that had
  already crossed 0.5 in the dose curve.
- HF/vLLM parity: residual cosine 0.998 (min 0.994), same first tool on 12/12 requests.

## Note: the published `tool_selection` demo used a different SAE

`../../tool_selection/index.html` embeds feature data that does not correspond to the SAE release
used here, so its indices, labels and L0 figures **must not be compared with the numbers below**.
See [`../README.md`](../README.md) for the measurements. Everything in this demo comes from the
shipped SAE (`output/layer_<N>/`) and the current sweep, so it is internally consistent.

## Files

| File | Purpose |
|---|---|
| `pairs.json` | The six pairs, emitted by `rank_flips.py --select-demo`. |
| `probes.json` | Hand-written paraphrases and keyword controls (29 prompts), validated against the assertions in `tests/test_tool_selection_demo.py`. |
| `index.html` | The page (reads `output/ui_data.json`, embedded fallback). |
| `output/` | `capture/`, `steering_layer34/`, `steering_layer27/`, `trace_layer34/`, `ui_data.json`. |

Drivers live in `../../tool_selection/` and are scenario-general — pass `--scenario customer_support`.

## Run

```bash
DOCKER="docker run --rm --gpus all -v $PWD:/workspace -v /path/to/models:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace/demo/tool_selection -w /workspace 575lab/kiji-inspector:dev"
MODEL=/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp

$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL --scenario customer_support
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/attribute_pairs.py --model-name '"$MODEL"' --scenario customer_support \
    --layer 34 --activations demo/steering/customer_support/output/capture/activations.npz'
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/trace_pairs.py --model-name '"$MODEL"' --scenario customer_support \
    --layer 34 --compare-layers 27'
uv run python demo/tool_selection/tool_selection_demo.py --scenario customer_support --layer 34 \
  --steering demo/steering/customer_support/output/steering_layer34/steering_results.json \
  --trace    demo/steering/customer_support/output/trace_layer34/trace_results.json
uv run python demo/home_repair/embed_ui_data.py \
  --ui-data demo/steering/customer_support/output/ui_data.json --html demo/steering/customer_support/index.html

python -m http.server 8001   # http://localhost:8001/demo/steering/customer_support/index.html
```

Wall clock on one H100: capture 3 min, attribute 2 min per layer, trace 3 min.
