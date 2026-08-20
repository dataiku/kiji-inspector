# Tool-selection demo — one cue, one flip

Pairs of near-identical requests where changing one cue changes the tool the model picks, and
what the SAE says about the flip. This is the decision-level companion of the home-repair demo
(`demo/home_repair/`): there the first tool choice turned out to be a function of the explicit
ask, and there was no hidden decision for an SAE to expose. In the `tool_selection` scenario
(8 tools, same SAE training data) the model's choice is a genuine decision — in a sweep of 650
training-pair requests 194 were split (p < 0.8) and 116 of 325 one-cue pairs flipped tool
(`demo/home_repair/output/sweep/tool_selection_sweep.json`).

**The pairs are chosen from that sweep by rule, not by hand** (`select_pairs.py` → `pairs.json`):
flip = min(p_A, p_B) when the two sides' top tools differ, score = flip × Jaccard(content words)
so that true one-cue edits rank above requests that differ throughout, flip ≥ 0.6, one pair per
theme, at most two per unordered tool combination, no tool named in the request. Seven pairs
qualify (internal vs external, specific vs broad, local vs remote, verified vs unverified, query
vs mutate, direct vs delegated, single vs batch); the rule is printed on the page. The sweep's
readout merges `file_read|file_write`, so the read/write themes cannot be scored from it; the
highest raw flips are pairs whose cue names the tool's domain ("in the internal knowledge base"
vs "from the public web"), which is why lexical overlap is in the score. An earlier hand-picked
set (three of which the rule also picks for their themes) is kept under `output/previous_pairs/`.

For each pair the page (`index.html`) shows:

1. **Both readouts** — the exact tool distribution at `I'll use the` on each request (modified
   vLLM backend; `file_read` / `file_write` share their first token, so a second token is read;
   capitalised surface forms such as ` API call` count toward the tool).
2. **Cue features** — the SAE features active on one side and weaker or silent on the other,
   ranked by the difference, near-duplicate labels merged into families.
3. **Ablation** — each cue family switched off on its own side at the decision token
   (HuggingFace backend, delta patch), against mass-matched random ablations; rows that beat
   the random band and 2 pp are tagged *load-bearing*, the rest *descriptive*; all families off
   together is reported too.
4. **Is it the word or the meaning?** — two paraphrases of each request with none of its cue
   words, and the other side's request with the cue word slipped in (`probes.json`, hand-written
   and style-checked by the tests): which cue families fire, the model's tool, and the cosine of
   the active set to each side. All captured with vLLM like the readouts.
5. **Cross-patch** — each side's cue families clamped into the *other* request at their
   activations here: does the other request's decision move toward this side's tool?
6. **Where the cue lives** — the activation of each side's top cue families at every token of
   the prompt, and p(tool) after switching all of them off at the decision token only / the
   request tokens only / everywhere but the decision token / everywhere, at layer 43 and, for
   comparison, layer 27 (`trace_pairs.py`).
7. **How much of the cue is needed** — dose-response: the other side's families clamped in at
   0×–3× of their activation, all together and the single strongest family, against matched
   random sets at the same scales.
8. **What the model writes** — greedy continuations with the other side's families clamped on
   every token, next to the unsteered continuation and a matched-random-set continuation.

The HuggingFace forward runs the fused Mamba kernels (see *Run*), so its residual stream matches
vLLM (cosine ≥ 0.995 at every layer, mean 0.999) and it picks the same first tool on 13 of the 14
requests; the one exception (`create_vs_update_A`, a near tie: vLLM `file_write` 0.42 vs HF
`internal_search` 0.47) has its causal column withheld and its cross-patch direction flagged, as
the page does for any HF/vLLM disagreement.

## Files

| File | Purpose |
|---|---|
| `select_pairs.py` / `pairs.json` | The selection rule and its output (see above); `pairs.json` is reproducible from the sweep files and a test checks that it is. |
| `probes.json` | Hand-written paraphrases and keyword controls per pair side. |
| `tool_selection_demo.py` | Loads the pairs/probes, prompt builder, tool-token readout tree, cue-feature analysis, probe evidence, `build_ui_data` (`--layer`, `--steering`, `--trace`). Generic helpers are imported from `../home_repair/home_repair_demo.py`. |
| `capture_decisions.py` | **Canonical producer** (modified vLLM in Docker): decision-position residuals for all trained SAE layers, exact tool readout, per-layer SAE features with labels, for the pair sides and (unless `--no-probes`) the probes → `output/capture/evaluation.json`, `activations.npz`. |
| `attribute_pairs.py` | HF-backend causal check for one layer: per-family ablation with matched controls, cross-patching, parity → `output/steering_layer<L>/steering_results.json`. |
| `trace_pairs.py` | HF-backend trace for the page layer: per-token cue map, position-resolved ablation (page layer + `--compare-layers`), dose-response cross-patch, steered generations → `output/trace_layer<L>/trace_results.json`. |
| `index.html` | The page (reads `output/ui_data.json`, embedded fallback). |

## Run

```bash
DOCKER="docker run --rm --gpus all -v $PWD:/workspace -v /path/to/models:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace/demo/tool_selection -w /workspace 575lab/kiji-inspector:dev"
MODEL=/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp

python demo/tool_selection/select_pairs.py                                            # pairs.json from the sweep
$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL            # vLLM, ~7 min (pairs + probes)
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/attribute_pairs.py --model-name $MODEL --layer 43 \
  --activations demo/tool_selection/output/capture/activations.npz'                   # HF, ~25 min
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/trace_pairs.py --model-name $MODEL --layer 43'               # HF, ~30 min
uv run python demo/tool_selection/tool_selection_demo.py --layer 43 \
  --steering demo/tool_selection/output/steering_layer43/steering_results.json \
  --trace demo/tool_selection/output/trace_layer43/trace_results.json
uv run python demo/home_repair/embed_ui_data.py \
  --ui-data demo/tool_selection/output/ui_data.json --html demo/tool_selection/index.html
python -m http.server 8001   # open http://localhost:8001/demo/tool_selection/index.html
```

Needs the trained SAE run under `output/layer_<N>/` (checkpoint + `feature_descriptions.json`).
The HF step needs the fused Mamba kernels (`kernels` hub package, in the `full` extra of
`pyproject.toml`; the `pip install` above covers images built before it was added) — without
them the HF forward of this hybrid Mamba model runs the naive scan and drifts from vLLM
(residual cosine ≈ 0.92–0.98, different first tool on 4 of 14 requests); with them it matches
(cosine ≈ 0.99+, 13 of 14). `attribute_pairs.py` records the status as `hfFastPath` and the page
shows the parity summary in its header.

## What the current run shows (2026-08-19, rule-selected pairs)

**Readouts** (vLLM, exact token tree; sweep readout in brackets where it differed): internal
policy → `internal_search` 1.00 / public trends → `web_search` 1.00; the Q3 report at a path →
`file_read` 0.95 / all quarterly reports → `internal_search` 0.95; local config file →
`file_read` 1.00 / remote URL → `api_call` 0.61 [sweep: web_search]; verified, in our docs →
`internal_search` 1.00 / public web → `web_search` 1.00; check the value → `internal_search`
0.67 / update it to 5 → `file_read` 0.64 [sweep bucket file_read|file_write]; calculate the
average → `database_query` 0.66 / analyse the root causes → `internal_search` 0.95; one account
by ID → `database_query` 0.65 / all accounts from the last day → `internal_search` 0.77. All
seven flip. HF (fused kernels) agrees with vLLM on 14/14 baselines, residual cosine 0.995–1.000.

**Is it the word or the meaning? The meaning.** Over the 28 paraphrases (none of the side's
cue words) 98 % of the side's cue families still fire (min 83 %), the active set's cosine to the
own side is 0.89 on average (to the other side 0.43) and the tool is the same on 22/28 — the
misses are requests the paraphrase genuinely made ambiguous ("Set MAX_RETRY_LIMIT to 5 in the
configuration" without "file" → internal_search; a bare URL → file_read/web_search split). Over
the 14 keyword controls (the other side's request with this side's cue word slipped in) 48 % of
the families fire, weakly, cosine to the own side 0.45 and to the other side 0.94, and the model
takes the other side's tool every time. E.g. *Local Configuration File Reading* is 15.6 on "read
the local config file", 77–79 on the two paraphrases ("Open /etc/app/config.json on this
machine…"), 11.8 on "fetch the remote … used by the local team"; *Retrieval of official
internal compliance documents* 16.0 / 15.2 / 12.9 / 0.0 on the internal-policy request, its
paraphrases and "public trends … for our internal newsletter". The separation is sharpest where
the cue features are specific (file, internal/public) and weaker for the generic
*Entity Status Update Requests*-type families of the single/batch and direct/delegated pairs
(cosine to the other side 0.7–0.8 even for paraphrases).

**Which decisions the layer-43 cue features carry.** The two top-ranked pairs by the rule are
the ones whose cue names the tool's domain, and they are saturated (p = 1.00 both sides):
switching all cue families off moves them ≤ 1 pp, cross-patching ≤ 3 pp, and the dose-response
needs 3× before p(other tool) reaches 0.5–0.7. The decision there is overdetermined; the
families are descriptive. In the four pairs with a real split the families carry it:

| pair | side | all cue families off (random band) | single families beyond band + 2 pp | cross-patch into the other side (1×) | dose: first scale with p ≥ 0.5 |
|---|---|---|---|---|---|
| local vs remote | A (file_read 1.00) | **−41 pp** (0.0) | none (each ≤ 2) | A→B **+95 pp → flips to file_read** | ½× 0.34, **1× 0.95** |
| | B (api_call 0.61) | **−46 pp** (25, wide), flips | *Remote Configuration Retrieval* −27 | B→A 0 | never (3×: 0.05) |
| specific vs broad | A (file_read 0.95) | **−47 pp** (3.8), flips | *Local Configuration File Reading* −15, *Reading local config.json file* −7 | A→B +14 | 2× 0.53, 3× 0.88 |
| | B (internal_search 0.95) | +1 (1.8) | none | B→A +22 | 2× 0.53 |
| single vs batch | A (database_query 0.65) | **−28 pp** (6.9), flips | *Entity Status Update Requests* −10, *Database record status updates* −9 | A→B +13 | 1½× 0.51 |
| | B (internal_search 0.77) | −11 pp (4.6) | *Retrieval of official internal compliance documents* −8 | B→A +11 | 2× 0.52 |
| direct vs delegated | A (database_query 0.66) | **−25 pp** (12.5), flips | none | A→B +1 | never (3×: 0.21) |
| | B (internal_search 0.95) | +3 (2.3) | none | B→A +6 | 3× 0.51 |
| check vs update | A / B | −8 / −3 pp (5.7 / 5.6) | *Retrieval of official internal compliance documents* −9 (A) | +11 / +8 | never / 3× 0.59 |

**Where the cue lives.** As with the earlier set the families are nearly silent on the request
tokens and peak at the decision token. Removing them from every position but the decision token
does little (file_read 1.00 → 1.00 local, 0.95 → 0.85 specific); at the decision token it takes
the split decisions most of the way (local 1.00 → 0.59, remote 0.61 → 0.14, specific 0.95 →
0.44, single 0.65 → 0.39); everywhere at once flips them (0.21 / 0.06 / 0.37 / 0.39). Layer 27
is not inert on this set but clearly weaker: its families removed everywhere leave local at
0.99, specific at 0.56, single at 0.56, and move the already-split remote request (0.61 → 0.09
everywhere, 0.33 at the decision token) — layer 27 has more to say about a 0.61 decision than
about a 1.00 one.

**What the model writes.** One pair flips in text — the remote request with the local-file
families clamped in: "API call tool to fetch the remote configuration file. Let me make the
request." → "**file_read** tool to fetch the contents of the remote configuration file. However,
I need to…" (matched random set: unchanged). Everywhere else the dose curve does not cross 0.5
at 1× and the text keeps its tool with at most wording changes.

**Compared with the earlier hand-picked set** (`output/previous_pairs/`): the rule keeps the
same causal picture (file-related cues load-bearing at layer 43, cross-patch flips where the
dose curve crosses 0.5 at 1×) but trades the hand-picked "cached vs live" and "create vs update"
pairs for the two saturated domain-naming pairs, which is what the sweep's top flips look like.

**Caveats.** Layer-43 labels are a little broad (*Q3 Sales Data Retrieval and Analysis*,
*Complex business root cause analysis* appear as side-specific on several pairs and are, as the
ablations show, descriptive). The random band is wide where the decision is split (25 pp on the
0.61 api_call request). vLLM activations vary slightly with batch composition. The "all cue
families off" numbers are *set* effects; the page orders rows by their individual effect and
greys the rest as descriptive.

## `ui_data.json`

`runMetadata` {model, saeLayer, availableLayers, backend, hf {fastPath, cosineMean, cosineMin,
baselineAgreement, baselineCompared}, trace? {compareLayers, scales, genTokens}},
`pairSelection` {source, rule} (from `pairs.json`), `scenario` {systemPrompt, tools, prefill},
`pairs[*]` {id, title, cue, flipped, overlap, shared, `a`/`b` {request, cueWords, modelChoice,
numActive, features[{index, label, activation, other, delta, merged, causal?}], attribution? |
causalWithheld?, probes? {paraphrases[{step, request, modelChoice, sameTool, familiesFiring,
familiesTotal, families[{label, activation, base, fires}], cosineToBase, cosineToOther}],
keyword {… + cue}}}, crossPatch {a_into_b, b_into_a: rows,
allRows, allBase, controlThreshold, intoBaselineChoice, intoBaselineMismatch}, positions
{a/b: tokens, requestSpan, targetTool, hfChoice, baseline, layers {L: numFamilies, numFeatures,
profile? [{index, label, familySize, perToken}], ablate {decision|request|allButDecision|all:
positions, p, choice, distribution, controlBand?}}}}, dose {a_into_b/b_into_a: targetTool,
baselineP, baselineChoice, scales, allRows[{scale, p, choice}], bestRow {label, curve},
controlBand[], numFeatures}, generations {a_into_b/b_into_a: baseline, steered, control,
numFeatures, controlSize}}.

## How far does this hold? (spec-sheet strip)

The page ends with a strip of context measured by `demo/spec_sheet` (all from
existing data): the 7 pairs sit in a census of 7,478 training pairs of which
44.7% flip the model's first tool (2,303 at ≥0.6 both sides); the same ablation
battery run at every trained depth shows effects beginning only around layer
27–34 and strongest at 43 (8/14 sides, where this page lives); the effect
survives three retrained dictionaries including one sharing zero feature
directions with this page's SAE; a held-out bag-of-words probe is competitive
with the features at predicting the tool (the features' value is causal
handles and legibility, not extra signal); and feature indices are
run-specific — across a retraining seed almost no decoder directions match,
though ~63% of features keep a functional counterpart.

The strip renders from `ui_data.json`'s optional `specSheet` block, which
`tool_selection_demo.py` attaches automatically when
`demo/spec_sheet/output/ui_data.json` exists (condensed by
`home_repair_demo.spec_sheet_note`). Pages built before the spec sheet ran
render unchanged. Full details: `../spec_sheet/index.html`.
