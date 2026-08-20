# Home Repair Agent Demo

A scripted multi-step agent that diagnoses three appliance problems (dishwasher leak, stuck
garbage disposal, noisy water heater). For each problem the demo captures **one** sparse
autoencoder (SAE) snapshot at the moment the model picks its first tool, and the page shows:

1. **What the model actually chose** — next-token probabilities over the four tool names at
   the `I'll use the` prefill (renormalised over the tools; raw coverage shown). The base
   request's ask all but names the tool ("find me a replacement gasket with price and
   availability" → `parts_search`), so this readout is only a check that the decision is sharp
   enough to attribute — it is not a prediction being confirmed, and the page does not call it
   "expected". The model's *own* choice is item 7.
2. **Which SAE features stood out** — raw activation, deviation from the three-problem average,
   near-duplicate labels merged, and a *not stated in request* badge when a feature carries
   specific content (gas, warranty, urgent, …) the request never mentions.
3. **A contrast** — the same situation with a different ask (professional quote instead of a
   part / video; DIY video instead of a plumber's quote): which features appear, disappear or
   shift, and how the tool choice flips.
4. **A causal ranking of the snapshot rows** — each row's feature family is ablated on its own
   at the decision token and the tool distribution re-read (HuggingFace backend; see caveat
   below), compared against mass-matched random ablations; rows that do not beat the random band
   are greyed as *descriptive*, the others tagged *load-bearing* (in the current run: all
   descriptive — see results).
5. **Contrast-dimension evidence** — the five configured scenario dimensions scored from *all*
   active features via the training contrastive map (Cohen's d, anchor/contrast direction),
   flagged *insufficient* when too few mapped features fire.
6. **"Is this just keyword matching?"** — three paraphrases per problem (same meaning, none of
   the base request's key terms) and two keyword controls (the word added or removed without
   changing the meaning), all captured with the canonical vLLM backend.
7. **Without the ask** — the situation alone ("My 3-year-old dishwasher leaks from the bottom
   because the door gasket is cracked."): the one prompt where the first tool is the model's own
   call. The page shows its choice and distribution, the snapshot rows followed into it (which
   survive without the ask, which were carried by the ask) and the features that appear only
   without it. The causal ranking (item 4) stays on the explicit-ask request.

The four manual/parts/tutorial/quote panels are scripted evidence gathering (mock data), not
model decisions; no SAE readings are attached to them.

## Files

| File | Purpose |
|---|---|
| `home_repair_demo.py` | Prompt/contrast definitions, feature helpers, HF end-to-end pipeline, and `--ui-from-evaluation` builder for `ui_data.json`. |
| `compare_sae_backends.py` | **Canonical producer** (modified vLLM in Docker): decision-position activations for base + contrast + probe prompts (paraphrases, keyword controls, situation-only "open" requests), tool-choice readout, per-layer SAE evaluation → `vllm_native_evaluation.json` (`--no-probes` for the short set). |
| `evaluate_sae_layers.py` | Shared SAE evaluation (`_evaluate_layer`) and HF-only capture; metrics are computed over base prompts only. |
| `steer_tool_choice.py` | Causal check (HF backend in the same container): ablate each snapshot row's feature family on its own against mass-matched random controls (`attribution`, shown on the page) and, for the record, ablate/clamp the contrast-discriminating features → `steering_results.json` (`--hazard-experiments` adds theme-based checks, `--no-attribution` skips the per-row pass). |
| `sweep_tool_choice.py` | Read the model's tool choice for a list of candidate requests — single-turn or with prior turns (`{"request", "history"}`), optionally for another scenario (`--scenario`). Used to design the prompts and for the sweeps below (`output/sweep/`). |
| `embed_ui_data.py` | Embeds `output/ui_data.json` into `index.html` as the first-paint fallback. |
| `home_repair.json` | Scenario config — system prompt, tool list, contrast type descriptions. |
| `index.html` | Interactive viewer for `output/ui_data.json`. |
| `home_repair_colab.ipynb` | Colab notebook walkthrough of the HF pipeline. |

## Canonical run (modified vLLM in Docker)

Every observational result on the page comes from the modified-vLLM backend
(`575lab/kiji-inspector:dev`); only the interventions (which vLLM cannot do mid-forward) run on
HuggingFace transformers inside the same container. **The HF path must run the fused Mamba
kernels**: transformers resolves `causal-conv1d` / `mamba-ssm` from the `kernels` hub package
(`kernels>=0.15.2,<0.16`, now part of the `full` extra in `pyproject.toml`; images built before
that need the one-line `pip install` below). Without them NemotronH falls back to the naive
PyTorch scan and its residual stream drifts from vLLM (layer-27 cosine ≈ 0.93, different first
tool on 1 of 3 base prompts); with them the two agree (cosine ≈ 0.99+). `HFEngine` prints the
fast-path status, records it as `hfFastPath`, and the page shows the parity summary. Requires
the trained SAE run under `output/layer_<N>/` (`sae_checkpoints/sae_final.pt`,
`activations/feature_descriptions.json`, `activations/contrastive_features.json`).

```bash
DOCKER="docker run --rm --gpus all -v $PWD:/workspace -v /path/to/models:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src -w /workspace 575lab/kiji-inspector:dev"
MODEL=/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp

# 0) cheap check of the tool-choice readout (1 prompt, layer 27)
$DOCKER python demo/home_repair/compare_sae_backends.py --model-name $MODEL --smoke

# 1) activations + tool-choice readout + SAE evaluation (base + contrast + probe prompts)
$DOCKER python demo/home_repair/compare_sae_backends.py --model-name $MODEL \
  --results-dir demo/home_repair/output/prompt_alignment

# 2) per-row causal attribution (HF backend inside the same container, fused Mamba kernels;
#    parity recorded). The pip install is only needed on images without the `kernels` package.
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/home_repair/steer_tool_choice.py --model-name $MODEL --layer 27 \
  --vllm-report demo/home_repair/output/prompt_alignment/vllm_native_evaluation.json \
  --vllm-activations demo/home_repair/output/prompt_alignment/vllm_six_layer_demo_activations.npz'

# 3) build the UI payload and refresh the embedded fallback
uv run python demo/home_repair/home_repair_demo.py \
  --ui-from-evaluation demo/home_repair/output/prompt_alignment/vllm_native_evaluation.json \
  --steering demo/home_repair/output/steering/steering_results.json \
  --sae-layer 27 --output-dir demo/home_repair/output
uv run python demo/home_repair/embed_ui_data.py
```

Serve from the repo root (`python -m http.server 8001`) and open
`http://localhost:8001/demo/home_repair/index.html`.

### Caveats shown on the page

- **Causal check backend.** vLLM cannot intervene mid-forward, so the per-row ablations run with
  HuggingFace transformers (*delta patch*: `x -= rms_scale · a · W_dec[i]` for every feature of
  the family, exact for the SAE decoder, no reconstruction error). The HF forward runs the fused
  Mamba kernels, so it matches vLLM (residual cosine ≥ 0.996 on all nine prompts, same first tool
  on 9/9; the page header shows this). The causal column is still shown only where the HF
  baseline picks the same tool as the vLLM readout — today everywhere — and rows the HF pass
  cannot ablate (feature silent under HF) are marked as such instead of reporting a zero.
- **Dimension evidence tracks tool choice.** In the training pairs the anchor side of each
  contrast type is strongly associated with one tool (e.g. `warranty_covered → pro_quote`), so
  the contrast-dimension bars partly reflect the tool the model is about to pick.
- **Small maps.** `diy_vs_professional` and `safe_vs_hazardous` have few mapped features at
  layer 27 (max |d| < 0.8); expect them to be flagged insufficient for some problems.

## What the current run shows (2026-08-19, vLLM layer 27)

The decision prompts carry the situation *and* an explicit ask. That is a design choice, not a
finding: a sweep of 48 candidate requests (`sweep_tool_choice.py`, results under
`output/sweep/`) showed that the model's first tool follows the ask — open questions ("should I
flush it myself or call a plumber?") go to `manual_check` regardless of hazard, while "get me a
licensed plumber's quote" / "find me a replacement gasket" / "show me a video" pick `pro_quote` /
`parts_search` / `tutorial_search` with p ≈ 1. Hazard cues alone barely move the first choice
(gas/9-year/rusty vs electric/2-year/clear with the same ask: `pro_quote` 1.00 → 0.94). The
explicit ask gives the attribution a sharp decision to move; the readout on it is therefore a
restatement of the request, and the page says so. The model's own choice is measured on the
situation-only request (below).

- **Tool choice on the explicit ask.** Dishwasher → `parts_search` 0.99, disposal →
  `tutorial_search` 1.00, water heater → `pro_quote` 1.00 — as the asks dictate. Each contrast
  keeps the situation and changes the ask, and flips the decision (→ `pro_quote`, `pro_quote`,
  `tutorial_search`).
- **Without the ask.** The situation-only requests all go to `manual_check` (0.99 / 1.00 /
  1.00) — the model's own first move on a bare complaint is to look up the manual. And the
  snapshot families largely switch off: dishwasher 0/6 rows keep half their strength (*Appliance
  door gasket replacement search* 7.9 → 2.3, *Appliance repair part lookup* 8.2 → 0), disposal
  2/6 (the video-search family off; the generic *Home appliance repair and maintenance* rises
  5.7 → 10.8), water heater 0/6 (*Appliance repair warranty service request* 8.8 → 0, *Request
  for professional appliance repair* 6.7 → 1.1 — and *Gas appliance safety warnings* 4.7 → 0,
  so the "safety not stated in the request" reading above is itself carried by the ask). The
  active-set Jaccard between a situation-only request and its explicit-ask version is
  0.04–0.09, *below* the overlap between different problems' explicit-ask requests (0.12–0.15).
  What fires instead are situation-statement features (*Appliance door seal or gasket failure*
  10.2, *Old appliance door leak diagnosis* 6.4, *Garbage disposal humming jam detection*
  3.5 → 6.3), several of them generic — *Appliance door seal or gasket failure* and *Gas
  appliance ignition and repair* fire at 7–10 on all three open requests. Reading: at the
  decision token the layer-27 representation is organised by the ask more than by the appliance;
  the snapshot rows are features *of the ask's meaning* (they survive paraphrase and ignore bare
  keywords), not of the situation — and, per the ablations below, they describe the ask rather
  than carry the decision.
- **Can the ask be put back through the features? No.** Clamping each snapshot family into the
  open request at its explicit-ask activation (HF backend, delta patch) moves p(asked tool) by at
  most +0.9 pp; all families together +2.4 / +0.1 / +0.2 pp; *every* feature active on the
  explicit-ask request (38 / 41 / 59 features), each at its value there, +4.5 / +0.3 / +0.5 pp,
  tool unchanged; matched random sets ≤ 0.5 pp (`steering_results.json` → `injection`). So the
  layer-27 features at the decision token neither create the decision (injection) nor carry much
  of it (ablation, next bullet) — it is settled by the ask elsewhere in the network.
- **Is there any situation-only decision to explain? Not in this scenario.** Two sweeps with
  `sweep_tool_choice.py` (results under `output/sweep/`): 1 127 situation-only training-pair
  prompts (age, warranty, gas/electric, urgency, DIY cues; no ask words) → `manual_check`
  1 125 / 1 127, p ≥ 0.95 on all but 12 (`training_pairs_sweep.json`); the same pairs at the
  *second* tool choice, after a `manual_check` result → `parts_search` 293 / 306, softer (p
  mostly 0.5–0.9) but the same tool whether the appliance is 3 or 18 years old, electric or
  smelling of gas — `pro_quote` never wins unasked (`step2_sweep.json`). The model's tool
  choice in this scenario is a function of the explicit ask; there is no hidden situational
  reasoning at the tool-choice token for an SAE to expose. A sweep of the `tool_selection`
  scenario (8 tools; `tool_selection_sweep.json`) looks different: 194 / 650 prompts split
  (p < 0.8) and 116 / 325 one-cue pairs flip (cached vs live, single vs batch, create vs update,
  local vs remote) — that is where a decision-level SAE readout has something to explain; see
  `demo/tool_selection/` for that demo (cue features, per-family ablation and cross-patching on
  seven one-cue pairs).
- **Features.** The snapshots are decision-linked: *Appliance door gasket replacement search* /
  *part replacement* for the dishwasher; *repair guide search* / *video search* for the
  disposal; *Request for professional appliance repair*, *Appliance repair warranty service
  request* and *Gas appliance safety warnings* (safety not stated in the request) for the water
  heater. The contrasts swap exactly these families in and out.
- **Dimensions.** Water heater sits on the professional (0.85) and hazardous (0.83) sides,
  dishwasher on the quick-part-swap side (0.08), disposal near the safe side (0.42).
- **Single-feature ablate/clamp (recorded, not shown).** Ablating the 5 *individual* features
  that discriminate each base decision from its contrast moved the readout only marginally beyond
  a reconstruction-only control (water heater −10 pp vs −9 pp control; disposal −8 vs −3;
  dishwasher none) and clamping them onto the contrast prompt did nothing — the decision is
  carried by *families* of split features, which the per-row attribution below ablates whole.
  Those experiments stay in `steering_results.json` but were dropped from the page as
  uninformative. (An earlier run without the fused Mamba kernels — naive fallback, layer-27
  cosine ≈ 0.93, two wrong HF baselines — reported −14 to −17 pp single-family ablation effects;
  those were artefacts of the drifted forward and are superseded by the numbers below.)
- **Is this just keyword matching?** No, at least not at the level of tokens. *Paraphrases*
  (same meaning, none of the base request's key terms — "rubber seal … look up a new seal … what it
  costs" for the gasket request, "certified plumber to price the repair" for the plumber's quote):
  a snapshot row counts as firing when its family (the feature plus merged twins) reaches at
  least half of its base-request activation — a binary "any member > 0" would be met almost
  anywhere by families of 8–24 split features. On that measure the six snapshot rows fire 6/6,
  6/6, 6/6 on the dishwasher paraphrases (references — same situation with a different ask, the
  other two problems — 1/6 each), 3/6, 6/6, 6/6 on the disposal paraphrases (references 1–3/6;
  the 3/6 is the "how-to clip" wording that goes to `manual_check`), 4/6, 5/6, 5/6 on the water
  heater paraphrases (references 0–1/6). Active-set Jaccard with the base is 0.17–0.46 for
  paraphrases (the disposal "how-to clip" wording 0.14) against 0.12–0.15 for the references
  (cosine 0.62–0.88 vs 0.19–0.29); 8/9 paraphrases pick the same tool as the base — as they
  should, since they restate the ask. *Keyword controls* (the word without the meaning): "warranty expired" /
  "a professional plumber confirmed" on the gasket request, "the gas company visited" / "my
  neighbour is a professional plumber" on the disposal request, and "I watched a video" on the
  plumber's-quote request leave the warranty / professional-quote / video-search features silent
  (largest response 0.9 on a feature that peaks at 8.6); dropping the word "gas" from the water
  heater request leaves *Gas appliance safety warnings* and the other gas features unchanged
  (4.72 → 4.91), so those fire on "water heater", not on the token.
- **Which rows carry the decision? At layer 27, none of them.** With HF matching vLLM the
  explicit-ask decisions are saturated (p(asked tool) = 0.99–1.00 on both backends), and ablating
  each snapshot family on its own moves them by at most −1.6 pp (water heater *Appliance repair
  warranty service request*), inside or barely outside mass-matched random bands of 0.1–0.3 pp;
  all snapshot families off together: −1.9 / −3.2 / −2.6 pp, tool unchanged (22 / 33 / 42
  features). The page therefore shows every row as *descriptive* — the features describe what the
  request is about, they do not carry the first tool choice. That is the honest result for this
  scenario; the decision-level demo in `../tool_selection/` shows what load-bearing features look
  like (set effects of 20–56 pp and cross-patch flips at layer 43).
- vLLM activations vary slightly run to run with batch composition (residual cosine ≈ 0.995 for
  the same prompt), and the fork only returns per-token logprobs for the first request of a
  `generate()` batch, so the readout is issued one request per call.

## `ui_data.json` sections

`runMetadata`, `problems`, `toolResults`, `recommendations` (grounded, unchanged),
`decisionFeatures[pid]` (`modelChoice`, `features` with `activation`/`delta`/`share`/`notStated`/`merged`,
`sharedAcrossProblems`, `alsoFired`, `themeEvidence`, and — when the steering file carries an
`attribution` block and the HF baseline agrees with the vLLM readout — `features[*].causal`
{`deltaTarget`, `descriptive`, `inactiveUnderHf`} and `attribution` {`targetTool`, `hfChoice`,
`controlThreshold`, `allRows`}; otherwise `causalWithheld` {`hfChoice`, `vllmChoice`}), `contrasts[pid]`
(`gained`/`lost`/`shifted`, `modelChoice`), `probes[pid]` (`paraphrase` {per-paraphrase tool
choice, active-set overlap with the base, which snapshot rows fire; `comparisons` = the contrast
and the other problems as calibration}, `controls` [{`direction`, `keyword`, `targets` with base →
control activation, `verdict`}]), `openRequests[pid]` (the situation-only request: `modelChoice`,
`baseChoice`, active-set `overlap` with the base, the snapshot `rows` followed into it with
`fires` at ≥ ½ base strength, and `gained`/`lost`/`shifted`), and `themes[*].evidence`.
`decisionFeatures[pid].askTool` records the tool the ask describes (a design input; nothing is
"expected" and the readout carries no `matchesExpected`). Sections a run did not produce are
`null`/absent and the page hides them.

## HF pipeline (Colab / no vLLM)

```bash
pip install 'kiji-inspector[huggingface]'
huggingface-cli login
uv run python demo/home_repair/home_repair_demo.py
```

The script writes `analysis_results.json`, `agent_output.txt`, `per_problem_analyses.json`, and
`ui_data.json` to `demo/home_repair/output/`. The HF path records the same base + contrast
prompts but has no tool-choice readout (`modelChoice: null`) and no causal check.

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dataiku/kiji-inspector/blob/main/demo/home_repair/home_repair_colab.ipynb)

The notebook needs an **A100 high-RAM** runtime (the base model is
`nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16`, ~30B parameters). Add a Colab
Secret named `HF_TOKEN` before running. Optionally add `YOUTUBE_API_KEY` to
fetch real tutorial results instead of the mock data.

## Hosting `index.html` from Colab

`index.html` loads its data with `fetch('output/ui_data.json')`, so it needs a real HTTP server — opening it through a `file://` URL or a notebook iframe won't work. The notebook's last cell handles this for you, but here's what it does:

1. **Stage the files** the way the page expects them — `index.html` at the root, `ui_data.json` under an `output/` subdirectory:

   ```python
   import shutil, urllib.request
   from pathlib import Path

   SERVE_ROOT = Path("/content/serve")
   (SERVE_ROOT / "output").mkdir(parents=True, exist_ok=True)

   urllib.request.urlretrieve(
       "https://raw.githubusercontent.com/dataiku/kiji-inspector/main/demo/home_repair/index.html",
       SERVE_ROOT / "index.html",
   )
   shutil.copy(OUTPUT_DIR / "ui_data.json", SERVE_ROOT / "output" / "ui_data.json")
   ```

2. **Start a static server** in the background:

   ```python
   import subprocess

   PORT = 8000
   subprocess.Popen(
       ["python", "-m", "http.server", str(PORT), "--directory", str(SERVE_ROOT)],
       stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL,
   )
   ```

3. **Open it through Colab's port proxy** — Colab forwards localhost ports to a signed `*.googleusercontent.com` URL:

   ```python
   from google.colab import output

   output.serve_kernel_port_as_window(PORT)  # opens a new tab
   # output.serve_kernel_port_as_iframe(PORT, height="900")  # inline in the cell
   ```

`serve_kernel_port_as_window` pops the viewer into a new tab; `serve_kernel_port_as_iframe` embeds it directly in the notebook output. Pick whichever you prefer.

Re-running the cell spawns a second `http.server` on the same port; the duplicate fails silently. If you regenerate `ui_data.json`, just refresh the proxy tab — the running server picks up the new file automatically.

## How far does this hold? (spec-sheet strip)

The page ends with a strip of context measured by `demo/spec_sheet`: the
layer-27 null reported here now sits on a depth curve — on the scenario that
*does* have flippable pairs (tool_selection), the same ablation battery moves
the decision on only 6/14 sides at layer 27 and 0–1/14 below it, with effects
concentrating at layers 34–43 (description early, causal leverage late); a
held-out bag-of-words probe predicts the tool at 86.3% vs 76.7% for SAE
features — quantifying this page's descriptive reading; and feature indices
are run-specific across retraining seeds. The strip renders from
`ui_data.json`'s optional `specSheet` block, attached automatically by
`home_repair_demo.py` (`attach_spec_sheet`) when
`demo/spec_sheet/output/ui_data.json` exists; pages built before the spec
sheet ran render unchanged. Full details: `../spec_sheet/index.html`.
