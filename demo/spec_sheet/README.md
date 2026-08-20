# The instrument's spec sheet

The two demo pages (`demo/home_repair`, `demo/tool_selection`) show what the
tool-choice SAEs can do. This demo measures **where they stop working** — and
does it entirely from data the project already had: the 95,386 stored
decision-token vectors, the 50K-pair parquet, the shipped feature labels, and
retrained dictionaries that cost ~18 seconds of GPU each. No new prompts are
generated anywhere; the only model forwards are a readout sweep over the
parquet's own requests (vLLM, canonical) and the existing causal battery re-run
at more depths and with more dictionaries (HF, fused kernels).

## Questions it answers

1. **Cross-scenario reach.** Train a dictionary on one scenario's decision
   tokens, evaluate on the other's held-out prompts. Both the SAEs and a
   PCA-75 baseline fail catastrophically across scenarios (EV < 0 vs ~0.9
   in-domain) — the two scenarios occupy different affine subspaces of the
   residual stream, so this is a property of the representation, not an SAE
   defect. Affine alignment (the norm-matched control) recovers part of the
   gap. Out of domain the instrument does not fail silently: L0 explodes from
   ~100 to ~800–2,150.
2. **Dictionary stability.** Retrain the joint dictionary with a different
   seed: decoder directions are almost entirely different (mean best cosine
   ~0.26 among features firing ≥1%, ~0% above 0.7 — barely above the Gaussian
   null), yet ~63% of frequently-firing features keep a *functional*
   counterpart (best activation correlation ≥0.7 on held-out prompts).
   Same-seed comparisons without a firing-rate cut are inflated to ~0.9 decoder
   cosine by dead rows frozen at shared initialization — the page prints this
   trap. Cross-scenario functional matching sits at the permutation null: a
   clean double dissociation.
3. **Decision signal.** A leak-free probe comparison (train on training-split
   prompts, test on held-out components; same split for every representation):
   bag-of-words 86/90% (home_repair / tool_selection), raw residual 87–93%,
   SAE features 76–90%. The sparse code keeps ~96% of the dense signal on
   tool_selection and loses ~10 pp on home_repair; words alone are competitive
   throughout — an honest baseline the demo pages now cite.
4. **Label ↔ signal alignment.** On held-out pairs, the top side-features'
   labels share content words with the matching clause of the pair's
   `distinguishing_signal` annotation at 0.86–0.91, vs 0.53–0.55 shuffled null
   on tool_selection (home_repair null is higher, 0.76–0.80, because its five
   themes share vocabulary).
5. **The population, not the showcase.** The first tool choice for every
   unique tool_selection request in the parquet (~9.3K), with the exact-logprob
   readout and second-token disambiguation for `file_read`/`file_write` —
   flip census by contrast type, seen/unseen-by-SAE strata, and a reliability
   curve against the generator's intent.
6. **Depth and dictionary robustness of the causal result.** The demo's causal
   battery (family ablations, cross-patching, mass-matched controls) at all
   six trained depths on the same 7 pairs, and again at layer 43 with
   dictionaries the capture never used (`--active-from-sae` re-derives the
   families from the captured vLLM residuals). If the effect survives in
   dictionaries with unrelated feature directions, it belongs to the model's
   representation, not to one training run.

## Files

| File | What it does |
|---|---|
| `build_splits.py` | Leak-free splits of the existing shards at prompt-pair-graph component level (4,785 components, none scenario-mixed) |
| `train_split_saes.py` | 24 dictionaries (4 × 6 layers), shipped configuration exactly, ~18 s each |
| `evaluate_transfer.py` | Held-out EV / L0 / affine-aligned control / PCA-75 baseline + decoder-cosine and activation-correlation feature matching with nulls |
| `feature_workbench.py` | Probes (SAE vs residual vs bag-of-words) and blind signal recovery, all on stored activations |
| `population_sweep.py` | vLLM readout of every unique tool_selection request (resumable JSONL) |
| `population_report.py` | Pair-level flip census, strata, reliability (CPU) |
| `run_hf_session.sh` | Depth battery (layers 6/13/20/34) + dictionary robustness at 43, one Docker HF step each |
| `build_ui.py` | Assembles `output/ui_data.json` from whatever exists and embeds it into `index.html` |
| `index.html` | The spec-sheet page (sections hide until their data exists) |

Tests: `tests/test_spec_sheet_{splits,transfer,workbench,population,ui}.py`.
The `--active-from-sae` flag lives in `demo/tool_selection/attribute_pairs.py`;
the `min_shared_mass` readout skip in `demo/tool_selection/capture_decisions.py`.

## How to run

```bash
# 1. CPU/local GPU (no Docker): splits -> dictionaries -> transfer -> workbench
uv run python demo/spec_sheet/build_splits.py
uv run python demo/spec_sheet/train_split_saes.py
uv run python demo/spec_sheet/evaluate_transfer.py
uv run python demo/spec_sheet/feature_workbench.py

# 2. One vLLM session (canonical backend, ~1.5-2 h; resumable)
docker run --rm --gpus all -v $PWD:/workspace -v /home/shadeform/models:/models:ro \
  -v /ephemeral/cache/huggingface:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src -w /workspace 575lab/kiji-inspector:dev \
  python demo/spec_sheet/population_sweep.py \
  --model-name /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp --gpu-memory-utilization 0.85
uv run python demo/spec_sheet/population_report.py

# 3. One HF session (~1-1.5 h; GPU-exclusive with step 2)
bash demo/spec_sheet/run_hf_session.sh

# 4. Assemble the page
uv run python demo/spec_sheet/build_ui.py
python -m http.server 8001   # open /demo/spec_sheet/index.html
```

`sudo chown -R shadeform:shadeform demo/spec_sheet/output demo/tool_selection/output`
after Docker steps if files come back root-owned.

## Results (this machine, 2026-08-19)

Filled by the run; see `output/ui_data.json` and the page. Headline entries:

* Transfer (layer 43, held out): in-domain EV 0.94 / 0.94; cross-domain −3.1 /
  −9.3 (PCA-75: −1.6 / −4.6); affine-aligned recovers to 0.33 / 0.19;
  joint dictionary 0.91 / 0.93 at L0 63/98.
* Stability (layer 43, rate ≥ 1%): decoder mean best cosine 0.26 (0.1% ≥ 0.7,
  null 0.06); functional mean best correlation 0.75 (63% ≥ 0.7, null 0.11).
  Cross-scenario functional matching ≈ null (10.6% vs 10.8%).
* Probes (layer 43, held out): tool_selection SAE 89.1% / residual 93.2% /
  BoW 89.9% (majority 15.5%); home_repair SAE 76.7% / residual 86.7% /
  BoW 86.3% (majority 37.1%).
* Signal recovery (layer 43): tool_selection 86.9% vs 54.2±1.2% null;
  home_repair 90.7% vs 78.6±0.9% null.
* Population census (9,291 prompts → 7,478 pairs): **3,342 flip the first
  tool (44.7%), 2,303 at ≥0.6 confidence on both sides**. Flip rates range
  from 4.7% (shallow_vs_deep) to 75.5% (verified_vs_unverified); the model's
  choices are decisive (61% of sides at p ≥ 0.9) but agree with the pair
  generator's intent only 52% overall (67% within the p ≥ 0.9 bin) — intent
  is not ground truth, and cached_vs_live agreement is 6%. The unseen-by-SAE
  stratum (407 pairs) flips at 18% vs 46% seen, but it concentrates in
  low-flip contrast types, so the strata are not directly comparable.
  internal_search takes 48% of all choices; code_execute 0.4%.
* Depth curve (same 7 pairs, all fast-path): sides where ablating all cue
  families beats the control band — layer 6: 1/14, 13: 0/14, 20: 1/14,
  27: 6/14, 34: 5/14, 43: 8/14; mean |Δ target| rises 0.008 → 0.154; the
  first cross-patch flips appear at layer 34. Features are readable from
  layer 6 but the decision only leans on this basis late.
* Dictionary robustness (layer 43): the battery still moves decisions under
  all three retrained dictionaries — tool_selection_only 11/14 sides,
  joint 9/14, joint_seed123 11/14, with 4–5 cross-patch flips each —
  including under a seed whose decoder directions share nothing with the
  original (stability card). The causal handle belongs to the model's
  representation, not to one training run. Families for these runs are
  built from unlabeled features, so granularity differs from the shipped
  run: qualitative comparison, not effect-size-matched.

## Standing caveats

* Decision-token, two-scenario instruments — every number here is about that
  regime; nothing is claimed for other scenarios, positions, or turns.
* Pairs, labels, and signal annotations come from the same LLM family;
  "meaning" claims partially inherit the generator's notion of meaning.
* Observational results: modified-vLLM backend. Interventions: HF with
  `kernels==0.15.2` (fused Mamba path; residual cosine ≥ 0.99 vs vLLM), same
  threshold offset as the demo runs.
* The shipped SAEs saw every eval vector during training; where they appear
  they are marked and never used for held-out claims.
* All-families ablation arms are compared against per-family mass-matched
  controls (smaller sets); per-family rows are the like-for-like comparison.
