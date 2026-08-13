## kiji-inspector-v0.7.0 (2026-08-13)

### Feat

- register Nemotron 3.5 SAE (#58)
- improve causal ablation and adaptive L1 experiments (#57)

## kiji-inspector-v0.6.0 (2026-08-06)

### Feat

- **core**: add `SAE.normalize_input` / `SAE.denormalize_output`, applying the full
  `(x - mean_vec) / rms_scale` transform the SAE was trained under, and load `mean_vec`
  from the checkpoint alongside `rms_scale`. Raw activations must go through
  `normalize_input` before `encode` — scaling by `rms_scale` alone skips the centering
  and reads feature activations against thresholds that never saw uncentered input (#55)
- **training**: center activations before RMS scaling and persist `mean_vec`. A residual
  stream carries a large constant offset that dominates `E[x^2]` (94-98% of it at most
  layers of gemma-4-E4B), so the uncentered RMS normalized the offset rather than the
  signal. **SAEs trained before this produce different activations — retrain** (#55)
- **core**: register the `google/gemma-4-E4B-it` SAE repo (#56)
- retarget quickstart at gemma-4-E4B-it layer-30 SAE with correct normalization (#56)
- add a per-layer ablation runner, `samples/run_ablation_per_layer.sh` (#55)
- support local model checkpoints (#51)
- add Qwen3.6-35B-A3B SAE quickstart notebook (quickstart_g4) (#50)

### Fix

- **extraction**: capture the layer's *input* via a forward pre-hook, not its output.
  `residual_N` is the stream entering layer N, so the HF path was off by one layer
  relative to the vLLM activations the SAEs were trained on (#55)
- **extraction**: drop the trailing space from the assistant prefill — both tokenizer
  families fold a leading space into the following token, so the extracted "decision
  token" was not a decision point at all (#55)
- **ablation**: apply the SAE's full input normalization when writing a reconstruction
  back into the residual stream, guard reports against a degenerate baseline pass rate,
  and exclude pairs where `anchor_tool == contrast_tool` (#55)
- **pipeline**: auto-suppress reasoning for the subject model, detected from the
  tokenizer's own chat template (#55)
- **paper**: correct layer types in Nemotron architecture diagram (#53)

## kiji-inspector-v0.5.1 (2026-07-24)

### Feat

- memory-safe shard loading and activation shard validation (#52)

## kiji-inspector-v0.5.0 (2026-07-20)

### Feat

- register released Qwen3.6-35B-A3B SAE repo; drop unpublished FP8 entry (#49)
- migrate vLLM extractor to native hidden-states connector; fix hybrid-MoE judge (#48)
- add Qwen3.6-35B-A3B subject-model support (HF + vLLM)
- Enhance ablation metrics with CATE and Wilcoxon statistical testing (#31)
- Align activation across Nemotron and Gemma3 models (#27)
- using Doubleword to generate pairs for sae training (#28)

### Fix

- never resample dead SAE features near end of training (#46)
- resolve ruff lint errors in ablation.py

### Perf

- stream contrastive feature activation collection (#45)

## kiji-inspector-v0.0.3 (2026-03-13)

### Feat

- Add Nemotron Nano FP8 and Gemma 3 27B to model registry (#21)
- Add SAE describe method and tests (#19)

## kiji-inspector-v0.0.2 (2026-03-13)

### Feat

- Add Nemotron Nano FP8 and Gemma 3 27B to model registry (#21)
- Add SAE describe method and tests (#19)
