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
