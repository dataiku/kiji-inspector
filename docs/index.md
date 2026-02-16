# Contrastive SAE Analysis of Agent Tool-Selection Decisions

## Overview

This project implements a complete mechanistic interpretability pipeline for understanding **how** an AI agent internally decides which tool to use. Rather than treating tool selection as a black-box classification, we decompose the agent's hidden representations into interpretable features using a **Sparse Autoencoder (SAE)** trained on contrastive activation data.

The pipeline extracts activations from the **NVIDIA Nemotron-3-Nano-30B-A3B-BF16** model at the precise moment it commits to a tool choice, trains a JumpReLU SAE to discover monosemantic features in that activation space, then uses contrastive analysis to identify which features correspond to specific decision factors.

## Architecture

```
                                          Subject Model
                                    (Nemotron-3-Nano-30B)
                                            |
User Request -----> Formatted Prompt -----> | -----> Hidden State at Decision Token
                    (ChatML + tools)        |              |
                                                           v
                                                    JumpReLU SAE
                                                     Encoder
                                                        |
                                                        v
                                                Sparse Feature Vector
                                                  (d_sae = 16384)
                                                        |
                                            +-----------+-----------+
                                            |                       |
                                    Contrastive              Feature
                                    Analysis                 Interpretation
                                    (Cohen's d)             (LLM labeling)
                                            |                       |
                                            v                       v
                                    Decision-Relevant        Human-Readable
                                    Features                 Explanations
```

## Pipeline Steps

The pipeline consists of six sequential steps, each building on the outputs of the previous:

| Step | Name | Input | Output | Key Algorithm |
|------|------|-------|--------|---------------|
| [1](step1_contrastive_pair_generation.md) | Contrastive Pair Generation | Scenario configs | Parquet shards of contrastive pairs | LLM-based synthetic data generation via vLLM |
| [2](step2_activation_extraction.md) | Activation Extraction | Contrastive pairs | NumPy shards of activation vectors | Forward hooks on transformer layers |
| [3](step3_sae_training.md) | SAE Training | Activation shards | Trained JumpReLU SAE checkpoint | JumpReLU with tanh sparsity loss |
| [4](step4_contrastive_activation_analysis.md) | Contrastive Activation Analysis | SAE + contrastive pairs | Ranked feature lists per contrast type | Cohen's d effect size |
| [5](step5_feature_interpretation.md) | Feature Interpretation | SAE + activations + features | Feature labels and decision report | LLM auto-labeling of max-activating examples |
| [6](step6_fuzzing_evaluation.md) | Fuzzing Evaluation | Feature labels + per-token activations | Accuracy metrics per feature | A/B LLM judge on token-highlighted texts |

## Key Design Decisions

### Decision Token Extraction

Every formatted prompt ends with the string `"I'll use the "`. The hidden state at this final token -- the **decision token** -- captures the model's internal representation at the moment it is about to commit to a tool name. All activation extraction targets this single position.

### Contrastive Pairs as Post-Hoc Probes

The SAE is trained on **all** activations from Step 2 without any contrastive signal. Contrastive pairs are used only in Step 4 as a post-hoc statistical test to identify which of the SAE's learned features correspond to decision-relevant dimensions. This separation means the SAE discovers the model's natural feature vocabulary, and the contrastive analysis merely queries that vocabulary.

### GPU Memory Isolation via Subprocesses

The pipeline uses two large models that cannot coexist in GPU memory: the generation model (Qwen3-VL-235B, used in Steps 1, 5c, and 6c) and the subject model (Nemotron-3-Nano-30B, used in Steps 2, 4, and 6a). The orchestrator spawns subprocesses via `multiprocessing.spawn` to ensure each model fully releases GPU memory before the next model loads.

### Multi-Domain Scenario System

The pipeline supports training on activations from diverse agent scenarios (tool selection, investment analysis, manufacturing, supply chain, customer support) through a JSON-based scenario configuration system. Each scenario defines its own system prompt, tool set, and contrast types. When no specific scenarios are selected, all `scenarios/*.json` files are loaded automatically.

## Models Used

| Model | Role | Steps | Loading |
|-------|------|-------|---------|
| Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 | Generator / Labeler / Judge | 1, 5c, 6c | vLLM with tensor + expert parallelism |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | Subject model (activation source) | 2, 4, 6a | HuggingFace Transformers with `device_map="auto"` |

Nemotron-3-Nano is a Mixture-of-Experts model (30B total parameters, 3B active per token) with a hidden dimension of 4096.

## Quick Start

```bash
# Full pipeline (steps 1 + 2 + 3)
uv run python -m pipeline 500000

# Individual steps
uv run python -m pipeline 500000 --step 1   # Generate pairs
uv run python -m pipeline --step 2           # Extract activations
uv run python -m pipeline --step 3           # Train SAE
uv run python -m pipeline --step 4           # Contrastive analysis
uv run python -m pipeline --step 5           # Interpret features
uv run python -m pipeline --step 6           # Fuzzing evaluation
```

## Output Directory Structure

```
output/
    pairs/
        shard_00000.parquet          # Contrastive pairs (Step 1)
        scenarios_meta.json          # Scenario configs used
    activations/
        shard_000000.npy             # Activation vectors, float16 (Step 2)
        metadata.json                # Model, layer, dimensions
        prompts.json                 # User request per activation vector
        contrastive_features.json    # Ranked features per contrast type (Step 4)
        feature_descriptions.json    # Feature labels and examples (Step 5)
        decision_report.json         # Per-contrast-type explanations (Step 5)
        fuzzing_results.json         # Per-feature evaluation scores (Step 6)
        fuzzing_summary.json         # Aggregate evaluation metrics (Step 6)
    sae_checkpoints/
        sae_final.pt                 # Trained SAE model (Step 3)
        feature_health.json          # Post-training feature health analysis
        metrics.jsonl                # Training loss curves
        config.json                  # Training hyperparameters
```

## Source Files

The codebase is organized into subpackages by functional area:

| Package | File | Purpose |
|---------|------|---------|
| (root) | `src/pipeline.py` | Main CLI orchestrator for all 6 steps |
| `data` | `src/data/scenario.py` | Scenario configuration loading and validation |
| `data` | `src/data/generator.py` | Contrastive pair generation via vLLM |
| `data` | `src/data/contrastive_dataset.py` | `ContrastivePair` / `ContrastiveDataset` data structures, Parquet I/O |
| `extraction` | `src/extraction/extractor.py` | Agent prompt formatting and shard-based activation extraction |
| `extraction` | `src/extraction/activation_extractor.py` | Low-level forward-hook-based activation capture from HuggingFace models |
| `sae` | `src/sae/model.py` | JumpReLU SAE architecture with STE gradients |
| `sae` | `src/sae/trainer.py` | Training loop with cosine LR, sparsity warmup, dead feature resampling |
| `analysis` | `src/analysis/contrastive_features.py` | Contrastive feature identification (Step 4) |
| `analysis` | `src/analysis/feature_interpreter.py` | Feature labeling via LLM and report generation |
| `analysis` | `src/analysis/fuzzing_evaluator.py` | A/B evaluation of feature explanations |
| `experiments` | `src/experiments/ablation.py` | Causal intervention experiments |
| `experiments` | `src/experiments/baselines.py` | Linear probe and PCA+k-means baselines |
| `experiments` | `src/experiments/layer_sweep.py` | Multi-layer SAE pipeline comparison |
| `utils` | `src/utils/stats.py` | Shared statistical functions (confidence intervals, hypothesis tests) |
