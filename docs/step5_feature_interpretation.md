# Step 5: Feature Interpretation

## Purpose

Transform the abstract SAE feature indices from Step 4 into human-readable labels and explanations. For each decision-relevant feature, this step identifies which prompts maximally activate it, generates a natural-language label via an LLM, and produces a per-contrast-type decision report explaining how the model makes tool-selection decisions.

## Sub-Steps

| Sub-step | Name | Input | Output |
|----------|------|-------|--------|
| 5a | Load activations from shards | Step 2 numpy shards + prompts.json | Deduplicated (prompts, activations) arrays |
| 5b | SAE encode + collect examples | SAE checkpoint + activations | Top/bottom activating prompts per feature |
| 5c | Label features via LLM | Max-activating examples | feature_descriptions.json |
| 5d | Generate decision report | Labels + contrastive features | decision_report.json |

## Source Files

| File | Key Components |
|------|----------------|
| `generate_training_set.py` | `_run_step5()` |
| `src/feature_interpreter.py` | `load_activations_from_shards()`, `collect_max_activating_examples()`, `label_features_via_llm()`, `generate_explanation_report()` |

## Sub-Step 5a: Load Activations from Shards

Rather than re-running Nemotron inference, this step loads the cached activation vectors and prompt texts saved by Step 2:

1. Read `metadata.json` and `prompts.json` from the activations directory
2. Load and concatenate all `shard_*.npy` files
3. **Deduplicate**: Keep only the first activation for each unique prompt string

Deduplication is necessary because the same user request may appear in multiple contrastive pairs (as anchor in one pair, contrast in another), resulting in duplicate activation vectors.

```python
seen = set()
for prompt, act in zip(all_prompts, all_activations):
    if prompt not in seen:
        seen.add(prompt)
        unique_prompts.append(prompt)
        unique_activations.append(act)
```

The output is a pair: `(prompts: list[str], activations: ndarray of shape (N, d_model))` with activations cast to float32.

## Sub-Step 5b: SAE Encode and Collect Examples

### Feature Selection

The features to analyze come from Step 4's `contrastive_features.json`. All unique feature indices across all contrast types are collected into a deduplicated set.

### Encoding

All activation vectors are encoded through the SAE in chunks of 4096 to avoid GPU OOM:

```python
for i in range(0, N, 4096):
    chunk = torch.from_numpy(activations[i:i+4096]).to(device, dtype=sae_dtype)
    features = sae.encode(chunk)  # (chunk_size, d_sae)
    all_features.append(features.cpu())

feature_matrix = torch.cat(all_features, dim=0)  # (N, d_sae)
```

### Collecting Top/Bottom Examples

For each target feature index $j$:

1. Extract the feature column: `col = feature_matrix[:, j]` -- shape `(N,)`
2. Find the top-$n$ prompts with highest activation via `torch.topk(col, n)`
3. Find the bottom-$n$ prompts with lowest activation via `torch.topk(col, n, largest=False)`
4. Record statistics:
   - `mean_activation`: Mean of the feature across all prompts
   - `max_activation`: Maximum activation value
   - `frac_nonzero`: Fraction of prompts where the feature is active (> 0)

Default: `top_n=20`, `bottom_n=10`.

### Output

```python
{
    feature_index: {
        "top": [{"prompt": "...", "activation": 12.345}, ...],     # 20 entries
        "bottom": [{"prompt": "...", "activation": 0.0}, ...],     # 10 entries
        "mean_activation": 0.234,
        "max_activation": 15.678,
        "frac_nonzero": 0.156
    }
}
```

## Sub-Step 5c: Label Features via LLM

### GPU Memory Isolation

Labeling requires the Qwen3-VL-235B model, which cannot coexist with the SAE or Nemotron in GPU memory. The labeling runs in a `multiprocessing.spawn` subprocess that loads vLLM, generates all labels, saves to a temp file, and exits.

### LLM Prompt Template

For each feature, the LLM receives the top-15 highest-activating prompts with their activation values and the bottom-8 near-zero prompts:

```
You are analyzing features learned by a Sparse Autoencoder (SAE) trained
on an AI agent's internal activations at the moment it decides which
tool to use.

For feature #{feature_index}, here are the prompts that MOST activate
this feature:
  [12.3456] "Search for the latest quarterly earnings report for NVDA"
  [11.2345] "Find the most recent revenue data for Tesla"
  [10.8901] "Look up current financial statements for Apple"
  ...

And here are prompts where this feature is INACTIVE (near-zero):
  [0.0000] "Write a Python script to sort a list of numbers"
  [0.0001] "Create a new config file for the deployment"
  ...

This feature is active in 15.6% of all prompts, with mean activation
0.234 and max 15.678.

Based on these examples, provide:
1. A short label (3-8 words)
2. A one-sentence description
3. Your confidence (high/medium/low)

Output as JSON:
{"label": "...", "description": "...", "confidence": "high|medium|low"}
```

### Sampling Parameters

```python
SamplingParams(
    temperature=0.3,    # Low temperature for deterministic labels
    top_p=0.9,
    max_tokens=500,
)
```

The low temperature (0.3 vs. 0.7 for generation) produces more consistent, deterministic labels.

### Error Handling

If the LLM output cannot be parsed as JSON, the feature receives:
```json
{"label": "parse_error", "description": "<first 200 chars of raw output>", "confidence": "low"}
```

## Sub-Step 5d: Generate Decision Report

### feature_descriptions.json

Combines labels, activation statistics, and example prompts for each feature:

```json
{
  "7342": {
    "label": "Financial Data Retrieval Intent",
    "description": "Activates when the request involves looking up financial metrics or market data.",
    "confidence": "high",
    "mean_activation": 0.234,
    "max_activation": 15.678,
    "frac_nonzero": 0.156,
    "top_examples": ["Search for quarterly earnings...", ...],
    "bottom_examples": ["Write a Python script...", ...]
  }
}
```

### decision_report.json

For each contrast type, takes the top-10 features from Step 4 and generates a plain-language explanation:

```json
{
  "read_vs_write": {
    "num_pairs": 2500,
    "explanation": "When deciding between read vs write tools, the model relies on features like \"Data Retrieval Intent\" (stronger in anchor prompt), \"Modification Action Signal\" (stronger in contrast prompt), and \"Schema Inspection Pattern\" (stronger in anchor prompt).",
    "key_features": [
      {
        "feature_index": 7342,
        "label": "Data Retrieval Intent",
        "description": "Activates for data lookup requests",
        "confidence": "high",
        "mean_abs_diff": 0.284,
        "anchor_mean_activation": 0.156,
        "contrast_mean_activation": 0.042
      }
    ]
  }
}
```

### Plain Language Explanation Algorithm

```python
def _build_plain_language_explanation(contrast_type, features):
    labeled = [f for f in features if f["label"] not in ("unlabeled", "parse_error")]
    top3 = labeled[:3]

    for f in top3:
        direction = "anchor" if f["anchor_mean"] > f["contrast_mean"] else "contrast"
        # e.g. '"Data Retrieval Intent" (stronger in anchor prompt)'

    return f"When deciding between {contrast_type}, the model relies on features like {parts}."
```

The explanation identifies the top 3 labeled features and indicates which side of the contrast (anchor or contrast) each feature fires more strongly on.

## Output Files

```
output/activations/
    feature_descriptions.json    # Per-feature labels, stats, examples
    decision_report.json         # Per-contrast-type explanations
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--label-top-n` | 20 | Top-activating examples per feature |
| `--label-bottom-n` | 10 | Near-zero examples per feature |
| `--qwen-model` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | LLM for labeling |
| `--generation-tp-size` | 4 | Tensor parallel size |
| `--max-model-len` | 16384 | Max sequence length |

## Interpretation Caveats

When the SAE is trained on one domain (e.g., tool selection) but applied to another (e.g., investment analysis), the feature **labels** reflect the training domain's vocabulary. However, the activation **patterns** (which features fire, their relative strengths, how they change across steps) still capture real differences in the model's internal processing. The labels are human-readable approximations, not ground truth -- the fuzzing evaluation in Step 6 quantifies their accuracy.
