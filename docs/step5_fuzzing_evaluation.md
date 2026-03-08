# Step 5: Fuzzing Evaluation

## Purpose

Evaluate whether the feature labels from Step 4 correctly identify **which tokens** activate each feature, not just which texts. This catches explanations that are "right for the wrong reasons" -- a label might correctly predict which prompts activate a feature but for the wrong conceptual reason. By testing at the token level, we verify that the label identifies the actual signal the SAE feature detects.

This approach is based on [Eleuther AI's autointerp methodology](https://blog.eleuther.ai/autointerp/).

## Sub-Steps

| Sub-step | Name | Input | Output |
|----------|------|-------|--------|
| 5a | Per-token activation extraction | Nemotron + formatted prompts | Per-token activation matrices |
| 5b | Build fuzzing examples | SAE + per-token activations | A/B comparison pairs |
| 5c | LLM judge evaluation | Qwen3-VL judge prompts | Raw judgments |
| 5d | Compute metrics | Judgments + ground truth | fuzzing_results.json, fuzzing_summary.json |

## Source Files

| File | Key Components |
|------|----------------|
| `src/pipeline.py` | `_run_step5()` |
| `src/analysis/fuzzing_evaluator.py` | `FuzzingExample`, `extract_per_token_activations()`, `build_fuzzing_examples()`, `evaluate_fuzzing()`, `compute_fuzzing_metrics()` |

## Architecture

```
feature_descriptions.json (Step 4)
    |
    v
Select top/bottom example prompts per feature
    |
    v
Nemotron-3-Nano-30B (token_positions="all")
    |
    v
Per-token activation matrices: (seq_len, d_model) per prompt
    |
    v
SAE.encode() per token
    |
    v
Per-token feature activations: (seq_len, d_sae) per prompt
    |
    v
Find user-request span via ChatML markers
    |
    v
Highlight top-K tokens within user request
    |
    v
Build A/B pairs (randomized order)
    |
    v
Qwen3-VL LLM judge (subprocess)
    |
    v
Parse A/B answers, compute accuracy
```

## Sub-Step 5a: Per-Token Activation Extraction

Unlike Step 1 (which extracts only the decision token), Step 5a extracts activations for **every token** in each prompt by setting `token_positions="all"`:

```python
config = ActivationConfig(
    model_name=nemotron_model,
    layers=layers,
    dtype=torch.bfloat16,
    token_positions="all",    # All tokens, not just the last
)
```

For each prompt, the output is:
- `token_strings`: List of token strings (from `convert_ids_to_tokens`)
- `token_activations`: NumPy array of shape `(seq_len, d_model)`

These are extracted in batches and stored in memory for Sub-step 5b.

## Sub-Step 5b: Build Fuzzing Examples

### User Request Span Detection

The formatted prompt contains system, user, and assistant turns in ChatML format. We need to isolate only the **user request tokens** for highlighting, since highlighting system prompt or tool description tokens would be meaningless.

The span is found deterministically using ChatML structural markers:

```python
def _find_user_request_span(token_strings, user_request, tokenizer):
    # Find <|im_start|> positions
    im_start_positions = [i for i, tok in enumerate(token_strings) if "<|im_start|>" in tok]

    # User turn is the 2nd <|im_start|> block (index 1)
    # System turn is index 0
    user_turn_start = im_start_positions[1]
    user_turn_end = im_end_positions[1]

    # Skip past "<|im_start|>user\n" role tokens
    content_start = user_turn_start + 1
    while token_text in ("user", ""):
        content_start += 1

    return (content_start, user_turn_end)
```

### Token Highlighting

For each (prompt, feature) combination:

1. Encode per-token activations through the SAE: `feat_acts = sae.encode(tok_acts)` -- shape `(seq_len, d_sae)`
2. Extract the feature column: `feature_col = feat_acts[:, feature_id]` -- shape `(seq_len,)`
3. Restrict to user request span tokens
4. Select top-K tokens by activation (adaptive K: at most 1/3 of user tokens)
5. Reconstruct user request text with `<<double angle brackets>>` around highlighted tokens

```python
k = max(1, min(top_k_tokens, req_token_count // 3))
sorted_indices = np.argsort(req_activations)
top_k_local = sorted_indices[-k:]
```

Example output:
```
Show me the <<current>> database <<schema>> for the <<users>> table
```

### Two Types of Fuzzing Examples

#### Token-Level Examples

Pairs a highlighted top-activating prompt with a highlighted bottom-activating prompt. The A/B order is **randomized** to prevent position bias:

```
Feature: "Database Schema Inspection"

Text A (with highlights):
  Show me the <<current>> database <<schema>> for the users table

Text B (with highlights):
  Write a <<Python>> script to <<sort>> a list of numbers

Which text's highlights better match the feature?
```

The judge must identify which highlighted text better matches the feature label.

#### Prompt-Level Examples

Pairs an unhighlighted top-activating prompt with a bottom-activating prompt:

```
Feature: "Database Schema Inspection"

Text A: Show me the current database schema for the users table
Text B: Write a Python script to sort a list of numbers

Which text would more strongly activate this feature?
```

This tests whether the label predicts the correct prompt (without token-level hints).

### A/B Randomization

For both types, the correct answer (top-activating text) is randomly assigned to position A or B with 50% probability. The `is_correctly_fuzzed` field tracks which position contains the correct answer:

```python
if random.random() < 0.5:
    text_a, text_b = top_text, bottom_text
    correct_is_a = True
else:
    text_a, text_b = bottom_text, top_text
    correct_is_a = False
```

## Sub-Step 5c: LLM Judge Evaluation

### Judge Prompt Templates

**Token-level judge prompt:**
```
You are evaluating which text has better token highlighting for a feature.

Feature label: "{label}"
Feature description: "{description}"

Text A (with highlights):
{text_a}

Text B (with highlights):
{text_b}

In which text do the <<highlighted>> tokens better match the feature
"{label}"? Consider both:
1. Whether the highlighted tokens are relevant to the described concept
2. Whether the overall text context relates to the feature

Answer with a single letter: A or B.
```

**Prompt-level judge prompt:**
```
You are evaluating which text better matches a feature explanation.

Feature label: "{label}"
Feature description: "{description}"

Text A:
{text_a}

Text B:
{text_b}

Which text would more strongly activate a feature described as "{label}"?

Answer with a single letter: A or B.
```

### Subprocess Execution

The judge runs in a `multiprocessing.spawn` subprocess to isolate GPU memory:

```python
SamplingParams(
    temperature=0.2,     # Very low for deterministic judgments
    top_p=0.9,
    max_tokens=20,       # Short answer: just "A" or "B"
)
```

### Answer Parsing

The judge's response is searched for `A` or `B` using a word-boundary regex:

```python
picked_a = bool(re.search(r"\bA\b", raw_judgment))
predicted_correct = (picked_a == example.is_correctly_fuzzed)
```

## Sub-Step 5d: Compute Metrics

### Per-Feature Metrics

For each feature, accuracy is computed separately for token-level and prompt-level examples:

$$\text{acc}_{\text{token}} = \frac{\text{correct token-level judgments}}{\text{total token-level examples}}$$

$$\text{acc}_{\text{prompt}} = \frac{\text{correct prompt-level judgments}}{\text{total prompt-level examples}}$$

The **combined score** weights token-level accuracy more heavily:

$$\text{combined} = 0.7 \cdot \text{acc}_{\text{token}} + 0.3 \cdot \text{acc}_{\text{prompt}}$$

Token-level accuracy is weighted 70% because it tests whether the label identifies the correct *mechanism* (which specific tokens trigger the feature), not just the correct text.

### Aggregate Metrics

| Metric | Description |
|--------|-------------|
| Mean combined score | Average across all features |
| Std combined score | Standard deviation across features |
| Mean token-level accuracy | Average token-level accuracy |
| By confidence tier | Mean score grouped by label confidence (high/medium/low) |

### Quality Tiers

| Tier | Combined Score | Interpretation |
|------|---------------|----------------|
| Excellent | > 0.8 | Label reliably predicts both which prompts and which tokens activate |
| Good | 0.6 - 0.8 | Label captures the general concept but may miss specific token patterns |
| Poor | < 0.6 | Label is unreliable; feature may be polysemantic or mislabeled |

## Output Files

### fuzzing_results.json

```json
{
  "per_feature": {
    "7342": {
      "label": "Database Schema Inspection",
      "confidence": "high",
      "token_level": {"accuracy": 0.9, "num_examples": 10},
      "prompt_level": {"accuracy": 1.0, "num_examples": 1},
      "combined_score": 0.93
    }
  },
  "details": [
    {
      "feature_id": 7342,
      "kind": "token_level",
      "text": "A: Show me the...\nB: Write a Python...",
      "fuzzed_text": "Show me the <<current>> database <<schema>>...",
      "fuzzed_text_b": "Write a <<Python>> script to <<sort>>...",
      "is_correctly_fuzzed": true,
      "predicted_correct": true,
      "raw_judgment": "A",
      "is_prediction_correct": true
    }
  ]
}
```

### fuzzing_summary.json

```json
{
  "num_features_evaluated": 150,
  "num_examples_total": 1650,
  "combined_score": {"mean": 0.782, "std": 0.145},
  "token_level_accuracy": {"mean": 0.756, "std": 0.183},
  "by_confidence": {
    "high": {"count": 45, "mean_score": 0.891},
    "medium": {"count": 72, "mean_score": 0.764},
    "low": {"count": 33, "mean_score": 0.623}
  },
  "quality_tiers": {
    "excellent_above_0.8": 67,
    "good_0.6_to_0.8": 52,
    "poor_below_0.6": 31
  }
}
```

## Baseline and Interpretation

Random guessing yields 50% accuracy (binary A/B choice). Scores significantly above 50% indicate the feature labels carry real predictive information about the feature's activation pattern.

| Score Range | Interpretation |
|-------------|----------------|
| ~50% | Label is no better than random; feature may be mislabeled |
| 60-70% | Label captures some signal but is imprecise |
| 70-80% | Label is a good description of the feature |
| 80-90% | Label is highly accurate |
| >90% | Label is an excellent description of both concept and mechanism |

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fuzz-top-k-tokens` | 5 | Tokens to highlight per example |
| `--fuzz-examples-per-feature` | 10 | Maximum A/B pairs per feature |
| `--fuzz-batch-size` | 64 | GPU batch size for per-token extraction |
| `--qwen-model` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | LLM judge model |
| `--generation-tp-size` | 4 | Tensor parallel size for judge |
| `--max-model-len` | 16384 | Max sequence length for judge |
