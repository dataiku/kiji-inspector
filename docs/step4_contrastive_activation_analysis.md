# Step 4: Contrastive Activation Analysis

## Purpose

Identify which SAE features are **decision-relevant** by measuring how they respond differently to anchor vs. contrast prompts within each contrastive pair. A feature that consistently activates more strongly for one side of a contrast type (e.g., higher for "read" requests than "write" requests) is a feature that encodes the corresponding decision factor.

This step does **not** modify the SAE. It uses the trained SAE from Step 3 as a fixed feature extractor and applies statistical tests to rank features by their discriminative power.

## Source Files

| File | Key Components |
|------|----------------|
| `src/pipeline.py` | `_run_step4()` |
| `src/analysis/contrastive_features.py` | `identify_contrastive_features()` |
| `src/extraction/extractor.py` | `build_agent_prompt()` (prompt formatting) |
| `src/extraction/activation_extractor.py` | `ActivationExtractor` (live extraction) |
| `src/sae/model.py` | `JumpReLUSAE.encode()` |

## Architecture

```
ContrastivePair
    |
    +-------> build_agent_prompt(anchor)  -----+
    |                                           |
    +-------> build_agent_prompt(contrast) -----+
                                                |
                                                v
                                    Nemotron-3-Nano-30B
                                    (extract at layer 20)
                                                |
                                    +-----------+-----------+
                                    |                       |
                              anchor_act (4096)     contrast_act (4096)
                                    |                       |
                                    v                       v
                                SAE.encode()           SAE.encode()
                                    |                       |
                              anchor_feat (16384)   contrast_feat (16384)
                                    |                       |
                                    +--------> Per-feature statistics
                                               |
                                               v
                                    Cohen's d effect size
                                    Mean absolute difference
                                               |
                                               v
                                    Top-K features per contrast type
```

## Algorithm

### 1. Group Pairs by Contrast Type

All contrastive pairs are grouped by their `contrast_type` field (e.g., `"read_vs_write"`, `"internal_vs_external"`). Each group is analyzed independently.

### 2. Extract Live Activations

For each pair, both the anchor and contrast prompts are formatted using the pair's scenario-specific tools and system prompt, then fed through Nemotron to extract the decision-token activation:

```python
for pair in ct_pairs:
    scenario = scenarios_meta.get(pair.scenario_name, default_scenario())
    anchor_prompt = build_agent_prompt(scenario.system_prompt, scenario.tools, pair.anchor_prompt)
    contrast_prompt = build_agent_prompt(scenario.system_prompt, scenario.tools, pair.contrast_prompt)
```

Extraction is batched for throughput: all anchor prompts and all contrast prompts for a contrast type are concatenated and processed in a single batch.

### 3. Encode Through SAE

Raw activations are encoded through the trained SAE:

$$\mathbf{f}_{\text{anchor}} = \text{SAE.encode}(\mathbf{x}_{\text{anchor}}) \in \mathbb{R}^{d_{\text{sae}}}$$
$$\mathbf{f}_{\text{contrast}} = \text{SAE.encode}(\mathbf{x}_{\text{contrast}}) \in \mathbb{R}^{d_{\text{sae}}}$$

### 4. Compute Per-Feature Statistics

For each feature $j$ across all $n$ pairs of a given contrast type:

**Mean activations:**
$$\bar{f}^{(a)}_j = \frac{1}{n_a} \sum_{i=1}^{n_a} f^{(a)}_{ij}, \quad \bar{f}^{(c)}_j = \frac{1}{n_c} \sum_{i=1}^{n_c} f^{(c)}_{ij}$$

**Variances:**
$$s^2_{a,j} = \text{Var}(f^{(a)}_{:,j}), \quad s^2_{c,j} = \text{Var}(f^{(c)}_{:,j})$$

**Mean absolute difference:**
$$\Delta_j = \frac{1}{n} \sum_{i=1}^{n} |f^{(a)}_{ij} - f^{(c)}_{ij}|$$

### 5. Cohen's d Effect Size

Cohen's d measures the standardized difference between the anchor and contrast distributions for each feature:

$$d_j = \frac{|\bar{f}^{(a)}_j - \bar{f}^{(c)}_j|}{s_{\text{pooled},j}}$$

where the pooled standard deviation is:

$$s_{\text{pooled},j} = \sqrt{\frac{(n_a - 1) \cdot s^2_{a,j} + (n_c - 1) \cdot s^2_{c,j}}{n_a + n_c - 2}}$$

A small epsilon ($10^{-8}$) is added to the denominator for numerical stability.

#### Interpretation of Cohen's d

| $d$ value | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible effect |
| 0.2 - 0.5 | Small effect |
| 0.5 - 0.8 | Medium effect |
| > 0.8 | Large effect |

### 6. Feature Filtering

Two filters are applied before ranking:

**Effect size filter:** Only features with $d_j \geq d_{\min}$ (default: 0.3) pass. This removes features with negligible systematic differences.

**Minimum activation filter:** At least one condition must have meaningful activation:

$$\max(|\bar{f}^{(a)}_j|, |\bar{f}^{(c)}_j|) > a_{\min}$$

where $a_{\min}$ defaults to 0.01. This removes features that are near-zero on both sides (not decision-relevant even if the tiny difference is statistically significant).

### 7. Top-K Selection

The remaining features are ranked by mean absolute difference $\Delta_j$ and the top $K$ (default: 200) are selected for each contrast type.

## Output Format

### contrastive_features.json

```json
{
  "read_vs_write": {
    "num_pairs": 2500,
    "top_features": [
      {
        "rank": 0,
        "feature_index": 7342,
        "mean_abs_diff": 0.284531,
        "anchor_mean_activation": 0.156234,
        "contrast_mean_activation": 0.042891,
        "cohens_d": 1.423567,
        "anchor_std": 0.089123,
        "contrast_std": 0.067234
      }
    ],
    "num_filtered_by_effect_size": 14200,
    "num_filtered_by_min_activation": 312
  },
  "_summary": {
    "total_feature_slots": 2600,
    "unique_features": 1847,
    "dedup_ratio": 0.7104,
    "features_in_multiple_contrasts": 423,
    "max_contrast_overlap": 5,
    "top_k_per_type": 200,
    "min_effect_size": 0.3,
    "min_activation": 0.01
  }
}
```

### Summary Block

The `_summary` key provides deduplication statistics:

| Field | Description |
|-------|-------------|
| `total_feature_slots` | Sum of top-K across all contrast types |
| `unique_features` | Number of distinct feature indices |
| `dedup_ratio` | `unique / total` -- lower means more feature sharing across contrast types |
| `features_in_multiple_contrasts` | Features appearing in 2+ contrast type rankings |
| `max_contrast_overlap` | Maximum number of contrast types sharing one feature |

A dedup_ratio near 1.0 means each contrast type uses mostly unique features. A lower ratio suggests shared decision mechanisms across contrast types.

## Per-Feature Output Fields

| Field | Description |
|-------|-------------|
| `rank` | Position in the ranking (0 = most discriminative) |
| `feature_index` | SAE feature index (0 to d_sae-1) |
| `mean_abs_diff` | Average absolute difference between anchor and contrast activations |
| `anchor_mean_activation` | Mean feature activation across anchor prompts |
| `contrast_mean_activation` | Mean feature activation across contrast prompts |
| `cohens_d` | Standardized effect size |
| `anchor_std` | Standard deviation of feature activation across anchor prompts |
| `contrast_std` | Standard deviation of feature activation across contrast prompts |

## Why Live Extraction Instead of Cached?

Step 4 re-extracts activations through Nemotron rather than reusing the Step 2 numpy shards. This is because Step 2 saves only the raw `d_model`-dimensional activation vectors without pairing information (which vector came from which pair's anchor vs. contrast). Step 4 needs the paired structure to compute per-pair feature differences.

An alternative would be to save pair indices in Step 2, but the current approach keeps Step 2 simple (just a flat stream of activation vectors suitable for SAE training) and Step 4 self-contained.

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sae-checkpoint` | Auto-detected from Step 3 | Path to trained SAE |
| `--top-k-features` | 200 | Features per contrast type |
| `--min-effect-size` | 0.3 | Minimum Cohen's d threshold |
| `--min-activation` | 0.01 | Minimum mean activation threshold |
| `--nemotron-model` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Subject model |
| `--layers` | `[20]` | Layers to hook |
| `--batch-size` | 512 | GPU batch size for extraction |
