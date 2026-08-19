# Step 2: SAE Training

## Purpose

Step 2 trains one JumpReLU sparse autoencoder (SAE) per selected transformer
layer using the decision-token activation vectors produced by Step 1. The SAE
learns a sparse dictionary whose decoder reconstructs the normalized residual
stream and whose active dimensions become the candidate features analyzed in
Steps 3-5.

When multiple layers are passed to the pipeline, they are trained sequentially
and each layer receives its own checkpoint directory.

## Source Files

| File | Key components |
|------|----------------|
| `src/kiji_inspector/pipeline.py` | `_run_step2()`, `train_sae_step()`, Step 2 CLI arguments |
| `src/kiji_inspector/training/model.py` | Training-time `JumpReLUSAE`, loss computation, initialization |
| `src/kiji_inspector/training/trainer.py` | `SAETrainingConfig`, activation buffer, calibration, adaptive L1, resampling, checkpointing, health analysis |
| `src/kiji_inspector/core/sae_core.py` | Shared JumpReLU operation and inference-time SAE methods |

## Recommended Invocation

The calibrated, target-L0 configuration used for the six-layer Nemotron run is:

```bash
uv run python -m kiji_inspector.pipeline \
  --step 2 \
  --pairs-dir output/pairs \
  --output-dir output \
  --subject-model /home/shadeform/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp \
  --layers 6 13 20 27 34 43 \
  --d-sae 10752 \
  --sae-batch-size 256 \
  --sae-epochs 10 \
  --target-l0 75 \
  --auto-calibrate-threshold \
  --no-sae-resampling
```

This writes each SAE to
`output/layer_<N>/sae_checkpoints/sae_final.pt`. Keep the default per-layer
checkpoint directories for a multi-layer run; a single shared
`--sae-checkpoint-dir` would make the layers write into the same directory.

Step 2 reads its training vectors from `--output-dir`. `--pairs-dir` is used
only to display scenario metadata at pipeline startup; if
`scenarios_meta.json` is absent, the loader falls back to the built-in default
scenario.

## Input Normalization

`CachedActivationBuffer` reads `shard_*.npy` and `metadata.json` from each
layer's Step 1 activation directory. It computes, over all finite vectors:

$$
\boldsymbol{\mu} = \mathbb{E}[\mathbf{x}], \qquad
s = \sqrt{\frac{1}{d_{\text{model}}}
    \sum_k \operatorname{Var}(x_k)}
$$

and trains on centered, RMS-scaled activations:

$$
\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{s}.
$$

Centering is important because residual streams can have a large constant
offset. Without centering, the same threshold and sparsity settings can produce
very different behavior at different layers.

The mean vector and centered RMS scale are embedded in `sae_final.pt`. Inference
code operating on raw model activations must use:

```python
x_norm = sae.normalize_input(x_raw)
features = sae.encode(x_norm)
reconstruction_raw = sae.denormalize_output(sae.decode(features))
```

Reported training and feature-health reconstruction MSE values are in this
normalized space.

The buffer shuffles shard order and rows for training, drops non-finite rows,
converts vectors to bfloat16, and drops incomplete batches at the end of each
shard. Consequently, the available number of steps is the sum of complete
batches per shard multiplied by `--sae-epochs`. A larger `--sae-steps` value is
clamped to that available count.

## JumpReLU SAE Architecture

For normalized input $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$:

$$
\mathbf{z} = (\mathbf{x} - \mathbf{b}_{\text{dec}})W_{\text{enc}}
             + \mathbf{b}_{\text{enc}},
$$

$$
\mathbf{f} = \operatorname{JumpReLU}(\mathbf{z}, \boldsymbol{\theta})
            = \mathbf{z} \odot \mathbb{1}[\mathbf{z} > \boldsymbol{\theta}],
$$

$$
\hat{\mathbf{x}} = \mathbf{f}W_{\text{dec}} + \mathbf{b}_{\text{dec}}.
$$

The decoder bias is shared: it is subtracted before encoding and added after
decoding.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| $W_{\text{enc}}$ | $(d_{\text{model}}, d_{\text{sae}})$ | Encoder directions |
| $\mathbf{b}_{\text{enc}}$ | $(d_{\text{sae}},)$ | Encoder bias |
| $\boldsymbol{\theta}$ | $(d_{\text{sae}},)$ | Learnable per-feature thresholds |
| $W_{\text{dec}}$ | $(d_{\text{sae}}, d_{\text{model}})$ | Decoder directions |
| $\mathbf{b}_{\text{dec}}$ | $(d_{\text{model}},)$ | Shared decoder/pre-encoder bias |

The pipeline auto-selects $d_{\text{sae}} = 4d_{\text{model}}$ when `--d-sae`
is omitted. The parameter count is
$2d_{\text{model}}d_{\text{sae}} + 2d_{\text{sae}} + d_{\text{model}}$.

### Gradients through JumpReLU

The forward pass creates exact zeros. For the pre-activation, gradients pass
only through active features:

$$
\frac{\partial \mathcal{L}}{\partial z_j}
= \frac{\partial \mathcal{L}}{\partial f_j}
  \mathbb{1}[z_j > \theta_j].
$$

The threshold gradient uses a rectangular approximation around the jump:

$$
\frac{\partial \mathcal{L}}{\partial \theta_j}
= -\sum_i \frac{\partial \mathcal{L}}{\partial f_{ij}} z_{ij}
  \frac{\mathbb{1}[|z_{ij}-\theta_j| < \epsilon]}{2\epsilon},
$$

where `bandwidth` $\epsilon$ defaults to `0.001`.

## Loss and Sparsity Control

The reconstruction loss is mean squared error:

$$
\mathcal{L}_{\text{recon}} = \operatorname{MSE}(\hat{\mathbf{x}}, \mathbf{x}).
$$

The differentiable sparsity term is a tanh-smoothed approximation to L0:

$$
\mathcal{L}_{\text{sparse}}
= \frac{1}{B}\sum_i\sum_j
  \operatorname{ReLU}\left(
    \tanh\left(\frac{z_{ij}-\theta_j}{\epsilon}\right)
  \right).
$$

The total loss is:

$$
\mathcal{L}_{\text{total}}
= \mathcal{L}_{\text{recon}} + \lambda(t)\mathcal{L}_{\text{sparse}}.
$$

The true L0 used for monitoring is the mean number of nonzero SAE features per
input vector.

### Fixed-L1 Mode

If `--target-l0` is omitted, the trainer uses `--l1-coefficient` as the target
coefficient after sparsity warmup. This preserves the original training mode,
but one coefficient can yield different L0 values across layers.

### Target-L0 Mode

If `--target-l0` is set, an adaptive controller adjusts the L1 coefficient at
each logging interval after sparsity warmup. It:

- tracks an exponential moving average of L0;
- computes error in log space so over- and undershoot are treated symmetrically;
- uses proportional-integral updates with a maximum 1.2x change per update;
- clamps L1 between `1e-5` and `--l1-max` (default `0.1`);
- freezes escalation when changing L1 has no measurable authority over L0, and
  re-arms if L0 later moves into a new regime.

Inspect `sparsity/l0`, `sparsity/l0_ema`,
`sparsity/current_l1_coef`, and `sparsity/l1_controller_frozen` in
`metrics.jsonl` when diagnosing convergence.

### Threshold Auto-Calibration

`--auto-calibrate-threshold` requires `--target-l0`. Before training, the
trainer collects eight activation batches, computes each feature's empirical
pre-activation quantile, and initializes its threshold to a common target firing
rate. The CLI calibration target is four times the requested training L0:

$$
L_{0,\text{calibration}}
= \min(d_{\text{sae}} - 1,\;4L_{0,\text{target}}).
$$

For example, `--target-l0 75` warm-starts at approximately L0 300. The looser
initial target leaves room for features to learn before the adaptive sparsity
controller brings the run toward its final target.

Calibration is skipped when `--sae-resume` is supplied because the checkpoint
already contains trained thresholds.

## Training Schedule

AdamW uses a peak learning rate of `3e-4`, betas `(0.9, 0.999)`, zero weight
decay, fused CUDA kernels when available, and gradient clipping at norm `1.0`.
After every optimizer step, each decoder row is normalized to unit norm.

The learning rate has a 5% linear warmup followed by cosine decay with a floor
of 10% of the peak learning rate. The sparsity coefficient has an independent
10% linear warmup.

With `auto_scale_steps=True` (the default), step-based settings are derived from
the effective total step count:

| Setting | Fraction of training |
|---------|---------------------|
| Learning-rate warmup | 5% |
| Sparsity warmup | 10% |
| Dead-feature check interval | 20% |
| Checkpoint interval | 25% |
| Metrics logging interval | 2% |

Pass `--no-auto-scale-steps` to retain the dataclass values instead. On CUDA,
the SAE is compiled with `torch.compile(mode="max-autotune", fullgraph=True)`.
SAE training uses one GPU; it does not use `DataParallel`.

## Dead-Feature Resampling

Dead-feature resampling is enabled by default and can be disabled with
`--no-sae-resampling`. At each eligible resampling interval, the trainer checks
20 batches and marks as dead every feature that never exceeds
`dead_feature_threshold=1e-6`.

The last interval is reserved for recovery: a resampling event runs only if at
least one full `resample_every` interval remains before training ends.

For up to as many dead features as there are candidate inputs, the trainer:

1. Collects the top 10% highest-reconstruction-loss inputs from five batches.
2. Uses a normalized high-loss input plus normalized Gaussian noise (`0.2`
   scale) as a new encoder and decoder direction.
3. Resets the feature's encoder bias to zero.
4. Sets its threshold from the empirical pre-activation quantile corresponding
   to `target_l0 / d_sae` in target-L0 mode. In fixed-L1 mode, it preserves the
   trained threshold scale by using the current median threshold.
5. Clears the Adam moments for the resampled encoder column, decoder row,
   encoder bias, and threshold so stale optimizer state cannot immediately push
   the replacement back toward its old state.

Target-aware thresholds prevent a large resampling event from making roughly
half of the replacement features active and destroying calibrated sparsity.

For a controlled calibrated baseline, use `--no-sae-resampling`. Enable
resampling when intermediate diagnostics show that dead capacity is the problem
and the run leaves enough post-resampling steps for replacement features to
train.

## Checkpointing and Resume

Periodic checkpoints contain model, optimizer, scheduler, step, configuration,
recent metrics, and a timestamp. Only the latest three `step_*.pt` checkpoints
are retained. `sae_final.pt` is the compact inference checkpoint and includes
the input mean and RMS scale.

`--sae-resume PATH` restores the model, optimizer, scheduler, and step. Threshold
auto-calibration is skipped on resume. The adaptive L0 controller's internal EMA
and integral state are not checkpointed; target-L0 control restarts from the
configured `--l1-coefficient`.

Per-layer outputs are:

```text
output/layer_<N>/sae_checkpoints/
    sae_final.pt          # Compact final model for Steps 3-5
    step_<K>.pt           # Up to three resumable checkpoints
    config.json           # Effective training configuration
    metrics.jsonl         # Logged loss, L0, threshold, and controller metrics
    feature_health.json   # Aggregate post-training diagnostics
    firing_rates.npy      # Per-feature firing rates
```

## Post-Training Feature Health

The final health pass processes up to 200 batches and reports:

| Metric | Definition |
|--------|------------|
| Alive features | Fire on more than 0.1% of analyzed vectors |
| Dead features | Never fire |
| Ultra-rare features | Fire, but on less than 0.01% of vectors |
| Moderate features | Fire on 0.01%-0.1% of vectors |
| L0 | Active features per vector, including distribution and bootstrap CI |
| Reconstruction MSE | Error in normalized activation space, with bootstrap CI |

There is no universal healthy percentage of alive features or universal MSE
cutoff: both depend on `d_sae`, the intended L0, the layer, and the dataset.
Evaluate whether achieved L0 is near the target, reconstruction is stable,
enough features survive for downstream analysis, and later labeling/fuzzing
results remain useful.

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d-sae` | Auto (`4 * d_model`) | SAE dictionary width |
| `--sae-lr` | `3e-4` | Peak SAE learning rate |
| `--sae-batch-size` | `8192` | Activation vectors per training batch |
| `--sae-epochs` | `10` | Passes over the activation shards |
| `--sae-steps` | Available steps | Optional step cap; cannot exceed available complete batches |
| `--l1-coefficient` | `5e-3` | Fixed L1 or initial adaptive-L1 value |
| `--target-l0` | None | Enable adaptive L1 toward this mean L0; 50-100 is the CLI guidance |
| `--l1-max` | `0.1` | Upper bound for the adaptive L1 controller |
| `--auto-calibrate-threshold` | False | Initialize thresholds from activation quantiles; requires `--target-l0` |
| `--no-sae-resampling` | False | Disable dead-feature resampling |
| `--no-auto-scale-steps` | False | Disable percentage-based step scheduling |
| `--sae-checkpoint-dir` | Per-layer directory | Override checkpoint output; safest for a single-layer run |
| `--sae-resume` | None | Resume from a periodic `step_*.pt` checkpoint |

## Initialization

| Parameter | Initialization |
|-----------|----------------|
| $W_{\text{enc}}$ | Kaiming uniform for ReLU |
| $W_{\text{dec}}$ | Copy of $W_{\text{enc}}^T$ |
| $\mathbf{b}_{\text{enc}}$ | Zeros |
| $\mathbf{b}_{\text{dec}}$ | Zeros |
| $\boldsymbol{\theta}$ | `0.01`, unless auto-calibrated |
