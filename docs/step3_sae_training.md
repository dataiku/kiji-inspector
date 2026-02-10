# Step 3: SAE Training

## Purpose

Train a **JumpReLU Sparse Autoencoder** on the raw activation vectors from Step 2 to discover a dictionary of monosemantic features in the model's hidden representation space. The SAE learns to reconstruct each activation vector using a sparse linear combination of learned feature directions, where each feature ideally corresponds to a single interpretable concept.

## Source Files

| File | Key Components |
|------|----------------|
| `generate_training_set.py` | `_run_step3()`, `train_sae_step()` |
| `src/sae_model.py` | `JumpReLUSAE`, `JumpReLUFunction` |
| `src/sae_trainer.py` | `SAETrainingConfig`, `CachedActivationBuffer`, `train_sae()`, `analyze_feature_health()` |

## JumpReLU SAE Architecture

### Mathematical Formulation

Given an input activation vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$:

**Encoder:**

$$\mathbf{z} = W_{\text{enc}} (\mathbf{x} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{enc}}$$

$$\mathbf{f} = \text{JumpReLU}(\mathbf{z}, \boldsymbol{\theta}) = \mathbf{z} \odot H(\mathbf{z} - \boldsymbol{\theta})$$

where $H$ is the Heaviside step function and $\boldsymbol{\theta} \in \mathbb{R}^{d_{\text{sae}}}$ are learnable per-feature thresholds.

**Decoder:**

$$\hat{\mathbf{x}} = W_{\text{dec}} \mathbf{f} + \mathbf{b}_{\text{dec}}$$

Note that $\mathbf{b}_{\text{dec}}$ is shared: it is subtracted from the input before encoding and added back after decoding.

### Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| $W_{\text{enc}}$ | $(d_{\text{model}}, d_{\text{sae}})$ | Encoder weight matrix |
| $\mathbf{b}_{\text{enc}}$ | $(d_{\text{sae}},)$ | Encoder bias |
| $\boldsymbol{\theta}$ | $(d_{\text{sae}},)$ | Per-feature JumpReLU thresholds |
| $W_{\text{dec}}$ | $(d_{\text{sae}}, d_{\text{model}})$ | Decoder weight matrix |
| $\mathbf{b}_{\text{dec}}$ | $(d_{\text{model}},)$ | Shared decoder/pre-encoder bias |

With default dimensions ($d_{\text{model}} = 4096$, $d_{\text{sae}} = 16384$):
- Total parameters: $2 \times 4096 \times 16384 + 2 \times 16384 + 4096 \approx 134M$

### JumpReLU Activation Function

The JumpReLU function creates **exact sparsity** (true zeros) rather than near-zero values:

$$\text{JumpReLU}(z_j, \theta_j) = \begin{cases} z_j & \text{if } z_j > \theta_j \\ 0 & \text{otherwise} \end{cases}$$

The Heaviside step function $H$ is non-differentiable. We use a **Straight-Through Estimator (STE)** for the gradient of $\mathbf{z}$ and a **rectangular kernel approximation** for the gradient of $\boldsymbol{\theta}$.

### Gradient Computation

**Forward pass:**
```python
mask = (z > threshold)  # Boolean, shape (batch, d_sae)
output = z * mask       # Exact zeros where inactive
```

**Backward pass for $\mathbf{z}$** (STE):
$$\frac{\partial \mathcal{L}}{\partial z_j} = \frac{\partial \mathcal{L}}{\partial f_j} \cdot \mathbb{1}[z_j > \theta_j]$$

The gradient passes through only where the feature is active.

**Backward pass for $\boldsymbol{\theta}$** (rectangular kernel):

$$\frac{\partial \mathcal{L}}{\partial \theta_j} = -\sum_{i} \frac{\partial \mathcal{L}}{\partial f_{ij}} \cdot z_{ij} \cdot \frac{\mathbb{1}[|z_{ij} - \theta_j| < \epsilon]}{2\epsilon}$$

where $\epsilon$ is the `bandwidth` hyperparameter (default: 0.001). This approximates the Dirac delta with a rectangular window of width $2\epsilon$, providing a smooth gradient signal for threshold learning.

## Loss Function

The total loss combines reconstruction accuracy with a differentiable sparsity penalty:

### Reconstruction Loss

$$\mathcal{L}_{\text{recon}} = \text{MSE}(\hat{\mathbf{x}}, \mathbf{x}) = \frac{1}{d_{\text{model}}} \|\hat{\mathbf{x}} - \mathbf{x}\|_2^2$$

### Tanh Sparsity Loss

A smooth, differentiable approximation of the L0 norm that provides gradients to both the encoder weights and the thresholds:

$$\mathcal{L}_{\text{sparse}} = \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{d_{\text{sae}}} \text{ReLU}\left(\tanh\left(\frac{z_{ij} - \theta_j}{\epsilon}\right)\right)$$

The $\tanh$ smoothly transitions from 0 to 1 around $z_j = \theta_j$, and the ReLU clips negative values (where $z_j \ll \theta_j$).

### L0 Pseudo-Norm (Monitoring Only)

$$L_0 = \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{d_{\text{sae}}} \mathbb{1}[|f_{ij}| > 0]$$

This is the true count of active features per input. It is non-differentiable and used only for logging.

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda(t) \cdot \mathcal{L}_{\text{sparse}}$$

where $\lambda(t)$ is the sparsity coefficient with warmup (see below).

## Training Loop

### Data Loading: CachedActivationBuffer

The buffer loads pre-computed numpy shards from Step 2 and yields shuffled batches:

```
For each epoch:
    1. Shuffle shard file order
    2. For each shard:
        a. Load numpy array from disk
        b. Shuffle rows in-place
        c. Convert to PyTorch tensor (CPU, bfloat16)
        d. Yield batches of size batch_size to GPU
        e. Free shard memory, empty CUDA cache
```

### Learning Rate Schedule

Cosine decay with linear warmup:

$$\text{lr}(t) = \begin{cases}
\text{lr}_{\text{peak}} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\
\text{lr}_{\text{peak}} \cdot \max\left(0.1, \; \frac{1 + \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)}{2}\right) & \text{otherwise}
\end{cases}$$

The minimum LR ratio is 0.1 (LR decays to 10% of peak, not to zero).

### Sparsity Warmup

The sparsity coefficient ramps linearly from 0 to the target value:

$$\lambda(t) = \begin{cases}
\lambda_{\text{target}} \cdot \frac{t}{t_{\text{sparse\_warmup}}} & \text{if } t < t_{\text{sparse\_warmup}} \\
\lambda_{\text{target}} & \text{otherwise}
\end{cases}$$

This prevents the sparsity penalty from dominating early training when the encoder hasn't learned meaningful directions yet.

### Auto-Scaling of Step Parameters

When `auto_scale_steps=True` (default), step-based hyperparameters are set as fractions of `total_steps`:

| Parameter | Fraction | Example (100K steps) |
|-----------|----------|---------------------|
| `warmup_steps` | 5% | 5,000 |
| `sparsity_warmup_steps` | 10% | 10,000 |
| `resample_every` | 20% | 20,000 |
| `checkpoint_every` | 25% | 25,000 |
| `log_every` | 2% | 2,000 |

### Decoder Weight Normalization

After each optimizer step, decoder weight rows are normalized to unit norm:

$$W_{\text{dec}}[j, :] \leftarrow \frac{W_{\text{dec}}[j, :]}{\|W_{\text{dec}}[j, :]\|_2}$$

This prevents the decoder from compensating for low feature activations by scaling up weight norms, which would undermine the sparsity incentive.

### Optimizer

AdamW with the following settings:

| Parameter | Value |
|-----------|-------|
| Learning rate | $3 \times 10^{-4}$ |
| Betas | $(0.9, 0.999)$ |
| Weight decay | 0.0 |
| Gradient clipping | Max norm 1.0 |
| Fused | True (on CUDA) |

### Training Step Pseudocode

```python
for batch in activation_buffer:                  # (batch_size, d_model)
    current_l1 = sparsity_warmup(step, target_l1, warmup_steps)
    loss, metrics = sae.compute_loss(batch, l1_coefficient=current_l1)
    loss.backward()
    clip_grad_norm_(sae.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    sae.normalize_decoder()                       # Unit-norm decoder rows
```

## Dead Feature Resampling

Features that never activate (or activate below `dead_feature_threshold=1e-6`) are "dead" and waste representational capacity.

### Detection

Every `resample_every` steps, the trainer checks which features fired across 20 batches:

```python
activated = torch.zeros(d_sae, dtype=torch.bool)
for batch in buffer[:20]:
    features = sae.encode(batch)
    activated |= (features.abs() > 1e-6).any(dim=0)
dead_indices = torch.where(~activated)[0]
```

### Resampling Algorithm

1. Collect the **top 10% highest-loss inputs** from 5 batches (inputs the SAE reconstructs worst)
2. For each dead feature index $j$:
   - Sample a high-loss input vector $\mathbf{v}$
   - Normalize: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$
   - Add noise: $\hat{\mathbf{v}}' = \hat{\mathbf{v}} + 0.2 \cdot \mathcal{N}(0, I)$, then re-normalize
   - Set $W_{\text{enc}}[:, j] = \hat{\mathbf{v}}'$ and $W_{\text{dec}}[j, :] = \hat{\mathbf{v}}'$
   - Reset $b_{\text{enc}}[j] = 0$ and $\theta_j = 0.01$

This gives dead features a fresh direction pointing toward inputs the SAE currently fails to reconstruct.

## Post-Training: Feature Health Analysis

After training completes, `analyze_feature_health()` iterates over up to 200 batches and reports:

| Metric | Definition | Healthy Range |
|--------|-----------|--------------|
| **Alive features** | Fire on >0.1% of inputs | 50-80% of total |
| **Dead features** | Never fire (0%) | <20% |
| **Ultra-rare** | Fire but <0.01% | Low count |
| **L0 mean** | Average active features per input | 50-200 |
| **Reconstruction MSE** | Mean squared error | <0.01 |

Results are saved to `feature_health.json`.

## Weight Initialization

| Parameter | Initialization |
|-----------|---------------|
| $W_{\text{enc}}$ | Kaiming uniform (ReLU nonlinearity) |
| $W_{\text{dec}}$ | Copy of $W_{\text{enc}}^T$ |
| $\mathbf{b}_{\text{enc}}$ | Zeros |
| $\mathbf{b}_{\text{dec}}$ | Zeros |
| $\boldsymbol{\theta}$ | Constant 0.01 |

## torch.compile

When CUDA is available, the SAE is compiled with:
```python
torch.compile(sae, mode="max-autotune", fullgraph=True)
```
This enables kernel fusion and memory planning optimizations for the training loop.

## Checkpointing

Checkpoints are saved every `checkpoint_every` steps (auto-scaled to 25% of total by default). Each checkpoint contains:

- `model_state_dict`: SAE weights
- `optimizer_state_dict`: AdamW state (momentum terms)
- `scheduler_state_dict`: LR scheduler state
- `step`: Current training step
- `config`: Full `SAETrainingConfig`
- `metrics`: Recent loss values

Only the last 3 checkpoints are retained. A final `sae_final.pt` is saved at the end with a simplified format (weights + config only).

## Output Files

```
output/sae_checkpoints/
    sae_final.pt          # Final trained model
    step_75000.pt         # Recent checkpoint
    step_50000.pt         # Older checkpoint
    config.json           # Training hyperparameters
    metrics.jsonl         # Per-step loss curves
    feature_health.json   # Post-training health analysis
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d-sae` | 16384 | SAE hidden dimension (typically 4x d_model) |
| `--sae-lr` | 3e-4 | Learning rate |
| `--sae-batch-size` | 8192 | Tokens per training batch |
| `--sae-epochs` | 10 | Passes over the activation data |
| `--sae-steps` | Auto | Total training steps (auto-computed if not set) |
| `--l1-coefficient` | 5e-3 | Sparsity penalty weight |
| `--sae-checkpoint-dir` | `output/sae_checkpoints` | Output directory |
| `--sae-resume` | None | Resume from checkpoint path |
| `--no-auto-scale-steps` | False | Disable auto-scaling of step parameters |

## Hyperparameter Guidance

| Parameter | Lower | Higher | Effect |
|-----------|-------|--------|--------|
| `d_sae` | 8192 | 65536 | More features = more specific concepts, but more dead features |
| `l1_coefficient` | 1e-3 | 1e-2 | Higher = sparser features (lower L0), may hurt reconstruction |
| `bandwidth` | 0.0005 | 0.005 | Wider = smoother threshold gradients, slower threshold learning |
| `threshold_init` | 0.001 | 0.1 | Higher = more features start dead, may need more resampling |
| `batch_size` | 4096 | 16384 | Larger = more stable gradients, higher GPU memory |
