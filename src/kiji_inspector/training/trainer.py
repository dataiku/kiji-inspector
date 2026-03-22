"""
SAE training loop with cached activation loading.

Loads pre-computed activations from numpy shards (produced by Step 2)
and trains a JumpReLU SAE.  Includes cosine LR schedule with warmup,
sparsity warmup, dead feature resampling, gradient clipping, and
periodic checkpointing.

Adapted from yaak-inspector-demo/sae_train/{train,cached_activations,utils}.py.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from kiji_inspector.training.model import JumpReLUSAE
from kiji_inspector.utils.stats import bootstrap_ci_mean as _bootstrap_ci
from kiji_inspector.utils.stats import wilson_score_ci as _wilson_score_ci

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SAETrainingConfig:
    """All hyperparameters for SAE training."""

    # Architecture
    d_sae: int = 16384
    bandwidth: float = 0.001
    threshold_init: float = 0.01

    # Optimiser
    batch_size: int = 8192
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Sparsity
    l1_coefficient: float = 5e-3
    target_l0: float | None = None  # If set, auto-tune l1_coefficient to hit this L0
    sparsity_warmup_steps: int = 10000

    # Dead feature resampling
    resample_dead_features: bool = True
    resample_every: int = 25000
    dead_feature_threshold: float = 1e-6

    # Run control
    total_steps: int | None = None
    num_epochs: int = 10
    checkpoint_every: int = 5000
    log_every: int = 500
    use_torch_compile: bool = True
    seed: int = 42
    auto_scale_steps: bool = True  # Scale warmup/checkpoint/resample to total_steps

    # Paths
    output_dir: str = "checkpoints"
    resume_from: str | None = None


# ---------------------------------------------------------------------------
# Cached activation buffer
# ---------------------------------------------------------------------------


class CachedActivationBuffer:
    """Load pre-computed activation shards and yield shuffled batches.

    Expects a directory with ``shard_*.npy`` (float32) and ``metadata.json``.
    Compatible with the output of Step 2 (``RawActivationExtractor``).
    """

    def __init__(
        self,
        activations_dir: str | Path,
        batch_size: int = 8192,
        shuffle: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        num_epochs: int = 1,
        device: str = "cuda",
    ):
        self.activations_dir = Path(activations_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.num_epochs = num_epochs
        self.device = device

        metadata_path = self.activations_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.d_model = self.metadata["d_model"]

        self.shard_files = sorted(self.activations_dir.glob("shard_*.npy"))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard_*.npy files in {activations_dir}")

        # Count total tokens and compute global RMS across all shards
        self._total_tokens = 0
        sum_sq = 0.0
        total_elements = 0
        for path in self.shard_files:
            arr = np.load(path, mmap_mode="r")
            self._total_tokens += arr.shape[0]
            # Accumulate sum of squares for RMS (cast to float64 to avoid overflow)
            arr_f64 = arr.astype(np.float64)
            finite_mask = np.isfinite(arr_f64)
            arr_clean = np.where(finite_mask, arr_f64, 0.0)
            sum_sq += (arr_clean**2).sum()
            total_elements += finite_mask.sum()

        self.rms_scale = float(np.sqrt(sum_sq / total_elements))

        print(
            f"Activation buffer: {len(self.shard_files)} shards, "
            f"{self._total_tokens:,} vectors, d_model={self.d_model}, "
            f"rms_scale={self.rms_scale:.4f}"
        )

    def estimate_total_tokens(self) -> int:
        return self._total_tokens

    def estimate_total_steps(self) -> int:
        return (self._total_tokens * self.num_epochs) // self.batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _epoch in range(self.num_epochs):
            shard_files = list(self.shard_files)
            if self.shuffle:
                random.shuffle(shard_files)

            for shard_path in shard_files:
                shard_data = np.load(shard_path)
                if self.shuffle:
                    np.random.shuffle(shard_data)

                shard_cpu = torch.from_numpy(shard_data).to(dtype=self.dtype)
                del shard_data

                # Drop rows containing NaN or Inf (can happen with FP8/hybrid models)
                finite_mask = torch.isfinite(shard_cpu).all(dim=-1)
                if not finite_mask.all():
                    n_bad = (~finite_mask).sum().item()
                    print(f"  Warning: dropping {n_bad} non-finite vectors from {shard_path.name}")
                    shard_cpu = shard_cpu[finite_mask]
                    if shard_cpu.shape[0] == 0:
                        continue

                # Normalize by global RMS so hyperparameters are layer-agnostic
                if self.rms_scale > 0:
                    shard_cpu /= self.rms_scale

                n = shard_cpu.shape[0]
                for i in range(0, n - self.batch_size + 1, self.batch_size):
                    batch = shard_cpu[i : i + self.batch_size].to(
                        device=self.device, non_blocking=True
                    )
                    yield batch

                del shard_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------


def _cosine_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _sparsity_warmup(step: int, target: float, warmup_steps: int) -> float:
    if step >= warmup_steps:
        return target
    return target * (step / warmup_steps)


class _AdaptiveL1Controller:
    """PI controller that adjusts l1_coefficient to hit a target L0.

    After the sparsity warmup period, measures the running average L0
    and nudges l1_coefficient up/down to converge on the target.
    """

    def __init__(
        self,
        target_l0: float,
        initial_l1: float,
        kp: float = 0.01,
        ki: float = 0.001,
        l1_min: float = 1e-5,
        l1_max: float = 1.0,
    ):
        self.target_l0 = target_l0
        self.l1 = initial_l1
        self.kp = kp
        self.ki = ki
        self.l1_min = l1_min
        self.l1_max = l1_max
        self._integral = 0.0

    def update(self, current_l0: float) -> float:
        """Update l1_coefficient based on current L0. Returns new l1."""
        # Positive error → L0 too high → need more sparsity → increase l1
        error = (current_l0 - self.target_l0) / self.target_l0  # normalized
        self._integral += error

        # Multiplicative adjustment (works better than additive for log-scale)
        adjustment = 1.0 + self.kp * error + self.ki * self._integral
        adjustment = max(0.5, min(2.0, adjustment))  # clamp per-step change
        self.l1 = max(self.l1_min, min(self.l1_max, self.l1 * adjustment))
        return self.l1


# ---------------------------------------------------------------------------
# Dead feature resampling
# ---------------------------------------------------------------------------


def _resample_dead_features(
    sae: JumpReLUSAE,
    dead_indices: torch.Tensor,
    activation_buffer: CachedActivationBuffer,
    num_batches: int = 5,
    noise_scale: float = 0.2,
) -> int:
    if len(dead_indices) == 0:
        return 0

    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype

    # Collect high-loss inputs
    high_loss_inputs = []
    sae.eval()
    with torch.inference_mode():
        for i, batch in enumerate(activation_buffer):
            if i >= num_batches:
                break
            batch = batch.to(device)
            reconstruction, _, _ = sae.forward(batch)
            losses = (batch - reconstruction).pow(2).mean(dim=-1)
            k = max(1, len(losses) // 10)
            _, top_idx = losses.topk(k)
            high_loss_inputs.append(batch[top_idx])

    if not high_loss_inputs:
        return 0

    high_loss_inputs = torch.cat(high_loss_inputs, dim=0)
    n = min(len(dead_indices), len(high_loss_inputs))

    with torch.no_grad():
        for i in range(n):
            idx = dead_indices[i].item()
            direction = high_loss_inputs[i % len(high_loss_inputs)]
            direction = direction / (direction.norm() + 1e-8)
            noise = torch.randn_like(direction) * noise_scale
            direction = direction + noise
            direction = direction / (direction.norm() + 1e-8)

            sae.W_enc[:, idx] = direction.to(dtype)
            sae.W_dec[idx, :] = direction.to(dtype)
            sae.b_enc[idx] = 0.0
            sae.threshold[idx] = 0.01

    return n


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    sae: JumpReLUSAE,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    config: SAETrainingConfig,
    metrics: dict,
    keep_last_n: int = 3,
) -> str:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    path = output / f"step_{step}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": sae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")

    # Save config as JSON for inspection
    config_path = output / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Keep only last N checkpoints
    existing = sorted(output.glob("step_*.pt"))
    for old in existing[:-keep_last_n]:
        old.unlink()

    return str(path)


def _load_checkpoint(
    path: str,
    sae: JumpReLUSAE,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    device: str = "cuda",
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    sae.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    step = checkpoint.get("step", 0)
    print(f"Resumed from step {step}")
    return step


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------


class _MetricsLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.output_dir / "metrics.jsonl"

    def log(self, step: int, metrics: dict):
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def print_metrics(self, step: int, metrics: dict, lr: float):
        tqdm.write(
            f"Step {step:6d} | "
            f"Loss: {metrics.get('loss/total', 0):.4f} "
            f"(recon: {metrics.get('loss/reconstruction', 0):.4f}, "
            f"sparse: {metrics.get('loss/sparsity', 0):.4f}) | "
            f"L0: {metrics.get('sparsity/l0', 0):.1f} | "
            f"Active: {metrics.get('sparsity/feature_activity', 0):.1%} | "
            f"LR: {lr:.2e}"
        )


# ---------------------------------------------------------------------------
# Post-training feature health analysis
# ---------------------------------------------------------------------------


def analyze_feature_health(
    sae: JumpReLUSAE,
    activations_dir: str,
    batch_size: int,
    output_dir: str,
    device: str = "cuda",
    max_batches: int = 200,
) -> dict:
    """Run a post-training analysis pass to assess feature health.

    Iterates over the activation data, encodes through the SAE, and
    reports per-feature firing rates, alive/dead counts, L0 distribution,
    and reconstruction quality.

    Returns:
        Health metrics dict (also saved to ``{output_dir}/feature_health.json``).
    """
    import torch.nn.functional as F

    buffer = CachedActivationBuffer(
        activations_dir=activations_dir,
        batch_size=batch_size,
        shuffle=False,
        dtype=torch.bfloat16,
        num_epochs=1,
        device=device,
    )

    d_sae = sae.d_sae
    feature_fire_count = torch.zeros(d_sae, device=device)
    total_vectors = 0
    l0_values: list[float] = []
    recon_losses: list[float] = []

    sae.eval()
    pbar = tqdm(desc="Analyzing feature health", unit="batch")
    with torch.inference_mode():
        for i, batch in enumerate(buffer):
            if i >= max_batches:
                break
            batch = batch.to(device)
            reconstruction, features, _ = sae.forward(batch)

            active_mask = features.abs() > 0
            feature_fire_count += active_mask.float().sum(dim=0)
            total_vectors += batch.shape[0]
            l0_values.append(active_mask.float().sum(dim=-1).mean().item())
            recon_losses.append(F.mse_loss(reconstruction, batch).item())
            pbar.update(1)

    pbar.close()

    # Compute per-feature firing rates
    firing_rates = (feature_fire_count / max(total_vectors, 1)).cpu().numpy()

    alive = int((firing_rates > 0.001).sum())  # fire on >0.1% of inputs
    dead = int((firing_rates == 0).sum())
    ultra_rare = int(((firing_rates > 0) & (firing_rates < 0.0001)).sum())
    moderate = d_sae - alive - dead - ultra_rare

    l0_arr = np.array(l0_values)
    recon_arr = np.array(recon_losses)

    if len(l0_arr) == 0:
        print("  WARNING: No valid activation batches — all vectors were non-finite.")
        print("  The activation shards are likely corrupt. Re-extract with a working backend.")
        health = {
            "total_features": d_sae,
            "total_vectors_analyzed": 0,
            "error": "no valid batches — all activation vectors were non-finite",
        }
        health_path = Path(output_dir) / "feature_health.json"
        health_path.write_text(json.dumps(health, indent=2))
        return health

    from scipy.stats import sem as _sem

    l0_ci = _bootstrap_ci(l0_arr)
    recon_ci = _bootstrap_ci(recon_arr)
    alive_ci = _wilson_score_ci(alive, d_sae)
    dead_ci = _wilson_score_ci(dead, d_sae)

    health = {
        "total_features": d_sae,
        "total_vectors_analyzed": total_vectors,
        "alive_features": alive,
        "dead_features": dead,
        "ultra_rare_features": ultra_rare,
        "moderate_features": moderate,
        "alive_pct": round(100 * alive / d_sae, 2),
        "alive_pct_ci_95": [round(100 * alive_ci[0], 2), round(100 * alive_ci[1], 2)],
        "dead_pct": round(100 * dead / d_sae, 2),
        "dead_pct_ci_95": [round(100 * dead_ci[0], 2), round(100 * dead_ci[1], 2)],
        "l0": {
            "mean": round(float(l0_arr.mean()), 2),
            "std": round(float(l0_arr.std(ddof=1)), 2),
            "sem": round(float(_sem(l0_arr)), 2),
            "ci_95": [round(l0_ci[0], 2), round(l0_ci[1], 2)],
            "median": round(float(np.median(l0_arr)), 2),
            "p5": round(float(np.percentile(l0_arr, 5)), 2),
            "p95": round(float(np.percentile(l0_arr, 95)), 2),
            "min": round(float(l0_arr.min()), 2),
            "max": round(float(l0_arr.max()), 2),
            "n_batches": len(l0_values),
        },
        "reconstruction_mse": {
            "mean": round(float(recon_arr.mean()), 6),
            "std": round(float(recon_arr.std(ddof=1)), 6),
            "sem": round(float(_sem(recon_arr)), 6),
            "ci_95": [round(recon_ci[0], 6), round(recon_ci[1], 6)],
            "n_batches": len(recon_losses),
        },
    }

    # Save
    out_path = Path(output_dir) / "feature_health.json"
    with open(out_path, "w") as f:
        json.dump(health, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  Feature Health Analysis")
    print(f"  Vectors analyzed  : {total_vectors:,}")
    print(
        f"  Alive (>0.1%)     : {alive:,} ({health['alive_pct']}%, "
        f"95% CI: [{health['alive_pct_ci_95'][0]}%, {health['alive_pct_ci_95'][1]}%])"
    )
    print(
        f"  Dead (0%)         : {dead:,} ({health['dead_pct']}%, "
        f"95% CI: [{health['dead_pct_ci_95'][0]}%, {health['dead_pct_ci_95'][1]}%])"
    )
    print(f"  Ultra-rare        : {ultra_rare:,}")
    print(
        f"  L0                : {health['l0']['mean']:.1f} +/- {health['l0']['sem']:.1f} (SEM), "
        f"95% CI: [{health['l0']['ci_95'][0]:.1f}, {health['l0']['ci_95'][1]:.1f}], "
        f"median: {health['l0']['median']:.1f}"
    )
    print(
        f"  Recon MSE         : {health['reconstruction_mse']['mean']:.6f} "
        f"+/- {health['reconstruction_mse']['sem']:.6f} (SEM)"
    )
    print(f"  Saved             : {out_path}")
    print("=" * 70)

    return health


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_sae(
    activations_dir: str,
    config: SAETrainingConfig,
) -> str:
    """Train a JumpReLU SAE on cached activations.

    Args:
        activations_dir: Path to directory with ``shard_*.npy`` + ``metadata.json``.
        config: Training hyperparameters.

    Returns:
        Path to the final saved model checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load activation buffer
    buffer = CachedActivationBuffer(
        activations_dir=activations_dir,
        batch_size=config.batch_size,
        shuffle=True,
        dtype=dtype,
        num_epochs=config.num_epochs,
        device=device,
    )

    d_model = buffer.d_model
    total_steps = config.total_steps or buffer.estimate_total_steps()

    # Auto-scale step-based parameters relative to total_steps
    if config.auto_scale_steps:
        config.warmup_steps = max(1, int(total_steps * 0.05))
        config.sparsity_warmup_steps = max(1, int(total_steps * 0.10))
        config.resample_every = max(1, int(total_steps * 0.20))
        config.checkpoint_every = max(1, int(total_steps * 0.25))
        config.log_every = max(1, int(total_steps * 0.02))

    print("=" * 70)
    print("  JumpReLU SAE Training")
    print(f"  Activations      : {activations_dir}")
    print(f"  d_model           : {d_model}")
    print(f"  d_sae             : {config.d_sae}")
    print(f"  Total tokens      : {buffer.estimate_total_tokens():,}")
    print(f"  Epochs            : {config.num_epochs}")
    print(f"  Batch size        : {config.batch_size:,}")
    print(f"  Total steps       : {total_steps:,}")
    print(f"  Learning rate     : {config.learning_rate}")
    print(f"  L1 coefficient    : {config.l1_coefficient}")
    if config.target_l0 is not None:
        print(f"  Target L0         : {config.target_l0} (adaptive l1)")
    print(
        f"  Warmup steps      : {config.warmup_steps} ({100 * config.warmup_steps / total_steps:.1f}%)"
    )
    print(
        f"  Sparsity warmup   : {config.sparsity_warmup_steps} ({100 * config.sparsity_warmup_steps / total_steps:.1f}%)"
    )
    print(
        f"  Resample every    : {config.resample_every} ({100 * config.resample_every / total_steps:.1f}%)"
    )
    print(
        f"  Checkpoint every  : {config.checkpoint_every} ({100 * config.checkpoint_every / total_steps:.1f}%)"
    )
    print(f"  Auto-scaled       : {config.auto_scale_steps}")
    print(f"  Device            : {device}")
    print("=" * 70)

    # Initialise SAE
    sae = JumpReLUSAE(
        d_model=d_model,
        d_sae=config.d_sae,
        dtype=dtype,
        bandwidth=config.bandwidth,
        threshold_init=config.threshold_init,
    ).to(device)
    print(f"SAE parameters: {sae.get_num_parameters():,}")

    if config.use_torch_compile and torch.cuda.is_available():
        print("Compiling SAE with torch.compile...")
        sae = torch.compile(sae, mode="max-autotune", fullgraph=True)

    # Note: DataParallel is not used for SAE training because the SAE is
    # small (~58M params) and compute_loss() includes reduction steps that
    # are incompatible with DP's scatter/gather.  A single GPU is sufficient.

    def _unwrap(model):
        return model

    # Optimiser + scheduler
    optimizer = AdamW(
        sae.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )
    scheduler = _cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    logger = _MetricsLogger(config.output_dir)

    # Resume
    start_step = 0
    if config.resume_from:
        start_step = _load_checkpoint(config.resume_from, sae, optimizer, scheduler, device)

    # Adaptive L0 controller (if target_l0 is set)
    l1_controller = (
        _AdaptiveL1Controller(target_l0=config.target_l0, initial_l1=config.l1_coefficient)
        if config.target_l0 is not None
        else None
    )

    # Training loop
    step = start_step
    running: dict[str, list[float]] = {}
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training SAE")

    for activations in buffer:
        if step >= total_steps:
            break

        activations = activations.to(device=device, dtype=dtype)

        base_l1 = l1_controller.l1 if l1_controller is not None else config.l1_coefficient
        current_l1 = _sparsity_warmup(step, base_l1, config.sparsity_warmup_steps)
        loss, metrics = sae.compute_loss(activations, l1_coefficient=current_l1)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Step {step}] NaN/Inf loss detected — skipping update")
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            pbar.update(1)
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(sae.parameters(), config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        _unwrap(sae).normalize_decoder()

        for k, v in metrics.items():
            running.setdefault(k, []).append(v)

        step += 1
        pbar.update(1)

        # Log
        if step % config.log_every == 0 and running:
            avg = {k: sum(v) / len(v) for k, v in running.items()}
            avg["sparsity/current_l1_coef"] = current_l1
            logger.log(step, avg)
            logger.print_metrics(step, avg, scheduler.get_last_lr()[0])

            # Adaptive L0: adjust l1_coefficient after sparsity warmup
            if l1_controller is not None and step >= config.sparsity_warmup_steps:
                avg_l0 = avg.get("sparsity/l0", 0)
                l1_controller.update(avg_l0)

            running = {}

        # Dead feature resampling
        if config.resample_dead_features and step > 0 and step % config.resample_every == 0:
            print(f"\n[Step {step}] Checking for dead features...")
            raw_sae = _unwrap(sae)
            dev = next(raw_sae.parameters()).device
            activated = torch.zeros(raw_sae.d_sae, dtype=torch.bool, device=dev)

            raw_sae.eval()
            with torch.inference_mode():
                for i, batch in enumerate(buffer):
                    if i >= 20:
                        break
                    features = raw_sae.encode(batch.to(dev))
                    activated |= (features.abs() > config.dead_feature_threshold).any(dim=0)

            dead = torch.where(~activated)[0]
            if len(dead) > 0:
                print(f"  {len(dead)} dead features ({100 * len(dead) / raw_sae.d_sae:.1f}%)")
                n = _resample_dead_features(raw_sae, dead, buffer)
                print(f"  Resampled {n} features")
            else:
                print("  No dead features found")
            sae.train()

        # Checkpoint
        if step % config.checkpoint_every == 0:
            avg = {k: sum(v) / len(v) for k, v in running.items()} if running else {}
            _save_checkpoint(_unwrap(sae), optimizer, scheduler, step, config, avg)

    pbar.close()

    # Final save (unwrap DataParallel for serialisation)
    raw_sae = _unwrap(sae)
    _save_checkpoint(raw_sae, optimizer, scheduler, step, config, {})
    final_path = str(Path(config.output_dir) / "sae_final.pt")
    raw_sae.save_pretrained(final_path, config={"rms_scale": buffer.rms_scale})
    print(f"Saved final model: {final_path}")

    # Post-training feature health analysis
    analyze_feature_health(
        sae=raw_sae,
        activations_dir=activations_dir,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        device=device,
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    return final_path
