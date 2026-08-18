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
    auto_calibrate_threshold: bool = False
    threshold_calibration_multiplier: float = 4.0
    threshold_calibration_batches: int = 8

    # Optimiser
    batch_size: int = 8192
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Sparsity
    l1_coefficient: float = 5e-3
    target_l0: float | None = None  # If set, auto-tune l1_coefficient to hit this L0
    l1_max: float = 0.1  # Ceiling for the adaptive controller; raise if l1 saturates
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

        # Count total tokens and compute the global per-dimension mean and the
        # RMS *about that mean* across all shards.
        #
        # Centering matters: a residual stream carries a large constant offset
        # that dominates E[x^2] (94-98% of it at most layers of gemma-4-E4B).
        # Scaling by the uncentered RMS therefore normalizes the offset rather
        # than the signal, and leaves the variance the SAE actually has to
        # explain differing by >10x across layers — so a threshold/l1 pair
        # tuned on one layer collapses or diverges on another. Centering first
        # makes the post-normalization variance ~1 everywhere, which is what
        # makes the hyperparameters layer-agnostic.
        self._total_tokens = 0
        self._batches_per_epoch = 0
        sum_vec = np.zeros(self.d_model, dtype=np.float64)
        sum_sq_vec = np.zeros(self.d_model, dtype=np.float64)
        total_rows = 0
        # Accumulate in row-chunks: a full float64 cast of one shard is ~10 GB
        # at 500k x 2560, and filtering it would double that.
        stat_chunk = 65_536
        for path in self.shard_files:
            arr = np.load(path, mmap_mode="r")
            self._total_tokens += arr.shape[0]
            shard_rows = 0
            for start in range(0, arr.shape[0], stat_chunk):
                block = np.asarray(arr[start : start + stat_chunk], dtype=np.float64)
                # __iter__ drops whole rows containing any non-finite value, so
                # the statistics must cover exactly those kept rows.
                block = block[np.isfinite(block).all(axis=1)]
                if block.shape[0] == 0:
                    continue
                sum_vec += block.sum(axis=0)
                sum_sq_vec += (block**2).sum(axis=0)
                shard_rows += block.shape[0]
            total_rows += shard_rows
            # __iter__ batches within a shard, so the remainder is dropped
            # per shard rather than across the whole set.
            self._batches_per_epoch += shard_rows // self.batch_size

        if total_rows == 0:
            raise ValueError(f"No finite activation rows found in {activations_dir}")

        self.mean_vec = (sum_vec / total_rows).astype(np.float32)
        # Var[x] = E[x^2] - E[x]^2, averaged over dimensions; clamp for fp error.
        var_vec = np.maximum(sum_sq_vec / total_rows - (sum_vec / total_rows) ** 2, 0.0)
        self.rms_scale = float(np.sqrt(var_vec.mean()))

        print(
            f"Activation buffer: {len(self.shard_files)} shards, "
            f"{self._total_tokens:,} vectors, d_model={self.d_model}, "
            f"rms_scale={self.rms_scale:.4f} (centered), "
            f"mean_norm={float(np.linalg.norm(self.mean_vec)):.4f}"
        )

    def estimate_total_tokens(self) -> int:
        return self._total_tokens

    def estimate_total_steps(self) -> int:
        """Exact number of batches ``__iter__`` will yield across all epochs."""
        return self._batches_per_epoch * self.num_epochs

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

                # Center, then normalize by the RMS about the mean, so that
                # hyperparameters are layer-agnostic (see __init__).
                shard_cpu -= torch.from_numpy(self.mean_vec).to(dtype=shard_cpu.dtype)
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


@torch.no_grad()
def _auto_calibrate_thresholds(
    sae: JumpReLUSAE,
    activation_buffer: CachedActivationBuffer,
    target_l0: float,
    num_batches: int = 8,
) -> dict[str, float]:
    """Set per-feature thresholds from empirical pre-activation quantiles.

    Each feature receives the quantile that gives it the requested activation
    rate on a representative sample. Averaged across features, that rate
    corresponds to ``target_l0 / d_sae`` and therefore warm-starts the SAE at
    approximately ``target_l0`` active features per vector.
    """
    if not 0 < target_l0 < sae.d_sae:
        raise ValueError(f"Calibration L0 must be between 0 and d_sae ({sae.d_sae})")
    if num_batches < 1:
        raise ValueError("threshold_calibration_batches must be at least 1")

    device = next(sae.parameters()).device
    was_training = sae.training
    sae.eval()
    pre_activations = []

    for batch_idx, batch in enumerate(activation_buffer):
        if batch_idx >= num_batches:
            break
        batch = batch.to(device=device, dtype=sae.W_enc.dtype)
        pre_activation = (batch - sae.b_dec) @ sae.W_enc + sae.b_enc
        pre_activations.append(pre_activation.float())

    if not pre_activations:
        raise ValueError("Activation buffer produced no batches for threshold calibration")

    sample = torch.cat(pre_activations, dim=0)
    quantile = 1.0 - target_l0 / sae.d_sae
    calibrated = torch.quantile(sample, quantile, dim=0)
    sae.threshold.copy_(calibrated.to(dtype=sae.threshold.dtype))

    # Measure after casting thresholds to the model dtype; bf16 rounding can
    # move values that land directly on a quantile boundary.
    achieved_l0 = (sample > sae.threshold.float()).float().sum(dim=-1).mean().item()
    stats = {
        "requested_l0": float(target_l0),
        "achieved_l0": achieved_l0,
        "sample_vectors": float(sample.shape[0]),
        "threshold_mean": sae.threshold.float().mean().item(),
        "threshold_std": sae.threshold.float().std().item(),
        "threshold_min": sae.threshold.float().min().item(),
        "threshold_max": sae.threshold.float().max().item(),
    }

    sae.train(was_training)
    return stats


class _AdaptiveL1Controller:
    """PI controller that adjusts l1_coefficient to hit a target L0.

    After the sparsity warmup period, measures the running average L0
    and nudges l1_coefficient up/down to converge on the target.

    The per-window L0 measurement is extremely noisy early in training (a
    real run recorded window averages of 525 → 6089 → 2207 across adjacent
    windows), so all control and safeguard decisions run on an EMA of L0
    rather than the raw window value.

    The error is computed in **log space** (``log(l0_ema / target)``) and the
    per-update multiplicative adjustment is clamped to a narrow band. Both
    are load-bearing: a linear normalized error is unbounded above but
    bounded at -1 below, and combined with a wide clamp a previous version
    slammed l1 from 0.005 to l1_max in ~8 updates on an early L0 spike, then
    could not descend faster than ~1% per update after L0 had crashed two
    orders of magnitude below target — l1 stayed pinned at max for the last
    40% of the run and reconstruction MSE doubled. Log-space error responds
    symmetrically to overshoot and undershoot, and the narrow clamp spreads
    large corrections over many windows so the (lagging) EMA can catch up
    before l1 saturates.

    Three safeguards keep the loop from destroying a run when it cannot
    actually reach the target — without letting a noisy early measurement
    permanently disable it (which is exactly what a previous version did:
    the authority reference was set from the first noisy window and the
    controller froze for the rest of the run after four updates):

    * **Anti-windup.** The integral only accumulates while the controller has
      authority, so a large constant error cannot ratchet it without bound.
    * **Deferred authority check.** A multiplicative controller facing a
      constant error raises l1 exponentially regardless of the integral, so
      anti-windup alone is not enough. If l1 rises by ``authority_ratio``
      without the (EMA) L0 falling meaningfully, the L1 penalty is not what
      is binding — in a JumpReLU SAE that is typically the threshold — and
      further escalation only destroys reconstruction. The controller then
      freezes l1 and warns instead of running to ``l1_max``. The check only
      arms after ``min_updates_before_guard`` updates, so the reference point
      reflects a stabilized L0 estimate rather than one noisy window.
    * **Re-arm on movement.** A freeze is not terminal: if the EMA L0 later
      moves by more than ``authority_l0_drop`` relative to its value at
      freeze time, the controller unfreezes, resets its references and
      integral, and resumes adjusting.
    """

    def __init__(
        self,
        target_l0: float,
        initial_l1: float,
        kp: float = 0.1,
        ki: float = 0.01,
        l1_min: float = 1e-5,
        l1_max: float = 0.1,
        authority_ratio: float = 4.0,
        authority_l0_drop: float = 0.10,
        min_updates_before_guard: int = 10,
        l0_ema_alpha: float = 0.3,
        max_step: float = 1.2,
    ):
        self.target_l0 = target_l0
        self.l1 = initial_l1
        self.kp = kp
        self.ki = ki
        self.l1_min = l1_min
        self.l1_max = l1_max
        self.authority_ratio = authority_ratio
        self.authority_l0_drop = authority_l0_drop
        self.min_updates_before_guard = min_updates_before_guard
        self.l0_ema_alpha = l0_ema_alpha
        self.max_step = max_step
        self._integral = 0.0
        self._frozen = False
        self._frozen_l0: float | None = None
        self._n_updates = 0
        self._l0_ema: float | None = None
        # Reference point for the authority check: l1 and L0 when last re-armed.
        self._ref_l1: float | None = None
        self._ref_l0: float | None = None

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def l0_ema(self) -> float | None:
        return self._l0_ema

    def update(self, current_l0: float) -> float:
        """Update l1_coefficient based on current L0. Returns new l1."""
        self._n_updates += 1
        a = self.l0_ema_alpha
        self._l0_ema = (
            current_l0 if self._l0_ema is None else (1 - a) * self._l0_ema + a * current_l0
        )
        l0 = self._l0_ema

        if self._frozen:
            # Re-arm: a freeze means "l1 had no authority at the time", not
            # "never adjust again". If L0 has since moved meaningfully, the
            # regime changed — resume control from a fresh reference.
            if abs(l0 - self._frozen_l0) / max(self._frozen_l0, 1e-9) > self.authority_l0_drop:
                self._frozen = False
                self._frozen_l0 = None
                self._ref_l1, self._ref_l0 = self.l1, l0
                self._integral = 0.0
                print(
                    f"\n  Adaptive L1 re-armed: L0 moved to ~{l0:.0f} since the freeze; "
                    f"resuming control from l1={self.l1:.4g}."
                )
            else:
                return self.l1

        # The authority guard only arms once the EMA has had enough windows
        # to stabilize; a reference taken from the first noisy window makes
        # the guard fire on noise.
        if self._n_updates >= self.min_updates_before_guard:
            if self._ref_l1 is None:
                self._ref_l1, self._ref_l0 = self.l1, l0
            # Authority check: has a large l1 increase actually moved L0?
            elif self.l1 >= self._ref_l1 * self.authority_ratio:
                if l0 > self._ref_l0 * (1.0 - self.authority_l0_drop):
                    self._frozen = True
                    self._frozen_l0 = l0
                    print(
                        f"\n  WARNING: adaptive L1 has no authority over L0 — l1 rose "
                        f"{self.l1 / self._ref_l1:.1f}x ({self._ref_l1:.4g} -> {self.l1:.4g}) "
                        f"while L0 held at ~{l0:.0f} (target {self.target_l0:.0f}).\n"
                        f"           Sparsity is bound by the JumpReLU threshold, not the "
                        f"L1 penalty; escalating further would wreck reconstruction.\n"
                        f"           Freezing l1 at {self.l1:.4g}; will re-arm if L0 moves."
                    )
                    return self.l1
                # L0 did respond — re-arm the reference and keep going.
                self._ref_l1, self._ref_l0 = self.l1, l0

        # Positive error → L0 too high → need more sparsity → increase l1.
        # Log-space error responds symmetrically to overshoot and undershoot:
        # a linear (l0 - target) / target error is bounded at -1 below but
        # unbounded above, which made descent from an overshoot ~100x slower
        # than the climb that caused it.
        error = float(np.log(max(l0, 1e-6) / self.target_l0))

        # Conditional integration: don't wind up while the output is saturated
        # or already pinned to an l1 bound in the direction of the error.
        lo, hi = 1.0 / self.max_step, self.max_step
        raw = 1.0 + self.kp * error + self.ki * (self._integral + error)
        saturated = raw > hi or raw < lo
        at_bound = (self.l1 >= self.l1_max and error > 0) or (self.l1 <= self.l1_min and error < 0)
        if not (saturated or at_bound):
            self._integral += error
        elif at_bound:
            self._integral *= 0.9  # decay so the loop can recover when L0 moves

        adjustment = 1.0 + self.kp * error + self.ki * self._integral
        # Narrow per-step clamp: large corrections spread over many windows so
        # the lagging EMA can catch up before l1 saturates at a bound.
        adjustment = max(lo, min(hi, adjustment))
        self.l1 = max(self.l1_min, min(self.l1_max, self.l1 * adjustment))
        return self.l1


# ---------------------------------------------------------------------------
# Dead feature resampling
# ---------------------------------------------------------------------------


def _resample_dead_features(
    sae: JumpReLUSAE,
    dead_indices: torch.Tensor,
    activation_buffer: CachedActivationBuffer,
    optimizer: AdamW | None = None,
    target_l0: float | None = None,
    num_batches: int = 5,
    noise_scale: float = 0.2,
) -> int:
    if len(dead_indices) == 0:
        return 0

    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype

    # Collect high-loss inputs for new decoder directions, plus the complete
    # sample used to calibrate each new feature's activation rate.
    high_loss_inputs = []
    sampled_inputs = []
    sae.eval()
    with torch.inference_mode():
        for i, batch in enumerate(activation_buffer):
            if i >= num_batches:
                break
            batch = batch.to(device)
            sampled_inputs.append(batch)
            reconstruction, _, _ = sae.forward(batch)
            losses = (batch - reconstruction).pow(2).mean(dim=-1)
            k = max(1, len(losses) // 10)
            _, top_idx = losses.topk(k)
            high_loss_inputs.append(batch[top_idx])

    if not high_loss_inputs:
        return 0

    high_loss_inputs = torch.cat(high_loss_inputs, dim=0)
    sampled_inputs = torch.cat(sampled_inputs, dim=0)
    n = min(len(dead_indices), len(high_loss_inputs))
    resampled = dead_indices[:n]

    with torch.no_grad():
        for i in range(n):
            idx = resampled[i].item()
            direction = high_loss_inputs[i]
            direction = direction / (direction.norm() + 1e-8)
            noise = torch.randn_like(direction)
            noise = noise / (noise.norm() + 1e-8)
            direction = direction + noise_scale * noise
            direction = direction / (direction.norm() + 1e-8)

            sae.W_enc[:, idx] = direction.to(dtype)
            sae.W_dec[idx, :] = direction.to(dtype)
            sae.b_enc[idx] = 0.0

        # A fixed 0.01 threshold makes every resampled feature fire on roughly
        # half the data and destroys a calibrated L0. Instead, assign the
        # empirical quantile corresponding to the target per-feature firing
        # rate. With N resampled features, their expected L0 contribution is
        # N * target_l0 / d_sae rather than approximately N / 2.
        pre_activation = (
            (sampled_inputs - sae.b_dec) @ sae.W_enc[:, resampled] + sae.b_enc[resampled]
        ).float()
        if target_l0 is not None:
            if not 0 < target_l0 < sae.d_sae:
                raise ValueError(f"Resampling target L0 must be between 0 and d_sae ({sae.d_sae})")
            quantile = 1.0 - target_l0 / sae.d_sae
            thresholds = torch.quantile(pre_activation, quantile, dim=0)
        else:
            # Fixed-L1 runs have no requested firing rate. Preserve the
            # trained threshold scale instead of returning to the dense init.
            thresholds = sae.threshold.float().median().expand(n)
        sae.threshold[resampled] = thresholds.to(dtype)

        achieved_rate = (
            pre_activation > sae.threshold[resampled].float()
        ).float().mean().item()

        # Adam moments attached to the old, dead feature would otherwise push
        # the newly initialized direction and threshold on the next update.
        if optimizer is not None:
            feature_axes = (
                (sae.W_enc, 1),
                (sae.W_dec, 0),
                (sae.b_enc, 0),
                (sae.threshold, 0),
            )
            for parameter, feature_axis in feature_axes:
                for value in optimizer.state.get(parameter, {}).values():
                    if not torch.is_tensor(value) or value.shape != parameter.shape:
                        continue
                    if feature_axis == 0:
                        value[resampled] = 0
                    else:
                        value[:, resampled] = 0

    print(
        f"  Resampled feature firing rate: {100 * achieved_rate:.3f}% "
        f"(expected L0 contribution: {n * achieved_rate:.2f})"
    )

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

    from scipy.stats import sem as _sem

    if len(l0_arr) == 0:
        raise ValueError(
            "No batches were analyzed — activation buffer is empty. Check value --sae-batch-size and number of shards."
        )
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

    # Persist per-feature firing rates alongside the aggregates. The ablation
    # experiment needs these for its frequency-matched random control and
    # otherwise has to recompute them from the activation shards.
    np.save(Path(output_dir) / "firing_rates.npy", firing_rates)

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
    # Clamp to what the buffer can actually supply: the training loop ends
    # when the buffer is exhausted, so a larger requested step count would
    # silently shift "end of training" earlier and break every step-based
    # schedule computed from total_steps (including the resample guard below).
    available_steps = buffer.estimate_total_steps()
    total_steps = config.total_steps or available_steps
    if total_steps > available_steps:
        print(
            f"Warning: requested {total_steps:,} steps but the activation buffer "
            f"only supplies {available_steps:,} ({config.num_epochs} epochs); "
            f"training for {available_steps:,} steps."
        )
        total_steps = available_steps

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
    if config.auto_calibrate_threshold:
        print(
            f"  Threshold init    : auto ({config.threshold_calibration_multiplier:g}x target, "
            f"{config.threshold_calibration_batches} batches)"
        )
    else:
        print(f"  Threshold init    : {config.threshold_init}")
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

    if config.auto_calibrate_threshold:
        if config.target_l0 is None:
            raise ValueError("--auto-calibrate-threshold requires --target-l0")
        if config.threshold_calibration_multiplier <= 0:
            raise ValueError("threshold_calibration_multiplier must be positive")
        if config.resume_from:
            print("Skipping threshold calibration because training is resuming from a checkpoint")
        else:
            calibration_l0 = min(
                config.d_sae - 1.0,
                config.target_l0 * config.threshold_calibration_multiplier,
            )
            stats = _auto_calibrate_thresholds(
                sae,
                buffer,
                target_l0=calibration_l0,
                num_batches=config.threshold_calibration_batches,
            )
            print(
                "Threshold calibration: "
                f"requested L0={stats['requested_l0']:.1f}, "
                f"achieved L0={stats['achieved_l0']:.1f} on "
                f"{int(stats['sample_vectors']):,} vectors; "
                f"threshold mean={stats['threshold_mean']:.4f}, "
                f"std={stats['threshold_std']:.4f}, "
                f"range=[{stats['threshold_min']:.4f}, {stats['threshold_max']:.4f}]"
            )

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
        _AdaptiveL1Controller(
            target_l0=config.target_l0,
            initial_l1=config.l1_coefficient,
            l1_max=config.l1_max,
        )
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

            # Adaptive L0: adjust l1_coefficient after sparsity warmup.
            # The controller's EMA and frozen state go into the log so a
            # misbehaving run is diagnosable from metrics.jsonl alone.
            if l1_controller is not None and step >= config.sparsity_warmup_steps:
                avg_l0 = avg.get("sparsity/l0", 0)
                l1_controller.update(avg_l0)
                if l1_controller.l0_ema is not None:
                    avg["sparsity/l0_ema"] = l1_controller.l0_ema
                avg["sparsity/l1_controller_frozen"] = float(l1_controller.frozen)

            logger.log(step, avg)
            logger.print_metrics(step, avg, scheduler.get_last_lr()[0])

            running = {}

        # Dead feature resampling. Never resample near the end of training:
        # reinitialised features need optimizer steps to become useful (or be
        # re-killed) before the final save. Without this guard, a resample
        # landing on the last step (e.g. total_steps divisible by
        # resample_every, which auto-scaling makes likely) ships thousands of
        # random, dense features in sae_final.pt (L0 ~50% of d_sae).
        can_resample = step + config.resample_every <= total_steps
        if (
            config.resample_dead_features
            and can_resample
            and step > 0
            and step % config.resample_every == 0
        ):
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
                n = _resample_dead_features(
                    raw_sae,
                    dead,
                    buffer,
                    optimizer=optimizer,
                    target_l0=config.target_l0,
                )
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
    raw_sae.save_pretrained(
        final_path,
        config={
            "rms_scale": buffer.rms_scale,
            "mean_vec": buffer.mean_vec.tolist(),
        },
    )
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
