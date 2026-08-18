import pytest
import torch

from kiji_inspector.training.model import JumpReLUSAE
from kiji_inspector.training.trainer import _auto_calibrate_thresholds, _resample_dead_features


def test_auto_calibration_hits_requested_l0():
    torch.manual_seed(7)
    sae = JumpReLUSAE(d_model=8, d_sae=128, dtype=torch.float32)
    batches = [torch.randn(256, 8) for _ in range(4)]

    stats = _auto_calibrate_thresholds(sae, batches, target_l0=16, num_batches=4)

    assert stats["sample_vectors"] == 1024
    assert stats["achieved_l0"] == pytest.approx(16, abs=0.2)
    assert stats["threshold_mean"] > 0.01


@pytest.mark.parametrize("target_l0", [0, 128])
def test_auto_calibration_rejects_invalid_l0(target_l0):
    sae = JumpReLUSAE(d_model=4, d_sae=128, dtype=torch.float32)

    with pytest.raises(ValueError, match="between 0 and d_sae"):
        _auto_calibrate_thresholds(
            sae,
            [torch.randn(32, 4)],
            target_l0=target_l0,
            num_batches=1,
        )


def test_dead_feature_resampling_calibrates_firing_rate_and_clears_momentum():
    torch.manual_seed(11)
    sae = JumpReLUSAE(d_model=8, d_sae=16, dtype=torch.float32)
    batches = [torch.randn(64, 8) for _ in range(4)]
    dead = torch.tensor([0, 1, 2])

    optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)
    sum(parameter.sum() for parameter in sae.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()

    n = _resample_dead_features(
        sae,
        dead,
        batches,
        optimizer=optimizer,
        target_l0=4,
        num_batches=4,
    )

    sample = torch.cat(batches)
    pre_activation = (sample - sae.b_dec) @ sae.W_enc[:, dead] + sae.b_enc[dead]
    firing_rate = (pre_activation > sae.threshold[dead]).float().mean().item()

    assert n == 3
    assert firing_rate == pytest.approx(4 / 16, abs=0.02)
    assert not torch.allclose(sae.threshold[dead], torch.full((3,), 0.01))
    assert torch.count_nonzero(optimizer.state[sae.W_enc]["exp_avg"][:, dead]) == 0
    assert torch.count_nonzero(optimizer.state[sae.threshold]["exp_avg"][dead]) == 0


def test_dead_feature_resampling_preserves_threshold_scale_without_target():
    torch.manual_seed(13)
    sae = JumpReLUSAE(d_model=4, d_sae=8, dtype=torch.float32)
    sae.threshold.data.fill_(1.5)

    _resample_dead_features(
        sae,
        torch.tensor([0]),
        [torch.randn(32, 4)],
        target_l0=None,
        num_batches=1,
    )

    assert sae.threshold[0].item() == pytest.approx(1.5)
