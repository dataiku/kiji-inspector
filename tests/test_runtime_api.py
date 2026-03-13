import json

import torch

from kiji_inspector import SAE
from kiji_inspector.core.sae_core import JumpReLUSAE


def _write_checkpoint(tmp_path):
    model = JumpReLUSAE(d_model=4, d_sae=3, dtype=torch.float32)
    with torch.no_grad():
        model.W_enc.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                    [0.5, 0.5, 1.0],
                    [1.0, -0.5, 0.25],
                ],
                dtype=torch.float32,
            )
        )
        model.b_enc.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
        model.threshold.copy_(torch.tensor([0.0, 0.25, 0.5], dtype=torch.float32))
        model.W_dec.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5, 0.0],
                    [0.5, 0.5, 1.0, 0.25],
                ],
                dtype=torch.float32,
            )
        )
        model.b_dec.copy_(torch.tensor([0.0, 0.1, -0.1, 0.2], dtype=torch.float32))

    checkpoint_path = tmp_path / "sae_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": model.d_model,
                "d_sae": model.d_sae,
                "dtype": "float32",
                "bandwidth": model.bandwidth,
                "threshold_init": 0.01,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_runtime_public_imports():
    assert SAE.__name__ == "SAE"
    assert JumpReLUSAE.__name__ == "JumpReLUSAE"


def test_sae_from_pretrained_downloads_and_loads(monkeypatch, tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    descriptions_path = tmp_path / "feature_descriptions.json"
    expected_descriptions = {"0": {"label": "test feature"}}
    descriptions_path.write_text(json.dumps(expected_descriptions))

    downloads = []

    def fake_download(repo_id, filename, cache_dir=None, token=None):
        downloads.append((repo_id, filename, cache_dir, token))
        if filename.endswith("sae_checkpoints/sae_final.pt"):
            return str(checkpoint_path)
        if filename.endswith("activations/feature_descriptions.json"):
            return str(descriptions_path)
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr("kiji_inspector.core.sae.hf_hub_download", fake_download)

    sae, feature_descriptions = SAE.from_pretrained(
        base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        layer=20,
        cache_dir=str(tmp_path / "cache"),
        token="secret-token",
    )

    assert isinstance(sae, SAE)
    assert not sae.training
    assert feature_descriptions == expected_descriptions
    assert downloads == [
        (
            "hanneshapke/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "layer_20/sae_checkpoints/sae_final.pt",
            str(tmp_path / "cache"),
            "secret-token",
        ),
        (
            "hanneshapke/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "layer_20/activations/feature_descriptions.json",
            str(tmp_path / "cache"),
            "secret-token",
        ),
    ]

    activations = torch.tensor(
        [
            [1.0, 0.5, -0.5, 0.25],
            [0.1, 0.4, 0.8, -0.2],
        ],
        dtype=torch.float32,
    )
    features = sae.encode(activations)
    reconstruction = sae.decode(features)

    assert features.shape == (2, 3)
    assert reconstruction.shape == activations.shape
    assert torch.equal(features[:, 0] >= 0, torch.tensor([True, True]))
