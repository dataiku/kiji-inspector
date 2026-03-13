import json

import pytest
import torch

from kiji_inspector import SAE
from kiji_inspector.core.sae_core import JumpReLUSAE


def _write_checkpoint(tmp_path, d_model=4, d_sae=3):
    model = JumpReLUSAE(d_model=d_model, d_sae=d_sae, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(model.W_enc)
    torch.nn.init.xavier_uniform_(model.W_dec)
    checkpoint_path = tmp_path / "sae_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": d_model,
                "d_sae": d_sae,
                "dtype": "float32",
                "bandwidth": model.bandwidth,
                "threshold_init": 0.01,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _fake_download_factory(checkpoint_path, descriptions_path=None, downloads=None):
    """Return a fake hf_hub_download that serves local files."""

    def fake_download(repo_id, filename, cache_dir=None, token=None):
        if downloads is not None:
            downloads.append((repo_id, filename))
        if filename.endswith("sae_checkpoints/sae_final.pt"):
            return str(checkpoint_path)
        if filename.endswith("activations/feature_descriptions.json"):
            if descriptions_path is not None:
                return str(descriptions_path)
            raise FileNotFoundError("no descriptions")
        raise FileNotFoundError(f"unexpected: {filename}")

    return fake_download


class TestSAEIsSubclass:
    def test_inherits_from_jumprelu(self):
        assert issubclass(SAE, JumpReLUSAE)

    def test_instance_type(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt),
        )
        sae, _ = SAE.from_pretrained(repo_id="fake/repo", layer=0)
        assert isinstance(sae, SAE)
        assert isinstance(sae, JumpReLUSAE)


class TestFromPretrainedWithRepoId:
    """Bypass the registry by passing repo_id directly."""

    def test_loads_checkpoint(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt),
        )
        sae, desc = SAE.from_pretrained(repo_id="custom/repo", layer=5)
        assert sae.d_model == 4
        assert sae.d_sae == 3
        assert not sae.training  # eval mode
        assert desc is None  # no descriptions file

    def test_loads_with_descriptions(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        desc_path = tmp_path / "feature_descriptions.json"
        expected = {"0": {"label": "some feature"}, "1": {"label": "other"}}
        desc_path.write_text(json.dumps(expected))

        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt, desc_path),
        )
        _, desc = SAE.from_pretrained(repo_id="custom/repo", layer=0)
        assert desc == expected


class TestFromPretrainedWithBaseModel:
    """Use the registry to resolve base_model → repo_id."""

    def test_resolves_registered_model(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        downloads = []
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt, downloads=downloads),
        )
        SAE.from_pretrained(
            base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", layer=20
        )
        for repo_id, _ in downloads:
            assert repo_id == "hanneshapke/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    def test_unregistered_model_raises_keyerror(self):
        with pytest.raises(KeyError, match="No SAE repo registered"):
            SAE.from_pretrained(base_model="unknown/model", layer=0)


class TestFromPretrainedValidation:
    def test_neither_base_model_nor_repo_id_raises(self):
        with pytest.raises(ValueError, match="Provide either base_model or repo_id"):
            SAE.from_pretrained()

    def test_missing_checkpoint_raises_file_not_found(self, monkeypatch):
        def always_fail(repo_id, filename, cache_dir=None, token=None):
            raise Exception("network error")

        monkeypatch.setattr("kiji_inspector.core.sae.hf_hub_download", always_fail)
        with pytest.raises(FileNotFoundError, match="Could not download"):
            SAE.from_pretrained(repo_id="fake/repo", layer=0)


class TestFromPretrainedDownloadPaths:
    """Verify the correct HF paths are requested for a given layer."""

    def test_layer_subfolder_in_paths(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        downloads = []
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt, downloads=downloads),
        )
        SAE.from_pretrained(repo_id="org/repo", layer=42)
        filenames = [fn for _, fn in downloads]
        assert filenames[0] == "layer_42/sae_checkpoints/sae_final.pt"
        assert filenames[1] == "layer_42/activations/feature_descriptions.json"

    def test_layer_zero_default(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        downloads = []
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt, downloads=downloads),
        )
        SAE.from_pretrained(repo_id="org/repo")
        assert downloads[0][1].startswith("layer_0/")


class TestFromPretrainedEncodeDecode:
    """Loaded SAE should produce correct-shaped outputs."""

    def test_encode_decode_shapes(self, tmp_path, monkeypatch):
        d_model, d_sae = 8, 16
        ckpt = _write_checkpoint(tmp_path, d_model=d_model, d_sae=d_sae)
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt),
        )
        sae, _ = SAE.from_pretrained(repo_id="fake/repo", layer=0)

        x = torch.randn(3, d_model)
        features = sae.encode(x)
        reconstruction = sae.decode(features)

        assert features.shape == (3, d_sae)
        assert reconstruction.shape == (3, d_model)

    def test_forward_returns_reconstruction_and_features(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt),
        )
        sae, _ = SAE.from_pretrained(repo_id="fake/repo", layer=0)

        x = torch.randn(2, 4)
        reconstruction, features = sae(x)

        assert reconstruction.shape == x.shape
        assert features.shape == (2, 3)

    def test_features_are_non_negative(self, tmp_path, monkeypatch):
        ckpt = _write_checkpoint(tmp_path)
        monkeypatch.setattr(
            "kiji_inspector.core.sae.hf_hub_download",
            _fake_download_factory(ckpt),
        )
        sae, _ = SAE.from_pretrained(repo_id="fake/repo", layer=0)

        x = torch.randn(10, 4)
        features = sae.encode(x)
        assert (features >= 0).all()
