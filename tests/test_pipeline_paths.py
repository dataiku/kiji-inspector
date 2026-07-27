"""Tests for the --subject-model local-path resolution in the pipeline CLI."""

import pytest

from kiji_inspector.pipeline import _resolve_subject_model


def make_model_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return path


class TestLocalPaths:
    def test_absolute_path_resolved(self, tmp_path):
        model = make_model_dir(tmp_path / "my-model")
        assert _resolve_subject_model(str(model)) == str(model)

    def test_dot_relative_path_resolved(self, tmp_path, monkeypatch):
        model = make_model_dir(tmp_path / "my-model")
        monkeypatch.chdir(tmp_path)
        assert _resolve_subject_model("./my-model") == str(model)

    def test_dotdot_relative_path_resolved(self, tmp_path, monkeypatch):
        model = make_model_dir(tmp_path / "my-model")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert _resolve_subject_model("../my-model") == str(model)

    def test_tilde_path_resolved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        model = make_model_dir(tmp_path / "my-model")
        assert _resolve_subject_model("~/my-model") == str(model)

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _resolve_subject_model(str(tmp_path / "nope"))

    def test_missing_relative_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            _resolve_subject_model("./typo-model")

    def test_file_not_directory_raises(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="not a directory"):
            _resolve_subject_model(str(f))

    def test_missing_config_json_raises(self, tmp_path):
        d = tmp_path / "empty-model"
        d.mkdir()
        with pytest.raises(ValueError, match="no config.json"):
            _resolve_subject_model(str(d))

    def test_docker_hint_in_error(self, tmp_path):
        with pytest.raises(ValueError, match="volume-mounted"):
            _resolve_subject_model(str(tmp_path / "nope"))


class TestHubIds:
    def test_org_name_passthrough(self):
        assert (
            _resolve_subject_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
            == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        )

    def test_bare_name_passthrough(self):
        assert _resolve_subject_model("gpt2") == "gpt2"

    def test_org_name_passthrough_even_if_local_dir_exists(self, tmp_path, monkeypatch):
        make_model_dir(tmp_path / "org" / "name")
        monkeypatch.chdir(tmp_path)
        assert _resolve_subject_model("org/name") == "org/name"
