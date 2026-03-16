import pytest

from kiji_inspector.core.registry import MODEL_REGISTRY, resolve_repo_id


class TestResolveRepoId:
    def test_known_model_returns_repo(self):
        repo = resolve_repo_id("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        assert repo == "575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    def test_unknown_model_raises_keyerror(self):
        with pytest.raises(KeyError, match="No SAE repo registered"):
            resolve_repo_id("unknown/model")

    def test_error_lists_available_models(self):
        with pytest.raises(KeyError, match="Available models"):
            resolve_repo_id("missing/model")

    def test_error_suggests_repo_id_bypass(self):
        with pytest.raises(KeyError, match="repo_id directly"):
            resolve_repo_id("missing/model")


class TestModelRegistry:
    def test_registry_is_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_all_entries_are_strings(self):
        for key, value in MODEL_REGISTRY.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_all_entries_look_like_hf_repos(self):
        for key, value in MODEL_REGISTRY.items():
            assert "/" in key, f"base_model key missing '/': {key}"
            assert "/" in value, f"repo_id value missing '/': {value}"
