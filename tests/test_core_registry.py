import pytest

from kiji_inspector.core.registry import (
    BASE_MODEL_REVISIONS,
    MODEL_REGISTRY,
    MODEL_REVISIONS,
    resolve_repo_id,
    resolve_revision,
)


class TestResolveRepoId:
    def test_known_model_returns_repo(self):
        repo = resolve_repo_id("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        assert repo == "575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    def test_qwen_model_returns_repo(self):
        repo = resolve_repo_id("Qwen/Qwen3.6-35B-A3B")
        assert repo == "575-lab/kiji-inspector-Qwen-Qwen3.6-35B-A3B"

    def test_nemotron_3_5_nano_model_returns_repo(self):
        repo = resolve_repo_id("nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16")
        assert repo == "575-lab/kiji-inspector-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

    def test_gemma4_model_returns_repo(self):
        repo = resolve_repo_id("google/gemma-4-E4B-it")
        assert repo == "575-lab/kiji-inspector-google-gemma-4-E4B-it"

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


class TestRevisionPinning:
    """``main`` moves; published results must not.

    ``hf_hub_download`` resolves ``main`` unless given a revision, so without
    these pins a rerun can silently load different weights than the paper was
    written from -- and the base model proves it happens: the runs used
    ``d468880b`` and upstream has since moved past it.
    """

    def test_every_registered_repo_is_pinned(self):
        unpinned = sorted(set(MODEL_REGISTRY.values()) - set(MODEL_REVISIONS))
        assert not unpinned, f"registered but unpinned, so they track main: {unpinned}"

    def test_pins_are_full_commit_hashes(self):
        for repo, revision in {**MODEL_REVISIONS, **BASE_MODEL_REVISIONS}.items():
            assert len(revision) == 40, f"{repo}: not a full commit hash"
            assert set(revision) <= set("0123456789abcdef"), f"{repo}: not hex"

    def test_no_pin_for_an_unregistered_repo(self):
        stale = sorted(set(MODEL_REVISIONS) - set(MODEL_REGISTRY.values()))
        assert not stale, f"pinned but no longer registered: {stale}"

    def test_resolve_revision_returns_the_pin(self):
        repo = resolve_repo_id("nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16")
        assert resolve_revision(repo) == "2380c95cbeddeb8a7aca17a122bc18df5ae62aaa"

    def test_unknown_repo_falls_back_to_main(self):
        assert resolve_revision("someone/not-ours") is None


class TestLoaderUsesThePin:
    def test_download_is_given_the_pinned_revision(self, monkeypatch):
        """The pin is worth nothing if it never reaches ``hf_hub_download``."""
        import kiji_inspector.core.sae as sae_module

        seen = {}

        def fake_download(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here; the call is what we are checking")

        monkeypatch.setattr(sae_module, "hf_hub_download", fake_download)
        with pytest.raises(FileNotFoundError):
            sae_module.SAE.from_pretrained(
                base_model="nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16", layer=43
            )
        assert seen["revision"] == "2380c95cbeddeb8a7aca17a122bc18df5ae62aaa"

    def test_explicit_revision_wins(self, monkeypatch):
        import kiji_inspector.core.sae as sae_module

        seen = {}

        def fake_download(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

        monkeypatch.setattr(sae_module, "hf_hub_download", fake_download)
        with pytest.raises(FileNotFoundError):
            sae_module.SAE.from_pretrained(
                base_model="nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16",
                layer=43,
                revision="main",
            )
        assert seen["revision"] == "main"
