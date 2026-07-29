import sys
import types

import numpy as np
import pytest
import torch

from kiji_inspector.extraction.activation_extractor import ActivationConfig, ActivationExtractor
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
)


def test_hf_hook_stores_full_sequence():
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="decision",
    )
    extractor._hooks = []
    extractor._activations = {}

    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # Pre-hook signature: (module, args, kwargs) — hidden_states is args[0].
    extractor._make_hook("residual_20")(None, (activation,), {})

    stored = extractor._activations["residual_20"]
    assert stored.shape == (2, 3, 4)
    assert torch.equal(stored, activation)


def test_hf_hook_reads_hidden_states_from_kwargs():
    """Some HF decoder layers are invoked with hidden_states as a kwarg."""
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="decision",
    )
    extractor._hooks = []
    extractor._activations = {}

    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    extractor._make_hook("residual_20")(None, (), {"hidden_states": activation})

    stored = extractor._activations["residual_20"]
    assert stored.shape == (2, 3, 4)
    assert torch.equal(stored, activation)


def test_hf_hook_registered_as_pre_hook_captures_layer_input():
    """Regression test for the HF/vLLM layer-indexing off-by-one.

    vLLM's aux-hidden-state extraction (used to build SAE training data)
    records the residual stream *entering* layer N, with N=0 seeded at the
    embedding output before any layer runs — the same convention as HF's own
    ``output_hidden_states=True`` tuple (``hidden_states[N]`` is the input to
    layer N). Registering the HF hook as a forward *hook* (capturing a
    layer's output) would be off by one relative to that convention;
    registering it as a forward *pre-hook* (capturing the layer's input)
    matches it exactly, including at N=0 where there is no ``layers[-1]``.

    This test builds a tiny deterministic decoder-style stack — each "layer"
    adds a distinct offset — and checks that the pre-hook captures the exact
    tensor a layer receives, for both an early and a later layer, matching a
    reference computed by calling the layers directly (equivalent to HF's
    own ``output_hidden_states`` tuple entries).
    """

    class FakeLayer(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, hidden_states):
            return hidden_states + self.offset

    torch.manual_seed(0)
    layers = torch.nn.ModuleList([FakeLayer(offset=i + 1) for i in range(4)])
    embedding = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    # Reference, computed with no hooks attached: what each layer actually
    # receives as input (HF's output_hidden_states[N] convention — N=0 is
    # the embedding itself, before any layer runs).
    expected_input_to_layer = {}
    hidden_states = embedding
    for idx, layer in enumerate(layers):
        expected_input_to_layer[idx] = hidden_states
        hidden_states = layer(hidden_states)

    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="fake/model",
        layers=[0, 2],
        token_positions="all",
    )
    extractor._hooks = []
    extractor._activations = {}
    extractor._get_model_layers = lambda: layers
    extractor._register_hooks()

    # Now run the monitored forward pass the hooks are attached to.
    hidden_states = embedding
    for layer in layers:
        hidden_states = layer(hidden_states)

    for hook in extractor._hooks:
        hook.remove()

    assert torch.equal(extractor._activations["residual_0"], expected_input_to_layer[0])
    assert torch.equal(extractor._activations["residual_2"], expected_input_to_layer[2])
    # Layer 0's captured input must be the raw embedding, not layer 0's output.
    assert not torch.equal(extractor._activations["residual_0"], layers[0](embedding))


def test_hf_activation_to_numpy_uses_last_token_by_default():
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="decision",
    )

    activation = np.arange(12, dtype=np.float32).reshape(3, 4)

    np.testing.assert_array_equal(
        extractor._activation_to_numpy(activation),
        activation[-1],
    )
    np.testing.assert_array_equal(
        extractor._activation_to_numpy(activation, decision_token_offset=-2),
        activation[-2],
    )


def test_hf_hook_preserves_full_sequence_for_all_mode():
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="all",
    )
    extractor._hooks = []
    extractor._activations = {}

    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    extractor._make_hook("residual_20")(None, (activation,), {})

    stored = extractor._activations["residual_20"]
    assert stored.shape == (2, 3, 4)
    assert torch.equal(stored, activation)


def test_vllm_activation_to_numpy_uses_last_token_by_default():
    extractor = VLLMActivationExtractor.__new__(VLLMActivationExtractor)
    extractor.config = VLLMActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="decision",
    )

    activation = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    np.testing.assert_array_equal(
        extractor._activation_to_numpy(activation),
        activation[-1].numpy(),
    )
    np.testing.assert_array_equal(
        extractor._activation_to_numpy(activation, decision_token_offset=-2),
        activation[-2].numpy(),
    )


def _install_fake_vllm(monkeypatch, captured, hidden_size=64):
    """Install fake ``vllm`` modules matching the native connector API.

    Registers ``vllm``, ``vllm.config.kv_transfer`` and the
    ``example_hidden_states_connector`` module used by the extractor, capturing
    the ``LLM(...)`` kwargs and recording connector load/cleanup calls.
    """

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.model_config = types.SimpleNamespace(
                hf_text_config=types.SimpleNamespace(hidden_size=hidden_size)
            )

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params, use_tqdm=False):
            return captured.get("outputs", [])

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeKVTransferConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_kv = types.ModuleType("vllm.config.kv_transfer")
    fake_kv.KVTransferConfig = FakeKVTransferConfig

    # Connector helper module: from ...kv_connector.v1 import
    # example_hidden_states_connector
    fake_connector = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector"
    )

    def load_hidden_states(path):
        captured.setdefault("loaded", []).append(path)
        return captured["load_result"]

    def cleanup_hidden_states(path):
        captured.setdefault("cleaned", []).append(path)

    fake_connector.load_hidden_states = load_hidden_states
    fake_connector.cleanup_hidden_states = cleanup_hidden_states

    fake_v1 = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1")
    fake_v1.example_hidden_states_connector = fake_connector

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config.kv_transfer", fake_kv)
    monkeypatch.setitem(sys.modules, "vllm.distributed.kv_transfer.kv_connector.v1", fake_v1)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector",
        fake_connector,
    )


def test_recommended_vllm_kwargs_qwen_hybrid():
    from kiji_inspector.extraction.vllm_activation_extractor import recommended_vllm_kwargs

    # Hybrid multimodal Qwen models skip the vision tower and pin the Triton
    # MoE/attention backends (the known-good hidden-state extraction config).
    expected = {
        "language_model_only": True,
        "attention_backend": "TRITON_ATTN",
        "moe_backend": "triton",
        "enable_chunked_prefill": False,
    }
    assert recommended_vllm_kwargs("Qwen/Qwen3.6-35B-A3B") == expected
    assert recommended_vllm_kwargs("Qwen/Qwen3.6-35B-A3B-FP8") == expected
    # Non-hybrid models get no special defaults.
    assert recommended_vllm_kwargs("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16") == {}
    assert recommended_vllm_kwargs("google/gemma-3-27b-it") == {}


class _FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template


_QWEN_STYLE_TEMPLATE = "{% if enable_thinking %}<think>{% endif %}{{ messages }}"
_PLAIN_TEMPLATE = "{{ messages }}"


def test_recommended_chat_template_kwargs_by_name():
    from kiji_inspector.extraction.vllm_activation_extractor import (
        recommended_chat_template_kwargs,
    )

    # Canonical IDs, no tokenizer available: name heuristic.
    assert recommended_chat_template_kwargs("Qwen/Qwen3.6-35B-A3B") == {"enable_thinking": False}
    assert recommended_chat_template_kwargs("Qwen/Qwen3-Next-80B-A3B") == {"enable_thinking": False}
    assert recommended_chat_template_kwargs("meta-llama/Llama-3.3-70B") == {}
    assert recommended_chat_template_kwargs("nvidia/Nemotron-H-56B") == {}


def test_recommended_chat_template_kwargs_by_tokenizer_template():
    from kiji_inspector.extraction.vllm_activation_extractor import (
        recommended_chat_template_kwargs,
    )

    # A renamed fine-tune / local path still loads the Qwen chat template:
    # detection must key off the template, not the user-provided name.
    tok = _FakeTokenizer(_QWEN_STYLE_TEMPLATE)
    assert recommended_chat_template_kwargs("/models/judge", tok) == {"enable_thinking": False}
    # Named-template dict form.
    tok = _FakeTokenizer({"default": _QWEN_STYLE_TEMPLATE})
    assert recommended_chat_template_kwargs("/models/judge", tok) == {"enable_thinking": False}
    # Template known and lacking the knob: no kwargs, even for a Qwen name.
    tok = _FakeTokenizer(_PLAIN_TEMPLATE)
    assert recommended_chat_template_kwargs("Qwen/Qwen3.6-35B-A3B", tok) == {}
    # Tokenizer without a template falls back to the name heuristic.
    tok = _FakeTokenizer(None)
    assert recommended_chat_template_kwargs("Qwen/Qwen3.6-35B-A3B", tok) == {
        "enable_thinking": False
    }
    assert recommended_chat_template_kwargs("/models/judge", tok) == {}


def test_vllm_extractor_uses_native_connector_config(monkeypatch):
    captured = {}
    _install_fake_vllm(monkeypatch, captured)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            layers=[20],
        )
    )

    kwargs = captured["kwargs"]
    # No more bespoke extract_activation_layers kwarg.
    assert "extract_activation_layers" not in kwargs
    assert kwargs["dtype"] == "bfloat16"

    # Native extract_hidden_states speculator method drives layer selection.
    spec = kwargs["speculative_config"]
    assert spec["method"] == "extract_hidden_states"
    assert spec["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [20]

    # ExampleHiddenStatesConnector installed as a kv_producer.
    kv = kwargs["kv_transfer_config"].kwargs
    assert kv["kv_connector"] == "ExampleHiddenStatesConnector"
    assert kv["kv_role"] == "kv_producer"
    assert kv["kv_connector_extra_config"]["allow_custom_save_path"] is True

    assert extractor.hidden_size == 64
    assert extractor.tokenizer.pad_token == extractor.tokenizer.eos_token
    extractor.cleanup()


def test_vllm_extractor_forwards_hybrid_moe_knobs(monkeypatch):
    captured = {}
    _install_fake_vllm(monkeypatch, captured, hidden_size=2048)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name="Qwen/Qwen3.6-35B-A3B",
            layers=[1, 2, 3, 4],
            enforce_eager=True,
            attention_backend="TRITON_ATTN",
            moe_backend="triton",
            enable_chunked_prefill=False,
            language_model_only=True,
            revision="deadbeef",
        )
    )

    kwargs = captured["kwargs"]
    assert kwargs["enforce_eager"] is True
    assert kwargs["attention_backend"] == "TRITON_ATTN"
    assert kwargs["moe_backend"] == "triton"
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["language_model_only"] is True
    assert kwargs["revision"] == "deadbeef"
    assert kwargs["speculative_config"]["draft_model_config"]["hf_config"][
        "eagle_aux_hidden_state_layer_ids"
    ] == [1, 2, 3, 4]
    extractor.cleanup()


def test_vllm_output_to_activations_maps_columns_to_layers(monkeypatch):
    captured = {}
    _install_fake_vllm(monkeypatch, captured)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            layers=[8, 20],
            token_positions="decision",
        )
    )

    # hidden states shape [T=3, L=2, H=64]; column c -> layers[c].
    hs = torch.arange(3 * 2 * 64, dtype=torch.float32).reshape(3, 2, 64)
    captured["load_result"] = {
        "hidden_states": hs,
        "token_ids": torch.tensor([11, 22, 33]),
    }
    output = types.SimpleNamespace(
        kv_transfer_params={"hidden_states_path": "/tmp/req_0.safetensors"},
        prompt_token_ids=[11, 22, 33],
    )

    result = extractor._output_to_activations(output)

    assert set(result) == {"residual_8", "residual_20"}
    # "decision" -> last token (index -1) of each layer column.
    np.testing.assert_array_equal(result["residual_8"], hs[-1, 0, :].numpy())
    np.testing.assert_array_equal(result["residual_20"], hs[-1, 1, :].numpy())
    # File is always cleaned up.
    assert captured["cleaned"] == ["/tmp/req_0.safetensors"]
    extractor.cleanup()


def test_vllm_output_to_activations_cleans_up_on_shape_error(monkeypatch):
    captured = {}
    _install_fake_vllm(monkeypatch, captured)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            layers=[8, 20],
        )
    )

    # Wrong layer count (1 column vs 2 configured) must raise but still clean up.
    hs = torch.zeros(3, 1, 64, dtype=torch.float32)
    captured["load_result"] = {
        "hidden_states": hs,
        "token_ids": torch.tensor([11, 22, 33]),
    }
    output = types.SimpleNamespace(
        kv_transfer_params={"hidden_states_path": "/tmp/req_err.safetensors"},
        prompt_token_ids=[11, 22, 33],
    )

    with pytest.raises(ValueError):
        extractor._output_to_activations(output)
    assert captured["cleaned"] == ["/tmp/req_err.safetensors"]
    extractor.cleanup()


def test_vllm_output_to_activations_detects_token_misalignment(monkeypatch):
    captured = {}
    _install_fake_vllm(monkeypatch, captured)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            layers=[20],
        )
    )

    hs = torch.zeros(3, 1, 64, dtype=torch.float32)
    captured["load_result"] = {
        "hidden_states": hs,
        "token_ids": torch.tensor([11, 22, 99]),  # mismatch on last token
    }
    output = types.SimpleNamespace(
        kv_transfer_params={"hidden_states_path": "/tmp/req_mis.safetensors"},
        prompt_token_ids=[11, 22, 33],
    )

    with pytest.raises(ValueError):
        extractor._output_to_activations(output)
    assert captured["cleaned"] == ["/tmp/req_mis.safetensors"]
    extractor.cleanup()
