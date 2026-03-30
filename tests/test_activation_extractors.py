import sys
import types

import numpy as np
import torch

from kiji_inspector.extraction.activation_extractor import ActivationConfig, ActivationExtractor
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
)


def test_hf_hook_keeps_only_last_token_for_decision_mode():
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="decision",
    )
    extractor._hooks = []
    extractor._activations = {}

    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    extractor._make_hook("residual_20")(None, None, activation)

    stored = extractor._activations["residual_20"]
    assert stored.shape == (2, 4)
    assert torch.equal(stored, activation[:, -1, :])


def test_hf_hook_preserves_full_sequence_for_all_mode():
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.config = ActivationConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        token_positions="all",
    )
    extractor._hooks = []
    extractor._activations = {}

    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    extractor._make_hook("residual_20")(None, None, activation)

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


def test_vllm_extractor_uses_compiled_activation_config(monkeypatch):
    captured = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.model_config = types.SimpleNamespace(
                hf_text_config=types.SimpleNamespace(hidden_size=64)
            )

        def get_tokenizer(self):
            return FakeTokenizer()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCompilationConfig:
        def __init__(self, mode):
            self.mode = mode

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_compilation = types.ModuleType("vllm.config.compilation")
    fake_compilation.CompilationConfig = FakeCompilationConfig
    fake_compilation.CompilationMode = types.SimpleNamespace(
        STOCK_TORCH_COMPILE="stock_torch_compile"
    )

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", fake_compilation)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            layers=[20],
        )
    )

    kwargs = captured["kwargs"]
    assert kwargs["extract_activation_layers"] == (20,)
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["compilation_config"].mode == "stock_torch_compile"
    assert extractor.hidden_size == 64
    assert extractor.tokenizer.pad_token == extractor.tokenizer.eos_token
