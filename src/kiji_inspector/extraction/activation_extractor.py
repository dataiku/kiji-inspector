"""
Activation extractor for HuggingFace causal language models.

Designed for multi-GPU inference. The model is sharded across GPUs
via device_map="auto" and activations are captured via forward hooks
at specified layers.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ActivationConfig:
    """Configuration for activation extraction."""

    model_name: str = ""  # Required — HuggingFace model ID (e.g. "Qwen/Qwen2.5-3B-Instruct")
    layers: list[int] = field(default_factory=lambda: [8, 12, 16, 20, 24])
    extract_residual_stream: bool = True
    extract_attention: bool = False
    extract_mlp: bool = False
    token_positions: str = "decision"  # "all", "last", "decision"
    dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True
    max_memory: dict[int, str] | None = None  # e.g. {0: "180GiB", 1: "180GiB", ...}


class ActivationExtractor:
    """Extract activations from any HuggingFace causal language model.

    Supports models using standard architectures (Llama, Qwen, Mistral,
    Nemotron, GPT-NeoX, etc.). The model is automatically sharded across
    all available GPUs via device_map="auto".
    """

    def __init__(self, config: ActivationConfig):
        if not config.model_name:
            raise ValueError("ActivationConfig.model_name is required.")
        self.config = config
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        num_gpus = torch.cuda.device_count()
        print(f"Loading model: {self.config.model_name}")
        print(f"  dtype: {self.config.dtype}")
        print(f"  available GPUs: {num_gpus}")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {name} ({mem:.0f} GiB)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        # Pad token needed for batched inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "dtype": self.config.dtype,
            "device_map": "auto",
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.max_memory is not None:
            load_kwargs["max_memory"] = self.config.max_memory

        # Nemotron-H's custom modelling code needs mamba_ssm for a Triton-based
        # rmsnorm, but mamba_ssm's __init__ unconditionally imports the CUDA kernel
        # selective_scan_cuda which isn't compiled for all platforms (e.g. GB200).
        # Inject a stub so the import succeeds; the model never calls this kernel.
        import sys
        import types

        if "selective_scan_cuda" not in sys.modules:
            sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **load_kwargs)
        self.model.eval()

        # Blackwell (SM ≥ 100): mamba-ssm's Triton kernels crash with illegal
        # memory access.  Fall back to the pure-PyTorch path for Mamba mixer
        # blocks by clearing the use_cuda_kernels flag that the HF modelling
        # code checks in its forward().
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 10:
            _patched = 0
            for mod in self.model.modules():
                if getattr(mod, "use_cuda_kernels", False):
                    mod.use_cuda_kernels = False
                    _patched += 1
            if _patched:
                print(f"  Blackwell GPU: disabled CUDA kernels on {_patched} Mamba mixer block(s)")

        # FP8 quantized models (e.g. ModelOpt FP8) store weights in float8_e4m3fn.
        # HuggingFace doesn't auto-dequantize these, so F.linear fails with a
        # dtype mismatch.  Cast any FP8 parameters to the target dtype.
        _fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
        n_cast = 0
        for param in self.model.parameters():
            if param.dtype in _fp8_dtypes:
                param.data = param.data.to(self.config.dtype)
                n_cast += 1
        if n_cast:
            print(f"  Cast {n_cast} FP8 parameters to {self.config.dtype}")

        # Warn if running on Blackwell without P2P mitigations
        if num_gpus > 1:
            import os

            major, _ = torch.cuda.get_device_capability(0)
            if major >= 10 and os.environ.get("NCCL_P2P_DISABLE") != "1":
                print(
                    "  WARNING: Blackwell GPU detected with P2P enabled. "
                    "This may cause host OOM from NVLink memory mapping. "
                    "Use --disable-p2p auto (or set NCCL_P2P_DISABLE=1) to prevent this."
                )

        # Enable CUDA optimizations for inference
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

        # Determine where the model's embedding layer lives — inputs must go there
        self._input_device = self._find_input_device()
        print(f"  input device: {self._input_device}")

        # Storage for hooked activations
        self._activations: dict[str, torch.Tensor] = {}

        # Register hooks on target layers
        self._register_hooks()

        # Cache hidden size for downstream consumers (Gemma3 nests under text_config)
        self.hidden_size = (
            getattr(self.model.config, "hidden_size", None)
            or getattr(self.model.config, "text_config", self.model.config).hidden_size
        )
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  hooked layers: {self.config.layers}")
        print(f"  model type: {type(self.model).__name__}")
        if hasattr(self.model, "model"):
            print(
                f"  model.model attrs: {[a for a in dir(self.model.model) if not a.startswith('_')]}"
            )

    def _get_inner_model(self):
        """Get the transformer body, skipping lm_head to avoid allocating logits.

        Walks common inner-model attribute paths used by various architectures:
        - language_model       (multimodal wrappers: Gemma3, LLaVA, etc.)
        - model                (standard: Llama, Qwen, Mistral, Gemma, etc.)
        - backbone             (NemotronH)
        - transformer          (GPT-NeoX, GPT-2)
        """
        for attr in ("language_model", "model", "backbone", "transformer"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        return self.model

    def _find_input_device(self) -> torch.device:
        """Find the device of the model's embedding layer.

        When using device_map="auto", different layers live on different GPUs.
        Input tensors must be sent to whatever device holds the embedding layer.
        """
        # Try common embedding attribute names
        for attr_path in (
            "language_model.model.embed_tokens",
            "language_model.embed_tokens",
            "model.embed_tokens",
            "backbone.embed_tokens",
            "backbone.embedding",
            "transformer.wte",
            "model.embed_in",
        ):
            parts = attr_path.split(".")
            obj = self.model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return next(obj.parameters()).device
            except (AttributeError, StopIteration):
                continue

        # Fallback: first parameter's device
        return next(self.model.parameters()).device

    def _get_model_layers(self):
        """Get the layer list, handling different model architectures."""
        # Gemma3 multimodal (Gemma3ForConditionalGeneration): language_model.model.layers
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
            # Some versions expose layers directly on language_model
            if hasattr(lm, "layers"):
                return lm.layers
        # Standard Llama/Nemotron/Gemma architecture
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # NemotronH architecture uses 'backbone' instead of 'model'
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone
            if hasattr(backbone, "layers"):
                return backbone.layers
            if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "layers"):
                return backbone.decoder.layers
        # NemotronH architecture (model.decoder.layers)
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            if hasattr(self.model.model.decoder, "layers"):
                return self.model.model.decoder.layers
        # Alternative: direct decoder access
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            return self.model.decoder.layers
        # GPT-NeoX architecture
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        # GPT-2 style
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            return self.model.transformer.layers

        # Debug: print backbone structure if it exists
        if hasattr(self.model, "backbone"):
            backbone_attrs = [attr for attr in dir(self.model.backbone) if not attr.startswith("_")]
            raise AttributeError(
                f"Cannot locate transformer layers for {type(self.model).__name__}. "
                f"Found 'backbone' but could not find layers. "
                f"Backbone attributes: {backbone_attrs}"
            )

        raise AttributeError(
            f"Cannot locate transformer layers for {type(self.model).__name__}. "
            "Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, "
            "MistralForCausalLM, NemotronForCausalLM, NemotronHForCausalLM, "
            "GPTNeoXForCausalLM, and others using model.model.layers. "
            f"Model structure: {[attr for attr in dir(self.model) if not attr.startswith('_')]}"
        )

    def _register_hooks(self):
        """Register forward hooks to capture activations at specified layers."""
        layers = self._get_model_layers()
        num_layers = len(layers)

        for layer_idx in self.config.layers:
            if layer_idx >= num_layers:
                print(
                    f"  WARNING: layer {layer_idx} requested but model only has "
                    f"{num_layers} layers, skipping"
                )
                continue

            layer = layers[layer_idx]

            if self.config.extract_residual_stream:
                hook = layer.register_forward_hook(self._make_hook(f"residual_{layer_idx}"))
                self._hooks.append(hook)

            if self.config.extract_attention:
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                if attn is not None:
                    hook = attn.register_forward_hook(self._make_hook(f"attention_{layer_idx}"))
                    self._hooks.append(hook)

            if self.config.extract_mlp:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None:
                    hook = mlp.register_forward_hook(self._make_hook(f"mlp_{layer_idx}"))
                    self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook function that stores activations.

        Activations are moved to CPU and cast to float32 immediately.
        This is critical for multi-GPU: hooked tensors may live on any GPU,
        and we need a uniform representation for numpy conversion.
        """

        def hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            self._activations[name] = activation.detach().cpu().to(torch.float32)

        return hook

    def extract(
        self,
        prompt: str,
        decision_token_offset: int = -1,
    ) -> dict[str, np.ndarray]:
        """
        Extract activations for a single prompt.

        Args:
            prompt: The full formatted prompt text.
            decision_token_offset: Token position to extract from.
                -1 means last token (the decision point).

        Returns:
            Dictionary mapping layer names to activation vectors (1D numpy arrays).
        """
        self._activations = {}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
        seq_len = inputs.input_ids.shape[1]

        # Run the transformer body only — skip lm_head to avoid
        # allocating the huge (batch × seq × vocab) logit tensor.
        inner_model = self._get_inner_model()
        with torch.no_grad():
            inner_model(**inputs)

        if self.config.token_positions in ("last", "decision"):
            position = seq_len + decision_token_offset
            result = {
                name: act[:, position, :].squeeze(0).numpy()
                for name, act in self._activations.items()
            }
        elif self.config.token_positions == "all":
            result = {name: act.squeeze(0).numpy() for name, act in self._activations.items()}
        else:
            raise ValueError(f"Unknown token_positions: {self.config.token_positions}")

        return result

    def extract_batch(
        self,
        prompts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, np.ndarray]]:
        """Extract activations for multiple prompts with true batched forward passes.

        Prompts are left-padded so the decision token (last real token) aligns
        across the batch — this is important because we extract at a fixed
        offset from each sequence's end.
        """
        self.tokenizer.padding_side = "left"
        results: list[dict[str, np.ndarray]] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            self._activations = {}

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._input_device)

            attention_mask = inputs.attention_mask  # (B, T)

            # Run the transformer body only — skip lm_head to avoid
            # allocating the huge (batch × seq × vocab) logit tensor.
            inner_model = self._get_inner_model()
            with torch.no_grad():
                inner_model(**inputs)

            # For each item in the batch, find its last real token position
            for b in range(len(batch_prompts)):
                # Last non-pad token index
                real_len = attention_mask[b].sum().item()
                position = real_len - 1  # 0-indexed last real token

                if self.config.token_positions in ("last", "decision"):
                    item = {
                        name: act[b, position, :].numpy() for name, act in self._activations.items()
                    }
                elif self.config.token_positions == "all":
                    # Return only non-padded tokens
                    item = {
                        name: act[b, :real_len, :].numpy()
                        for name, act in self._activations.items()
                    }
                else:
                    raise ValueError(f"Unknown token_positions: {self.config.token_positions}")
                results.append(item)

        return results

    def cleanup(self):
        """Remove hooks, delete model, and free GPU memory."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

        if hasattr(self, "model"):
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()
