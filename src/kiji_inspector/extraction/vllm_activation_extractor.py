"""
Activation extractor using vLLM's native hidden-state extraction connector.

Drop-in replacement for ActivationExtractor that uses vLLM's
``extract_hidden_states`` speculator method together with the
``ExampleHiddenStatesConnector`` instead of HuggingFace Transformers with
manual forward hooks.  Significantly faster for bulk extraction thanks to
vLLM's continuous batching and compiled kernels.

How it works (native connector API):

* The model is loaded with ``speculative_config={"method":
  "extract_hidden_states", ...}`` and the target layers are supplied via
  ``draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids``.
* A ``kv_transfer_config`` installs the ``ExampleHiddenStatesConnector``
  (``kv_producer``) which writes the captured hidden states to safetensors
  files on disk.
* Each request is given a unique ``hidden_states_path`` via
  ``SamplingParams.extra_args["kv_transfer_params"]`` so files never collide
  under continuous batching.
* After generation, the per-request path is read from
  ``output.kv_transfer_params["hidden_states_path"]``, loaded with the
  connector's ``load_hidden_states`` helper (a single tensor of shape
  ``[num_tokens, num_layers, hidden_size]`` — the layer axis is the *middle*
  axis, ordered by ``eagle_aux_hidden_state_layer_ids``), split back into
  ``residual_{layer}`` entries, and the file is removed with
  ``cleanup_hidden_states``.

This requires the hidden-state-enabled vLLM build shipped in the
``575lab/kiji-inspector:dev`` image (Davidnet/vllm fork); the public
v0.19.0 wheel does not contain the connector.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class VLLMActivationConfig:
    """Configuration for vLLM-based activation extraction."""

    model_name: str = ""
    layers: list[int] = field(default_factory=lambda: [20])
    token_positions: str = "decision"  # "all", "last", "decision"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    max_num_seqs: int = 512
    trust_remote_code: bool = True
    # Optional model revision (pin a HuggingFace commit).  Useful for hybrid
    # MoE models such as Qwen3.6 whose reference activations were captured at a
    # specific revision.
    revision: str | None = None
    # Native connector knobs.
    num_speculative_tokens: int = 1
    # Optional vLLM engine knobs.  Left unset (None/False) they are omitted so
    # existing models (Nemotron, Gemma3) behave exactly as before.  Hybrid MoE
    # models such as Qwen3.6-35B-A3B can set these to pin a known-good backend
    # (see samples / the residual-stream-model-comparisons reference).
    enforce_eager: bool = False
    language_model_only: bool = False
    attention_backend: str | None = None
    moe_backend: str | None = None
    enable_chunked_prefill: bool | None = None
    # Escape hatch: extra kwargs forwarded verbatim to ``vllm.LLM(...)``.
    extra_llm_kwargs: dict = field(default_factory=dict)


def recommended_vllm_kwargs(model_name: str) -> dict:
    """Model-family defaults for the vLLM extractor.

    Some subject models need non-default engine settings to load reliably for
    hidden-state extraction.  The Qwen3.5/3.6 hybrid MoE family in particular:

    * is multimodal — loading the vision tower makes engine startup stall on
      multimodal profiling, so ``language_model_only=True`` skips it (we only
      need the language-model residual stream);
    * hangs during the profiling forward pass on this image's FlashInfer MoE
      backend, so we pin the Triton MoE + attention backends and disable chunked
      prefill (the known-good config from the residual-stream reference).  The
      extractor additionally sets ``VLLM_USE_FLASHINFER_SAMPLER=0`` when the
      Triton MoE backend is selected.

    Callers can override any of these.
    """
    name = model_name.lower()
    if any(tag in name for tag in ("qwen3.6", "qwen3.5", "qwen3_5", "qwen3-next", "qwen3_next")):
        return {
            "language_model_only": True,
            "attention_backend": "TRITON_ATTN",
            "moe_backend": "triton",
            "enable_chunked_prefill": False,
        }
    return {}


def recommended_chat_template_kwargs(model_name: str) -> dict:
    """Model-family kwargs for ``tokenizer.apply_chat_template``.

    Judge/labeling subprocesses run models with small token budgets and
    machine-parsed outputs, so reasoning modes must be disabled at the chat
    template. The mechanism is model-family specific:

    * Qwen3 family templates implement ``enable_thinking``; without
      ``enable_thinking=False`` the model meta-reasons in plain text (no
      ``<think>`` tags to strip) and exhausts the budget before answering.
    * Other families ignore unknown template kwargs, so returning Qwen's
      knob for unknown models would be harmless but ineffective — extend
      this dispatch when adding a judge family with a different convention
      (e.g. Nemotron toggles reasoning via a system-prompt phrase).
    """
    name = model_name.lower()
    if "qwen3" in name or "qwen-3" in name:
        return {"enable_thinking": False}
    return {}


class VLLMActivationExtractor:
    """Extract activations using vLLM's native hidden-state connector.

    Presents the same interface as ``ActivationExtractor`` (``extract()``,
    ``extract_batch()``, ``cleanup()``, ``tokenizer``, ``hidden_size``)
    so it can be used as a drop-in replacement wherever the HuggingFace
    Transformers-based extractor is used.
    """

    def __init__(self, config: VLLMActivationConfig):
        if not config.model_name:
            raise ValueError("VLLMActivationConfig.model_name is required.")
        if config.token_positions not in {"all", "last", "decision"}:
            raise ValueError(f"Unknown token_positions: {config.token_positions}")
        self.config = config

        from vllm import LLM
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector,
        )

        # Helper module exposing load_hidden_states / cleanup_hidden_states.
        self._connector = example_hidden_states_connector

        # Per-request safetensors files land here; cleaned up in cleanup().
        self._storage_dir = tempfile.mkdtemp(prefix="kiji_hs_")
        # Column index -> layer id, matching eagle_aux_hidden_state_layer_ids.
        self._layer_order = list(config.layers)

        # The ExampleHiddenStatesConnector (a KV connector) is incompatible with
        # PyTorch's expandable-segments allocator: the CUDA VMM allocator can
        # remap KV-cache virtual addresses, invalidating registered KV memory.
        # Callers such as the pipeline enable it for Blackwell host-OOM
        # mitigation, so strip just that token here (keep the rest).
        alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:true" in alloc.lower():
            kept = [
                part
                for part in alloc.split(",")
                if part.strip().lower() != "expandable_segments:true"
            ]
            if kept:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(kept)
            else:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            print(
                "  Note: removed expandable_segments:True from PYTORCH_CUDA_ALLOC_CONF "
                "(incompatible with the hidden-states KV connector)."
            )

        # The FlashInfer sampler is unstable on the same hybrid-MoE path where
        # we pin the Triton MoE backend; avoid it too (matches the reference
        # config).  setdefault so an explicit env var still wins.
        if config.moe_backend == "triton":
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        print(f"Loading model via vLLM (native hidden-states connector): {config.model_name}")
        print(f"  layers (eagle_aux_hidden_state_layer_ids): {self._layer_order}")
        print(f"  tensor_parallel_size: {config.tensor_parallel_size}")

        llm_kwargs: dict = {
            "model": config.model_name,
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tensor_parallel_size,
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.max_num_seqs,
            "disable_log_stats": True,
            "speculative_config": {
                "method": "extract_hidden_states",
                "num_speculative_tokens": config.num_speculative_tokens,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": list(config.layers),
                    },
                },
            },
            "kv_transfer_config": KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "shared_storage_path": self._storage_dir,
                    "allow_custom_save_path": True,
                    "use_synchronization_lock": True,
                },
            ),
        }

        # Optional engine knobs — only forwarded when explicitly set.
        if config.revision:
            llm_kwargs["revision"] = config.revision
        if config.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if config.language_model_only:
            llm_kwargs["language_model_only"] = True
        if config.attention_backend:
            llm_kwargs["attention_backend"] = config.attention_backend
        if config.moe_backend:
            llm_kwargs["moe_backend"] = config.moe_backend
        if config.enable_chunked_prefill is not None:
            llm_kwargs["enable_chunked_prefill"] = config.enable_chunked_prefill
        llm_kwargs.update(config.extra_llm_kwargs)

        self.llm = LLM(**llm_kwargs)

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hf_text_config = getattr(self.llm.model_config, "hf_text_config", None)
        self.hidden_size = getattr(hf_text_config, "hidden_size", None) or getattr(
            self.llm.model_config, "hidden_size", None
        )
        if self.hidden_size is None:
            raise AttributeError("Could not determine hidden_size from vLLM model_config.")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  token_positions: {config.token_positions}")

    def _activation_to_numpy(
        self,
        activation: torch.Tensor,
        decision_token_offset: int = -1,
    ) -> np.ndarray:
        """Convert a ``[T, H]`` per-layer hidden-state slice into the requested view."""
        array = activation.to(device="cpu", dtype=torch.float32).numpy()

        if self.config.token_positions == "all":
            return array
        if self.config.token_positions in ("last", "decision"):
            if array.ndim == 1:
                return array
            position = array.shape[0] + decision_token_offset
            return array[position]
        raise ValueError(f"Unknown token_positions: {self.config.token_positions}")

    def _build_sampling_params(self, index: int):
        """Build per-request SamplingParams with a unique hidden-states path."""
        from vllm import SamplingParams

        path = os.path.join(self._storage_dir, f"req_{index}.safetensors")
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extra_args={
                "kv_transfer_params": {
                    "hidden_states_path": path,
                    "include_output_tokens": False,
                }
            },
        )

    def _output_to_activations(
        self,
        output,
        decision_token_offset: int = -1,
    ) -> dict[str, np.ndarray]:
        """Load one request's hidden states from disk and map columns to layers.

        The connector writes a single tensor of shape ``[T, L, H]`` where the
        layer axis (``L``) is ordered by ``eagle_aux_hidden_state_layer_ids``.
        The file is always removed afterwards, even if validation fails.
        """
        kv_params = getattr(output, "kv_transfer_params", None)
        if not kv_params or not kv_params.get("hidden_states_path"):
            raise RuntimeError(
                "vLLM did not return a hidden_states_path. Check the "
                "ExampleHiddenStatesConnector / extract_hidden_states config."
            )
        path = kv_params["hidden_states_path"]

        try:
            obj = self._connector.load_hidden_states(path)
            hidden_states = obj["hidden_states"]  # torch.Tensor [T, L, H]
            token_ids = obj["token_ids"]

            if hidden_states.ndim != 3:
                raise ValueError(
                    f"Expected hidden states with ndim==3 [T, L, H], "
                    f"got shape {tuple(hidden_states.shape)}."
                )
            n_tokens, n_layers, n_hidden = hidden_states.shape
            if n_layers != len(self._layer_order):
                raise ValueError(
                    f"Layer-axis mismatch: got {n_layers} layers but expected "
                    f"{len(self._layer_order)} (layers={self._layer_order})."
                )
            if n_hidden != self.hidden_size:
                raise ValueError(
                    f"Hidden-size mismatch: got {n_hidden}, expected {self.hidden_size}."
                )

            # Guard against silent token misalignment: the connector's token
            # ids (prompt tokens, since include_output_tokens=False) must match
            # the request's prompt token ids.
            prompt_ids = list(output.prompt_token_ids or [])
            extracted_ids = (
                token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            )
            if prompt_ids and extracted_ids != prompt_ids:
                raise ValueError(
                    "Extracted token ids do not match prompt token ids "
                    f"({len(extracted_ids)} vs {len(prompt_ids)} tokens); "
                    "hidden-state / token alignment is not trustworthy."
                )

            result: dict[str, np.ndarray] = {}
            for col in range(n_layers):
                key = f"residual_{self._layer_order[col]}"
                result[key] = self._activation_to_numpy(
                    hidden_states[:, col, :], decision_token_offset
                )
            return result
        finally:
            self._connector.cleanup_hidden_states(path)

    def extract(
        self,
        prompt: str,
        decision_token_offset: int = -1,
    ) -> dict[str, np.ndarray]:
        """Extract activations for a single prompt.

        Args:
            prompt: The full formatted prompt text.
            decision_token_offset: Token position to extract from.
                -1 means last token (the decision point).

        Returns:
            Dictionary mapping layer names to activation vectors.
        """
        sp = self._build_sampling_params(0)
        outputs = self.llm.generate([prompt], sp, use_tqdm=False)
        return self._output_to_activations(outputs[0], decision_token_offset)

    def extract_batch(
        self,
        prompts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, np.ndarray]]:
        """Extract activations for multiple prompts.

        vLLM handles batching internally via its scheduler, so all prompts
        are submitted at once.  The ``batch_size`` parameter is accepted
        for API compatibility but chunking is only used as a memory safety
        measure for very large prompt lists.

        Each request gets a unique ``hidden_states_path`` so the connector's
        per-request safetensors files never collide under continuous batching.
        """
        results: list[dict[str, np.ndarray]] = []

        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            params = [self._build_sampling_params(j) for j in range(len(chunk))]
            outputs = self.llm.generate(chunk, params, use_tqdm=False)

            for output in outputs:
                results.append(self._output_to_activations(output))

        return results

    def cleanup(self):
        """Free vLLM resources and GPU memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

        # Remove the per-request safetensors staging directory.
        storage_dir = getattr(self, "_storage_dir", None)
        if storage_dir:
            shutil.rmtree(storage_dir, ignore_errors=True)
            self._storage_dir = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()


def _dp_shard_worker(
    rank: int,
    prompts: list[str],
    config_kwargs: dict,
    batch_size: int,
    layer_keys: list[str],
    layer_dirs: dict[str, str],
    shard_size: int,
    shard_offset: int,
) -> None:
    """Worker process for data-parallel extraction with direct shard writing.

    Pins to a single GPU via ``CUDA_VISIBLE_DEVICES``, extracts activations,
    and writes numpy shards directly to the output directories.  No temp
    files or cross-process data transfer needed.
    """
    import os
    from pathlib import Path

    from tqdm import tqdm

    tp_size = config_kwargs.get("tensor_parallel_size", 1)
    # Assign tp_size consecutive GPUs to this worker
    gpu_ids = [str(rank * tp_size + i) for i in range(tp_size)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    # Disable P2P only for single-GPU workers to avoid host OOM on
    # Blackwell NVLink-C2C.  Multi-GPU (TP>1) workers need P2P for
    # all-reduce/all-gather across GPUs.
    if tp_size == 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    config = VLLMActivationConfig(**config_kwargs)
    extractor = VLLMActivationExtractor(config)

    # Per-layer shard state
    shard_buffers: dict[str, list[np.ndarray]] = {lk: [] for lk in layer_keys}
    shard_counts: dict[str, int] = dict.fromkeys(layer_keys, 0)
    shard_indices: dict[str, int] = dict.fromkeys(layer_keys, shard_offset)
    total_written: dict[str, int] = dict.fromkeys(layer_keys, 0)

    pbar = tqdm(
        total=len(prompts),
        desc=f"[GPU {rank}] Extracting",
        unit="prompt",
        position=rank,
    )

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        acts_list = extractor.extract_batch(chunk, batch_size=len(chunk))

        for act_dict in acts_list:
            for lk in layer_keys:
                vec = act_dict[lk]
                shard_buffers[lk].append(vec.astype(np.float32))
                shard_counts[lk] += 1

                if shard_counts[lk] >= shard_size:
                    _flush_shard(Path(layer_dirs[lk]), shard_indices[lk], shard_buffers[lk])
                    total_written[lk] += shard_counts[lk]
                    shard_indices[lk] += 1
                    shard_buffers[lk] = []
                    shard_counts[lk] = 0

        pbar.update(len(chunk))

    # Flush remaining buffers
    for lk in layer_keys:
        if shard_buffers[lk]:
            _flush_shard(Path(layer_dirs[lk]), shard_indices[lk], shard_buffers[lk])
            total_written[lk] += shard_counts[lk]
            shard_indices[lk] += 1

    pbar.close()
    extractor.cleanup()

    print(
        f"  [GPU {rank}] Wrote {total_written[layer_keys[0]]} vectors "
        f"across {shard_indices[layer_keys[0]] - shard_offset} shard(s)"
    )


def _flush_shard(output_dir, shard_idx: int, buffer: list[np.ndarray]):
    """Stack buffered vectors and save as a numpy shard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_data = np.stack(buffer, axis=0)
    shard_path = output_dir / f"shard_{shard_idx:06d}.npy"
    np.save(shard_path, shard_data)


def run_dp_extraction_to_shards(
    prompts: list[str],
    dp_size: int,
    config_kwargs: dict,
    batch_size: int,
    layer_keys: list[str],
    layer_dirs: dict[str, str],
    shard_size: int,
) -> dict[str, int]:
    """Run data-parallel extraction, writing shards directly to output dirs.

    Each worker writes its own shard files with non-overlapping indices.
    No temp files or merging needed.

    Args:
        prompts: All prompts to extract.
        dp_size: Number of data-parallel workers (one per GPU).
        config_kwargs: Keyword arguments for ``VLLMActivationConfig``.
        batch_size: Batch size per worker.
        layer_keys: Layer keys (e.g. ``["residual_8", "residual_20"]``).
        layer_dirs: Dict mapping layer_key to output directory path (str).
        shard_size: Vectors per shard file.

    Returns:
        Dict mapping layer_key to total number of vectors written.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")

    tp_size = config_kwargs.get("tensor_parallel_size", 1)
    config_kwargs = {**config_kwargs, "tensor_parallel_size": tp_size}

    # Split prompts into exactly dp_size contiguous chunks
    base, remainder = divmod(len(prompts), dp_size)
    chunks: list[list[str]] = []
    start = 0
    for r in range(dp_size):
        size = base + (1 if r < remainder else 0)
        chunks.append(prompts[start : start + size])
        start += size

    # Compute shard index offsets so workers don't collide.
    # Each worker gets enough shard indices to cover its chunk.
    shard_offsets = []
    offset = 0
    for chunk in chunks:
        shard_offsets.append(offset)
        # Max shards this worker could write
        max_shards = (len(chunk) + shard_size - 1) // shard_size + 1
        offset += max_shards

    # Spawn workers
    processes = []
    for rank, (chunk, shard_offset) in enumerate(zip(chunks, shard_offsets, strict=True)):
        p = ctx.Process(
            target=_dp_shard_worker,
            args=(
                rank,
                chunk,
                config_kwargs,
                batch_size,
                layer_keys,
                layer_dirs,
                shard_size,
                shard_offset,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"DP worker (pid={p.pid}) failed with exit code {p.exitcode}")

    # Count total vectors per layer from the shard files
    from pathlib import Path

    totals: dict[str, int] = {}
    for lk in layer_keys:
        total = 0
        for shard_path in sorted(Path(layer_dirs[lk]).glob("shard_*.npy")):
            total += np.load(shard_path, mmap_mode="r").shape[0]
        totals[lk] = total

    return totals
