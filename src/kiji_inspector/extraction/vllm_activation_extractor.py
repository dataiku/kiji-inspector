"""
Activation extractor using vLLM's built-in activation capture.

Drop-in replacement for ActivationExtractor that uses vLLM's
``extract_activation_layers`` API instead of HuggingFace Transformers
with manual forward hooks.  Significantly faster for bulk extraction
thanks to vLLM's continuous batching and optimized kernels.

Requires a patched vLLM build that supports ``extract_activation_layers``
on ``ModelConfig`` (engine-level) and ``extract_activations`` on
``SamplingParams`` (request-level).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class VLLMActivationConfig:
    """Configuration for vLLM-based activation extraction."""

    model_name: str = ""
    layers: list[int] = field(default_factory=lambda: [20])
    token_positions: str = "decision"  # "all", "last", "decision"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    trust_remote_code: bool = True


class VLLMActivationExtractor:
    """Extract activations using vLLM's activation extraction API.

    Presents the same interface as ``ActivationExtractor`` (``extract()``,
    ``extract_batch()``, ``cleanup()``, ``tokenizer``, ``hidden_size``)
    so it can be used as a drop-in replacement wherever the HuggingFace
    Transformers-based extractor is used.
    """

    def __init__(self, config: VLLMActivationConfig):
        if not config.model_name:
            raise ValueError("VLLMActivationConfig.model_name is required.")
        self.config = config

        from transformers import AutoConfig
        from vllm import LLM

        # Validate requested layers against actual model depth
        hf_config = AutoConfig.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is not None:
            invalid = [idx for idx in config.layers if idx >= num_layers]
            if invalid:
                raise ValueError(
                    f"Requested layers {invalid} but {config.model_name} "
                    f"only has {num_layers} layers (0-{num_layers - 1}). "
                    f"Use --layers with values < {num_layers}."
                )

        print(f"Loading model via vLLM: {config.model_name}")
        print(f"  layers: {config.layers}")
        print(f"  tensor_parallel_size: {config.tensor_parallel_size}")

        # FlashInfer has a bug with block_size=16 + head_size=256 (e.g. Gemma 3).
        # Use block_size=32 when head_size is 256 to avoid the assertion.
        head_size = getattr(hf_config, "head_dim", None) or getattr(
            hf_config, "hidden_size", 0
        ) // getattr(hf_config, "num_attention_heads", 1)
        block_size = 32 if head_size == 256 else None

        llm_kwargs = {
            "model": config.model_name,
            "extract_activation_layers": config.layers,
            "enforce_eager": True,
            "trust_remote_code": config.trust_remote_code,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tensor_parallel_size,
            "max_model_len": config.max_model_len,
            "disable_log_stats": True,
        }
        if block_size is not None:
            llm_kwargs["block_size"] = block_size

        self.llm = LLM(**llm_kwargs)

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.llm.model_config.hf_text_config.hidden_size
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  token_positions: {config.token_positions}")

        if config.token_positions == "all":
            import warnings

            warnings.warn(
                "vLLM backend with token_positions='all' only returns activations "
                "from the last forward step (decode), not the full prompt sequence. "
                "Use --backend hf for full per-token activations.",
                stacklevel=2,
            )

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
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extract_activations=True,
        )
        outputs = self.llm.generate([prompt], sp, use_tqdm=False)
        activations = outputs[0].outputs[0].activations

        if not activations:
            raise RuntimeError("vLLM did not return activations. Check extract_activation_layers.")

        result = {}
        for layer_idx, tensor in activations.items():
            # vLLM overwrites activations per step (prefill then decode).
            # With max_tokens=1 the returned tensor is from the decode step:
            # shape (1, hidden_size) — the hidden state at the decision point.
            key = f"residual_{layer_idx}"

            if self.config.token_positions in ("last", "decision"):
                result[key] = tensor[-1].float().numpy()
            elif self.config.token_positions == "all":
                result[key] = tensor.float().numpy()
            else:
                raise ValueError(f"Unknown token_positions: {self.config.token_positions}")

        return result

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
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extract_activations=True,
        )
        results: list[dict[str, np.ndarray]] = []

        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            outputs = self.llm.generate(chunk, sp, use_tqdm=False)

            for output in outputs:
                activations = output.outputs[0].activations
                if not activations:
                    raise RuntimeError("vLLM did not return activations.")

                item = {}
                for layer_idx, tensor in activations.items():
                    key = f"residual_{layer_idx}"

                    if self.config.token_positions in ("last", "decision"):
                        item[key] = tensor[-1].float().numpy()
                    elif self.config.token_positions == "all":
                        item[key] = tensor.float().numpy()
                    else:
                        raise ValueError(f"Unknown token_positions: {self.config.token_positions}")

                results.append(item)

        return results

    def cleanup(self):
        """Free vLLM resources and GPU memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

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
