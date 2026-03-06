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

        from vllm import LLM

        print(f"Loading model via vLLM: {config.model_name}")
        print(f"  layers: {config.layers}")
        print(f"  tensor_parallel_size: {config.tensor_parallel_size}")

        self.llm = LLM(
            model=config.model_name,
            extract_activation_layers=config.layers,
            enforce_eager=True,
            trust_remote_code=config.trust_remote_code,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            disable_log_stats=True,
        )

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


def _dp_worker(
    rank: int,
    prompts: list[str],
    config_kwargs: dict,
    batch_size: int,
    output_path: str,
) -> None:
    """Worker process for data-parallel extraction.

    Pins to a single GPU via ``CUDA_VISIBLE_DEVICES``, creates a
    ``VLLMActivationExtractor``, processes its prompt chunk, and
    saves results to a temporary numpy file.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    config = VLLMActivationConfig(**config_kwargs)
    extractor = VLLMActivationExtractor(config)

    results = extractor.extract_batch(prompts, batch_size=batch_size)
    extractor.cleanup()

    # Serialise results: list of dicts with string keys → numpy arrays.
    # Save as a single dict: {"{idx}:{layer_key}": array, ...}
    flat: dict[str, np.ndarray] = {}
    for i, item in enumerate(results):
        for key, arr in item.items():
            flat[f"{i}:{key}"] = arr
    flat["__count__"] = np.array([len(results)])

    np.savez(output_path, **flat)


def extract_batch_data_parallel(
    prompts: list[str],
    dp_size: int,
    config_kwargs: dict,
    batch_size: int = 512,
) -> list[dict[str, np.ndarray]]:
    """Run extraction across multiple GPUs using data parallelism.

    Splits prompts evenly across ``dp_size`` worker processes, each
    pinned to a separate GPU.  Results are merged in original order.

    Args:
        prompts: All prompts to extract.
        dp_size: Number of data-parallel workers (one per GPU).
        config_kwargs: Keyword arguments for ``VLLMActivationConfig``.
            ``tensor_parallel_size`` is forced to 1.
        batch_size: Batch size per worker.

    Returns:
        List of activation dicts in the same order as ``prompts``.
    """
    import tempfile
    from multiprocessing import Process

    config_kwargs = {**config_kwargs, "tensor_parallel_size": 1}

    # Split prompts across workers
    chunk_size = (len(prompts) + dp_size - 1) // dp_size
    chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    # Create temp files for output
    tmp_dir = tempfile.mkdtemp(prefix="kiji_dp_")
    output_paths = [f"{tmp_dir}/rank_{r}.npz" for r in range(len(chunks))]

    # Spawn workers
    processes = []
    for rank, (chunk, out_path) in enumerate(zip(chunks, output_paths)):
        p = Process(
            target=_dp_worker,
            args=(rank, chunk, config_kwargs, batch_size, out_path),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"DP worker (pid={p.pid}) failed with exit code {p.exitcode}")

    # Merge results in order
    all_results: list[dict[str, np.ndarray]] = []
    for out_path in output_paths:
        data = np.load(out_path, allow_pickle=False)
        count = int(data["__count__"][0])
        layer_keys = {k.split(":", 1)[1] for k in data.files if ":" in k}
        for i in range(count):
            item = {lk: data[f"{i}:{lk}"] for lk in layer_keys}
            all_results.append(item)

    # Cleanup temp files
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return all_results
