#!/usr/bin/env python3
"""
Compare decision-token activations: HuggingFace path vs. patched vLLM path.

The HF extraction path is the proven reference (see
``utils/nemotron/verify_decision_token.py``): it captures the hidden state at the last
token of the prompt (the "I'll use the " decision token). The patched vLLM
build (``patches/0.19``) captures prompt activations for ALL prompt tokens at
the requested layer via the Eagle3 ``aux_hidden_state_layers`` hook
(``hidden_states + residual``), and ``VLLMActivationExtractor`` takes ``[-1]``.

This harness checks, on a real model, whether the vLLM ``[-1]`` vector actually
equals the HF decision-token vector -- catching any layer-index, residual
definition (pre/post norm), or off-by-one mismatch.

Design: HF (device_map="auto") and patched vLLM (~90% VRAM) cannot share a CUDA
context, so each extraction runs as its own fresh ``sys.executable`` subprocess
(``--mode hf`` / ``--mode vllm``) and writes activations to a work dir; the
parent (``--mode all``) then compares on CPU. Fresh subprocesses also avoid the
cuDNN-in-spawned-child init failures seen with multiprocessing. Comparison is
aligned from the END of each sequence, so a leading-BOS tokenization difference
between the two tokenizer calls does not affect the decision-token result.

Requires the vLLM patches to be applied first:
    bash patches/0.19/apply-patch.sh

Usage:
    uv run --extra nemotron python utils/nemotron/compare_hf_vllm_activations.py
    uv run --extra nemotron python utils/nemotron/compare_hf_vllm_activations.py \
        --subject-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --layer 28
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Real artifacts from the dataset (output/pairs), so the comparison feeds the
# model exactly what the pipeline does.  Scenario "tool_selection" system prompt
# and 8 tools are copied verbatim from output/pairs/scenarios_meta.json; the two
# requests are the anchor + contrast of pair_id
# "tool_selection_shallow_vs_deep_0" (output/pairs/shard_00007.parquet) -- a
# causally significant contrast (internal_search vs delegate_agent).
SYSTEM_PROMPT = "You are a helpful assistant. Choose the best tool for each request."
TOOLS = [
    {"name": "internal_search", "description": "Search internal company documentation"},
    {"name": "web_search", "description": "Search the public web"},
    {"name": "file_read", "description": "Read a local file"},
    {"name": "file_write", "description": "Write or update a local file"},
    {"name": "database_query", "description": "Query a SQL database"},
    {"name": "api_call", "description": "Call an external REST API"},
    {"name": "code_execute", "description": "Execute code in a sandbox"},
    {"name": "delegate_agent", "description": "Delegate to a sub-agent for complex tasks"},
]
DEFAULT_REQUESTS = [
    # anchor -> internal_search
    "What's the latest API version according to internal docs?",
    # contrast -> delegate_agent
    "Can you recursively trace the API version history and dependencies across all internal documentation?",
]
PREFILL = "I'll use the "


# ---------------------------------------------------------------------------
# Extraction phases (each runs in its own fresh subprocess)
# ---------------------------------------------------------------------------


def run_hf(subject_model: str, layer: int, work_dir: Path) -> None:
    """Extract full per-token activations via the HF path; save to work_dir."""
    import torch

    from kiji_inspector.extraction.activation_extractor import (
        ActivationConfig,
        ActivationExtractor,
    )
    from kiji_inspector.extraction.extractor import build_agent_prompt

    # NemotronH's attention uses torch SDPA; the cuDNN SDPA backend can fail
    # intermittently with CUDNN_STATUS_NOT_INITIALIZED. Force the non-cuDNN
    # kernels (flash / mem-efficient / math) -- numerically equivalent here.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    extractor = ActivationExtractor(
        ActivationConfig(model_name=subject_model, layers=[layer], token_positions="all")
    )
    tok = extractor.tokenizer
    layer_key = f"residual_{layer}"

    formatted = [
        build_agent_prompt(
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
            user_request=req,
            tokenizer=tok,
            assistant_prefill=PREFILL,
        )
        for req in DEFAULT_REQUESTS
    ]
    (work_dir / "formatted_prompts.json").write_text(json.dumps(formatted))
    for i, prompt in enumerate(formatted):
        acts = extractor.extract(prompt)[layer_key]  # [seq, hidden], float32
        np.save(work_dir / f"hf_{i}.npy", np.asarray(acts, dtype=np.float32))
    extractor.cleanup()
    print(f"  HF: wrote {len(formatted)} activation files to {work_dir}")


def run_vllm(subject_model: str, layer: int, work_dir: Path, max_model_len: int) -> None:
    """Extract full per-token activations via the patched vLLM path."""
    from kiji_inspector.extraction.vllm_activation_extractor import (
        VLLMActivationConfig,
        VLLMActivationExtractor,
    )

    formatted = json.loads((work_dir / "formatted_prompts.json").read_text())
    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=subject_model,
            layers=[layer],
            token_positions="all",
            max_model_len=max_model_len,
        )
    )
    layer_key = f"residual_{layer}"
    for i, prompt in enumerate(formatted):
        acts = extractor.extract(prompt)[layer_key]  # [num_prompt_tokens, hidden]
        np.save(work_dir / f"vllm_{i}.npy", np.asarray(acts, dtype=np.float32))
    extractor.cleanup()
    print(f"  vLLM: wrote {len(formatted)} activation files to {work_dir}")


# ---------------------------------------------------------------------------
# Comparison (parent process, CPU only)
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _per_token_mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine over the overlapping tail (aligned from the end)."""
    k = min(a.shape[0], b.shape[0])
    a_tail, b_tail = a[-k:], b[-k:]
    an = a_tail / (np.linalg.norm(a_tail, axis=1, keepdims=True) + 1e-12)
    bn = b_tail / (np.linalg.norm(b_tail, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(an * bn, axis=1)))


def _subprocess(mode: str, args: argparse.Namespace, layer: int) -> None:
    """Re-invoke this script in a fresh interpreter for one extraction phase."""
    cmd = [
        sys.executable, __file__,
        "--mode", mode,
        "--subject-model", args.subject_model,
        "--layer", str(layer),
        "--work-dir", str(args.work_dir),
        "--max-model-len", str(args.max_model_len),
    ]
    subprocess.run(cmd, check=True)


def compare(args: argparse.Namespace, layer: int, num_layers: int) -> int:
    work = Path(args.work_dir)
    print("\n" + "=" * 70)
    print("  Per-prompt comparison (aligned from the decision token)")
    print("=" * 70)

    decision_cosines: list[float] = []
    for i, req in enumerate(DEFAULT_REQUESTS):
        hf = np.load(work / f"hf_{i}.npy")
        vllm = np.load(work / f"vllm_{i}.npy")

        d_cos = _cosine(hf[-1], vllm[-1])
        d_l2 = float(np.linalg.norm(hf[-1] - vllm[-1]))
        d_rel = d_l2 / (float(np.linalg.norm(hf[-1])) + 1e-12)
        tok_cos = _per_token_mean_cosine(hf, vllm)
        decision_cosines.append(d_cos)

        mismatch = "   <-- LENGTH MISMATCH" if hf.shape[0] != vllm.shape[0] else ""
        print(f"\n  Prompt {i}: {req!r}")
        print(f"    seq len   HF={hf.shape[0]:<4} vLLM={vllm.shape[0]:<4} "
              f"hidden HF={hf.shape[1]} vLLM={vllm.shape[1]}{mismatch}")
        print(f"    decision token : cosine={d_cos:.6f}  L2={d_l2:.4f}  rel_L2={d_rel:.4%}")
        print(f"    per-token tail : mean cosine={tok_cos:.6f}")

    mean_cos = float(np.mean(decision_cosines))
    min_cos = float(np.min(decision_cosines))
    passed = min_cos >= args.decision_cosine_threshold

    print("\n" + "=" * 70)
    print(f"  Decision-token cosine: mean={mean_cos:.6f}  min={min_cos:.6f}  "
          f"(threshold {args.decision_cosine_threshold})")
    if passed:
        print("  [PASS] vLLM reproduces the HF decision-token activation.")
        print("         The vLLM path captures the prompt's last token, same as HF.")
    else:
        print("  [FAIL] vLLM decision-token activation diverges from HF.")
        print(f"         Investigate layer indexing / residual definition at layer {layer},")
        print("         or whether vLLM is returning a different token position.")
    print("=" * 70)
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["all", "hf", "vllm"], default="all",
                   help="all (default): orchestrate both phases + compare. "
                   "hf/vllm: run a single extraction phase (used internally).")
    p.add_argument("--subject-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    p.add_argument("--layer", type=int, default=None, help="Default: ~2/3 depth (pipeline default).")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--work-dir", default="/tmp/hf_vllm_cmp")
    # 0.99, not 0.999: HF (eager) and vLLM (compiled) BF16 kernels accumulate
    # numerical error with depth, so the single decision-token position lands at
    # ~0.99-0.995 cosine on real prompts even when every layer is faithfully
    # aligned (verified by compare_hf_vllm_layer_alignment.py: 52/52 layers >=0.99
    # on the 30B). A structural mismatch (e.g. the 4B) instead collapses to ~0.4.
    p.add_argument("--decision-cosine-threshold", type=float, default=0.99)
    args = p.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Single-phase modes (run inside the fresh subprocesses): the orchestrator
    # always passes an explicit --layer, so no model-config load is needed here.
    if args.mode == "hf":
        run_hf(args.subject_model, args.layer, work)
        return
    if args.mode == "vllm":
        run_vllm(args.subject_model, args.layer, work, args.max_model_len)
        return

    # Orchestrator: resolve the layer (and true layer count for display).
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.subject_model, trust_remote_code=True)
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", cfg), "num_hidden_layers", 30
    )
    layer = args.layer if args.layer is not None else min(int(num_layers * 2 / 3), num_layers - 1)

    # Orchestrator: run each extraction in its own clean interpreter.
    print("=" * 70)
    print("  HF vs vLLM decision-token activation comparison")
    print(f"  Model    : {args.subject_model}")
    print(f"  Layer    : {layer} (of {num_layers})")
    print(f"  Requests : {len(DEFAULT_REQUESTS)}")
    print(f"  Work dir : {work}")
    print("=" * 70)

    print("\n[1/2] HF extraction (fresh subprocess)...")
    _subprocess("hf", args, layer)

    print("\n[2/2] vLLM extraction (fresh subprocess)...")
    _subprocess("vllm", args, layer)

    raise SystemExit(compare(args, layer, num_layers))


if __name__ == "__main__":
    main()
