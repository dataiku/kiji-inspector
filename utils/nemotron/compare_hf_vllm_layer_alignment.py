#!/usr/bin/env python3
"""
HF vs vLLM layer-alignment matrix for decision-token activations.

Companion to ``utils/nemotron/compare_hf_vllm_activations.py``. That harness found the
30B/layer-20 vLLM activation matches HF (cosine ~0.9998) but the 4B/layer-28
does NOT (cosine ~0.42). This diagnostic localizes the cause by capturing the
decision-token activation at *every* layer in a single model load (per backend)
and, for each HF layer L, comparing it against vLLM layers in a window
[L-band, L+band].

Reading the output:
  * diag cosine (L vs L) high everywhere      -> aligned; divergence is numeric.
  * best offset consistently +k / -k          -> systematic layer-index shift.
  * best cosine low even at the best offset    -> genuine divergence at those
                                                  layers (e.g. layer-type the
                                                  patch captures differently).

Each backend runs as its own fresh ``sys.executable`` subprocess (HF and
patched vLLM cannot share a CUDA context). Decision-token vectors for all
layers are saved as ``[num_prompts, num_layers, hidden]`` arrays; the parent
compares on CPU.

Requires the vLLM patches:  bash patches/0.19/apply-patch.sh

Usage:
    uv run --extra nemotron python utils/nemotron/compare_hf_vllm_layer_alignment.py \
        --subject-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
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
# Extraction phases (each in its own fresh subprocess)
# ---------------------------------------------------------------------------


def run_hf(subject_model: str, num_layers: int, work_dir: Path) -> None:
    import torch

    from kiji_inspector.extraction.activation_extractor import (
        ActivationConfig,
        ActivationExtractor,
    )
    from kiji_inspector.extraction.extractor import build_agent_prompt

    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)  # avoid flaky cuDNN SDPA init

    layers = list(range(num_layers))
    extractor = ActivationExtractor(
        ActivationConfig(model_name=subject_model, layers=layers, token_positions="decision")
    )
    tok = extractor.tokenizer
    formatted = [
        build_agent_prompt(SYSTEM_PROMPT, TOOLS, req, tokenizer=tok, assistant_prefill=PREFILL)
        for req in DEFAULT_REQUESTS
    ]
    (work_dir / "formatted_prompts.json").write_text(json.dumps(formatted))

    # Record the exact token ids HF feeds the model (decisive for the
    # "tokenization bug vs patch bug" question).
    token_ids = [tok(p).input_ids for p in formatted]
    (work_dir / "hf_token_ids.json").write_text(json.dumps(token_ids))

    hidden = extractor.hidden_size
    arr = np.full((len(formatted), num_layers, hidden), np.nan, dtype=np.float32)
    for i, prompt in enumerate(formatted):
        out = extractor.extract(prompt)  # {residual_L: [hidden]} for all layers
        for layer in layers:
            vec = out.get(f"residual_{layer}")
            if vec is not None:
                arr[i, layer] = vec
    np.save(work_dir / "hf_decision.npy", arr)
    extractor.cleanup()
    print(f"  HF: saved decision activations for {num_layers} layers, shape {arr.shape}")


def run_vllm(subject_model: str, num_layers: int, work_dir: Path, max_model_len: int) -> None:
    from kiji_inspector.extraction.vllm_activation_extractor import (
        VLLMActivationConfig,
        VLLMActivationExtractor,
    )

    formatted = json.loads((work_dir / "formatted_prompts.json").read_text())
    layers = list(range(num_layers))
    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=subject_model,
            layers=layers,
            token_positions="decision",
            max_model_len=max_model_len,
        )
    )
    # Record the exact token ids vLLM feeds the model, read from the
    # RequestOutput's prompt_token_ids (faithful to vLLM's own tokenization).
    from vllm import SamplingParams

    outs = extractor.llm.generate(
        formatted, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False
    )
    token_ids = [list(o.prompt_token_ids) for o in outs]
    (work_dir / "vllm_token_ids.json").write_text(json.dumps(token_ids))

    hidden = extractor.hidden_size
    arr = np.full((len(formatted), num_layers, hidden), np.nan, dtype=np.float32)
    for i, prompt in enumerate(formatted):
        out = extractor.extract(prompt)  # {residual_L: [hidden]}
        for layer in layers:
            vec = out.get(f"residual_{layer}")
            if vec is not None:
                arr[i, layer] = vec
    np.save(work_dir / "vllm_decision.npy", arr)
    extractor.cleanup()
    print(f"  vLLM: saved decision activations for {num_layers} layers, shape {arr.shape}")


def _subprocess(mode: str, args: argparse.Namespace, num_layers: int) -> None:
    cmd = [
        sys.executable, __file__,
        "--mode", mode,
        "--subject-model", args.subject_model,
        "--num-layers", str(num_layers),
        "--work-dir", str(args.work_dir),
        "--max-model-len", str(args.max_model_len),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Comparison (parent, CPU)
# ---------------------------------------------------------------------------


def _mean_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Mean cosine over prompts. x, y are [num_prompts, hidden]."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(xn * yn, axis=1)))


def _token_id_check(work: Path) -> None:
    """Compare the exact token ids each backend fed the model."""
    hf_ids = json.loads((work / "hf_token_ids.json").read_text())
    vllm_ids = json.loads((work / "vllm_token_ids.json").read_text())

    print("\n" + "=" * 70)
    print("  Token-id equality check (HF input_ids vs vLLM prompt_token_ids)")
    print("=" * 70)
    all_identical = True
    for i, (a, b) in enumerate(zip(hf_ids, vllm_ids)):
        if a == b:
            print(f"  Prompt {i}: IDENTICAL ({len(a)} tokens)")
            continue
        all_identical = False
        first = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
        print(f"  Prompt {i}: DIFFER  HF len={len(a)} vLLM len={len(b)}")
        print(f"    HF   [:6]={a[:6]}  [-3:]={a[-3:]}")
        print(f"    vLLM [:6]={b[:6]}  [-3:]={b[-3:]}")
        if first is not None:
            print(f"    first divergence at index {first}: HF={a[first]} vLLM={b[first]}")
    if all_identical:
        print("\n  => Inputs are identical. Divergence is NOT tokenization; it is in")
        print("     the model forward / patch residual capture.")
    else:
        print("\n  => Inputs DIFFER. The divergence is (at least partly) a tokenization")
        print("     / special-token mismatch between the HF and vLLM paths.")


def compare(work: Path, band: int, threshold: float) -> int:
    _token_id_check(work)

    hf = np.load(work / "hf_decision.npy")      # [P, L, H]
    vllm = np.load(work / "vllm_decision.npy")  # [P, L, H]
    _, num_layers, _ = hf.shape

    diag, best_off, best_cos = [], [], []
    for layer in range(num_layers):
        diag.append(_mean_cosine(hf[:, layer], vllm[:, layer]))
        cands = [
            (d, _mean_cosine(hf[:, layer], vllm[:, layer + d]))
            for d in range(-band, band + 1)
            if 0 <= layer + d < num_layers
        ]
        d, c = max(cands, key=lambda t: t[1])
        best_off.append(d)
        best_cos.append(c)

    print("\n" + "=" * 70)
    print(f"  Layer-alignment matrix  (decision token, ±{band} window)")
    print("=" * 70)
    print(f"  {'layer':>5}  {'diag cos (L vs L)':>18}  {'best vLLM layer':>15}  {'best cos':>9}")
    for layer in range(num_layers):
        d = best_off[layer]
        flag = ""
        if diag[layer] >= threshold:
            flag = "  aligned"
        elif best_cos[layer] >= threshold and d != 0:
            flag = f"  <-- shifted by {d:+d}"
        elif best_cos[layer] < 0.9:
            flag = "  <-- diverged"
        print(f"  {layer:>5}  {diag[layer]:>18.4f}  {layer + d:>15}  {best_cos[layer]:>9.4f}{flag}")

    aligned = sum(1 for c in diag if c >= threshold)
    shifted_offs = [best_off[layer] for layer in range(num_layers)
                    if diag[layer] < threshold and best_cos[layer] >= threshold]
    diverged = [layer for layer in range(num_layers)
                if best_cos[layer] < 0.9]

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Layers aligned at same index (diag cos >= {threshold}): {aligned}/{num_layers}")
    if shifted_offs:
        mode_off, mode_n = Counter(shifted_offs).most_common(1)[0]
        print(f"  Layers explained by an index shift: {len(shifted_offs)} "
              f"(dominant offset {mode_off:+d}, {mode_n} layers)")
    if diverged:
        print(f"  Layers that diverge even at best offset (cos < 0.9): "
              f"{len(diverged)} -> {diverged}")
    else:
        print("  No layers diverge once the best offset is applied.")

    # Verdict
    if aligned == num_layers:
        print("\n  VERDICT: HF and vLLM are layer-aligned everywhere (numeric match).")
    elif not diverged and shifted_offs:
        mode_off = Counter(shifted_offs).most_common(1)[0][0]
        print(f"\n  VERDICT: systematic layer-index shift of {mode_off:+d} "
              "between HF and vLLM. Likely an indexing convention mismatch in the patch.")
    else:
        print("\n  VERDICT: genuine per-layer divergence (not a simple shift) -- "
              "inspect layer types (Mamba/attention/MoE) at the diverged indices.")
    return 0 if aligned == num_layers else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["all", "hf", "vllm"], default="all")
    p.add_argument("--subject-model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    p.add_argument("--num-layers", type=int, default=None, help="Internal: passed to phases.")
    p.add_argument("--band", type=int, default=3, help="Offset window for alignment search.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--work-dir", default="/tmp/hf_vllm_align")
    p.add_argument("--threshold", type=float, default=0.99)
    args = p.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    if args.mode == "hf":
        run_hf(args.subject_model, args.num_layers, work)
        return
    if args.mode == "vllm":
        run_vllm(args.subject_model, args.num_layers, work, args.max_model_len)
        return

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.subject_model, trust_remote_code=True)
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", cfg), "num_hidden_layers", 30
    )

    print("=" * 70)
    print("  HF vs vLLM layer-alignment diagnostic")
    print(f"  Model      : {args.subject_model}")
    print(f"  Num layers : {num_layers}")
    print(f"  Requests   : {len(DEFAULT_REQUESTS)}")
    print("=" * 70)

    print("\n[1/2] HF extraction (all layers, fresh subprocess)...")
    _subprocess("hf", args, num_layers)
    print("\n[2/2] vLLM extraction (all layers, fresh subprocess)...")
    _subprocess("vllm", args, num_layers)

    raise SystemExit(compare(work, args.band, args.threshold))


if __name__ == "__main__":
    main()
