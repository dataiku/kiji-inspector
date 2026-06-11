#!/usr/bin/env python3
"""
Verify WHERE the SAE training activations are captured (HuggingFace path).

Question this answers empirically, on a real model:
    Is the hidden state extracted at the last token of the *prompt*
    (the "decision token", i.e. the trailing space of "I'll use the ")
    or at the last token of the *generation* (the emitted tool name)?

The position-selection logic is already unit-tested in
``tests/test_activation_extractors.py`` (offset -1 picks the last row).
This harness goes further: it loads a real Nemotron model and proves the
end-to-end behaviour through the *production* extraction path
(``ActivationExtractor`` + its forward hooks), with four checks:

  A. The formatted prompt ends with the prefill "I'll use the ".
  B. ``extractor.extract(prompt)`` returns exactly the hooked sequence's
     last row -> the pipeline captures token position N-1.
  C. Decision-token representation is generation-independent: re-hooking the
     full prompt+generation sequence yields the *same* vector at position N-1
     (cosine ~ 1.0). A causal model cannot let later tokens change it, which
     is precisely why this is a prompt/prefill activation.
  D. The decision token is NOT the last generated token: their hidden states
     differ (cosine < 1) and they decode to different tokens (a space vs. a
     tool-name token).

Note: Nemotron's KV cache is currently broken, so generation here runs with
``use_cache=False`` (a single prefill forward pass is unaffected).

Usage:
    uv run --extra nemotron python utils/nemotron/verify_decision_token.py
    uv run --extra nemotron python utils/nemotron/verify_decision_token.py \
        --subject-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --layer 20
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from kiji_inspector.extraction.activation_extractor import (
    ActivationConfig,
    ActivationExtractor,
)
from kiji_inspector.extraction.extractor import build_agent_prompt

# Real artifacts from the dataset (output/pairs), so the harness verifies on
# exactly what the pipeline feeds the model.  Scenario "tool_selection" system
# prompt and 8 tools are copied verbatim from output/pairs/scenarios_meta.json;
# the request is the anchor of pair_id "tool_selection_shallow_vs_deep_0"
# (output/pairs/shard_00007.parquet) -> internal_search.
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
DEFAULT_USER_REQUEST = "What's the latest API version according to internal docs?"
PREFILL = "I'll use the "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subject-model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        help="HF model ID to verify (default: the 4B Nemotron for a fast check; "
        "behaviour is identical for the 30B pipeline default).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Transformer layer to hook (default: ~2/3 depth, matching the pipeline).",
    )
    p.add_argument(
        "--user-request",
        default=DEFAULT_USER_REQUEST,
        help="User request used to build the agent prompt.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Tokens to greedily generate for the decision-vs-generation comparison.",
    )
    return p.parse_args()


def _hooked_sequence(extractor: ActivationExtractor, input_ids: torch.Tensor, layer_key: str) -> torch.Tensor:
    """Run the inner model on input_ids through the PRODUCTION hook path.

    Returns the full per-token activation sequence [seq_len, hidden] that the
    forward hook stored for ``layer_key`` (CPU, float32). This is the exact
    tensor ``ActivationExtractor.extract`` selects a single position from.
    """
    extractor._activations = {}
    inner_model = extractor._get_inner_model()
    with torch.no_grad():
        # use_cache=False: a single prefill pass, and Nemotron's KV cache is broken.
        inner_model(input_ids=input_ids.to(extractor._input_device), use_cache=False)
    return extractor._activations[layer_key].squeeze(0)


def _fmt_tok(tokenizer, token_id: int) -> str:
    """Human-readable rendering of a single token (piece + decoded text)."""
    piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    text = tokenizer.decode([int(token_id)])
    return f"id={int(token_id):<6} piece={piece!r:<14} decoded={text!r}"


def main() -> None:
    args = parse_args()

    # Resolve a sensible default layer (~2/3 depth) like the pipeline does.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.subject_model, trust_remote_code=True)
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", cfg), "num_hidden_layers", 30
    )
    layer = args.layer if args.layer is not None else int(num_layers * 2 / 3)
    layer = min(layer, num_layers - 1)
    layer_key = f"residual_{layer}"

    print("=" * 70)
    print("  Decision-token verification (HuggingFace path)")
    print(f"  Model       : {args.subject_model}")
    print(f"  Layers      : {num_layers} (hooking layer {layer})")
    print(f"  User request: {args.user_request!r}")
    print("=" * 70)

    extractor = ActivationExtractor(
        ActivationConfig(
            model_name=args.subject_model,
            layers=[layer],
            token_positions="decision",
        )
    )
    tok = extractor.tokenizer
    model = extractor.model
    dev = extractor._input_device

    prompt = build_agent_prompt(
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        user_request=args.user_request,
        tokenizer=tok,
        assistant_prefill=PREFILL,
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids
    n = input_ids.shape[1]

    checks: list[tuple[str, bool]] = []

    # ---- A. Prompt ends at the decision point ------------------------------
    print("\n[A] Prompt construction")
    print(f"    last 30 chars: {prompt[-30:]!r}")
    ends_with_prefill = prompt.endswith(PREFILL)
    print(f"    ends with {PREFILL!r}: {ends_with_prefill}")
    print(f"    prompt token count N = {n}")
    print("    last 5 tokens of the prompt:")
    for idx in range(n - 5, n):
        marker = "  <-- position N-1 (decision token)" if idx == n - 1 else ""
        print(f"      [{idx}] {_fmt_tok(tok, input_ids[0, idx])}{marker}")
    checks.append(("A. prompt ends with the prefill 'I'll use the '", ends_with_prefill))

    # ---- B. The pipeline captures position N-1 -----------------------------
    print("\n[B] What ActivationExtractor.extract() captures")
    extracted = extractor.extract(prompt)[layer_key]
    a_decision = torch.from_numpy(extracted).float()
    hs_prompt = _hooked_sequence(extractor, input_ids, layer_key)  # [N, hidden]
    captures_last_row = torch.allclose(a_decision, hs_prompt[-1], atol=1e-4)
    print(f"    extract() vector shape : {tuple(a_decision.shape)}")
    print(f"    hooked sequence shape  : {tuple(hs_prompt.shape)}")
    print(f"    extract() == hooked[N-1]: {captures_last_row}")
    print(f"    captured token (N-1)   : {_fmt_tok(tok, input_ids[0, n - 1])}")
    checks.append(("B. extract() returns the activation at prompt position N-1", captures_last_row))

    # ---- Generate the tool name the model would actually emit --------------
    print("\n[--] Greedy continuation (what the model emits AFTER the prompt)")
    with torch.no_grad():
        generated = model.generate(
            input_ids.to(dev),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,  # Nemotron's KV cache is broken
            pad_token_id=tok.pad_token_id,
        )
    full_ids = generated[0].cpu().unsqueeze(0)
    new_ids = full_ids[0, n:]
    print(f"    generated continuation : {tok.decode(new_ids)!r}")
    print(f"    first generated token  : {_fmt_tok(tok, new_ids[0])}")

    # Re-hook the FULL prompt+generation sequence in one forward pass.
    hs_full = _hooked_sequence(extractor, full_ids, layer_key)  # [N+G, hidden]
    a_decision_in_context = hs_full[n - 1]
    a_last_generated = hs_full[-1]

    # ---- C. Decision token is generation-independent -----------------------
    print("\n[C] Is the decision-token representation generation-independent?")
    cos_invariant = F.cosine_similarity(a_decision, a_decision_in_context, dim=0).item()
    print(f"    cosine(decision_token, same_token_within_full_sequence) = {cos_invariant:.6f}")
    print("    (~1.0 => later/generated tokens cannot change it => it is a prompt activation)")
    invariant = cos_invariant > 0.999
    checks.append(("C. decision-token representation is generation-independent", invariant))

    # ---- D. Decision token != last generated token -------------------------
    print("\n[D] Decision token vs. last generated token")
    cos_distinct = F.cosine_similarity(a_decision, a_last_generated, dim=0).item()
    print("    position  token                                          cosine-to-decision")
    print(f"    N-1 ({n - 1:>4}) {_fmt_tok(tok, full_ids[0, n - 1]):<48} 1.000000 (reference)")
    print(f"    last({n + len(new_ids) - 1:>4}) {_fmt_tok(tok, full_ids[0, -1]):<48} {cos_distinct:.6f}")
    different_token = int(full_ids[0, n - 1]) != int(full_ids[0, -1])
    distinct = cos_distinct < 0.999 and different_token
    print(f"    different token & representation: {distinct}")
    checks.append(("D. decision token differs from the last generated token", distinct))

    # ---- Verdict -----------------------------------------------------------
    print("\n" + "=" * 70)
    all_pass = all(ok for _, ok in checks)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 70)
    print(
        "\nConclusion: activations are captured at the LAST TOKEN OF THE PROMPT\n"
        "(the decision token, trailing space of \"I'll use the \"), NOT at the\n"
        "last token of the generation."
        if all_pass
        else "\nConclusion: one or more checks FAILED — inspect the output above."
    )

    extractor.cleanup()
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
