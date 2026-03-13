"""
Fuzzing evaluation of feature explanations (Step 6).

Tests whether Step 5's feature labels correctly identify WHICH tokens activate
each feature, not just which texts. This catches explanations that are
"right for the wrong reasons".

Based on Eleuther AI's autointerp approach:
https://blog.eleuther.ai/autointerp/

6a: Extract per-token activations from the subject model.
6b: Encode through SAE, build cross-prompt A/B fuzzing examples.
6c: LLM judge evaluates highlights in subprocess.
6d: Compute metrics and save report.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kiji_inspector.utils.stats import binomial_p_value as _binomial_p_value
from kiji_inspector.utils.stats import bootstrap_ci_mean as _bootstrap_ci_mean
from kiji_inspector.utils.stats import clopper_pearson_ci as _clopper_pearson_ci
from kiji_inspector.utils.stats import wilson_score_ci as _wilson_score_ci

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FuzzingExample:
    """A single fuzzing evaluation example."""

    feature_id: int
    text: str  # original user request(s) description
    fuzzed_text: str  # highlighted text A (token_level) or prompt pair text (prompt_level)
    fuzzed_text_b: str  # highlighted text B (token_level) or empty (prompt_level)
    is_correctly_fuzzed: bool  # True means A is the top-activating example
    kind: str  # "token_level" or "prompt_level"


# ---------------------------------------------------------------------------
# 6a: Per-token activation extraction
# ---------------------------------------------------------------------------


def _extraction_subprocess(
    formatted_prompts: list[str],
    subject_model: str,
    layers: list[int],
    batch_size: int,
    backend: str,
    output_dir: str,
) -> None:
    """Child process: load HF model, extract per-token activations, save to disk, exit.

    Running in a subprocess ensures all GPU memory is freed when the process
    exits, leaving the GPUs clean for the judge model in step 5c.

    Saves results incrementally to avoid accumulating all activations in RAM:
    - Token strings: single pickle file (small).
    - Activations: one ``.npz`` file per prompt, written as each batch completes.
    """
    import pickle

    from tqdm import tqdm

    from kiji_inspector.extraction import create_extractor

    out = Path(output_dir)
    acts_dir = out / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)

    layer_keys = [f"residual_{layer}" for layer in layers]

    extractor = create_extractor(
        backend=backend,
        model_name=subject_model,
        layers=layers,
        token_positions="all",
    )
    tokenizer = extractor.tokenizer

    all_token_strings: list[list[str]] = []
    prompt_idx = 0

    pbar = tqdm(total=len(formatted_prompts), desc="[6a] Per-token extraction", unit="prompt")

    for i in range(0, len(formatted_prompts), batch_size):
        batch_formatted = formatted_prompts[i : i + batch_size]

        acts = extractor.extract_batch(batch_formatted, batch_size=len(batch_formatted))

        for j, prompt_text in enumerate(batch_formatted):
            encoding = tokenizer(prompt_text, return_tensors="pt", truncation=True)
            token_ids = encoding.input_ids[0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            first_act = acts[j][layer_keys[0]]
            min_len = min(len(tokens), first_act.shape[0])

            all_token_strings.append(tokens[:min_len])

            # Write activations for this prompt to a single .npz file
            arrays = {lk: acts[j][lk][:min_len].astype(np.float16) for lk in layer_keys}
            np.savez(acts_dir / f"{prompt_idx}.npz", **arrays)
            prompt_idx += 1

        # Free batch activations immediately
        del acts

        pbar.update(len(batch_formatted))

    pbar.close()
    extractor.cleanup()

    # Save token strings (small — list of list of short strings)
    with open(out / "token_strings.pkl", "wb") as f:
        pickle.dump(all_token_strings, f)
    print(f"  [subprocess] Saved {prompt_idx} prompts to {output_dir}")


class DiskBackedActivations:
    """Lazy-loading activation store backed by per-prompt ``.npz`` files on disk.

    Only loads activations from disk when accessed, so all 6 layers' data
    never needs to reside in RAM simultaneously.
    """

    def __init__(self, acts_dir: Path, n_prompts: int):
        self._acts_dir = acts_dir
        self._n = n_prompts

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = np.load(self._acts_dir / f"{idx}.npz")
        return dict(data)

    def cleanup(self) -> None:
        """Remove temp directory."""
        import shutil

        shutil.rmtree(self._acts_dir.parent, ignore_errors=True)


def extract_per_token_activations(
    prompts: list[str],
    formatted_prompts: list[str],
    subject_model: str,
    layers: list[int],
    batch_size: int,
    backend: str = "vllm",
    dp_size: int = 1,
) -> tuple[list[list[str]], DiskBackedActivations]:
    """Extract per-token activations for a list of formatted prompts.

    Runs extraction in a subprocess so that all GPU memory is freed when
    the subprocess exits, leaving GPUs clean for the judge model.

    Returns:
        token_strings: List of per-prompt token string lists.
        activations: A ``DiskBackedActivations`` that lazily loads
            per-prompt activation dicts from disk.
    """
    import pickle
    import tempfile

    # Per-token extraction requires full sequence activations, which
    # the vLLM backend cannot provide (it only returns the decode step).
    # Force HF backend for this step.
    if backend == "vllm":
        print(
            "  Note: switching to HF backend for per-token extraction (vLLM only returns decode-step activations)"
        )
        backend = "hf"

    # Run in a subprocess so GPU memory is fully freed on exit
    tmp_dir = tempfile.mkdtemp(prefix="kiji_extraction_")
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_extraction_subprocess,
        args=(formatted_prompts, subject_model, layers, batch_size, backend, tmp_dir),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Extraction subprocess failed with exit code {p.exitcode}")

    tmp_path = Path(tmp_dir)

    # Load token strings (small)
    with open(tmp_path / "token_strings.pkl", "rb") as f:
        all_token_strings = pickle.load(f)

    # Return lazy-loading activation store instead of loading everything into RAM
    activations = DiskBackedActivations(tmp_path / "acts", len(all_token_strings))

    return all_token_strings, activations


# ---------------------------------------------------------------------------
# 6b: Build fuzzing examples
# ---------------------------------------------------------------------------


def _try_chatml_span(
    token_strings: list[str],
    tokenizer,
) -> tuple[int, int] | None:
    """Try to find user request span using ChatML markers.

    Works with Qwen, Nemotron, Yi, and other ChatML-based models.
    """
    im_start_str = "<|im_start|>"
    im_end_str = "<|im_end|>"

    im_start_positions = [i for i, tok in enumerate(token_strings) if im_start_str in tok]
    im_end_positions = [i for i, tok in enumerate(token_strings) if im_end_str in tok]

    # User turn is the 2nd im_start block (index 1); system is index 0
    if len(im_start_positions) >= 2 and len(im_end_positions) >= 2:
        user_turn_start_marker = im_start_positions[1]
        user_turn_end_marker = im_end_positions[1]

        # Skip past "<|im_start|>user\n" tokens (typically 1-3 tokens)
        content_start = user_turn_start_marker + 1
        while content_start < user_turn_end_marker:
            tok_text = (
                tokenizer.decode(
                    tokenizer.convert_tokens_to_ids([token_strings[content_start]]),
                    skip_special_tokens=False,
                )
                .strip()
                .lower()
            )
            if tok_text in ("user", ""):
                content_start += 1
            else:
                break

        if content_start < user_turn_end_marker:
            return content_start, user_turn_end_marker

    return None


def _try_llama3_span(
    token_strings: list[str],
    tokenizer,
) -> tuple[int, int] | None:
    """Try to find user request span using Llama 3 markers.

    Looks for <|start_header_id|>user<|end_header_id|> boundaries.
    """
    header_start = "<|start_header_id|>"
    eot = "<|eot_id|>"

    header_positions = [i for i, tok in enumerate(token_strings) if header_start in tok]
    eot_positions = [i for i, tok in enumerate(token_strings) if eot in tok]

    # Find the "user" header (typically the 2nd header after "system")
    user_header_idx = None
    for pos in header_positions:
        # Check if next few tokens contain "user"
        for offset in range(1, 4):
            if pos + offset < len(token_strings):
                decoded = (
                    tokenizer.decode(
                        tokenizer.convert_tokens_to_ids([token_strings[pos + offset]]),
                        skip_special_tokens=False,
                    )
                    .strip()
                    .lower()
                )
                if decoded == "user":
                    user_header_idx = pos
                    break
        if user_header_idx is not None:
            break

    if user_header_idx is None:
        return None

    # Find the end_header_id after the user header
    end_header = "<|end_header_id|>"
    content_start = None
    for i in range(user_header_idx, min(user_header_idx + 5, len(token_strings))):
        if end_header in token_strings[i]:
            content_start = i + 1
            break

    if content_start is None:
        return None

    # Skip whitespace tokens after the header
    while content_start < len(token_strings):
        decoded = tokenizer.decode(
            tokenizer.convert_tokens_to_ids([token_strings[content_start]]),
            skip_special_tokens=False,
        ).strip()
        if decoded == "":
            content_start += 1
        else:
            break

    # Find the eot_id that ends the user turn
    content_end = None
    for pos in eot_positions:
        if pos > content_start:
            content_end = pos
            break

    if content_end is not None and content_start < content_end:
        return content_start, content_end

    return None


def _try_text_search_span(
    token_strings: list[str],
    user_request: str,
    tokenizer,
) -> tuple[int, int] | None:
    """Find user request by text matching in the decoded token stream.

    Decodes each token, reconstructs the full text, finds the user_request
    substring, and maps character offsets back to token indices.
    """
    if not user_request:
        return None

    # Decode each token to get character boundaries
    decoded_texts: list[str] = []
    token_char_starts: list[int] = []
    char_pos = 0
    for tok_str in token_strings:
        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids([tok_str]),
            skip_special_tokens=False,
        )
        token_char_starts.append(char_pos)
        decoded_texts.append(text)
        char_pos += len(text)

    full_text = "".join(decoded_texts)
    idx = full_text.find(user_request)
    if idx == -1:
        return None

    end_char = idx + len(user_request)

    # Map character positions to token indices
    start_token = 0
    for i, start in enumerate(token_char_starts):
        if start <= idx:
            start_token = i

    end_token = len(token_strings)
    for i, start in enumerate(token_char_starts):
        if start >= end_char:
            end_token = i
            break

    if start_token < end_token:
        return start_token, end_token

    return None


def _find_user_request_span(
    token_strings: list[str],
    user_request: str,
    tokenizer,
) -> tuple[int, int]:
    """Find the token index range corresponding to the user request.

    Tries multiple strategies to support different chat template formats:
    1. ChatML markers (<|im_start|>/<|im_end|>) -- Qwen, Nemotron, Yi
    2. Llama 3 markers (<|start_header_id|>/<|end_header_id|>)
    3. Text-search fallback -- decode tokens, find user_request substring

    Returns:
        (start_idx, end_idx) -- inclusive start, exclusive end.
        The span covers only the user request content tokens.
    """
    # Strategy 1: ChatML markers
    span = _try_chatml_span(token_strings, tokenizer)
    if span is not None:
        return span

    # Strategy 2: Llama 3 markers
    span = _try_llama3_span(token_strings, tokenizer)
    if span is not None:
        return span

    # Strategy 3: Text search fallback
    span = _try_text_search_span(token_strings, user_request, tokenizer)
    if span is not None:
        return span

    # Fallback: warn and return full prompt
    warnings.warn(
        f"Could not locate user turn in token sequence. "
        f"Falling back to full prompt ({len(token_strings)} tokens).",
        stacklevel=2,
    )
    return 0, len(token_strings)


def _user_request_text_with_highlights(
    token_strings: list[str],
    highlight_indices: set[int],
    span_start: int,
    span_end: int,
    tokenizer,
) -> str:
    """Reconstruct ONLY the user request tokens with <<highlights>>.

    Args:
        token_strings: Full prompt token list.
        highlight_indices: Global token indices to highlight.
        span_start: Start of user request span (inclusive).
        span_end: End of user request span (exclusive).
        tokenizer: For decoding tokens to text.

    Returns:
        User request text with <<highlighted>> tokens.
    """
    parts = []
    for i in range(span_start, span_end):
        tok = token_strings[i]
        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids([tok]),
            skip_special_tokens=False,
        )
        if i in highlight_indices and text.strip():
            stripped = text.strip()
            leading = text[: len(text) - len(text.lstrip())]
            trailing = text[len(text.rstrip()) :]
            parts.append(f"{leading}<<{stripped}>>{trailing}")
        else:
            parts.append(text)
    return "".join(parts)


def _compute_highlighted_user_text(
    user_request: str,
    feat_id: int,
    prompt_to_idx: dict[str, int],
    token_strings_list: list[list[str]],
    token_activations_list: list[dict[str, np.ndarray]],
    layer_key: str,
    sae,
    device: str,
    sae_dtype,
    tokenizer,
    top_k_tokens: int,
) -> str | None:
    """Compute highlighted user-request text for a single prompt/feature.

    Encodes per-token activations through the SAE, finds the user request
    span via chat template markers, and highlights the top-K activated tokens.

    Returns:
        User request text with <<highlighted>> tokens, or None if the
        prompt cannot be processed.
    """
    if user_request not in prompt_to_idx:
        return None

    idx = prompt_to_idx[user_request]
    tok_strs = token_strings_list[idx]
    tok_acts = token_activations_list[idx][layer_key]  # (seq_len, d_model)

    # Encode through SAE
    with torch.no_grad():
        act_tensor = torch.from_numpy(tok_acts).to(device=device, dtype=sae_dtype)
        feat_acts = sae.encode(act_tensor)  # (seq_len, d_sae)
        feature_col = feat_acts[:, feat_id].float().cpu().numpy()  # (seq_len,)

    # Find user request span via ChatML markers
    req_start, req_end = _find_user_request_span(tok_strs, user_request, tokenizer)
    if req_end - req_start < 2:
        return None

    # Get feature activations within user request span
    req_activations = feature_col[req_start:req_end]
    req_token_count = req_end - req_start

    # Adaptive K: don't highlight more than 1/3 of user tokens
    k = max(1, min(top_k_tokens, req_token_count // 3))

    # Top-K tokens by activation within the span
    sorted_indices = np.argsort(req_activations)
    top_k_local = sorted_indices[-k:]
    top_k_global = {req_start + int(i) for i in top_k_local}

    # Reconstruct ONLY the user request text with highlights
    return _user_request_text_with_highlights(tok_strs, top_k_global, req_start, req_end, tokenizer)


def build_fuzzing_examples(
    feature_descriptions: dict[str, dict],
    prompt_to_idx: dict[str, int],
    token_strings_list: list[list[str]],
    token_activations_list: list[dict[str, np.ndarray]],
    layer_key: str,
    sae_checkpoint: str,
    tokenizer,
    top_k_tokens: int = 5,
    max_examples_per_feature: int = 10,
) -> list[FuzzingExample]:
    """Build cross-prompt A/B fuzzing examples for each feature.

    For token-level evaluation, pairs a highlighted top-activating prompt
    with a highlighted bottom-activating prompt. The judge picks which
    highlighted text better matches the feature label.

    For prompt-level evaluation, pairs an unhighlighted top-activating
    prompt with a bottom-activating prompt (same as before).

    Args:
        feature_descriptions: From feature_descriptions.json (keyed by str feature ID).
        prompt_to_idx: Maps user request text -> index into token_strings/activations.
        token_strings_list: Per-prompt list of token strings.
        token_activations_list: Per-prompt dicts mapping layer_key to (seq_len, d_model).
        layer_key: Which layer's activations to use for SAE encoding.
        sae_checkpoint: Path to trained SAE.
        tokenizer: The subject model tokenizer (for text reconstruction).
        top_k_tokens: Number of tokens to highlight.
        max_examples_per_feature: Max fuzzing examples per feature.

    Returns:
        List of FuzzingExample objects.
    """
    from kiji_inspector.core.sae_core import JumpReLUSAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(sae_checkpoint, device=device)
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype

    examples: list[FuzzingExample] = []

    feature_ids = sorted(int(k) for k in feature_descriptions.keys())
    pbar = tqdm(feature_ids, desc="[6b] Building fuzzing examples", unit="feature")

    for feat_id in pbar:
        desc = feature_descriptions[str(feat_id)]
        top_prompts = desc.get("top_examples", [])
        bottom_prompts = desc.get("bottom_examples", [])

        if not top_prompts or not bottom_prompts:
            continue

        # Compute highlighted text for top-activating prompts
        top_highlighted: list[tuple[str, str]] = []
        for user_request in top_prompts:
            result = _compute_highlighted_user_text(
                user_request,
                feat_id,
                prompt_to_idx,
                token_strings_list,
                token_activations_list,
                layer_key,
                sae,
                device,
                sae_dtype,
                tokenizer,
                top_k_tokens,
            )
            if result is not None:
                top_highlighted.append((user_request, result))

        # Compute highlighted text for bottom-activating prompts
        bottom_highlighted: list[tuple[str, str]] = []
        for user_request in bottom_prompts:
            result = _compute_highlighted_user_text(
                user_request,
                feat_id,
                prompt_to_idx,
                token_strings_list,
                token_activations_list,
                layer_key,
                sae,
                device,
                sae_dtype,
                tokenizer,
                top_k_tokens,
            )
            if result is not None:
                bottom_highlighted.append((user_request, result))

        if not top_highlighted or not bottom_highlighted:
            continue

        # Create cross-prompt A/B pairs for token-level evaluation
        n_pairs = min(len(top_highlighted), len(bottom_highlighted), max_examples_per_feature)
        for i in range(n_pairs):
            top_req, top_text = top_highlighted[i % len(top_highlighted)]
            bot_req, bot_text = bottom_highlighted[i % len(bottom_highlighted)]

            # Randomize A/B order to avoid position bias
            if random.random() < 0.5:
                text_a, text_b = top_text, bot_text
                req_a, req_b = top_req, bot_req
                correct_is_a = True
            else:
                text_a, text_b = bot_text, top_text
                req_a, req_b = bot_req, top_req
                correct_is_a = False

            examples.append(
                FuzzingExample(
                    feature_id=feat_id,
                    text=f"A: {req_a}\nB: {req_b}",
                    fuzzed_text=text_a,
                    fuzzed_text_b=text_b,
                    is_correctly_fuzzed=correct_is_a,
                    kind="token_level",
                )
            )

        # Prompt-level fuzzing: pick a top and bottom prompt
        top_prompt = top_prompts[0]
        bottom_prompt = bottom_prompts[0]

        # Randomly assign A/B order to avoid position bias
        if random.random() < 0.5:
            text_a, text_b = top_prompt, bottom_prompt
            correct_answer_is_a = True
        else:
            text_a, text_b = bottom_prompt, top_prompt
            correct_answer_is_a = False

        examples.append(
            FuzzingExample(
                feature_id=feat_id,
                text=f"A: {text_a}\nB: {text_b}",
                fuzzed_text=f"A: {text_a}\nB: {text_b}",
                fuzzed_text_b="",
                is_correctly_fuzzed=correct_answer_is_a,
                kind="prompt_level",
            )
        )

    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"  Built {len(examples)} fuzzing examples "
        f"({sum(1 for e in examples if e.kind == 'token_level')} token-level, "
        f"{sum(1 for e in examples if e.kind == 'prompt_level')} prompt-level)"
    )

    return examples


# ---------------------------------------------------------------------------
# 6c: LLM judge (runs in subprocess for GPU isolation)
# ---------------------------------------------------------------------------

_TOKEN_JUDGE_PROMPT = """You are evaluating which text has better token highlighting for a feature.

Feature label: "{label}"
Feature description: "{description}"

Two user requests are shown below. In each, certain tokens are marked with <<double angle brackets>> to indicate they are claimed to be most relevant to this feature.

Text A (with highlights):
{text_a}

Text B (with highlights):
{text_b}

In which text do the <<highlighted>> tokens better match the feature "{label}"? Consider both:
1. Whether the highlighted tokens are relevant to the described concept
2. Whether the overall text context relates to the feature

Answer with a single letter: A or B."""

_PROMPT_JUDGE_PROMPT = """You are evaluating which text better matches a feature explanation.

Feature label: "{label}"
Feature description: "{description}"

Text A:
{text_a}

Text B:
{text_b}

Which text would more strongly activate a feature described as "{label}"?

Answer with a single letter: A or B."""


def _build_judge_prompts(
    examples: list[FuzzingExample],
    feature_descriptions: dict[str, dict],
) -> list[str]:
    """Build ChatML judge prompts for all fuzzing examples.

    Note: These prompts target the generator/judge model (e.g. Qwen),
    not the subject model. ChatML is used because the generator model
    is assumed to be a Qwen model.
    """
    system = (
        "You evaluate whether token highlights match feature descriptions. "
        "Answer concisely with A or B as instructed."
    )

    prompts = []
    for ex in examples:
        desc = feature_descriptions.get(str(ex.feature_id), {})
        label = desc.get("label", "unknown")
        description = desc.get("description", "")

        if ex.kind == "token_level":
            user_content = _TOKEN_JUDGE_PROMPT.format(
                label=label,
                description=description,
                text_a=ex.fuzzed_text[:1000],
                text_b=ex.fuzzed_text_b[:1000],
            )
        else:
            # prompt_level: text field contains "A: ...\nB: ..."
            parts = ex.text.split("\nB: ", 1)
            text_a = parts[0].removeprefix("A: ") if len(parts) == 2 else ex.text
            text_b = parts[1] if len(parts) == 2 else ""

            user_content = _PROMPT_JUDGE_PROMPT.format(
                label=label,
                description=description,
                text_a=text_a,
                text_b=text_b,
            )

        chatml = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(chatml)

    return prompts


def _run_judge_subprocess(
    chatml_prompts: list[str],
    judging_model: str,
    tp_size: int,
    max_model_len: int,
    output_path: str,
    gpu_ids: str | None = None,
) -> None:
    """Child process: load vLLM, judge all examples, save results, exit."""
    import os

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    from vllm import LLM, SamplingParams

    print(f"  [subprocess] Loading vLLM model: {judging_model}")
    llm = LLM(
        model=judging_model,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        enable_expert_parallel=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=20,
    )

    print(f"  [subprocess] Judging {len(chatml_prompts)} examples...")
    outputs = llm.generate(chatml_prompts, sampling_params)

    judgments: list[str] = []
    for output in outputs:
        raw = output.outputs[0].text.strip().upper()
        judgments.append(raw)

    with open(output_path, "w") as f:
        json.dump(judgments, f)

    print(f"  [subprocess] Saved {len(judgments)} judgments to {output_path}")


def evaluate_fuzzing(
    examples: list[FuzzingExample],
    feature_descriptions: dict[str, dict],
    judging_model: str,
    tp_size: int,
    max_model_len: int,
    output_dir: str | Path,
    dp_size: int = 1,
) -> list[dict]:
    """Run LLM judge on fuzzing examples in a subprocess.

    When ``dp_size > 1``, spawns N model copies on N GPUs (each with
    ``tp_size=1``) to judge examples in parallel.

    Returns:
        List of dicts, one per example, with judgment results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chatml_prompts = _build_judge_prompts(examples, feature_descriptions)

    ctx = mp.get_context("spawn")

    if dp_size > 1:
        # Data-parallel: split prompts across N GPUs
        chunk_size = (len(chatml_prompts) + dp_size - 1) // dp_size
        prompt_chunks = [
            chatml_prompts[i : i + chunk_size] for i in range(0, len(chatml_prompts), chunk_size)
        ]
        output_paths = [
            str(output_dir / f"_judge_temp_rank{r}.json") for r in range(len(prompt_chunks))
        ]

        processes = []
        for rank, (chunk, out_path) in enumerate(zip(prompt_chunks, output_paths, strict=True)):
            p = ctx.Process(
                target=_run_judge_subprocess,
                args=(chunk, judging_model, 1, max_model_len, out_path, str(rank)),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Judge subprocess (pid={p.pid}) failed with exit code {p.exitcode}"
                )

        # Merge results from all ranks in order
        raw_judgments: list[str] = []
        for out_path in output_paths:
            with open(out_path) as f:
                raw_judgments.extend(json.load(f))
            Path(out_path).unlink(missing_ok=True)
    else:
        judgments_path = str(output_dir / "_judge_temp.json")
        p = ctx.Process(
            target=_run_judge_subprocess,
            args=(chatml_prompts, judging_model, tp_size, max_model_len, judgments_path),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"Judge subprocess failed with exit code {p.exitcode}")

        with open(judgments_path) as f:
            raw_judgments = json.load(f)

        Path(judgments_path).unlink(missing_ok=True)

    # Parse judgments -- both token-level and prompt-level use A/B format
    results = []
    for ex, raw in zip(examples, raw_judgments, strict=True):
        picked_a = bool(re.search(r"\bA\b", raw))
        predicted_correct = picked_a == ex.is_correctly_fuzzed

        results.append(
            {
                "feature_id": ex.feature_id,
                "kind": ex.kind,
                "text": ex.text[:200],
                "fuzzed_text": ex.fuzzed_text[:500],
                "fuzzed_text_b": ex.fuzzed_text_b[:500],
                "is_correctly_fuzzed": ex.is_correctly_fuzzed,
                "predicted_correct": predicted_correct,
                "raw_judgment": raw,
                "is_prediction_correct": predicted_correct,
            }
        )

    return results


# ---------------------------------------------------------------------------
# 6d: Compute metrics and save report
# ---------------------------------------------------------------------------


def compute_fuzzing_metrics(
    results: list[dict],
    feature_descriptions: dict[str, dict],
) -> tuple[dict[str, dict], dict]:
    """Compute per-feature and aggregate fuzzing metrics.

    Both token-level and prompt-level use A/B comparison format,
    so the metric is simple accuracy (did the judge pick the correct text).

    Args:
        results: List of judgment dicts from evaluate_fuzzing().
        feature_descriptions: For confidence tier grouping.

    Returns:
        (per_feature_results, aggregate_summary)
    """
    from collections import defaultdict

    # Group results by feature
    by_feature: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_feature[r["feature_id"]].append(r)

    per_feature: dict[str, dict] = {}

    for feat_id, feat_results in by_feature.items():
        # Split by kind
        token_results = [r for r in feat_results if r["kind"] == "token_level"]
        prompt_results = [r for r in feat_results if r["kind"] == "prompt_level"]

        # Token-level accuracy
        if token_results:
            tok_correct = sum(1 for r in token_results if r["is_prediction_correct"])
            tok_total = len(token_results)
            token_accuracy = tok_correct / tok_total
            tok_ci = _clopper_pearson_ci(tok_correct, tok_total)
            tok_p = _binomial_p_value(tok_correct, tok_total)
        else:
            token_accuracy = 0
            tok_correct = tok_total = 0
            tok_ci = (0.0, 0.0)
            tok_p = 1.0

        # Prompt-level accuracy
        if prompt_results:
            prm_correct = sum(1 for r in prompt_results if r["is_prediction_correct"])
            prm_total = len(prompt_results)
            prompt_accuracy = prm_correct / prm_total
            prm_ci = _clopper_pearson_ci(prm_correct, prm_total)
            prm_p = _binomial_p_value(prm_correct, prm_total)
        else:
            prompt_accuracy = 0
            prm_correct = prm_total = 0
            prm_ci = (0.0, 0.0)
            prm_p = 1.0

        # Combined score
        if token_results and prompt_results:
            combined = 0.7 * token_accuracy + 0.3 * prompt_accuracy
        elif token_results:
            combined = token_accuracy
        elif prompt_results:
            combined = prompt_accuracy
        else:
            combined = 0

        desc = feature_descriptions.get(str(feat_id), {})

        per_feature[str(feat_id)] = {
            "label": desc.get("label", "unknown"),
            "confidence": desc.get("confidence", "low"),
            "token_level": {
                "accuracy": round(token_accuracy, 4),
                "num_correct": tok_correct,
                "num_examples": tok_total,
                "ci_95": [round(tok_ci[0], 4), round(tok_ci[1], 4)],
                "p_value": round(tok_p, 6),
            },
            "prompt_level": {
                "accuracy": round(prompt_accuracy, 4),
                "num_correct": prm_correct,
                "num_examples": prm_total,
                "ci_95": [round(prm_ci[0], 4), round(prm_ci[1], 4)],
                "p_value": round(prm_p, 6),
            },
            "combined_score": round(combined, 4),
        }

    # Aggregate summary
    from scipy.stats import kruskal, ttest_1samp
    from scipy.stats import sem as _sem

    all_combined = [v["combined_score"] for v in per_feature.values()]
    all_token_acc = [
        v["token_level"]["accuracy"]
        for v in per_feature.values()
        if v["token_level"]["num_examples"] > 0
    ]

    # Group by confidence tier
    by_confidence: dict[str, list[float]] = defaultdict(list)
    for v in per_feature.values():
        by_confidence[v["confidence"]].append(v["combined_score"])

    # Quality tiers
    n_total_features = len(all_combined)
    excellent = sum(1 for s in all_combined if s > 0.8)
    good = sum(1 for s in all_combined if 0.6 <= s <= 0.8)
    poor = sum(1 for s in all_combined if s < 0.6)

    # Bootstrap CIs for aggregate means
    combined_ci = _bootstrap_ci_mean(all_combined)
    token_ci = _bootstrap_ci_mean(all_token_acc)

    # One-sample t-test vs 0.5 baseline
    if len(all_combined) >= 2:
        _, combined_p = ttest_1samp(all_combined, 0.5, alternative="greater")
    else:
        combined_p = 1.0
    if len(all_token_acc) >= 2:
        _, token_p = ttest_1samp(all_token_acc, 0.5, alternative="greater")
    else:
        token_p = 1.0

    # Kruskal-Wallis test across confidence tiers
    tier_groups = [scores for scores in by_confidence.values() if len(scores) >= 2]
    if len(tier_groups) >= 2:
        kw_stat, kw_p = kruskal(*tier_groups)
    else:
        kw_stat, kw_p = 0.0, 1.0

    # Wilson CIs for quality tier proportions
    excellent_ci = _wilson_score_ci(excellent, n_total_features) if n_total_features else (0, 0)
    good_ci = _wilson_score_ci(good, n_total_features) if n_total_features else (0, 0)
    poor_ci = _wilson_score_ci(poor, n_total_features) if n_total_features else (0, 0)

    summary = {
        "num_features_evaluated": n_total_features,
        "num_examples_total": len(results),
        "combined_score": {
            "mean": round(np.mean(all_combined), 4) if all_combined else 0,
            "std": round(float(np.std(all_combined, ddof=1)), 4) if len(all_combined) > 1 else 0,
            "sem": round(float(_sem(all_combined)), 4) if len(all_combined) > 1 else 0,
            "ci_95": [round(combined_ci[0], 4), round(combined_ci[1], 4)],
            "median": round(float(np.median(all_combined)), 4) if all_combined else 0,
            "p_value_vs_baseline": round(float(combined_p), 6),
        },
        "token_level_accuracy": {
            "mean": round(np.mean(all_token_acc), 4) if all_token_acc else 0,
            "std": round(float(np.std(all_token_acc, ddof=1)), 4) if len(all_token_acc) > 1 else 0,
            "sem": round(float(_sem(all_token_acc)), 4) if len(all_token_acc) > 1 else 0,
            "ci_95": [round(token_ci[0], 4), round(token_ci[1], 4)],
            "p_value_vs_baseline": round(float(token_p), 6),
        },
        "by_confidence": {
            tier: {
                "count": len(scores),
                "mean_score": round(np.mean(scores), 4) if scores else 0,
                "std": round(float(np.std(scores, ddof=1)), 4) if len(scores) > 1 else 0,
                "sem": round(float(_sem(scores)), 4) if len(scores) > 1 else 0,
            }
            for tier, scores in by_confidence.items()
        },
        "by_confidence_comparison": {
            "test": "Kruskal-Wallis",
            "statistic": round(float(kw_stat), 4),
            "p_value": round(float(kw_p), 6),
        },
        "quality_tiers": {
            "excellent_above_0.8": {
                "count": excellent,
                "proportion": round(excellent / n_total_features, 4) if n_total_features else 0,
                "ci_95": [round(excellent_ci[0], 4), round(excellent_ci[1], 4)],
            },
            "good_0.6_to_0.8": {
                "count": good,
                "proportion": round(good / n_total_features, 4) if n_total_features else 0,
                "ci_95": [round(good_ci[0], 4), round(good_ci[1], 4)],
            },
            "poor_below_0.6": {
                "count": poor,
                "proportion": round(poor / n_total_features, 4) if n_total_features else 0,
                "ci_95": [round(poor_ci[0], 4), round(poor_ci[1], 4)],
            },
        },
    }

    return per_feature, summary


def save_fuzzing_report(
    per_feature: dict[str, dict],
    summary: dict,
    results: list[dict],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save fuzzing results and summary to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "fuzzing_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"per_feature": per_feature, "details": results},
            f,
            indent=2,
        )

    summary_path = output_dir / "fuzzing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    cs = summary["combined_score"]
    ta = summary["token_level_accuracy"]
    qt = summary["quality_tiers"]

    print("\n  Fuzzing evaluation complete:")
    print(f"    Features evaluated: {summary['num_features_evaluated']}")
    print(f"    Total examples: {summary['num_examples_total']}")
    print(
        f"    Mean combined score: {cs['mean']:.3f} +/- {cs['sem']:.3f} (SEM), "
        f"95% CI: [{cs['ci_95'][0]:.3f}, {cs['ci_95'][1]:.3f}], "
        f"p={cs['p_value_vs_baseline']:.4f} vs 0.5 baseline"
    )
    print(
        f"    Token-level accuracy: {ta['mean']:.3f} +/- {ta['sem']:.3f} (SEM), "
        f"p={ta['p_value_vs_baseline']:.4f} vs 0.5 baseline"
    )
    print(
        f"    Quality tiers: "
        f"{qt['excellent_above_0.8']['count']} excellent "
        f"({qt['excellent_above_0.8']['proportion']:.1%}), "
        f"{qt['good_0.6_to_0.8']['count']} good "
        f"({qt['good_0.6_to_0.8']['proportion']:.1%}), "
        f"{qt['poor_below_0.6']['count']} poor "
        f"({qt['poor_below_0.6']['proportion']:.1%})"
    )
    kw = summary["by_confidence_comparison"]
    print(f"    Confidence tiers differ: {kw['test']} p={kw['p_value']:.4f}")
    print(f"    Results: {results_path}")
    print(f"    Summary: {summary_path}")

    return results_path, summary_path
