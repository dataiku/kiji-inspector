"""
Generate synthetic contrastive pairs using a local LLM via vLLM.

Each pair shares the same user intent but requires a different tool,
isolating the decision signal for SAE training.

The model (default: Qwen/Qwen3-VL-235B-A22B-Instruct-FP8) runs locally
on multi-GPU via vLLM with tensor + expert parallelism.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

from kiji_inspector.data.contrastive_dataset import ContrastiveDataset, ContrastivePair

CONTRASTIVE_PAIR_PROMPT = """
Generate {n_pairs} pairs of user requests where:
- Both requests have the SAME underlying intent/goal
- But they should use DIFFERENT tools due to a specific distinguishing factor

The distinguishing factor for this batch: {contrast_type}

Available tools:
{tool_list}

Contrast type explanation:
{contrast_explanation}

CRITICAL: The requests must be semantically VERY similar. Only subtle differences
should determine the tool choice. This is essential for training.

Output as a JSON array using EXACTLY these field names:
[
  {{
    "shared_intent": "what both requests try to accomplish",
    "anchor_request": "first user request text",
    "anchor_tool": "tool_name_for_anchor",
    "contrast_request": "second user request text",
    "contrast_tool": "tool_name_for_contrast",
    "distinguishing_signal": "what makes the difference"
  }}
]

No markdown fences, just raw JSON.
"""


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

_CHATML_SYSTEM = (
    "You are a dataset generator for machine learning research. "
    "Output only valid JSON arrays, no markdown fences or commentary."
)


def _format_chatml(user_content: str, system_content: str = _CHATML_SYSTEM) -> str:
    """Construct a ChatML prompt string for Qwen3-VL.

    ChatML is the stable chat template for all Qwen models.  Building it
    manually avoids loading the heavy AutoProcessor just for text-only
    generation -- vLLM handles tokenisation internally.
    """
    return (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------


def _parse_json_array(raw: str) -> list[dict]:
    """Parse a JSON array from potentially noisy LLM output.

    Recovery layers:
    1. Markdown fence stripping
    2. Outermost bracket extraction
    3. Trailing-comma removal
    4. Truncation recovery (find last complete ``}``)
    """
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    # Fast path
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract outermost brackets
    start = raw.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model output")

    end = raw.rfind("]")
    if end == -1 or end <= start:
        # Truncated output -- salvage up to last complete object
        last_brace = raw.rfind("}")
        if last_brace > start:
            raw = raw[start : last_brace + 1] + "]"
        else:
            raise ValueError("Cannot recover truncated JSON")
    else:
        raw = raw[start : end + 1]

    # Fix trailing commas (common LLM error)
    raw = re.sub(r",\s*]", "]", raw)
    raw = re.sub(r",\s*}", "}", raw)

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Flexible key lookup for LLM JSON output
# ---------------------------------------------------------------------------


def _fuzzy_get(d: dict, primary_key: str, fallbacks: list[str]) -> str:
    """Get a value from a dict, trying the primary key then fallbacks.

    LLMs sometimes rename JSON keys (e.g. "contrast_tool" becomes
    "best_tool_for_contrast").  This tries the canonical name first,
    then common variants, raising KeyError only if none match.
    """
    if primary_key in d:
        return d[primary_key]
    for key in fallbacks:
        if key in d:
            return d[key]
    raise KeyError(primary_key)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ContrastivePairGenerator:
    """Generates contrastive pairs using a local vLLM model."""

    def __init__(
        self,
        llm: LLM,
        tools: list[dict],
        contrast_types: dict[str, str],
        scenario_name: str = "",
        sampling_params: SamplingParams | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.contrast_types = contrast_types
        self.scenario_name = scenario_name
        self._malformed_count = 0
        self.tool_list = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
        if sampling_params is None:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                max_tokens=8000,
            )
        self.sampling_params = sampling_params

    def _build_prompt(
        self,
        contrast_type: str,
        n_pairs: int,
    ) -> str:
        """Build a ChatML prompt for a single contrast type."""
        explanation = self.contrast_types.get(contrast_type, f"Contrast based on: {contrast_type}")
        user_content = CONTRASTIVE_PAIR_PROMPT.format(
            n_pairs=n_pairs,
            contrast_type=contrast_type,
            tool_list=self.tool_list,
            contrast_explanation=explanation,
        )
        return _format_chatml(user_content)

    def generate_for_contrast_type(
        self,
        contrast_type: str,
        n_pairs: int = 50,
    ) -> list[ContrastivePair]:
        """Generate pairs for a specific contrast type (single prompt)."""
        prompt = self._build_prompt(contrast_type, n_pairs)
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        raw = outputs[0].outputs[0].text.strip()
        return self._parse_pairs(raw, contrast_type)

    def generate_batched(
        self,
        requests: list[tuple[str, int]],
    ) -> list[list[ContrastivePair]]:
        """Generate pairs for multiple (contrast_type, n_pairs) in one vLLM call.

        Returns a list of pair-lists, one per request, in the same order.
        vLLM schedules all prompts with continuous batching for maximum
        GPU utilization.
        """
        prompts = [self._build_prompt(ct, n) for ct, n in requests]

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        results: list[list[ContrastivePair]] = []
        for (ct, _n), output in zip(requests, outputs, strict=True):
            raw = output.outputs[0].text.strip()
            try:
                results.append(self._parse_pairs(raw, ct))
            except Exception:
                results.append([])
        return results

    def generate_full_dataset(
        self,
        pairs_per_contrast_type: int = 100,
    ) -> ContrastiveDataset:
        """Generate a complete contrastive dataset covering all contrast types."""
        requests = [(ct, pairs_per_contrast_type) for ct in self.contrast_types]
        results = self.generate_batched(requests)

        all_pairs: list[ContrastivePair] = []
        for pairs in results:
            all_pairs.extend(pairs)

        type_dist: dict[str, int] = {}
        tool_pair_dist: dict[tuple[str, str], int] = {}
        for pair in all_pairs:
            ct = pair.contrast_type
            type_dist[ct] = type_dist.get(ct, 0) + 1
            tp = (pair.anchor_tool, pair.contrast_tool)
            tool_pair_dist[tp] = tool_pair_dist.get(tp, 0) + 1

        return ContrastiveDataset(
            pairs=all_pairs,
            contrast_type_distribution=type_dist,
            tool_pair_distribution=tool_pair_dist,
        )

    def _parse_pairs(
        self,
        raw: str,
        contrast_type: str,
    ) -> list[ContrastivePair]:
        """Parse JSON output into ContrastivePair objects with error isolation."""
        pairs_data = _parse_json_array(raw)

        pairs: list[ContrastivePair] = []
        for i, p in enumerate(pairs_data):
            try:
                pairs.append(
                    ContrastivePair(
                        pair_id=f"{contrast_type}_{i}",
                        anchor_prompt=_fuzzy_get(
                            p, "anchor_request", ["anchor", "request_1", "first_request"]
                        ),
                        anchor_tool=_fuzzy_get(
                            p,
                            "anchor_tool",
                            ["anchor_best_tool", "best_tool_for_anchor", "tool_1", "first_tool"],
                        ),
                        contrast_prompt=_fuzzy_get(
                            p, "contrast_request", ["contrast", "request_2", "second_request"]
                        ),
                        contrast_tool=_fuzzy_get(
                            p,
                            "contrast_tool",
                            [
                                "contrast_best_tool",
                                "best_tool_for_contrast",
                                "tool_2",
                                "second_tool",
                            ],
                        ),
                        shared_intent=_fuzzy_get(
                            p, "shared_intent", ["intent", "common_intent", "goal"]
                        ),
                        semantic_similarity=0.9,
                        contrast_type=contrast_type,
                        distinguishing_signal=_fuzzy_get(
                            p,
                            "distinguishing_signal",
                            ["signal", "distinction", "difference", "key_difference"],
                        ),
                        scenario_name=self.scenario_name,
                    )
                )
            except (KeyError, TypeError):
                self._malformed_count += 1

        return pairs


def generate_minimal_pair_variants(
    llm: LLM,
    base_request: str,
    base_tool: str,
    target_tool: str,
    n_variants: int = 5,
    sampling_params: SamplingParams | None = None,
) -> list[dict]:
    """
    Generate minimal pair variants: same request with tiny modifications
    that flip the tool choice.

    Returns a list of dicts with keys:
        modified_request, words_changed, why_different_tool
    """
    if sampling_params is None:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=4000,
        )

    user_content = f"""
Take this request and create {n_variants} minimal modifications that would
change the best tool from {base_tool} to {target_tool}.

Original request: "{base_request}"
Original best tool: {base_tool}
Target tool: {target_tool}

Rules:
- Change as FEW words as possible
- The core intent should remain the same
- Only the subtle context should change

For each variant, provide:
- modified_request: The minimally modified request
- words_changed: Which words were changed
- why_different_tool: Why this now needs {target_tool}

Output as a JSON array. No markdown fences.
"""

    prompt = _format_chatml(user_content)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    raw = outputs[0].outputs[0].text.strip()

    return _parse_json_array(raw)
