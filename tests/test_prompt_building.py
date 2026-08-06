from kiji_inspector.extraction.extractor import (
    _DEFAULT_ASSISTANT_PREFILL,
    build_agent_prompt,
    build_agent_prompt_from_tokenizer,
)

TOOLS = [
    {"name": "internal_search", "description": "Search internal docs"},
    {"name": "web_search", "description": "Search the web"},
]


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer with a ChatML-style template."""

    chat_template = "chatml"

    def __init__(self, thinking_generation_prompt: bool = False):
        self.thinking_generation_prompt = thinking_generation_prompt
        self.last_template_kwargs: dict = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.last_template_kwargs = kwargs
        rendered = ""
        for m in messages:
            rendered += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            # Reasoning models (e.g. Qwen3.6) open a think block unless
            # enable_thinking=False is honored by the template.
            if self.thinking_generation_prompt and kwargs.get("enable_thinking", True):
                rendered += "<think>\n"
        return rendered


def test_prompt_ends_with_prefill_by_default():
    tok = FakeTokenizer()
    prompt = build_agent_prompt_from_tokenizer(tok, "You are an agent.", TOOLS, "Find API limits")
    assert prompt.endswith("<|im_start|>assistant\nI'll use the")
    assert tok.last_template_kwargs == {}


def test_default_prefill_has_no_trailing_whitespace():
    """The decision token must be the tool name, not whitespace.

    Both tokenizer families fold a leading space into the following token
    ("▁ticket" / "Ġticket"), so a prompt ending in a bare space is
    off-distribution: gemma-4 then predicts a newline and Nemotron starts a
    numbered list, and the extracted activation is not a decision point.
    """
    assert _DEFAULT_ASSISTANT_PREFILL == _DEFAULT_ASSISTANT_PREFILL.rstrip()

    for tok in (FakeTokenizer(), FakeTokenizer(thinking_generation_prompt=True)):
        prompt = build_agent_prompt_from_tokenizer(tok, "You are an agent.", TOOLS, "Find limits")
        assert prompt == prompt.rstrip(), "tokenizer-template prompt ends with whitespace"

    # Same invariant through the public entrypoint and the manual fallbacks.
    for model_type in ("auto", "nemotron", "llama", "mistral", "generic"):
        prompt = build_agent_prompt(
            "You are an agent.", TOOLS, "Find API limits", model_type=model_type
        )
        assert prompt == prompt.rstrip(), f"model_type={model_type} ends with whitespace"


def test_chat_template_kwargs_are_forwarded():
    tok = FakeTokenizer(thinking_generation_prompt=True)
    prompt = build_agent_prompt_from_tokenizer(
        tok,
        "You are an agent.",
        TOOLS,
        "Find API limits",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert tok.last_template_kwargs == {"enable_thinking": False}
    assert "<think>" not in prompt
    assert prompt.endswith("I'll use the")


def test_close_think_block_closes_open_think_tag():
    # Template that ignores enable_thinking and always opens a think block.
    tok = FakeTokenizer(thinking_generation_prompt=True)
    prompt = build_agent_prompt_from_tokenizer(
        tok,
        "You are an agent.",
        TOOLS,
        "Find API limits",
        close_think_block=True,
    )
    assert prompt.endswith("<think>\n\n</think>\n\nI'll use the")


def test_close_think_block_noop_without_open_think_tag():
    tok = FakeTokenizer(thinking_generation_prompt=False)
    prompt = build_agent_prompt_from_tokenizer(
        tok,
        "You are an agent.",
        TOOLS,
        "Find API limits",
        close_think_block=True,
    )
    assert "</think>" not in prompt
    assert prompt.endswith("<|im_start|>assistant\nI'll use the")


def test_build_agent_prompt_passes_thinking_options_through():
    tok = FakeTokenizer(thinking_generation_prompt=True)
    prompt = build_agent_prompt(
        system_prompt="You are an agent.",
        tools=TOOLS,
        user_request="Find API limits",
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
        close_think_block=True,
    )
    assert tok.last_template_kwargs == {"enable_thinking": False}
    assert "<think>" not in prompt
    assert prompt.endswith("I'll use the")


def test_reasoning_suppression_moves_prefill_out_of_think_block():
    """Auto-suppression must put the decision token at the answer position.

    Nemotron-3-Nano's template opens a <think> block in its generation prompt,
    so without enable_thinking=False the prefill — and therefore the extracted
    activation — lands inside the reasoning channel rather than where the tool
    name is emitted. gemma-4's template defaults the flag to false already, so
    the same handling is a no-op for it.
    """
    reasoning = FakeTokenizer(thinking_generation_prompt=True)
    plain = FakeTokenizer(thinking_generation_prompt=False)

    # Reasoning model, suppression off: prefill sits inside the think block.
    inside = build_agent_prompt_from_tokenizer(reasoning, "S", TOOLS, "U")
    assert "<think>" in inside
    assert inside.index("<think>") < inside.index(_DEFAULT_ASSISTANT_PREFILL)
    assert "</think>" not in inside

    # Reasoning model, suppression on: block is closed before the prefill.
    outside = build_agent_prompt_from_tokenizer(
        reasoning,
        "S",
        TOOLS,
        "U",
        chat_template_kwargs={"enable_thinking": False},
        close_think_block=True,
    )
    assert "<think>" not in outside
    assert outside.endswith(_DEFAULT_ASSISTANT_PREFILL)

    # Non-reasoning model: identical with and without suppression.
    assert build_agent_prompt_from_tokenizer(
        plain, "S", TOOLS, "U", chat_template_kwargs={"enable_thinking": False}
    ) == build_agent_prompt_from_tokenizer(plain, "S", TOOLS, "U")
