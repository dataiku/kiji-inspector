from kiji_inspector.extraction.extractor import (
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

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
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
    assert prompt.endswith("<|im_start|>assistant\nI'll use the ")
    assert tok.last_template_kwargs == {}


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
    assert prompt.endswith("I'll use the ")


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
    assert prompt.endswith("<think>\n\n</think>\n\nI'll use the ")


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
    assert prompt.endswith("<|im_start|>assistant\nI'll use the ")


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
    assert prompt.endswith("I'll use the ")
