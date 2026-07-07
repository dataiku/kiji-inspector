# Local vLLM patch workflow

## Requirements
Install vllm as:
```
uv pip install vllm --torch-backend=cu128
```

This repository includes scripts to apply and revert a local patch set on the installed `vllm` package in:

`.venv/lib/python3.12/site-packages/vllm`

The patch files are:

- `patches/01_allow_extract_hidden_states.patch`
- `patches/02_support_nemotron_models.patch`
- `patches/03_support_gemma3_models.patch`
- `patches/04_support_qwen3_5_models.patch`

They are applied in lexical order by `patches/apply-patch.sh`.

The patch set adds:

`Support extracting hidden states for Gemma3, Nemotron, and Qwen3.5/3.6 models`

At a high level, this patch adds support for configuring `extract_activation_layers`, capturing prompt activations for selected layers, and returning serialized activations in OpenAI-compatible chat and completion responses.

The patch set targets **vLLM v0.19.0** (all four patches apply cleanly against it).

### Patch 04: Qwen3.5 / Qwen3.6 (e.g. `Qwen/Qwen3.6-35B-A3B`)

Qwen3.5/3.6 checkpoints declare the `Qwen3_5MoeForConditionalGeneration`
(or `Qwen3_5ForConditionalGeneration`) architecture, whose inner causal LM
already ships EAGLE3 auxiliary-hidden-state support upstream. Patch 04 makes
two small changes:

1. `model_executor/models/qwen3_5.py`: overrides
   `set_aux_hidden_state_layers` on the `ForConditionalGeneration` wrapper to
   delegate to the wrapped language model (the default `SupportsEagle3`
   implementation asserts the inner model inherits `EagleModelMixin`, which
   `Qwen3NextModel` does not).
2. `model_executor/models/qwen3_next.py`: moves the auxiliary hidden-state
   capture from *before* each decoder layer to *after* it, so that a layer
   index passed to `extract_activation_layers` yields the same residual-stream
   vector as a HuggingFace forward hook on `model.layers[i]` (post-layer
   output), consistent with the Nemotron and Gemma3 patches.

Note: because of (2), do not use this patched vLLM build for EAGLE3/MTP
speculative decoding of Qwen3-Next-family models — the aux-hidden-state
semantics intentionally differ from upstream. The patched build is intended
for activation extraction.

## Apply the patch

Run from the repository root:

`./patches/apply-patch.sh`

## Assumptions

- Python is installed in `.venv`.
- The installed package path is `.venv/lib/python3.12/site-packages/vllm`.
- The installed `vllm` version matches the patch context.

The apply script copies the local `patches/` directory into the installed `vllm` directory, then applies every `*.patch` file in lexical order with `patch -p1`. It skips patches that are already present and can complete partially applied states as long as the final installed files match the requested patch content.
