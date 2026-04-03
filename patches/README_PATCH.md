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

They are applied in lexical order by `patches/apply-patch.sh`.

The patch set adds:

`Support extracting hidden states for Gemma3 and Nemotron models`

At a high level, this patch adds support for configuring `extract_activation_layers`, capturing prompt activations for selected layers, and returning serialized activations in OpenAI-compatible chat and completion responses.

## Apply the patch

Run from the repository root:

`./patches/apply-patch.sh`

## Assumptions

- Python is installed in `.venv`.
- The installed package path is `.venv/lib/python3.12/site-packages/vllm`.
- The installed `vllm` version matches the patch context.

The apply script copies the local `patches/` directory into the installed `vllm` directory, then applies every `*.patch` file in lexical order with `patch -p1`. It skips patches that are already present and can complete partially applied states as long as the final installed files match the requested patch content.
