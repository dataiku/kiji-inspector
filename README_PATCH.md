# Local vLLM patch workflow

## Requirements
Install vllm as:
```
uv pip install vllm --torch-backend=cu128
```

This repository includes scripts to apply and revert a local patch on the installed `vllm` package in:

`.venv/lib/python3.12/site-packages/vllm`

The patch file used by both scripts is:

`patches/0001-patch-venv.mbox.patch`

The patch subject is:

`Expose activations in OpenAI serve responses`

At a high level, this patch adds support for configuring `extract_activation_layers`, capturing prompt activations for selected layers, and returning serialized activations in OpenAI-compatible chat and completion responses.

## Apply the patch

Run from the repository root:

`./apply-patch.sh`

## Revert the patch

Run from the repository root:

`./revert-patch.sh`

## Assumptions

- You run the scripts from the repository root.
- Python is installed in `.venv`.
- The installed package path is `.venv/lib/python3.12/site-packages/vllm`.
- The local `patches/` directory exists.
- The installed `vllm` version matches the patch context.

Both scripts copy `patches/` into the installed `vllm` directory before running `git apply`. They also skip cleanly when the requested state is already satisfied.
