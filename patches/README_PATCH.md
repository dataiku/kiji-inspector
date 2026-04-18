# Local vLLM patch workflow

## Requirements
Install `vllm` into this project's virtual environment, for example:

```bash
uv pip install vllm --torch-backend=cu128
```

This repository applies local patches to the installed package at:

`.venv/lib/python3.12/site-packages/vllm`

## Patch files

Current patch set for the installed `vllm 0.19rc1` layout:

- `custom-patches/vllm-0.19rc1-gemma4-hidden-states.patch`

Historical reference patches from the older `0.18`-era layout:

- `custom-patches/01_allow_extract_hidden_states.patch`
- `custom-patches/02_support_nemotron_models.patch`
- `custom-patches/03_support_gemma3_models.patch`

The old numbered patches are kept as references for intent. They are not the
preferred apply path for the current environment.

The current `0.19rc1` patch adds:

- `extract_activation_layers` plumbing through `ModelConfig`, `EngineArgs`, and `LLM`
- prompt activation capture and request/output propagation in the V1 engine
- OpenAI-compatible `activations` serialization for chat and completion outputs
- Gemma 4 decoder-layer activation capture aligned with the Hugging Face
  `extract_all_hidden_states_npz.py` reference

`extract_activation_layers` is an exact `list[int]` API. For example,
`[12, 20]` captures only layers `12` and `20`.

Gemma 4 activation extraction now returns the last prompt-token hidden state for
each requested layer. That narrower contract is validated against the saved
Hugging Face `.npz` by slicing the reference activations at the last prompt
token during comparison, and it is intended to work on the fast-prefill path
without forcing eager mode.

## Apply the patch

Run from the repository root:

```bash
./custom-patches/apply-patch.sh
```

`apply-patch.sh` prefers `custom-patches/vllm-0.19*.patch`. If no current-version
patch is present, it falls back to the numbered legacy patches.

The script copies `custom-patches/` into the installed `vllm` tree as
`<site-packages>/vllm/patches/`, then applies the selected patch files with
`patch -p1`.

## Verification

After patching, verify the Gemma 4 activations against the saved Hugging Face
reference:

```bash
uv run python compare_vllm_hidden_states.py
```

The current comparison helper tolerates the missing batch dimension on the
vLLM-side activation tensor by squeezing singleton batch dimensions during
comparison only. The default pass threshold is `0.998` cosine similarity across
all layers.

## Assumptions

- Python is installed in `.venv`.
- The installed package path resolves to `.venv/lib/python3.12/site-packages/vllm`.
- The current patch file targets the installed `vllm 0.19rc1` code layout.
