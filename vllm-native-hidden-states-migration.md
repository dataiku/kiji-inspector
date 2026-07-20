# Migration plan: patched vLLM → native hidden-state extraction

Replace the patched vLLM v0.19.0 API (bespoke `extract_activation_layers` +
in-memory `RequestOutput.outputs[0].activations`) with vLLM's connector-based
`extract_hidden_states` speculator method + `ExampleHiddenStatesConnector`.
The target API currently comes from the feature fork built by this repository;
it must not be described as released upstream until an upstream tag is verified.

> [!IMPORTANT]
> **All implementation, tests, and exploratory API checks for this migration
> must run inside `575lab/kiji-inspector:dev`, the project image containing the
> hidden-state-enabled vLLM build.** The host
> environment and the vLLM v0.19.0 wheel do not contain the hidden-state
> connector used by this plan. The current image builds the feature-bearing
> `Davidnet/vllm` fork at branch `hidden-states-inline-return-squashed`; that
> exact source revision is the API contract until the feature is available in
> a pinned upstream release. Host-side tools may edit files, but must not be
> used to decide whether the vLLM API works.

## Decision gate before implementation

The current `Dockerfile` builds a feature branch, while `pyproject.toml` still
resolves the public v0.19.0 wheel. Therefore “native” in this document means
the connector/speculator API rather than “already released by upstream.” Before
changing application code:

1. Pull/use `575lab/kiji-inspector:dev` and record its immutable image digest,
   exact vLLM commit (`git -C /opt/vllm rev-parse HEAD`), and reported vLLM
   version from inside it. Do not rely on the mutable `:dev` tag alone in
   verification notes.
2. Run `samples/run_hidden_states_test.sh 575lab/kiji-inspector:dev`
   successfully on the GPU host.
3. Decide the distribution contract:
   - **Current recommended path:** make the Docker image the supported runtime
     and pin the fork by immutable commit (not only a movable branch); or
   - wait for an upstream release containing the same API, then pin that wheel.
4. Do not update `pyproject.toml` to an arbitrary newer/nightly vLLM wheel. Only
   point it at an artifact proven to contain the connector and model support.

The migration is blocked if the smoke test fails in the image; do not emulate
or mock the connector on the host to continue implementation.

## The big picture

| | Current (patched) | Target (native) |
|---|---|---|
| **Layers** | `LLM(..., extract_activation_layers=(20,))` kwarg | `speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids=[20]` via `method="extract_hidden_states"` |
| **Output** | in-memory `output.outputs[0].activations` = `{layer_idx: tensor[T, H]}` | `ExampleHiddenStatesConnector` writes **safetensors to disk**; read `output.kv_transfer_params["hidden_states_path"]` → `load_hidden_states(path)` → `{"token_ids", "hidden_states"}`, then `cleanup_hidden_states(path)` |
| **Shape** | dict of `[T, H]` per layer | single tensor `[T, num_layers, H]` — layer axis is the **middle** axis, ordered by `eagle_aux_hidden_state_layer_ids` |
| **Install** | pin vLLM v0.19.0 + apply 4 patch files | pin vLLM version that ships the connector; drop/reduce patches |

## Critical nuance: the 4 patches are NOT one thing

The patches split into two categories with different fates:

| Patch | Touches | Fate under native API |
|-------|---------|----------------------|
| **01_allow_extract_hidden_states** | `config/model.py`, `arg_utils.py`, `entrypoints/llm.py`, `outputs.py`, `v1/…/scheduler.py`, `gpu_model_runner.py`, openai serving | **Fully replaced** by the native connector. Drop entirely. |
| **02_support_nemotron_models** | `models/nemotron_h.py` | **Conditional** — implements `SupportsEagle3`/`EagleModelMixin` for the arch |
| **03_support_gemma3_models** | `models/gemma3.py` | **Conditional** |
| **04_support_qwen3_5_models** | `models/qwen3_5.py`, `qwen3_next.py` | **Conditional** |

Key realization: patches 02/03/04 implement the **exact same `set_aux_hidden_state_layers` / aux-hidden-state machinery** that the native
`extract_hidden_states` feature drives via `eagle_aux_hidden_state_layer_ids`.
So whether they can be dropped depends **entirely on whether the target vLLM
version ships upstream EAGLE3 aux-hidden-state support** for those architectures
(Nemotron-H, Gemma3, Qwen3-Next). `README_PATCH.md` documents that Qwen3-Next's
inner model does not inherit `EagleModelMixin` upstream — so patch 04 may still
be required unless upstream fixed it.

- **Patch 01 → guaranteed droppable.**
- **Patches 02/03/04 → verify per-model against the new vLLM version; if still
  needed, rebase them onto the new version's model files (line contexts will
  have shifted).**
- Default subject model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` and
  `Qwen3.6-35B-A3B` is also supported, so **patches 02 and 04 are the most
  likely to still be needed.**

## Semantic caveat (pre- vs post-layer residual)

The patches deliberately capture the residual **after** layer *i* (post-layer
output, matching an HF hook on `model.layers[i]`). Upstream EAGLE3 convention
typically captures the **input** to the layer. So `eagle_aux_hidden_state_layer_ids=[i]`
in the native API may point to a slightly different residual than the old
`residual_{i}`.

This is **not a correctness problem** — SAEs are retrained from scratch and
every step (1, 4, 5) uses the same extractor, so it stays internally consistent
— but numeric activation values and the effective layer will shift. Do **not**
expect bit-compatibility with existing SAE checkpoints / reports.

## Core extractor changes

Design principle: **keep `VLLMActivationExtractor`'s public interface unchanged**
(`extract()`, `extract_batch()`, `.tokenizer`, `.hidden_size`, `.cleanup()`,
`.config`). Then `RawActivationExtractor`, the DP worker path
(`_dp_shard_worker` / `run_dp_extraction_to_shards`), and the fuzzing evaluator
all keep working with **zero edits**.

### `src/kiji_inspector/extraction/vllm_activation_extractor.py`

**`__init__`:**
- Remove `extract_activation_layers=tuple(config.layers)`.
- Add `speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": list(config.layers)}}}`.
- Add `kv_transfer_config=KVTransferConfig(kv_connector="ExampleHiddenStatesConnector", kv_role="kv_producer", kv_connector_extra_config={"shared_storage_path": self._storage_dir, "allow_custom_save_path": True})`.
- Create a temp storage dir (`tempfile.mkdtemp`), store on `self._storage_dir`.
- Store `self._layer_order = list(config.layers)` for column → `residual_N` mapping.
- Import `from vllm.config.kv_transfer import KVTransferConfig` and import the
  connector module as demonstrated by `samples/test_hidden_states.py`; call
  `example_hidden_states_connector.load_hidden_states(...)` and
  `.cleanup_hidden_states(...)`. This avoids assuming that helper functions are
  re-exported from a package path.
- Likely drop the `VLLM_ALLOW_INSECURE_SERIALIZATION` env (line 21) — tensors no longer travel through `RequestOutput`. Keep until verified if unsure.

**`extract` / `extract_batch`:**
- Build **per-request** `SamplingParams` with a unique `hidden_states_path`
  (e.g. `f"{storage_dir}/req_{i}.safetensors"`) via
  `extra_args={"kv_transfer_params": {"hidden_states_path": ..., "include_output_tokens": False}}`
  — avoids file collisions under continuous batching.
- After `generate`, per output, validate that `kv_transfer_params` and the path
  are present, and use `try/finally` so files are cleaned even if tensor shape
  validation or NumPy conversion fails:
  `path = output.kv_transfer_params["hidden_states_path"]`;
  `hs = load_hidden_states(path)["hidden_states"]` (shape `[T, L, H]`);
  build `{f"residual_{self._layer_order[c]}": self._activation_to_numpy(hs[:, c, :]) for c in range(L)}`;
  then `cleanup_hidden_states(path)`.
- Assert `hs.ndim == 3`, `hs.shape[1] == len(self._layer_order)`, and
  `hs.shape[2] == self.hidden_size` before mapping columns. Also verify returned
  token IDs align with `output.prompt_token_ids`; silent token misalignment is
  more dangerous than a hard failure.
- `_activation_to_numpy` stays almost as-is (slices axis-0 for `decision`/`last`,
  returns full for `all`) — it's now fed a `[T, H]` slice per layer.

**`cleanup`:** also `shutil.rmtree(self._storage_dir, ignore_errors=True)`.

### Bonus: fuzzing per-token can now use vLLM

Because the default connector emits **all prompt-token** hidden states
(`[T, L, H]`), `token_positions="all"` now works under vLLM. The HF-backend
fallback in `fuzzing_evaluator.py:184-188` ("vLLM only returns the decode step")
can be removed — **optional**, and only after verifying the default request
actually emits prompt-position states (not just the generated token).

## Install / packaging changes

- **Docker image**: `575lab/kiji-inspector:dev` is the canonical development
  and runtime environment for this migration. Its `Dockerfile` must pin
  `Davidnet/vllm` to the verified immutable commit (or pin a released upstream
  tag once available), retain the CUDA/Torch compatibility constraints, and
  add an image-build import check for the connector. Add a development target
  (or install the project dev dependency group in this image) so `pytest` and
  `ruff` used below are guaranteed to exist; do not rely on tools inherited
  incidentally from vLLM's build environment.
- **`pyproject.toml`**: only bump the vLLM pin off the `v0.19.0` wheel if a
  distributable artifact containing the verified feature exists. Keep its
  source consistent with the Docker image and regenerate `uv.lock`. If the fork
  is image-only, document that the `full` extra alone is insufficient and keep
  Kiji installed with `--no-deps` inside the image as the Dockerfile does.
- **`patches/`**: delete `01_allow_extract_hidden_states.patch`. Keep 02/03/04
  only if verification shows upstream still lacks that model's aux-hidden-state
  support (and rebase them). If all can go: remove `patches/`, `apply-patch.sh`,
  `README_PATCH.md`, and every reference to `apply-patch.sh` in setup docs
  (these files currently live under `patches/`).

## Callers / tests to update (found by grep)

- `tests/test_activation_extractors.py:132` — asserts
  `kwargs["extract_activation_layers"] == (20,)`; rewrite to assert on
  `speculative_config` / `kv_transfer_config`.
- `src/kiji_inspector/utils/find_optimal_extraction.py:48,72` — uses
  `extract_activation_layers=[layer]` and `out.outputs[0].activations` directly;
  port to the new API (or route it through `VLLMActivationExtractor` so it's
  centralized).
- `README.md:88-93` and `docs/index.md` — patch-workflow instructions; update or
  remove.
- `demo/web_demo/app.py`, `experiments/ablation.py` — grep hit on "patch" but no
  API usage; verify no impact (ablation uses HF backend, unaffected).

## Docker development workflow

The existing smoke-test wrapper mounts the checkout read-only and executes the
sample from that mount. For iterative application development, run the checked
out source inside the feature image as well; do not accidentally import the
copy baked into `/opt/kiji-inspector`:

```bash
docker pull 575lab/kiji-inspector:dev
samples/run_hidden_states_test.sh 575lab/kiji-inspector:dev

docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src \
  -w /workspace \
  575lab/kiji-inspector:dev \
  pytest tests/test_activation_extractors.py
```

The example assumes the image's development target has installed the dev
dependency group described above.

Use the same container pattern for linting, focused integration tests, and the
small pipeline run. A test result from the host Python environment does not
count as migration verification. Rebuild the image whenever the vLLM revision,
Torch/CUDA stack, dependency metadata, or files copied by `Dockerfile` change.
Application-only edits under `src/` and `tests/` can use the bind mount plus
`PYTHONPATH` without rebuilding.

## Verification checklist (must run in the feature image on the GPU box)

1. Record the `575lab/kiji-inspector:dev` image digest, vLLM commit/version,
   Torch version, CUDA version, and GPU model in the verification notes.
   Confirm the target vLLM exports
   `vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector`
   and accepts the `extract_hidden_states` speculator method.
2. **Per subject model** (Nemotron-3-Nano-30B-A3B, Qwen3.6-35B-A3B):
   `eagle_aux_hidden_state_layer_ids` works **without** the model patches. If it
   raises on `set_aux_hidden_state_layers`/`EagleModelMixin`, that model's patch
   (02/04) is still required.
3. Default request emits **prompt-token** hidden states; shape is
   `[T, num_layers, H]`; layer ordering matches the
   `eagle_aux_hidden_state_layer_ids` order.
4. End-to-end: run `pipeline --step 1` then `--step 2` on a small pair set;
   confirm shards are `(N, H)` float32 and the SAE trains.
5. **Throughput**: the disk round-trip per request (vs the old in-memory attach)
   is the main perf risk under continuous batching — measure it; ensure
   per-request files are cleaned promptly.
6. Run unit tests in the same image, including failure-path tests that prove
   connector files and the extractor temp directory are removed after load or
   shape-validation errors.
7. Run the smoke test and focused tests in a fresh container after the final
   image rebuild, so success does not depend on state left in a development
   container.
