# Verification notes: native hidden-state extraction migration

Records the environment and results for migrating `VLLMActivationExtractor`
from the patched `extract_activation_layers` API to vLLM's native
`extract_hidden_states` speculator method + `ExampleHiddenStatesConnector`.

All checks below ran **inside** `575lab/kiji-inspector:dev` on the GPU host, per
the migration plan (host Python does not count as verification).

## Environment (decision gate)

| Item | Value |
|---|---|
| Image tag | `575lab/kiji-inspector:dev` |
| Image digest | `sha256:7c1e4a8652d65aacbbf4afdb637d7a7d747d6f6e7a3d8aa5eb492b538c213ff0` |
| Image ID | `sha256:246b40a44d70` |
| vLLM commit (`git -C /opt/vllm rev-parse HEAD`) | `b6455d43be849d4850bed6ecfb834489ba9f0a08` |
| vLLM branch | `hidden-states-inline-return-squashed` |
| vLLM version | `0.1.dev18596+gb6455d43b` |
| Torch | `2.11.0+cu129` |
| CUDA (torch) | `12.9` |
| GPU | NVIDIA RTX PRO 6000 Blackwell (97 GB) |
| ninja / flashinfer | `1.13.0` / `0.6.13` (present — no MoE-backend fallback needed) |

Connector exports confirmed present:
`vllm.config.kv_transfer.KVTransferConfig`,
`vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector`
with `load_hidden_states` and `cleanup_hidden_states`.

## Smoke test (plan gate step 2)

`samples/run_hidden_states_test.sh 575lab/kiji-inspector:dev` (Qwen3-8B) passes.
Both the disk-path and inline-return code paths return hidden states of shape
`[num_tokens, num_extracted_layers, hidden_size]` — e.g. `[5, 4, 4096]` for the
4-layer example — confirming the `[T, L, H]` layout with the layer axis in the
middle.

## Qwen3.6-35B-A3B (plan verification step 2 & 3)

Model: `Qwen/Qwen3.6-35B-A3B` @ revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`.

- `eagle_aux_hidden_state_layer_ids=[1,2,3,4]` is accepted **without** the
  `04_support_qwen3_5_models` patch applied separately — the fork already ships
  the aux-hidden-state support. Engine log:
  `Using auxiliary layers from speculative config: (1, 2, 3, 4)`.
- Model loading took 65.53 GiB (fits the 97 GB GPU); loaded with
  `enforce_eager=True`.
- The FlashInfer MoE backend (auto-selected because ninja/flashinfer are
  present) **stalls** the profiling forward pass on this model. Pinning the
  reference's known-good backend clears it: `attention_backend="TRITON_ATTN"`,
  `moe_backend="triton"`, `enable_chunked_prefill=False`, plus
  `VLLM_USE_FLASHINFER_SAMPLER=0`. With `language_model_only=True` the footprint
  is 64.69 GiB (matches the reference report). These defaults are now applied
  automatically for the Qwen3.5/3.6 family by
  `recommended_vllm_kwargs()`.
- Extractor checks via `VLLMActivationExtractor` — **all passed**:

| Check | Result |
|---|---|
| `hidden_size` | 2048 |
| `token_positions="all"` → per-layer `[T, H]` | (5, 2048) per layer; T == prompt token count |
| keys / ordering | `residual_1..4`, columns distinct (order preserved) |
| dtype / finiteness | float32, all finite |
| batch of 3 prompts | per-prompt shapes (5,2048), (9,2048), (9,2048) — per-request files do not collide |
| token alignment | extracted token ids == prompt token ids (internal assert did not trip) |
| `token_positions="decision"` (layer 20) | `residual_20` vector shape (2048,), finite |

## End-to-end pipeline (plan verification step 4)

Ran `pipeline --step 1` then `--step 2` on a synthetic 12-pair set
(`--subject-model Qwen/Qwen3.6-35B-A3B --no-thinking --layers 20`) through the
default `create_extractor` path (no explicit backend flags — the Qwen defaults
come from `recommended_vllm_kwargs`). The pipeline compiles + captures CUDA
graphs (the connector works in **compiled** mode, not just eager).

- **Step 1**: shards are `(24, 2048)` float32 (`SHARDS_OK`), 12 pairs × 2
  prompts = 24 decision-token vectors, `d_model=2048`.
- **Step 2**: the JumpReLU SAE trained and saved
  `layer_20/sae_checkpoints/sae_final.pt` and `step_3.pt`.

Two unrelated, pre-existing image gaps surfaced (the dev image installs kiji
with `--no-deps`, so its pure-python deps are absent): `pyarrow` (pairs parquet
I/O) and `scipy` (a *post-training* "feature health" analysis helper — it fails
only after the SAE is already trained and checkpointed). Both are packaging
gaps, not migration regressions; installing the kiji dependency group in the
image (see the Dockerfile note) resolves them.

## Two engine-startup gotchas found & fixed (Qwen3.6 / this image)

1. **FlashInfer MoE stall**: because ninja/flashinfer are present, vLLM
   auto-selects the FlashInfer MoE backend, which hangs the profiling forward
   pass. Fixed by pinning Triton (`moe_backend="triton"`,
   `attention_backend="TRITON_ATTN"`, `enable_chunked_prefill=False`,
   `VLLM_USE_FLASHINFER_SAMPLER=0`) — now applied automatically for the
   Qwen3.5/3.6 family via `recommended_vllm_kwargs()`.
2. **Allocator incompatibility**: the KV connector rejects
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (which the pipeline sets
   for Blackwell host-OOM mitigation). The extractor now strips that one token
   before engine construction.

## Unit tests

`pytest tests/test_activation_extractors.py` → **10 passed** in the image
(bind-mounted source, `PYTHONPATH=/workspace/src`), `ruff check` clean. Includes
failure-path tests proving the connector file is cleaned up on shape-validation
and token-misalignment errors, and a test for the Qwen-family recommended
kwargs.
