# Step 2: Activation Extraction

## Purpose

Extract the hidden-state activations from the subject model (Nemotron-3-Nano-30B) at the **decision token** -- the precise position where the model is about to output a tool name. These raw activation vectors form the training data for the SAE in Step 3.

Each contrastive pair contributes **two** activation vectors (one for the anchor prompt, one for the contrast prompt), yielding `2 * num_pairs` total vectors.

## Source Files

| File | Key Components |
|------|----------------|
| `src/pipeline.py` | `_run_step2()`, `extract_activations()` |
| `src/extraction/extractor.py` | `build_agent_prompt()`, `RawActivationExtractor`, `extract_to_shards()` |
| `src/extraction/activation_extractor.py` | `ActivationConfig`, `ActivationExtractor`, forward hooks |

## Architecture

```
ContrastivePair
    |
    v
build_agent_prompt()          Per-pair scenario lookup
    |                         (tools, system_prompt)
    v
Formatted ChatML Prompt
    |
    v
Nemotron-3-Nano-30B
(HuggingFace Transformers, device_map="auto", multi-GPU)
    |
    |--- Forward hooks capture residual stream at layer 20
    |
    v
Hidden state at last token: shape (d_model,) = (4096,)
    |
    v
Cast to float16, append to shard buffer
    |
    v
Flush to shard_XXXXXX.npy when buffer reaches shard_size
```

## The Decision Token

The prompt is formatted to end with `"I'll use the "`:

```
<|im_start|>system
{system_prompt}

Available tools:
- tool_name: description
...

When you decide to use a tool, respond with the tool name.<|im_end|>
<|im_start|>user
{user_request}<|im_end|>
<|im_start|>assistant
I'll use the
```

The last token of this prompt is the **decision token**. At this position, the model's hidden state encodes its full "reasoning" about which tool to name next. By extracting activations at exactly this point, we capture the model's internal decision state before it commits to a specific tool.

### Supported Model Formats

The `build_agent_prompt()` function supports four chat templates:

| `model_type` | Format | Example |
|--------------|--------|---------|
| `nemotron` | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) | Default |
| `llama` | Llama 3 Instruct (`<\|begin_of_text\|>`, `<\|start_header_id\|>`) | |
| `mistral` | Mistral instruct (`[INST]` / `[/INST]`) | |
| `generic` | Plain text (`System:` / `User:` / `Assistant:`) | Fallback |

## Forward Hook Mechanism

### Hook Registration

For each target layer, a forward hook is registered on the layer module:

```python
def _make_hook(name):
    def hook(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        self._activations[name] = activation.detach().cpu().to(torch.float32)
    return hook

# Register on layer 20's residual stream
layers[20].register_forward_hook(_make_hook("residual_20"))
```

The hook:
1. Intercepts the layer's output tensor
2. Detaches from the computation graph
3. Moves to CPU (critical for multi-GPU: tensors may live on different GPUs)
4. Casts to float32 for uniform representation

### Architecture Detection

The extractor auto-detects the model's internal structure to find the layer list:

| Architecture | Layer Path |
|-------------|-----------|
| Llama / Nemotron | `model.model.layers` |
| NemotronH | `model.backbone.layers` or `model.backbone.decoder.layers` |
| GPT-NeoX | `model.transformer.h` |
| GPT-2 | `model.transformer.layers` |

### Input Device Detection

With `device_map="auto"`, different layers live on different GPUs. Input tensors must go to the embedding layer's device:

```python
# Search common embedding attribute paths
for attr_path in ("model.embed_tokens", "backbone.embed_tokens",
                  "backbone.embedding", "transformer.wte", "model.embed_in"):
    # Find the device of the first parameter
```

## Extraction Process

### Single Prompt Extraction

```python
def extract(self, prompt, decision_token_offset=-1):
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
    seq_len = inputs.input_ids.shape[1]

    # Run only the transformer body (skip lm_head)
    inner_model = getattr(self.model, "model", self.model)
    with torch.no_grad():
        inner_model(**inputs)

    position = seq_len + decision_token_offset  # -1 = last token
    return {name: act[:, position, :].squeeze(0).numpy()
            for name, act in self._activations.items()}
```

The `lm_head` is skipped to avoid allocating the huge `(batch, seq_len, vocab_size)` logit tensor, which would waste GPU memory.

### Batched Extraction

For throughput, prompts are processed in batches using **left-padding**:

```python
self.tokenizer.padding_side = "left"
inputs = self.tokenizer(batch_prompts, return_tensors="pt",
                        padding=True, truncation=True)
```

Left-padding ensures the last real token (decision token) aligns across the batch at different absolute positions. For each batch item, the last non-pad token is found via `attention_mask.sum()`:

```python
for b in range(batch_size):
    real_len = attention_mask[b].sum().item()
    position = real_len - 1  # last real token
    item = {name: act[b, position, :].numpy() for name, act in self._activations.items()}
```

## Shard-Based Output

Activation vectors are accumulated in a buffer and flushed to disk when the buffer reaches `shard_size` (default: 500,000):

```python
shard_buffer.append(vec.astype(np.float16))
if len(shard_buffer) >= shard_size:
    shard_data = np.stack(shard_buffer, axis=0)  # (N, 4096), float16
    np.save(f"shard_{shard_idx:06d}.npy", shard_data)
```

### Output Files

```
output/activations/
    shard_000000.npy    # shape (N, 4096), dtype float16
    shard_000001.npy
    ...
    metadata.json       # Model, layer, d_model, total_tokens, num_shards
    prompts.json        # User request string per activation vector (shard order)
```

### metadata.json

```json
{
  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
  "layer": "residual_20",
  "d_model": 4096,
  "total_tokens": 1000000,
  "num_shards": 2,
  "shard_size": 500000,
  "dtype": "float16",
  "prompts_per_pair": 2,
  "total_pairs": 500000
}
```

### prompts.json

A JSON array of user request strings, one per activation vector, in the same order as the shard data. This enables Step 5 to map activations back to prompt text without re-running inference:

```json
["Show me the users table schema", "Add a timestamp to the users table", ...]
```

## Per-Pair Scenario Lookup

Each pair's `scenario_name` field maps to a `ScenarioConfig` via `scenarios_meta.json` (saved by Step 1). This ensures each pair's formatted prompt uses the correct system prompt and tool list:

```python
for pair in pairs:
    scenario = scenarios_meta.get(pair.scenario_name, default_scenario())
    anchor_prompt = build_agent_prompt(
        system_prompt=scenario.system_prompt,
        tools=scenario.tools,
        user_request=pair.anchor_prompt,
    )
```

Backward compatibility: if `scenarios_meta.json` is missing (older pipeline runs), the built-in `default_scenario()` (tool_selection with 8 tools) is used for all pairs.

## Memory Considerations

| Component | Typical Size |
|-----------|-------------|
| Nemotron-3-Nano-30B (bfloat16) | ~60 GB across 4 GPUs |
| Single activation vector | 4096 * 2 bytes = 8 KB (float16) |
| 1M activation vectors | ~8 GB on disk |
| Forward pass peak memory | Proportional to `batch_size * seq_len * d_model` |

The `lm_head` skip saves approximately `batch_size * seq_len * vocab_size * 2` bytes per batch (for Nemotron's vocab_size of 256,000, this is substantial).

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | `output/activations` | Directory for numpy shards |
| `--nemotron-model` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Subject model |
| `--layers` | `[20]` | Transformer layers to hook |
| `--layer-key` | `residual_20` | Layer key for extraction |
| `--batch-size` | 512 | Prompts per GPU forward pass |
| `--shard-size` | 500,000 | Activation vectors per numpy shard |
| `--pairs-dir` | `output/pairs` | Source of contrastive pairs |
