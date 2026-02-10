# Step 1: Contrastive Pair Generation

## Purpose

Generate synthetic pairs of user requests that share the same underlying intent but require different tools. These pairs isolate the **decision signal** -- the minimal difference in a request that causes the agent to choose one tool over another. The contrastive structure is not used during SAE training (Step 3) but is critical for post-hoc feature identification (Step 4).

## Source Files

| File | Key Components |
|------|----------------|
| `generate_training_set.py` | `_run_step1()`, `generate_pairs()`, `_run_generation_subprocess()` |
| `src/generator.py` | `ContrastivePairGenerator`, `_parse_json_array()`, `_fuzzy_get()` |
| `src/contrastive_dataset.py` | `ContrastivePair`, `ContrastiveDataset`, Parquet I/O |
| `src/scenario.py` | `ScenarioConfig`, `load_scenarios()`, `save_scenarios_meta()` |

## Architecture

```
scenarios/*.json
      |
      v
ScenarioConfig objects
      |
      v
ContrastivePairGenerator
      |  (one per scenario)
      |
      v
vLLM (Qwen3-VL-235B)  <-- runs in subprocess for GPU isolation
      |
      |  N prompts per contrast type, batched via continuous batching
      |
      v
Raw JSON arrays (LLM output)
      |
      v
_parse_json_array() + _fuzzy_get()
      |
      v
ContrastivePair objects
      |
      v
ContrastiveDataset.to_parquet()
      |
      v
output/pairs/shard_*.parquet + scenarios_meta.json
```

## Scenario Configuration

Each scenario defines a domain with its own tools and contrast types. The schema:

```json
{
  "name": "tool_selection",
  "system_prompt": "You are a helpful assistant. Choose the best tool...",
  "tools": [
    {"name": "internal_search", "description": "Search internal docs..."},
    {"name": "web_search", "description": "Search the web..."}
  ],
  "contrast_types": {
    "internal_vs_external": "One request needs internal docs, the other needs web results.",
    "read_vs_write": "One request reads data, the other modifies data."
  }
}
```

When no `--scenario` flags are provided, all `scenarios/*.json` files are auto-discovered and loaded. The built-in scenarios are:

| Scenario | Tools | Contrast Types |
|----------|-------|----------------|
| `tool_selection` | 8 (internal_search, web_search, file_read, file_write, database_query, api_call, code_execute, delegate_agent) | 13 |
| `investment` | 6 (market_data_lookup, financial_analysis, risk_assessment, news_search, portfolio_optimizer, sector_comparison) | 6 |
| `manufacturing` | 6 (equipment_monitor, quality_check, inventory_lookup, maintenance_scheduler, production_planner, defect_analyzer) | 6 |
| `supply_chain` | 6 (shipment_tracker, demand_forecaster, supplier_database, route_optimizer, inventory_manager, procurement_system) | 6 |
| `customer_support` | 6 (ticket_lookup, knowledge_base, escalation_system, customer_history, billing_system, product_docs) | 6 |

## ContrastivePair Data Structure

Each pair captures two semantically similar requests that require different tools:

```python
@dataclass
class ContrastivePair:
    pair_id: str               # e.g. "tool_selection_read_vs_write_3"
    anchor_prompt: str         # First user request
    anchor_tool: str           # Correct tool for anchor
    contrast_prompt: str       # Second user request
    contrast_tool: str         # Correct tool for contrast
    shared_intent: str         # What both requests have in common
    semantic_similarity: float # 0-1 similarity score (default: 0.9)
    contrast_type: str         # Category name (e.g. "read_vs_write")
    distinguishing_signal: str # Human-readable explanation of key difference
    scenario_name: str         # Which scenario produced this pair
```

**Example pair** (tool_selection, read_vs_write):

| Field | Anchor | Contrast |
|-------|--------|----------|
| Prompt | "Show me the current database schema for the users table" | "Add a created_at timestamp column to the users table" |
| Tool | `database_query` | `file_write` |
| Shared intent | "Working with the users table schema" |
| Distinguishing signal | "Reading schema vs. modifying schema" |

## LLM Prompt Template

The generator uses a structured prompt template in ChatML format:

```
<|im_start|>system
You are a dataset generator for machine learning research.
Output only valid JSON arrays, no markdown fences or commentary.<|im_end|>
<|im_start|>user
Generate {n_pairs} pairs of user requests where:
- Both requests have the SAME underlying intent/goal
- But they should use DIFFERENT tools due to a specific distinguishing factor

The distinguishing factor for this batch: {contrast_type}

Available tools:
- internal_search: Search internal documentation...
- web_search: Search the web...
[...]

Contrast type explanation:
{contrast_explanation}

CRITICAL: The requests must be semantically VERY similar. Only subtle
differences should determine the tool choice.

Output as a JSON array using EXACTLY these field names:
[
  {
    "shared_intent": "...",
    "anchor_request": "...",
    "anchor_tool": "...",
    "contrast_request": "...",
    "contrast_tool": "...",
    "distinguishing_signal": "..."
  }
]

No markdown fences, just raw JSON.<|im_end|>
<|im_start|>assistant
```

## Generation Process

### Sample Distribution

Given `num_samples` total pairs and `S` scenarios:

```
pairs_per_scenario = ceil(num_samples / S)
pairs_per_type     = ceil(pairs_per_scenario / num_contrast_types_in_scenario)
```

Each contrast type generates `pairs_per_type` pairs, split into chunks of at most `--generation-batch` (default: 50) pairs per LLM prompt.

### Batched Generation

Prompts are grouped into batches of at most `--vllm-batch-size` (default: 128) and sent to vLLM in a single `llm.generate()` call. vLLM's continuous batching schedules all prompts across GPU resources simultaneously.

```python
# SamplingParams for generation
SamplingParams(
    temperature=0.7,
    top_p=0.8,
    max_tokens=8000,
)
```

### Subprocess Isolation

The entire generation runs in a `multiprocessing.spawn` subprocess to ensure the vLLM model (Qwen3-VL-235B) releases all GPU memory before Step 2 loads Nemotron. The subprocess:

1. Loads vLLM with `gpu_memory_utilization=0.95`, `enable_expert_parallel=True`
2. Iterates over scenarios, generating pairs for each
3. Saves the combined dataset as Parquet shards
4. Exits, freeing all GPU memory

## JSON Parsing and Error Recovery

LLM output is parsed through multiple recovery layers in `_parse_json_array()`:

1. **Markdown fence stripping**: Remove ` ```json ... ``` ` wrappers
2. **Direct parse**: Try `json.loads()` on the stripped output
3. **Bracket extraction**: Find outermost `[...]` brackets
4. **Truncation recovery**: If `]` is missing (output truncated), find the last complete `}` and close the array
5. **Trailing comma removal**: Fix `,]` and `,}` (common LLM generation artifacts)

### Fuzzy Key Matching

LLMs sometimes rename JSON keys (e.g., `"contrast_tool"` becomes `"best_tool_for_contrast"`). The `_fuzzy_get()` function tries the canonical key first, then a list of known variants:

```python
_fuzzy_get(d, "anchor_tool",
    fallbacks=["anchor_best_tool", "best_tool_for_anchor", "tool_1", "first_tool"])
```

Entries where no key variant matches are silently counted as malformed and skipped. A per-scenario malformed count is printed at the end.

## Parquet Output Format

The `ContrastiveDataset.to_parquet()` method serializes pairs into sharded Parquet files using PyArrow:

```
output/pairs/
    shard_00000.parquet    # Up to 50,000 pairs per shard
    shard_00001.parquet
    ...
    scenarios_meta.json    # Serialized ScenarioConfig objects
```

Each Parquet file contains columns matching the `ContrastivePair` fields. The `scenarios_meta.json` file preserves the tool lists and system prompts so that later steps can reconstruct per-pair prompts without re-loading the scenario JSON files.

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | (required) | Total number of contrastive pairs to generate |
| `--qwen-model` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | Generation model |
| `--generation-tp-size` | 4 | Tensor parallel size for vLLM |
| `--max-model-len` | 16384 | Maximum sequence length |
| `--generation-batch` | 50 | Max pairs per LLM prompt |
| `--vllm-batch-size` | 128 | Max prompts per `llm.generate()` call |
| `--scenario` | All `scenarios/*.json` | Scenario config path (repeatable) |

## Verification

```bash
# Check pair distribution across scenarios and contrast types
python -c "
import pyarrow.parquet as pq
from collections import Counter
t = pq.read_table('output/pairs/shard_00000.parquet')
d = t.to_pydict()
print('Scenarios:', Counter(d['scenario_name']))
print('Contrast types:', Counter(d['contrast_type']))
print('Total pairs:', len(d['pair_id']))
"
```
