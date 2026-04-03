#!/usr/bin/env python3
"""
Generate contrastive pairs via DoubleWord's batch API.

Drop-in alternative to generate_pairs.py that uses DoubleWord's OpenAI-compatible
batch endpoint instead of a local vLLM + Qwen setup.

Usage:
    # Generate 1300 pairs using all scenarios (1h SLA)
    uv run python -m kiji_inspector.generate_pairs_doubleword 1300

    # Generate with 24h window (cheaper)
    uv run python -m kiji_inspector.generate_pairs_doubleword 1300 --completion-window 24h

    # Use specific scenarios
    uv run python -m kiji_inspector.generate_pairs_doubleword 1300 --scenario scenarios/tool_selection.json

Requires DOUBLEWORD_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

DOUBLEWORD_BASE_URL = "https://api.doubleword.ai/v1"

# Re-use the same prompt template from generator.py
CONTRASTIVE_PAIR_PROMPT = """
Generate {n_pairs} pairs of user requests where:
- Both requests have the SAME underlying intent/goal
- But they should use DIFFERENT tools due to a specific distinguishing factor

The distinguishing factor for this batch: {contrast_type}

Available tools:
{tool_list}

Contrast type explanation:
{contrast_explanation}

CRITICAL: The requests must be semantically VERY similar. Only subtle differences
should determine the tool choice. This is essential for training.

Output as a JSON array using EXACTLY these field names:
[
  {{
    "shared_intent": "what both requests try to accomplish",
    "anchor_request": "first user request text",
    "anchor_tool": "tool_name_for_anchor",
    "contrast_request": "second user request text",
    "contrast_tool": "tool_name_for_contrast",
    "distinguishing_signal": "what makes the difference"
  }}
]

No markdown fences, just raw JSON.
"""

SYSTEM_MESSAGE = (
    "You are a dataset generator for machine learning research. "
    "Output only valid JSON arrays, no markdown fences or commentary."
)

CONTRASTIVE_PAIR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "contrastive_pairs",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "shared_intent": {"type": "string"},
                            "anchor_request": {"type": "string"},
                            "anchor_tool": {"type": "string"},
                            "contrast_request": {"type": "string"},
                            "contrast_tool": {"type": "string"},
                            "distinguishing_signal": {"type": "string"},
                        },
                        "required": [
                            "shared_intent",
                            "anchor_request",
                            "anchor_tool",
                            "contrast_request",
                            "contrast_tool",
                            "distinguishing_signal",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["pairs"],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate contrastive pairs via DoubleWord batch API.",
    )
    p.add_argument(
        "num_samples",
        type=int,
        help="Number of contrastive pairs to generate.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/pairs",
        help="Directory for parquet shards and scenarios_meta.json (default: output/pairs).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-397B-A17B-FP8",
        help="Model to use on DoubleWord (default: Qwen/Qwen3-235B-A22B-FP8).",
    )
    p.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        choices=["1h", "24h"],
        help="Batch completion window: '1h' (faster) or '24h' (cheaper). Default: 24h.",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status checks (default: 30).",
    )
    p.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Path to scenario JSON config. Can be specified multiple times. "
        "Default: all *.json files in scenarios/.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _build_requests(
    scenarios: list,
    num_samples: int,
    model: str,
) -> list[dict]:
    """Build JSONL request dicts for the DoubleWord batch API (one pair per request)."""
    from tqdm import tqdm

    pairs_per_scenario = math.ceil(num_samples / len(scenarios))
    requests: list[dict] = []

    total_requests = sum(
        math.ceil(pairs_per_scenario / len(s.contrast_types)) * len(s.contrast_types)
        for s in scenarios
    )

    pbar = tqdm(total=total_requests, desc="Building requests", unit="req")

    for scenario in scenarios:
        tool_list = "\n".join(f"- {t['name']}: {t['description']}" for t in scenario.tools)
        n_types = len(scenario.contrast_types)
        pairs_per_type = math.ceil(pairs_per_scenario / n_types)

        for ct, explanation in scenario.contrast_types.items():
            for i in range(pairs_per_type):
                user_content = CONTRASTIVE_PAIR_PROMPT.format(
                    n_pairs=1,
                    contrast_type=ct,
                    tool_list=tool_list,
                    contrast_explanation=explanation,
                )

                custom_id = f"{scenario.name}_{ct}_{i}"
                requests.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": SYSTEM_MESSAGE},
                                {"role": "user", "content": user_content},
                            ],
                            "response_format": CONTRASTIVE_PAIR_SCHEMA,
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "max_tokens": 2048,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    }
                )
                pbar.update(1)

    pbar.close()
    return requests


# ---------------------------------------------------------------------------
# DoubleWord batch API helpers
# ---------------------------------------------------------------------------


def _get_client():
    """Create an OpenAI client pointed at DoubleWord."""
    from openai import OpenAI

    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        print("Error: DOUBLEWORD_API_KEY environment variable is required.", file=sys.stderr)
        print("Get your key at https://app.doubleword.ai", file=sys.stderr)
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url=DOUBLEWORD_BASE_URL)


def _upload_jsonl(client, requests: list[dict]) -> str:
    """Write requests to a temp JSONL file and upload to DoubleWord. Returns file ID."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
        tmp_path = f.name

    size = os.path.getsize(tmp_path)
    for unit in ("B", "KB", "MB", "GB"):  # noqa: B007
        if size < 1024:
            break
        size /= 1024
    print(f"  JSONL file size: {size:.1f} {unit}")

    try:
        result = client.files.create(file=open(tmp_path, "rb"), purpose="batch")
        return result.id
    finally:
        os.unlink(tmp_path)


def _create_batch(client, file_id: str, completion_window: str) -> str:
    """Create a batch job. Returns batch ID."""
    result = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": "kiji-inspector contrastive pair generation"},
    )
    return result.id


def _poll_batch(client, batch_id: str, poll_interval: int) -> object:
    """Poll until batch completes or fails. Returns the final batch object."""
    print(f"  Batch ID: {batch_id}")
    print(f"  Polling every {poll_interval}s...")

    last_completed = -1
    while True:
        batch = client.batches.retrieve(batch_id)
        total = batch.request_counts.total
        completed = batch.request_counts.completed
        failed = batch.request_counts.failed

        if completed != last_completed:
            print(
                f"  Status: {batch.status} | "
                f"Progress: {completed}/{total} completed, {failed} failed"
            )
            last_completed = completed

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch

        time.sleep(poll_interval)


def _download_results(client, output_file_id: str) -> list[dict]:
    """Download batch results as a list of response dicts."""
    import requests as req

    url = f"{DOUBLEWORD_BASE_URL}/files/{output_file_id}/content"
    headers = {"Authorization": f"Bearer {client.api_key}"}
    response = req.get(url, headers=headers)
    response.raise_for_status()

    results = []
    for line in response.text.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Parse results into ContrastivePairs
# ---------------------------------------------------------------------------


def _parse_response_to_pairs(result: dict) -> list:
    """Parse a single batch response into ContrastivePair objects."""
    import re

    from kiji_inspector.data.contrastive_dataset import ContrastivePair
    from kiji_inspector.data.generator import _fuzzy_get, _parse_json_array

    custom_id = result.get("custom_id", "")

    response_body = result.get("response", {}).get("body", {})
    choices = response_body.get("choices", [])
    if not choices:
        return []

    raw_content = choices[0].get("message", {}).get("content", "")
    if not raw_content:
        return []

    # Parse the JSON — structured output wraps in {"pairs": [...]}
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict) and "pairs" in parsed:
            pairs_data = parsed["pairs"]
        elif isinstance(parsed, list):
            pairs_data = parsed
        else:
            pairs_data = _parse_json_array(raw_content)
    except (json.JSONDecodeError, ValueError):
        try:
            pairs_data = _parse_json_array(raw_content)
        except (json.JSONDecodeError, ValueError):
            return []

    # Infer contrast_type and scenario_name from the custom_id
    # custom_id format: "{scenario}_{contrast_type}_{chunk_idx}"
    # We need to reconstruct. The last part is chunk_idx (digit).
    # Everything before is scenario + contrast_type separated by underscore.
    # We'll store the full prefix as context and try to split later.
    # For now, use the custom_id prefix as-is and extract from the first pair's data.
    scenario_name = ""
    contrast_type = ""

    # Parse custom_id: we know contrast_types contain underscores
    # e.g., "tool_selection_internal_vs_external_0"
    # Strategy: try known contrast type patterns (contains "_vs_")
    match = re.match(r"^(.+?)_((?:\w+_vs_\w+|\w+)?)_(\d+)$", custom_id)
    if match:
        scenario_name = match.group(1)
        contrast_type = match.group(2)

    pairs = []
    for i, p in enumerate(pairs_data):
        try:
            pairs.append(
                ContrastivePair(
                    pair_id=f"{custom_id}_{i}",
                    anchor_prompt=_fuzzy_get(
                        p, "anchor_request", ["anchor", "request_1", "first_request"]
                    ),
                    anchor_tool=_fuzzy_get(
                        p,
                        "anchor_tool",
                        ["anchor_best_tool", "best_tool_for_anchor", "tool_1", "first_tool"],
                    ),
                    contrast_prompt=_fuzzy_get(
                        p, "contrast_request", ["contrast", "request_2", "second_request"]
                    ),
                    contrast_tool=_fuzzy_get(
                        p,
                        "contrast_tool",
                        [
                            "contrast_best_tool",
                            "best_tool_for_contrast",
                            "tool_2",
                            "second_tool",
                        ],
                    ),
                    shared_intent=_fuzzy_get(
                        p, "shared_intent", ["intent", "common_intent", "goal"]
                    ),
                    semantic_similarity=0.9,
                    contrast_type=contrast_type,
                    distinguishing_signal=_fuzzy_get(
                        p,
                        "distinguishing_signal",
                        ["signal", "distinction", "difference", "key_difference"],
                    ),
                    scenario_name=scenario_name,
                )
            )
        except (KeyError, TypeError):
            continue

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    from kiji_inspector.data.scenario import discover_scenarios, save_scenarios_meta

    scenarios = discover_scenarios(args.scenarios)
    scenario_names = [s.name for s in scenarios]
    total_contrast_types = sum(len(s.contrast_types) for s in scenarios)

    print("=" * 60)
    print("  Contrastive Pair Generation (DoubleWord Batch API)")
    print(f"  Requested samples : {args.num_samples:,}")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Scenarios         : {', '.join(scenario_names)}")
    print(f"  Contrast types    : {total_contrast_types} total across {len(scenarios)} scenario(s)")
    print(f"  Model             : {args.model}")
    print(f"  Completion window : {args.completion_window}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Build all requests
    requests = _build_requests(scenarios, args.num_samples, args.model)
    print(f"\n  Built {len(requests)} batch requests")

    # Step 2: Upload and submit
    client = _get_client()

    print("  Uploading JSONL file...")
    file_id = _upload_jsonl(client, requests)
    print(f"  Uploaded file: {file_id}")

    print("  Creating batch...")
    batch_id = _create_batch(client, file_id, args.completion_window)

    # Step 3: Poll for completion
    print()
    batch = _poll_batch(client, batch_id, args.poll_interval)
    print()

    if batch.status != "completed":
        print(f"  Batch ended with status: {batch.status}", file=sys.stderr)
        if batch.request_counts.failed > 0:
            print(f"  {batch.request_counts.failed} requests failed", file=sys.stderr)
        sys.exit(1)

    # Step 4: Download and parse results
    print("  Downloading results...")
    if not batch.output_file_id:
        print("  Error: No output file available.", file=sys.stderr)
        sys.exit(1)

    raw_results = _download_results(client, batch.output_file_id)
    print(f"  Downloaded {len(raw_results)} responses")

    # Save raw JSONL for debugging
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "batch_output_raw.jsonl"
    with open(raw_path, "w") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved raw output: {raw_path}")

    # Step 5: Parse into ContrastivePairs
    from kiji_inspector.data.contrastive_dataset import ContrastiveDataset

    all_pairs = []
    malformed = 0
    for result in raw_results:
        pairs = _parse_response_to_pairs(result)
        if not pairs:
            malformed += 1
        all_pairs.extend(pairs)

    if malformed > 0:
        print(f"  {malformed} responses failed to parse")

    # Check for existing pairs to merge with
    output_path = Path(args.output_dir)
    existing_shards = sorted(output_path.glob("shard_*.parquet")) if output_path.exists() else []
    if existing_shards:
        existing_dataset = ContrastiveDataset.from_parquet(args.output_dir)
        print(
            f"  Found {len(existing_dataset.pairs)} existing pairs, merging with {len(all_pairs)} new"
        )
        all_pairs = existing_dataset.pairs + all_pairs

    # Trim to requested count (new pairs only, existing are kept)
    if not existing_shards:
        all_pairs = all_pairs[: args.num_samples]

    dataset = ContrastiveDataset(pairs=all_pairs)
    shards = dataset.to_parquet(args.output_dir)
    print(f"  Saved {len(shards)} shard(s) to {args.output_dir}")

    # Save scenarios metadata
    meta_path = save_scenarios_meta(scenarios, Path(args.output_dir))
    print(f"  Saved scenarios metadata: {meta_path}")

    elapsed = time.time() - t0
    print(f"\n  Generation complete ({elapsed:.1f}s)")
    print(f"  {len(all_pairs)} total pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
