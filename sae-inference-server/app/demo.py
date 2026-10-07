"""Live supply-chain comparison adapted from steering-demos; no stored results."""

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.routes import fetch_completion
from app.config import SAE_LAYER, SAE_REPO_ID, VLLM_CAPTURE_LAYERS, VLLM_MODEL
from app.dependencies import SAEEngine, get_engine
from app.inline import extract_inline_activation
from app.schemas import InterpretRequest

ROOT = Path(__file__).parent
DATA = ROOT / "demo_data"
SCENARIO = json.loads((DATA / "scenario.json").read_text())
PAIRS = json.loads((DATA / "pairs.json").read_text())["pairs"]
PROBES = json.loads((DATA / "probes.json").read_text())
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
router = APIRouter()


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: str = Field(min_length=1, max_length=4000)
    b: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=12, ge=1, le=50)


@lru_cache
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(VLLM_MODEL)


def decision_prompt(request, tokenizer):
    # Same helper, settings, and exact prefill as steering-demos build_prompts().
    from kiji_inspector.extraction.extractor import build_agent_prompt
    from kiji_inspector.extraction.vllm_activation_extractor import (
        recommended_chat_template_kwargs,
    )

    kwargs = recommended_chat_template_kwargs(VLLM_MODEL, tokenizer)
    return build_agent_prompt(
        system_prompt=SCENARIO["system_prompt"],
        tools=SCENARIO["tools"],
        user_request=request,
        tokenizer=tokenizer,
        chat_template_kwargs=kwargs,
        close_think_block=bool(kwargs),
        assistant_prefill="I'll use the",
    )


def generated_tool(text):
    # Report only an explicit leading tool name; never infer from SAE labels.
    text = text.strip().lstrip("\"'`*").lower()
    for tool in SCENARIO["tools"]:
        name = tool["name"]
        for form in (name, name.replace("_", " ")):
            if re.match(re.escape(form) + r"\b", text):
                return name
    return None


@router.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse(ROOT / "static" / "demo.html")


@router.get("/demo/config")
def demo_config():
    pairs = []
    for pair in PAIRS:
        item = {key: pair[key] for key in ("id", "title", "signal")}
        for side in ("a", "b"):
            probe = PROBES.get(pair["id"], {}).get(side, {})
            variants = [{"label": "Original", "request": pair[side]["request"]}]
            variants.extend(
                {"label": f"Paraphrase {i + 1}", "request": request}
                for i, request in enumerate(probe.get("paraphrases", []))
            )
            if probe.get("keyword"):
                variants.append(
                    {"label": "Keyword control", "request": probe["keyword"]["request"]}
                )
            item[side] = variants
        pairs.append(item)
    return {
        "pairs": pairs,
        "tools": SCENARIO["tools"],
        "model": VLLM_MODEL,
        "sae_repo": SAE_REPO_ID,
        "layer": SAE_LAYER,
        "provenance": PROVENANCE,
        "compatible": SAE_LAYER == 43,
    }


def run_side(request, tokenizer, engine, top_k):
    prompt = decision_prompt(request, tokenizer)
    response = fetch_completion(
        InterpretRequest(
            route="completion",
            request={
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 24,
                "add_special_tokens": False,
            },
        )
    )
    try:
        activation = extract_inline_activation(
            response, VLLM_CAPTURE_LAYERS, SAE_LAYER, -1, engine.sae.d_model
        )
        encoded = engine.encode_activation(activation)
        text = response["choices"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"Invalid vLLM capture: {exc}") from exc
    return {
        "request": request,
        "tool": generated_tool(text),
        "continuation": text,
        "formatted_prompt": prompt,
        "response_id": response.get("id"),
        "usage": response.get("usage"),
        **engine.describe_features(encoded, top_k),
    }, encoded


def feature_changes(a, b, engine, top_k):
    delta = b.float() - a.float()
    indices = torch.topk(delta.abs(), min(top_k, delta.numel())).indices.tolist()
    rows = []
    for index in indices:
        if delta[index].item() == 0:
            continue
        description = engine.sae._lookup_feature_description(engine.feature_descriptions, index)
        rows.append(
            {
                "feature_id": str(index),
                "description": engine._normalize_description(description),
                "a": a[index].item(),
                "b": b[index].item(),
                "delta": delta[index].item(),
            }
        )
    return rows


@router.post("/demo/compare")
def compare(payload: ComparisonRequest, engine: SAEEngine = Depends(get_engine)):
    if SAE_LAYER != 43:
        raise HTTPException(status_code=409, detail="This supply-chain demo requires SAE_LAYER=43")
    try:
        tokenizer = get_tokenizer()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not load the model tokenizer") from exc
    a, encoded_a = run_side(payload.a, tokenizer, engine, payload.top_k)
    b, encoded_b = run_side(payload.b, tokenizer, engine, payload.top_k)
    return {
        "a": a,
        "b": b,
        "tool_changed": a["tool"] != b["tool"] if a["tool"] and b["tool"] else None,
        "feature_changes": feature_changes(encoded_a, encoded_b, engine, payload.top_k),
        "layer": SAE_LAYER,
        "model": VLLM_MODEL,
        "sae_repo": SAE_REPO_ID,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provenance": PROVENANCE,
        "method": "Greedy continuation at the assistant prefill; observational SAE comparison.",
    }
