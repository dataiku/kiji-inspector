from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException

from app.config import SAE_LAYER, VLLM_CAPTURE_LAYERS, VLLM_MODEL, VLLM_URL
from app.dependencies import SAEEngine, get_engine
from app.inline import extract_inline_activation
from app.schemas import (
    BatchDescribeRequest,
    BatchDescribeResponse,
    DescribeByActivationRequest,
    DescribeBySampleResponseRequest,
    DescribeInlineRequest,
    DescribeRequest,
    DescribeResponse,
    InterpretRequest,
)

router = APIRouter()


def _extract_activation(payload: DescribeRequest) -> list[float]:
    if isinstance(payload, DescribeByActivationRequest):
        return payload.activation

    if isinstance(payload, DescribeBySampleResponseRequest):
        try:
            return payload.sample_response["choices"][0]["activations"][payload.activation_key]
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not extract activation from sample_response at "
                    f"choices[0].activations['{payload.activation_key}']"
                ),
            ) from exc

    raise HTTPException(status_code=422, detail="Unsupported request shape")


def _run_describe(payload: DescribeRequest, engine: SAEEngine) -> dict[str, Any]:
    activation = _extract_activation(payload)
    return engine.describe(activation=activation, top_k=payload.top_k)


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _describe_inline(response, token_index, top_k, engine):
    activation = extract_inline_activation(
        response, VLLM_CAPTURE_LAYERS, SAE_LAYER, token_index, engine.sae.d_model
    )
    return {
        "layer": SAE_LAYER,
        "token_index": token_index,
        **engine.describe(activation, top_k),
    }


@router.post("/describe/inline")
def describe_inline(payload: DescribeInlineRequest, engine: SAEEngine = Depends(get_engine)):
    try:
        return _describe_inline(payload.response, payload.token_index, payload.top_k, engine)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/interpret")
def interpret(payload: InterpretRequest, engine: SAEEngine = Depends(get_engine)):
    response = fetch_completion(payload)
    try:
        interpretation = _describe_inline(response, payload.token_index, payload.top_k, engine)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    # Avoid sending the full residual tensors back across the network again.
    response.pop("kv_transfer_params", None)
    return {"completion": response, "interpretation": interpretation}


def fetch_completion(payload: InterpretRequest) -> dict:
    """Request prompt activations and a completion from the co-located vLLM server."""
    body = dict(payload.request)
    if body.get("stream") or body.get("n", 1) != 1 or body.get("use_beam_search"):
        raise HTTPException(status_code=422, detail="Use stream=false, n=1, no beam search")
    body.setdefault("model", VLLM_MODEL)
    body.setdefault("max_tokens", 1)
    body.update(stream=False, n=1)
    body["kv_transfer_params"] = {"return_inline": True, "include_output_tokens": False}
    route = "/v1/chat/completions" if payload.route == "chat" else "/v1/completions"
    request = Request(
        VLLM_URL.rstrip("/") + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as upstream:
            response = json.load(upstream)
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="vLLM request failed") from exc
    return response


@router.post("/describe", response_model=DescribeResponse)
def describe(payload: DescribeRequest, engine: SAEEngine = Depends(get_engine)) -> dict[str, Any]:
    try:
        return _run_describe(payload, engine)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/describe/batch", response_model=BatchDescribeResponse)
def describe_batch(
    payload: BatchDescribeRequest, engine: SAEEngine = Depends(get_engine)
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for idx, item in enumerate(payload.items):
        try:
            results.append(_run_describe(item, engine))
        except HTTPException as exc:
            raise HTTPException(
                status_code=422,
                detail={"item_index": idx, "error": exc.detail},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"item_index": idx, "error": str(exc)},
            ) from exc

    return {"results": results}
