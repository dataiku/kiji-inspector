from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import SAEEngine, get_engine
from app.schemas import (
    BatchDescribeRequest,
    BatchDescribeResponse,
    DescribeByActivationRequest,
    DescribeBySampleResponseRequest,
    DescribeRequest,
    DescribeResponse,
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
