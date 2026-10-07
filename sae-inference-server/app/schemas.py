from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class DescribeByActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation: list[float] = Field(description="Raw activation vector matching SAE d_model size.")
    top_k: int = Field(default=10, ge=1, le=200)


class DescribeBySampleResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_response: dict[str, Any] = Field(
        description="Full upstream response containing choices[0].activations."
    )
    activation_key: str = Field(
        default="20",
        description="Key under choices[0].activations to extract the vector from.",
    )
    top_k: int = Field(default=10, ge=1, le=200)


DescribeRequest: TypeAlias = DescribeByActivationRequest | DescribeBySampleResponseRequest


class DescribeInlineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: dict[str, Any] = Field(description="Full vLLM inline HTTP response.")
    token_index: int = Field(default=-1, description="Prompt token position; -1 selects the last.")
    top_k: int = Field(default=10, ge=1, le=200)


class InterpretRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: dict[str, Any] = Field(description="vLLM completion or chat request body.")
    route: str = Field(default="chat", pattern="^(chat|completion)$")
    token_index: int = -1
    top_k: int = Field(default=10, ge=1, le=200)


class FeatureDescription(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    description: str
    confidence: str | None = None
    mean_activation: float | None = None
    max_activation: float | None = None
    frac_nonzero: float | None = None
    top_examples: list[str] = Field(default_factory=list)
    bottom_examples: list[str] = Field(default_factory=list)


class FeatureResult(BaseModel):
    feature_id: str
    activation: float
    description: FeatureDescription | None


class DescribeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_features: list[FeatureResult]
    num_active_features: int


class BatchDescribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[DescribeRequest] = Field(min_length=1, max_length=256)


class BatchDescribeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[DescribeResponse]
