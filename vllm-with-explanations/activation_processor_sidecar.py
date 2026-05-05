#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.136.1",
#   "kiji-inspector",
#   "uvicorn>=0.46.0",
# ]
#
# [tool.uv.sources]
# kiji-inspector = { git = "https://github.com/dataiku/kiji-inspector", branch = "feat/new-vllm-gemma4" }
# ///
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from kiji_inspector import SAE
from kiji_inspector.core.sae_core import JumpReLUSAE
from pydantic import BaseModel, Field


@dataclass
class LayerSAE:
    sae: SAE
    feature_descriptions: dict[int | str, str]
    device: str


class ActivationExplanationRequest(BaseModel):
    request_id: str
    model: str
    choice_index: int | None = None
    activations: dict[int, list[float]] = Field(default_factory=dict)


class ActivationExplanationResponse(BaseModel):
    explanations: dict[int, list["FeatureExplanation"]]


class FeatureExplanation(BaseModel):
    feature_id: int
    description: Any
    activation: float


app = FastAPI()


@app.post("/v1/activation_explanations")
async def create_activation_explanations(
    request: ActivationExplanationRequest,
) -> ActivationExplanationResponse:
    explanations: dict[int, list[FeatureExplanation]] = {}
    layer_saes: dict[int, LayerSAE] = app.state.layer_saes
    top_k: int = app.state.top_k

    for layer_idx, vector in request.activations.items():
        if layer_idx not in layer_saes:
            raise ValueError(
                f"No SAE was loaded for activation layer {layer_idx}. "
                f"Loaded layers: {sorted(layer_saes)}"
            )

        layer_sae = layer_saes[layer_idx]
        sae = layer_sae.sae
        x = torch.tensor(vector, device=layer_sae.device, dtype=sae.dtype)
        top_features = sae.describe(
            x,
            layer_sae.feature_descriptions,
            top_k=top_k,
        )
        explanations[layer_idx] = [
            FeatureExplanation(
                feature_id=feature_id,
                description=description,
                activation=float(activation),
            )
            for feature_id, description, activation in top_features
        ]

    return ActivationExplanationResponse(explanations=explanations)


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    seen_layers: set[int] = set()
    for raw_layer in value.replace(",", " ").split():
        layer_idx = int(raw_layer)
        if layer_idx < 0:
            raise ValueError("Activation layers must be non-negative integers.")
        if layer_idx not in seen_layers:
            layers.append(layer_idx)
            seen_layers.add(layer_idx)
    if not layers:
        raise ValueError("At least one activation layer must be configured.")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Activation processor sidecar for vLLM activation outputs."
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get(
            "ACTIVATION_PROCESSOR_SOCKET",
            "/tmp/vllm-activation-processor.sock",
        ),
        help="Unix domain socket path to bind.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ACTIVATION_PROCESSOR_LOG_LEVEL", "info"),
        help="Uvicorn log level.",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get(
            "ACTIVATION_BASE_MODEL",
            os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it"),
        ),
        help="Base model name used to load kiji-inspector SAEs.",
    )
    parser.add_argument(
        "--layers",
        default=os.environ.get("ACTIVATION_LAYERS", "8"),
        help="Space- or comma-separated activation layers to load SAEs for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("ACTIVATION_EXPLANATION_TOP_K", "5")),
        help="Number of active SAE features to return per layer.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("ACTIVATION_PROCESSOR_DEVICE", "cpu"),
        help="Device for SAE inference: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("ACTIVATION_PROCESSOR_CACHE_DIR"),
        help="Optional cache directory for SAE downloads.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for SAE downloads.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_layer_saes(args: argparse.Namespace) -> dict[int, LayerSAE]:
    device = resolve_device(args.device)
    layer_saes: dict[int, LayerSAE] = {}
    for layer_idx in parse_layers(args.layers):
        sae, feature_descriptions = SAE.from_pretrained(
            base_model=args.base_model,
            layer=layer_idx,
            device=device,
            cache_dir=args.cache_dir,
            token=args.hf_token,
        )
        layer_saes[layer_idx] = LayerSAE(
            sae=sae,
            feature_descriptions=feature_descriptions or {},
            device=device,
        )
        print(
            f"Loaded SAE for {args.base_model} layer {layer_idx} "
            f"on {device}: d_model={sae.d_model}, d_sae={sae.d_sae}",
            flush=True,
        )
    return layer_saes


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    app.state.layer_saes = load_layer_saes(args)
    app.state.top_k = args.top_k

    socket_dir = os.path.dirname(args.socket)
    if socket_dir:
        os.makedirs(socket_dir, exist_ok=True)
    if os.path.exists(args.socket):
        os.unlink(args.socket)

    uvicorn.run(
        app,
        uds=args.socket,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
