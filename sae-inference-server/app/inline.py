"""Decode the pinned vLLM fork's inline HTTP tensors without importing vLLM."""

import base64
import binascii
import math

import torch


def extract_inline_activation(response, layers, layer, token_index, d_model):
    try:
        params = response["kv_transfer_params"]
        if not isinstance(params, dict):
            raise ValueError("Missing inline kv_transfer_params")
        if params.get("hidden_states_inline") is not True:
            raise ValueError("Request vLLM activations with return_inline=true")
        tensor = params["hidden_states"]
        if not isinstance(tensor, dict):
            raise ValueError("Malformed hidden_states tensor")
        shape = tensor["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(size) is not int or size <= 0 for size in shape)
            or shape[1] != len(layers)
            or shape[2] != d_model
        ):
            raise ValueError("Expected [tokens, configured capture layers, SAE d_model]")
        if layer not in layers or len(set(layers)) != len(layers):
            raise ValueError("SAE layer must occur once in VLLM_CAPTURE_LAYERS")
        if not -shape[0] <= token_index < shape[0]:
            raise ValueError("token_index is outside the captured prompt")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(tensor["dtype"])
        if dtype is None:
            raise ValueError("Unsupported hidden-state dtype")
        raw = base64.b64decode(tensor["data"], validate=True)
        width = torch.empty((), dtype=dtype).element_size()
        if len(raw) != math.prod(shape) * width:
            raise ValueError("Hidden-state byte length does not match shape and dtype")
        # Only materialize the requested token/layer vector, not the full tensor.
        offset = ((token_index % shape[0]) * shape[1] + layers.index(layer)) * d_model * width
        vector = torch.frombuffer(bytearray(raw[offset : offset + d_model * width]), dtype=dtype)
        return vector.float().tolist()
    except (KeyError, TypeError, binascii.Error) as exc:
        raise ValueError("Malformed vLLM inline hidden-state response") from exc
