"""Sidecar HTTP contract, using a tiny deterministic SAE and no downloads/GPU."""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sae-inference-server"))

from app import demo, main
from app.api import routes
from app.dependencies import SAEEngine, get_engine
from app.inline import extract_inline_activation

from kiji_inspector.core.sae_core import JumpReLUSAE


def inline_response():
    hidden = torch.zeros(2, 6, 2, dtype=torch.bfloat16)
    hidden[-1, -1] = torch.tensor([5, 8])
    return {
        "choices": [{"message": {"content": "tool"}}],
        "kv_transfer_params": {
            "hidden_states_inline": True,
            "hidden_states": {
                "dtype": "bfloat16",
                "shape": list(hidden.shape),
                "data": base64.b64encode(hidden.view(torch.uint8).numpy().tobytes()).decode(),
            },
        },
    }


@pytest.fixture
def client(monkeypatch):
    sae = JumpReLUSAE(2, 2, dtype=torch.float32)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2))
    sae.mean_vec = np.array([1, 2], dtype=np.float32)
    sae.rms_scale = 2
    engine = SAEEngine(sae, {"1": "second feature"})
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    monkeypatch.setattr(routes, "SAE_LAYER", 43)
    monkeypatch.setattr(routes, "VLLM_CAPTURE_LAYERS", [6, 13, 20, 27, 34, 43])
    app = main.create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client


def test_inline_selects_layer_and_token_and_normalizes(client):
    response = client.post("/describe/inline", json={"response": inline_response(), "top_k": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["layer"] == 43
    assert result["num_active_features"] == 2  # Total, not the truncated top-k count.
    assert result["top_features"][0]["activation"] == 3  # (8 - 2) / 2
    assert result["top_features"][0]["description"]["label"] == "second feature"


@pytest.mark.parametrize("change", ["shape", "data", "dtype", "length"])
def test_malformed_inline_tensor_is_rejected(client, change):
    response = inline_response()
    tensor = response["kv_transfer_params"]["hidden_states"]
    tensor[change if change != "length" else "data"] = {
        "shape": [2, 5, 2],
        "data": "!invalid!",
        "dtype": "int64",
        "length": "AA==",
    }[change]
    assert client.post("/describe/inline", json={"response": response}).status_code == 422


def test_wrong_token_and_missing_sae_layer_are_rejected(client):
    response = inline_response()
    assert (
        client.post(
            "/describe/inline",
            json={
                "response": response,
                "token_index": 2,
            },
        ).status_code
        == 422
    )
    with pytest.raises(ValueError, match="SAE layer"):
        extract_inline_activation(response, [6, 13, 20, 27, 34, 42], 43, -1, 2)


def test_missing_inline_payload_is_rejected(client):
    assert (
        client.post(
            "/describe/inline",
            json={
                "response": {"kv_transfer_params": None},
            },
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "route,path", [("chat", "/v1/chat/completions"), ("completion", "/v1/completions")]
)
def test_interpret_requests_inline_and_returns_features(client, monkeypatch, route, path):
    def upstream(request, timeout):
        assert request.full_url.endswith(path)
        body = json.loads(request.data)
        assert body["kv_transfer_params"] == {"return_inline": True, "include_output_tokens": False}
        assert body["stream"] is False and body["n"] == 1
        return io.BytesIO(json.dumps(inline_response()).encode())

    monkeypatch.setattr(routes, "urlopen", upstream)
    response = client.post("/interpret", json={"route": route, "request": {"prompt": "test"}})
    assert response.status_code == 200
    result = response.json()
    assert "kv_transfer_params" not in result["completion"]
    assert result["interpretation"]["top_features"][0]["activation"] == 3


def test_streaming_and_upstream_failures(client, monkeypatch):
    assert client.post("/interpret", json={"request": {"stream": True}}).status_code == 422

    def failed(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(routes, "urlopen", failed)
    assert client.post("/interpret", json={"request": {"prompt": "test"}}).status_code == 502


def test_raw_activation_validation_and_health(client):
    assert client.get("/healthz").status_code == 200
    assert client.post("/describe", json={"activation": [1]}).status_code == 422
    response = client.post("/describe", json={"activation": [5, 8], "top_k": 1})
    assert response.status_code == 200
    assert response.json()["top_features"][0]["activation"] == 3


def test_demo_uses_frozen_prompts_without_historical_scores(client):
    response = client.get("/demo/config")
    assert response.status_code == 200
    config = response.json()
    assert len(config["pairs"]) == 4
    assert len(config["provenance"]["commit"]) == 40
    first = config["pairs"][0]
    assert (
        first["b"][0]["request"]
        == "Adjust inventory for items with rising forecasted customer order frequency"
    )
    assert first["b"][-1]["label"] == "Keyword control"
    assert "prob" not in json.dumps(config["pairs"])
    assert client.get("/demo").status_code == 200


def test_demo_preserves_decision_prompt_and_compares_full_codes(client, monkeypatch):
    class Tokenizer:
        chat_template = "enable_thinking"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            assert "Available tools:" in messages[0]["content"]
            return messages[0]["content"] + "\n" + messages[1]["content"] + "\n<think>"

    monkeypatch.setattr(demo, "SAE_LAYER", 43)
    monkeypatch.setattr(demo, "get_tokenizer", lambda: Tokenizer())
    captured = []

    def complete(payload):
        assert payload.route == "completion"
        assert payload.request["add_special_tokens"] is False
        prompt = payload.request["prompt"]
        assert prompt.endswith("</think>\n\nI'll use the")
        captured.append(prompt)
        response = inline_response()
        response["choices"] = [{"text": " inventory_manager"}]
        if "forecasted" in prompt:
            hidden = torch.zeros(2, 6, 2, dtype=torch.bfloat16)
            hidden[-1, -1] = torch.tensor([15, 6])
            response["kv_transfer_params"]["hidden_states"]["data"] = base64.b64encode(
                hidden.view(torch.uint8).numpy().tobytes()
            ).decode()
            response["choices"] = [{"text": " demand_forecaster"}]
        return response

    monkeypatch.setattr(demo, "fetch_completion", complete)
    result = client.post(
        "/demo/compare", json={"a": "rising orders", "b": "forecasted orders", "top_k": 1}
    )
    assert result.status_code == 200
    body = result.json()
    assert len(captured) == 2
    assert body["tool_changed"] is True
    assert body["a"]["top_features"][0]["feature_id"] == "1"
    # Feature 0 is below A's top-1, but the comparison must still use its true value.
    assert body["feature_changes"][0] == {
        "feature_id": "0",
        "description": None,
        "a": 2.0,
        "b": 7.0,
        "delta": 5.0,
    }
    assert "kv_transfer_params" not in body["a"]


def test_demo_wrong_layer_unknown_tools_and_failure(client, monkeypatch):
    monkeypatch.setattr(demo, "SAE_LAYER", 34)
    assert client.post("/demo/compare", json={"a": "a", "b": "b"}).status_code == 409
    assert demo.generated_tool(" demand forecaster tool") == "demand_forecaster"
    assert demo.generated_tool("Maybe inventory_manager could help") is None
    assert demo.generated_tool("inventory_managerial") is None
    monkeypatch.setattr(demo, "SAE_LAYER", 43)

    def fail():
        raise OSError("download unavailable")

    monkeypatch.setattr(demo, "get_tokenizer", fail)
    assert client.post("/demo/compare", json={"a": "a", "b": "b"}).status_code == 503
