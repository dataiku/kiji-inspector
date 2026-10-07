# SAE Feature Description Server

This project provides a FastAPI server for generating descriptions of features from a Sparse Autoencoder (SAE).

## Features

*   **FastAPI Server**: A lightweight and fast web server.
*   **Health Check**: A `/healthz` endpoint to monitor the server's status.
*   **Describe Endpoint**: A `/describe` endpoint that accepts a feature activation (a vector of floats) and returns a human-readable description.
*   **Batch Describe Endpoint**: A `/describe/batch` endpoint for processing multiple describe requests in a single call.
*   **Flexible Input**: The server can extract feature activations from different JSON request structures.

## API

### `/healthz`

*   **Method**: `GET`
*   **Description**: Returns the health status of the server.
*   **Success Response**: `{"status": "ok"}`

### `/describe`

*   **Method**: `POST`
*   **Description**: Describes a single feature activation.
*   **Request Body**: See `app/schemas.py` for the detailed request and response models. The server can accept `DescribeByActivationRequest` or `DescribeBySampleResponseRequest`.
*   **Success Response**: A JSON object containing the description.

### `/describe/batch`

*   **Method**: `POST`
*   **Description**: Describes a batch of feature activations.
*   **Request Body**: A JSON object containing a list of describe requests. See `app/schemas.py`.
*   **Success Response**: A JSON object containing a list of description results.

## Setup and Running

1.  **Create the virtual environment and install dependencies**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    ```

2.  **Run the server**:
    ```bash
    python main.py
    ```

The server will be available at `http://localhost:8000`.

## Kubernetes sidecar and inline activations

`GET /demo` serves the live supply-chain comparison UI. Its frozen pairs and
controls come from `steering-demos`; `POST /demo/compare` builds the matching
decision prompts and compares layer-43 activations from two vLLM calls.
See the [walkthrough](../demo/kubernetes/README.md) for setup and interpretation.

The [deployment guide](../README.md#sae-sidecar) includes image build steps and
examples for the sidecar on port **8001**. Build `Dockerfile` from the repository
root; it includes this app on top of `575lab/kiji-inspector:dev`.

- `POST /describe/inline`: accepts `{"response": <vLLM HTTP response>,
  "token_index": -1, "top_k": 10}`. Decodes the base64 tensor, selects the
  configured SAE layer and token, normalizes, and encodes it.
- `POST /interpret`: accepts `{"request": <vLLM request body>, "route": "chat",
  "token_index": -1, "top_k": 10}`. Calls vLLM with inline capture enabled and
  returns the completion plus SAE interpretation. `route="completion"` uses
  `/v1/completions` for preformatted prompts.
- `GET /healthz`: available once the startup checkpoint download/load finishes.

Set `SAE_REPO_ID`, `SAE_LAYER`, and `SAE_DEVICE=cpu` to choose the SAE.
`VLLM_URL` defaults to `http://127.0.0.1:8000`; `VLLM_MODEL` selects the served
model name. `VLLM_CAPTURE_LAYERS` is a comma-separated list whose order must
match the vLLM capture configuration. The deployment sets layer 43 from the
Nemotron 3.5 Lightning SAE repository.

All describe endpoints accept **raw residual vectors** and apply training
normalization before encoding. `num_active_features` counts all positive SAE
features, even when `top_k` truncates the returned list. These are observational
feature readouts; the sidecar does not modify vLLM's generation.
