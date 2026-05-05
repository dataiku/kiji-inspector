# vLLM Activations + Explanations

This project patches `vllm==0.20.1` so the OpenAI-compatible server can capture
prompt hidden-state vectors for selected Gemma 4 layers. Requests can ask vLLM to
send those vectors to a separate Unix-socket sidecar and return the processed
result as `choices[].explanations`.

## Start The Server

Install/sync the environment:

```bash
uv sync
```

Apply the vLLM patch to the installed package:

```bash
PYTHON=.venv/bin/python patches/0.20.1/apply_patch.sh
```

Start the activation processor sidecar in one terminal:

```bash
./serve_activation_processor.sh
```

By default this listens on:

```text
/tmp/vllm-activation-processor.sock
```

Start the Gemma 4 vLLM server in another terminal:

```bash
./serve_gemma4_activations.sh
```

The server starts `google/gemma-4-E4B-it` with activation extraction enabled for
layer `8`, and listens on:

```text
http://127.0.0.1:8000
```

## Start With Docker

Build the image:

```bash
docker build -t vllm-extras-activations .
```

Run the container with GPU access:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  vllm-extras-activations
```

The Docker entrypoint starts both processes:

1. `activation_processor_sidecar.py` on `/tmp/vllm-activation-processor.sock`
2. `vllm serve` on `0.0.0.0:8000`

Useful runtime overrides:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MODEL_NAME="google/gemma-4-E4B-it" \
  -e ACTIVATION_LAYERS="8" \
  -e ACTIVATION_EXPLANATION_TOP_K="5" \
  -e TENSOR_PARALLEL_SIZE="1" \
  -e VLLM_DTYPE="bfloat16" \
  vllm-extras-activations
```

Extra vLLM CLI flags can be appended after the image name:

```bash
docker run --rm --gpus all -p 8000:8000 \
  vllm-extras-activations \
  --max-model-len 8192
```

## Test Explanations

Send a request with `return_explanations: true`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [
      {"role": "user", "content": "Say hi in one word."}
    ],
    "max_tokens": 2,
    "temperature": 0,
    "return_explanations": true
  }'
```

The response includes processed sidecar output. For each activation layer, the
sidecar returns the top active SAE features from `kiji-inspector`:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello"
      },
      "explanations": {
        "8": [
          {
            "feature_id": 123,
            "description": "example feature description",
            "activation": 1.234
          }
        ]
      }
    }
  ]
}
```

Raw activation vectors are not returned unless the request also includes:

```json
{
  "return_activations": true
}
```

## Modify The Sidecar

The sidecar implementation is in:

```text
activation_processor_sidecar.py
```

At startup, the sidecar loads one `kiji-inspector` SAE for each configured layer.
It uses the captured activation vector to return the top `k` feature
descriptions. Configure `k` with:

```bash
ACTIVATION_EXPLANATION_TOP_K=10 ./serve_activation_processor.sh
```
