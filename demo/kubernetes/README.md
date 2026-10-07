# Live supply-chain interpretation on Kubernetes

This demo adapts `steering-demos`' supply-chain pairs to the vLLM + CPU SAE
sidecar in [`deploy/vllm.yml`](../../deploy/vllm.yml). Open **`/demo` on port
8001** to compare requests, see the model's generated tool, inspect layer-43
SAE features, and download the run as JSON.

The selected source is pinned in
[`provenance.json`](../../sae-inference-server/app/demo_data/provenance.json).
Pairs, controls, and scenario tool definitions are copied from that commit.
The UI uses no historical scores or feature indices. It uses the published
Nemotron 3.5 Lightning SAE downloaded by the sidecar.

## Deploy and open

Build the updated sidecar from the repository root. The HTML, Dataiku logos,
local Signifier/Roboto fonts, and frozen demo inputs are packaged inside the
image; no volume mounts or extra services are
needed. Replace `YOUR_REGISTRY` with a registry your cluster can pull from:

```bash
docker build --platform linux/amd64 -f sae-inference-server/Dockerfile \
  -t YOUR_REGISTRY/kiji-inspector-sae:nutanix-demo .
docker push YOUR_REGISTRY/kiji-inspector-sae:nutanix-demo
```

Set the `sae` container's image in `deploy/vllm.yml` to that reference. Keep
`SAE_LAYER=43` and both containers' capture-layer order consistent. Then:

```bash
kubectl apply -f deploy/vllm.yml
kubectl rollout status deployment/kiji-vllm --timeout=30m
kubectl port-forward service/kiji-vllm 8000:8000 8001:8001
# Open http://localhost:8001/demo
```

For another namespace, add `-n YOUR_NAMESPACE` to each kubectl command. The
deployment targets an 80 GB NVIDIA GPU for vLLM; the sidecar uses CPU. Its
first comparison also downloads the model tokenizer (not the model weights).

Before presenting, verify the real deployment's inline tensors from another
terminal using the branch's activation smoke test:

```bash
python samples/test_deployment_activations.py --url http://localhost:8000
```

It checks tensor shapes, byte lengths, token alignment, and finite, nonzero
activation values. This complements the synthetic tests below.

## A five-minute walkthrough

1. **Demand pull vs supply push.** Compare the original requests. B adds
   `forecasted` to A's customer-order-frequency request. Read the generated
   tool names first, then inspect the largest feature changes. The earlier
   branch run routed these toward inventory management and demand forecasting;
   this page reports whatever the deployed checkpoint actually does.
2. **Test meaning.** Change B to *Paraphrase 1* and run again. Does the routing
   and feature pattern survive when the exact cue word disappears?
3. **Test a keyword distraction.** Change B to *Keyword control*. It uses
   `forecasted` in a budget reference while preserving A's actual-demand
   request. Does the model follow the task or the distracting word? Feature
   labels can be imperfect, so use this to investigate their meaning.
4. **Local vs global sourcing.** Repeat with the supplier-delivery pair and
   its controls. The economical/express and single/multiple-supplier pairs
   offer two more examples, including less clear outcomes.
5. **Keep the evidence.** Download the run JSON. It includes both requests,
   formatted prompts, generated continuations, active feature counts, exact
   feature differences, UTC timestamp, model/SAE IDs, and the source commit.

## How the live measurements work

`POST /demo/compare` builds the same scenario prompt as the branch using
`build_agent_prompt` and model-specific chat-template options. It ends with
`I'll use the` **without trailing whitespace**. Each side is sent separately
to vLLM's `/v1/completions`, with inline prompt-state return, greedy decoding,
and `add_special_tokens=false` because the prompt is already templated.

The sidecar reads the last **prompt** token's residual at layer 43, applies
the SAE's stored mean and RMS normalization, and encodes it. Feature changes
are computed over the full code, before truncating for display; a feature
missing from a top-k panel is not assumed to have zero activation.

The chosen tool is parsed from the leading tool name in the generated text.
If it is not recognized, the page displays that uncertainty and the raw
continuation. This is a greedy-generation readout. The source branch's
token-tree probability measurement is not implemented here, so the UI does
not display probabilities or claim numerical parity with that measurement.

The comparison is **observational**. The source branch's ablation and
cross-patching runs use Hugging Face forward hooks, which this vLLM serving
deployment does not expose. No causal steering scores are carried into the
live page. The source also uses an MTP-free local GA checkpoint; matching the
public model name is not proof of checkpoint or numerical parity.

The customer-support demo was deliberately left out: its recommended SAE
layer is **34**, and the branch reports a degenerate feature readout at layer
43 for that scenario. This deployment's demo remains focused on supply chain.

## Troubleshooting

```bash
kubectl logs deployment/kiji-vllm -c sae --tail=100
kubectl logs deployment/kiji-vllm -c vllm --tail=100
curl --fail http://localhost:8001/healthz
```

An image pull error means the sidecar image reference or registry access needs
attention. A tokenizer error requires Hugging Face access from the sidecar.
If `/demo/compare` reports invalid capture data, confirm vLLM uses the fork
with inline HTTP returns and that `VLLM_CAPTURE_LAYERS` matches its layer axis.
Changing `SAE_LAYER` away from 43 makes this demo return a clear configuration
error. Inspect unusual feature labels as hypotheses, using the controls.

## Local verification

With the SAE server's FastAPI dependencies and pytest installed:

```bash
python -m pytest tests/test_sae_sidecar.py tests/test_prompt_building.py -q
```

The tests use synthetic tensors and a small deterministic SAE. They verify
the request/response flow, prompt position, layer selection, normalization,
full-code differences, and failures without downloading the real model.
GPU inference and the exact behavior of the deployed public checkpoint must
be checked in your cluster.
