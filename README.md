
# Kiji Inspector: Mechanistic Interpretability for AI Agent Tool Selection

<div align="center">
  <img src="https://raw.githubusercontent.com/dataiku/kiji-inspector/main/static/kiji_inspector_workflow.png" alt="Kiji Inspector Workflow" width="600">

  <p>
    <a href="https://github.com/dataiku/kiji-inspector/actions/workflows/ci-core.yml"><img src="https://github.com/dataiku/kiji-inspector/actions/workflows/ci-core.yml/badge.svg" alt="CI Core"></a>
    <a href="https://github.com/dataiku/kiji-inspector/actions/workflows/ci-extras.yml"><img src="https://github.com/dataiku/kiji-inspector/actions/workflows/ci-extras.yml/badge.svg" alt="CI Extras"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue" alt="License: Apache 2.0"></a>
    <a href="https://github.com/dataiku/kiji-inspector/stargazers"><img src="https://img.shields.io/github/stars/dataiku/kiji-inspector?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/dataiku/kiji-inspector/issues"><img src="https://img.shields.io/github/issues/dataiku/kiji-inspector" alt="GitHub Issues"></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white" alt="Python Version">
    <a href="https://colab.research.google.com/github/dataiku/kiji-inspector/blob/main/demo/quickstart_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/LLMs-responsible-blue" alt="Responsible AI">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
  </p>
</div>

## Status
This project is **under heavy active development**. We are planning to release a stable version of the framework in the coming weeks.

In the meantime, join our [Slack Community](https://join.slack.com/t/dataiku-opensource/shared_invite/zt-3o6yq14rp-FTtAHZYhyru~jLZ~S6xPLA)

Learn more about our approach and early results:

* [Paper](paper/Opening%20the%20Black%20Box%20Mechanistic%20Interpretability%20of%20Agent%20Tool%20Selection%20with%20Sparse%20Autoencoders%20(color).pdf)
* [Presentation](presentation/Opening%20the%20Black%20Box%20Mechanistic%20Interpretability%20of%20Agent%20Tool%20Selection%20with%20Sparse%20Autoencoders.pdf)

---

## What This Project Does

This project trains **Sparse Autoencoders (SAEs)** on the internal activations of an AI agent to understand *why* it selects specific tools. Given a user request like "Search our docs for API limits," the agent must choose between tools (e.g., `internal_search` vs `web_search`). We extract the model's hidden representations at the moment of that decision, decompose them into interpretable features using a JumpReLU SAE, and validate the resulting explanations through automated fuzzing and causal ablation experiments.

The key insight: train the SAE on **raw activations** (not difference vectors), then use **contrastive pairs** post-hoc to identify which learned features correspond to specific tool-selection decisions. This preserves the SAE's general feature dictionary while enabling targeted analysis of decision-relevant features.

## Install

For loading and running pretrained SAEs:

```bash
pip install kiji-inspector
```

For the HuggingFace-based training and analysis extras (accelerate etc.):

```bash
pip install 'kiji-inspector[full]'
```

The vLLM extraction path is not covered by any extra: upstream vLLM wheels do
not ship the hidden-states connector, so it requires the Docker image built
from the repository [Dockerfile](Dockerfile), which compiles vLLM from the
[`Davidnet/vllm`](https://github.com/Davidnet/vllm) fork.

## Quick Start

```python
from kiji_inspector import SAE

sae, feature_descriptions = SAE.from_pretrained(
    base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    layer=20,
)

features = sae.encode(activations)
reconstruction = sae.decode(features)
```

Training and data-generation entrypoints live under the package namespace:

```bash
python -m kiji_inspector.generate_pairs 1300
python -m kiji_inspector.pipeline --layers 10 20 30
```

## vLLM hidden-state extraction (native connector)

Activation extraction uses vLLM's native `extract_hidden_states` speculator
method together with the `ExampleHiddenStatesConnector`, which writes captured
hidden states to safetensors files that the extractor loads and cleans up per
request. This capability ships in the `575lab/kiji-inspector:dev` image, which
builds the [`Davidnet/vllm`](https://github.com/Davidnet/vllm) fork
(branch `hidden-states-inline-return-squashed`); the public v0.19.0 wheel does
**not** contain the connector.

Run all extraction, tests, and API checks inside that image:

```bash
docker pull 575lab/kiji-inspector:dev

# Smoke test the connector (Qwen3-8B):
samples/run_hidden_states_test.sh 575lab/kiji-inspector:dev

# Iterate on the checked-out source with a bind mount:
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src \
  -w /workspace \
  575lab/kiji-inspector:dev \
  python -m pytest tests/test_activation_extractors.py
```

The historical `patches/` directory (applied against a stock v0.19.0 wheel)
predates the fork-based image and is retained for reference only; the current
image bakes those changes into the fork and does not run `apply-patch.sh`. See
[patches/README_PATCH.md](patches/README_PATCH.md) for the legacy workflow.

To run the pipeline against a Qwen3.6 subject model (a reasoning model), pass `--no-thinking` so the decision token sits at the final-answer position:

```bash
python -m kiji_inspector.pipeline --subject-model Qwen/Qwen3.6-35B-A3B --no-thinking
```

### Using a locally downloaded model

`--subject-model` also accepts a local model directory (paths must start with
`/`, `./`, `../`, or `~`; anything else is treated as a hub ID). Inside Docker
the directory must be volume-mounted, and you pass the **container** path:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -v /home/user/models:/models:ro \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src \
  -w /workspace \
  575lab/kiji-inspector:dev \
  python -m kiji_inspector.pipeline \
    --subject-model /models/programs/downloaded-model
```

Note: `SAE.from_pretrained(base_model=...)` resolves SAE repos through an
exact-match registry of hub IDs, so it won't match a local path — pass
`repo_id` directly instead.

---

## 📓 Examples

Two end-to-end notebooks demonstrate the library. Both run on Colab — the first works on a free T4, the second needs an A100 high-RAM runtime.

| Notebook | What it shows | Open |
|---|---|---|
| [`quickstart_colab.ipynb`](demo/quickstart_colab.ipynb) | Minimal walkthrough: capture a hidden state from `google/gemma-4-E4B-it` via a forward hook, load a pretrained SAE with `SAE.from_pretrained`, and describe the top features firing on a single prompt. Also covers the vLLM extraction path. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dataiku/kiji-inspector/blob/main/demo/quickstart_colab.ipynb) |
| [`home_repair_colab.ipynb`](demo/home_repair/home_repair_colab.ipynb) | Full agent demo: a Nemotron-3-Nano-30B home repair advisor calls four tools across three appliance problems, the residual stream is captured at every decision point, and a trained JumpReLU SAE decomposes those activations into themed features. Includes the interactive `index.html` viewer served from Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dataiku/kiji-inspector/blob/main/demo/home_repair/home_repair_colab.ipynb) |

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing a bug, improving documentation, or proposing a new feature, your help is appreciated.

### Ways to Contribute

- **Report Bugs** - [Open an issue](https://github.com/dataiku/kiji-inspector/issues) with steps to reproduce
- **Improve Docs** - Documentation PRs are always welcome
- **Submit Features** - Open an issue to discuss your idea before submitting a PR
- **Share Feedback** - [Start a discussion](https://github.com/dataiku/kiji-inspector/discussions)

### Community

- **Slack** - [Join our community](https://join.slack.com/t/dataiku-opensource/shared_invite/zt-3o6yq14rp-FTtAHZYhyru~jLZ~S6xPLA) to ask questions and connect with other contributors
- **Contributors** - See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the list of people who have contributed

---

## 📄 License

Copyright (c) 2026 Dataiku SAS

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
