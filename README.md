
# Kiji Inspector: Mechanistic Interpretability for AI Agent Tool Selection

<div align="center">
  <img src="static/kiji_inspector_inverted.png" alt="Kiji Inspector" width="300">

  <p>
    <!--<a href="https://github.com/dataiku/kiji-inspector/actions/workflows/lint-and-test.yml"><img src="https://github.com/dataiku/kiji-inspector/actions/workflows/lint-and-test.yml/badge.svg" alt="Lint & Test"></a>-->
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue" alt="License: Apache 2.0"></a>
    <a href="https://github.com/dataiku/kiji-inspector/stargazers"><img src="https://img.shields.io/github/stars/dataiku/kiji-inspector?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/dataiku/kiji-inspector/issues"><img src="https://img.shields.io/github/issues/dataiku/kiji-inspector" alt="GitHub Issues"></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/python-%3E%3D3.13-3776AB?logo=python&logoColor=white" alt="Python Version">
  </p>

  <p>
    <img src="https://img.shields.io/badge/LLMs-responsible-blue" alt="Responsible AI">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
  </p>
</div>

## Status
This project is under active development. We are planning to release the framework in the coming weeks.

In the meantime, join our [Slack Community](https://join.slack.com/t/dataiku-opensource/shared_invite/zt-3o6yq14rp-FTtAHZYhyru~jLZ~S6xPLA)

Learn more about our approach and early results:

* [Paper](paper/Opening%20the%20Black%20Box%20Mechanistic%20Interpretability%20of%20Agent%20Tool%20Selection%20with%20Sparse%20Autoencoders.pdf)
* [Presentation](presentation/Opening%20the%20Black%20Box%20Mechanistic%20Interpretability%20of%20Agent%20Tool%20Selection%20with%20Sparse%20Autoencoders.pdf)

---

## What This Project Does

This project trains **Sparse Autoencoders (SAEs)** on the internal activations of an AI agent to understand *why* it selects specific tools. Given a user request like "Search our docs for API limits," the agent must choose between tools (e.g., `internal_search` vs `web_search`). We extract the model's hidden representations at the moment of that decision, decompose them into interpretable features using a JumpReLU SAE, and validate the resulting explanations through automated fuzzing and causal ablation experiments.

The key insight: train the SAE on **raw activations** (not difference vectors), then use **contrastive pairs** post-hoc to identify which learned features correspond to specific tool-selection decisions. This preserves the SAE's general feature dictionary while enabling targeted analysis of decision-relevant features.

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
