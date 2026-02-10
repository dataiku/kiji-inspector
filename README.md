
# Kiji Inspector: Mechanistic Interpretability for AI Agent Tool Selection

## What This Project Does

This project trains **Sparse Autoencoders (SAEs)** on the internal activations of an AI agent to understand *why* it selects specific tools. Given a user request like "Search our docs for API limits," the agent must choose between tools (e.g., `internal_search` vs `web_search`). We extract the model's hidden representations at the moment of that decision, decompose them into interpretable features using a JumpReLU SAE, and validate the resulting explanations through automated fuzzing and causal ablation experiments.

The key insight: train the SAE on **raw activations** (not difference vectors), then use **contrastive pairs** post-hoc to identify which learned features correspond to specific tool-selection decisions. This preserves the SAE's general feature dictionary while enabling targeted analysis of decision-relevant features.

## Status
This project is under active development. We are planning to release the framework in the coming weeks.

In the meantime, join our [Slack Community](https://join.slack.com/t/dataiku-opensource/shared_invite/zt-3o6yq14rp-FTtAHZYhyru~jLZ~S6xPLA)
