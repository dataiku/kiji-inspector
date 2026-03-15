import os

# SAE loading configuration — either from HuggingFace or a local checkpoint.
# Set SAE_BASE_MODEL to load from HF registry, or SAE_REPO_ID for a specific repo.
# Set SAE_CHECKPOINT_PATH to load from a local .pt file instead.
SAE_BASE_MODEL = os.environ.get("SAE_BASE_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
SAE_REPO_ID = os.environ.get("SAE_REPO_ID")
SAE_LAYER = int(os.environ.get("SAE_LAYER", "20"))
SAE_DEVICE = os.environ.get("SAE_DEVICE", "cpu")
SAE_CHECKPOINT_PATH = os.environ.get("SAE_CHECKPOINT_PATH")
