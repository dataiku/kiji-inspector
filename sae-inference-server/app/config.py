import os

# SAE loading configuration — either from HuggingFace or a local checkpoint.
# Set SAE_BASE_MODEL to load from HF registry, or SAE_REPO_ID for a specific repo.
# Set SAE_CHECKPOINT_PATH to load from a local .pt file instead.
SAE_BASE_MODEL = os.environ.get("SAE_BASE_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
SAE_REPO_ID = os.environ.get("SAE_REPO_ID")
SAE_LAYER = int(os.environ.get("SAE_LAYER", "20"))
SAE_DEVICE = os.environ.get("SAE_DEVICE", "cpu")
SAE_CHECKPOINT_PATH = os.environ.get("SAE_CHECKPOINT_PATH")
# Order of the layer axis in vLLM's inline hidden-state tensor.
VLLM_CAPTURE_LAYERS = [
    int(value) for value in os.environ.get("VLLM_CAPTURE_LAYERS", "6,13,20,27,34,43").split(",")
]
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16")
