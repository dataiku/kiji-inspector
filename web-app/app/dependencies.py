from functools import lru_cache

from app.config import FEATURE_DESCRIPTIONS_PATH, MODEL_PATH
from app.services.sae_inference import SAEInference


@lru_cache
def get_engine() -> SAEInference:
    return SAEInference(MODEL_PATH, FEATURE_DESCRIPTIONS_PATH)
