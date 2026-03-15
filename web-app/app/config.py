from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_PATH = ASSETS_DIR / "sae_final.pt"
FEATURE_DESCRIPTIONS_PATH = ASSETS_DIR / "feature_descriptions.json"
