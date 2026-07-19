"""Central configuration for MoodLens."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "moodlens_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

HF_BASE_URL = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split/"
HF_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_EMOJI = {
    "sadness": "😢", "joy": "😄", "love": "❤️",
    "anger": "😠", "fear": "😨", "surprise": "😲",
}

RANDOM_SEED = 42
