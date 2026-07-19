"""Dataset loading utilities.

Uses the `dair-ai/emotion` dataset (English tweets labeled with 6 emotions).
Downloads parquet splits from the HuggingFace Hub on first run and caches
them as CSVs under ``data/``.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, HF_BASE_URL, HF_FILES, LABELS


def download_split(split: str) -> pd.DataFrame:
    """Download one split from the HuggingFace Hub."""
    if split not in HF_FILES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(HF_FILES)}")
    return pd.read_parquet(HF_BASE_URL + HF_FILES[split])


def load_split(split: str, force_download: bool = False) -> pd.DataFrame:
    """Load a split from the local cache, downloading it if missing.

    Returns a DataFrame with columns: ``text`` (str), ``label`` (int),
    ``label_name`` (str).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{split}.csv"
    if force_download or not cache.exists():
        df = download_split(split)
        df.to_csv(cache, index=False)
    else:
        df = pd.read_csv(cache)

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(dict(enumerate(LABELS)))
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) DataFrames."""
    return load_split("train"), load_split("validation"), load_split("test")
