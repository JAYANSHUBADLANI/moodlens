"""Inference: load the trained pipeline and predict emotions.

CLI:  python -m moodlens.predict "i cant believe this actually worked"
"""
from __future__ import annotations

import sys
from functools import lru_cache

import joblib

from .config import LABEL_EMOJI, LABELS, MODEL_PATH


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m moodlens.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict(text: str, top_k: int = 3) -> dict:
    """Predict the emotion of a single text.

    Returns: {"text", "emotion", "confidence", "top": [{"emotion", "prob"}, ...]}
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    model = get_model()
    probs = model.predict_proba([text])[0]
    ranked = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)
    best_label, best_prob = ranked[0]
    return {
        "text": text,
        "emotion": best_label,
        "confidence": round(float(best_prob), 4),
        "top": [
            {"emotion": lab, "prob": round(float(p), 4)}
            for lab, p in ranked[:top_k]
        ],
    }


def predict_batch(texts: list[str], top_k: int = 3) -> list[dict]:
    return [predict(t, top_k=top_k) for t in texts]


def main() -> None:
    texts = sys.argv[1:] or [
        "i cant stop smiling today everything is going right",
        "why would they cancel it without even telling us",
        "im terrified about the results tomorrow",
    ]
    for text in texts:
        res = predict(text)
        emoji = LABEL_EMOJI.get(res["emotion"], "")
        top = ", ".join(f"{t['emotion']}={t['prob']:.2f}" for t in res["top"])
        print(f"{emoji} {res['emotion']:<9} ({res['confidence']:.2f})  {text}")
        print(f"   top-3: {top}")


if __name__ == "__main__":
    main()
