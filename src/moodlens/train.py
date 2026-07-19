"""Train and evaluate the MoodLens emotion classifier.

Pipeline: TF-IDF (word 1-2 grams + char 3-5 grams) -> Linear SVM,
calibrated with Platt scaling so the API can return probabilities.

Run:  python -m moodlens.train
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import (
    CONFUSION_MATRIX_PATH,
    LABELS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from .data import load_all


def build_pipeline() -> Pipeline:
    """TF-IDF features (word + char n-grams) feeding a calibrated linear SVM."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )
    clf = CalibratedClassifierCV(
        estimator=LinearSVC(C=0.5, random_state=RANDOM_SEED),
        method="sigmoid",
        cv=3,
    )
    return Pipeline([("features", features), ("clf", clf)])


def evaluate(pipeline: Pipeline, X, y, split_name: str) -> dict:
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")
    print(f"\n== {split_name} ==")
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")
    print(classification_report(y, preds, target_names=LABELS, digits=3))
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "report": classification_report(
            y, preds, target_names=LABELS, output_dict=True
        ),
        "predictions": preds,
    }


def plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False
    )
    ax.set_title("MoodLens: Test Confusion Matrix (row-normalized)")
    plt.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train, val, test = load_all()
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    pipeline = build_pipeline()

    print("Training ...")
    t0 = time.time()
    pipeline.fit(train["text"], train["label"])
    train_seconds = time.time() - t0
    print(f"done in {train_seconds:.1f}s")

    val_metrics = evaluate(pipeline, val["text"], val["label"], "validation")
    test_metrics = evaluate(pipeline, test["text"], test["label"], "test")
    plot_confusion_matrix(test["label"], test_metrics.pop("predictions"))
    val_metrics.pop("predictions", None)

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"\nModel saved -> {MODEL_PATH}")

    metrics = {
        "model": "TF-IDF (word 1-2g + char 3-5g) -> calibrated LinearSVC",
        "train_size": len(train),
        "train_seconds": round(train_seconds, 1),
        "validation": val_metrics,
        "test": test_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved -> {METRICS_PATH}")
    print(f"Confusion matrix -> {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
