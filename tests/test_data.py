from moodlens.config import LABELS
from moodlens.data import load_split


def test_load_train_split():
    df = load_split("train")
    assert {"text", "label", "label_name"} <= set(df.columns)
    assert len(df) > 10_000
    assert df["label"].between(0, 5).all()
    assert set(df["label_name"].unique()) <= set(LABELS)


def test_splits_disjoint_sizes():
    val = load_split("validation")
    test = load_split("test")
    assert len(val) == 2000
    assert len(test) == 2000
