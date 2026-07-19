import pytest

from moodlens.config import LABELS
from moodlens.predict import predict, predict_batch


def test_predict_returns_valid_label():
    res = predict("i am so happy today, everything went perfectly")
    assert res["emotion"] in LABELS
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top"]) == 3


def test_predict_obvious_joy():
    res = predict("im thrilled and delighted, this is the best day of my life")
    assert res["emotion"] == "joy"


def test_predict_obvious_sadness():
    res = predict("i feel so hopeless and miserable, i cried all night")
    assert res["emotion"] == "sadness"


def test_top_probs_sorted_and_normalized():
    res = predict("what a shocking turn of events", top_k=6)
    probs = [t["prob"] for t in res["top"]]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 0.01


def test_empty_text_raises():
    with pytest.raises(ValueError):
        predict("   ")


def test_batch():
    out = predict_batch(["i love you", "i am furious"])
    assert len(out) == 2
    assert out[0]["emotion"] in LABELS
