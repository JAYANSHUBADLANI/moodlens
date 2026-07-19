from fastapi.testclient import TestClient

from moodlens.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_labels():
    r = client.get("/labels")
    assert r.status_code == 200
    assert len(r.json()["labels"]) == 6


def test_predict_endpoint():
    r = client.post("/predict", json={"text": "i am scared of the dark"})
    assert r.status_code == 200
    body = r.json()
    assert body["emotion"] == "fear"
    assert "top" in body


def test_predict_validation_error():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_batch_endpoint():
    r = client.post(
        "/predict/batch",
        json={"texts": ["i adore my family", "this makes me so angry"]},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
