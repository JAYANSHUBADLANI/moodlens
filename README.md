# MoodLens 🔍: Emotion Detection from Text

[![tests](https://github.com/JAYANSHUBADLANI/moodlens/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/moodlens/actions/workflows/tests.yml)

Classify short text into **6 emotions** (sadness, joy, love, anger, fear, surprise), served through a production-style REST API.

Built end-to-end: data pipeline → training → evaluation → serving → tests → Docker.

## Results

| Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Validation | 0.900 | 0.873 | 0.899 |
| **Test** | **0.894** | **0.844** | **0.893** |

Model: TF-IDF (word 1-2 grams + char 3-5 grams) → calibrated Linear SVM. Trains in ~30s on CPU, 8.4 MB artifact, millisecond inference, a deliberately strong classical baseline before reaching for transformers. Per-class breakdown and a normalized confusion matrix are generated in `reports/` on every training run.

Dataset: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), 20k English tweets (16k train / 2k val / 2k test), auto-downloaded and cached on first run.

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: set PYTHONPATH=src

python -m moodlens.train         # downloads data, trains, writes model + reports
python -m moodlens.predict "i cant believe this actually worked"
uvicorn moodlens.api:app --app-dir src   # serve → http://localhost:8000/docs
pytest                           # 13 tests
```

### API example

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "why would they cancel it without telling us"}'
```

```json
{
  "text": "why would they cancel it without telling us",
  "emotion": "anger",
  "confidence": 0.71,
  "top": [
    {"emotion": "anger", "prob": 0.71},
    {"emotion": "sadness", "prob": 0.18},
    {"emotion": "fear", "prob": 0.06}
  ]
}
```

Endpoints: `POST /predict`, `POST /predict/batch`, `GET /labels`, `GET /health`.

### Docker

```bash
docker build -t moodlens .
docker run -p 8000:8000 moodlens
```

## Project structure

```
moodlens/
├── src/moodlens/
│   ├── config.py      # paths, labels, constants
│   ├── data.py        # HF download + local caching
│   ├── train.py       # pipeline, evaluation, reports
│   ├── predict.py     # cached model loading + CLI
│   └── api.py         # FastAPI service
├── tests/             # data, inference, and API tests
├── reports/           # metrics.json, confusion_matrix.png
├── models/            # trained pipeline (joblib)
└── data/              # cached CSV splits
```

## Design decisions

**Why a linear SVM and not BERT?** On this dataset a tuned classical pipeline reaches ~89% vs ~92-93% for fine-tuned BERT, at a fraction of the cost: CPU-only training in seconds, tiny artifact, no GPU serving. Establishing a strong cheap baseline first is the right engineering order; the pipeline abstraction means a transformer can be dropped in behind the same `predict()` contract later.

**Why word + char n-grams?** Tweets are noisy ("soooo happpy"). Char 3-5 grams capture misspellings and elongations that word tokens miss, worth ~1.5 points of macro F1 over word-only in my runs.

**Why calibration?** Raw SVM margins aren't probabilities. Sigmoid (Platt) calibration lets the API return meaningful confidence scores, which downstream consumers can threshold.

**Known limits.** `surprise` is the weakest class (F1 ≈ 0.69): it has only 572 training examples and overlaps semantically with joy/fear. Next steps: class-weighted loss, data augmentation, or a DistilBERT fine-tune.

## License

MIT
