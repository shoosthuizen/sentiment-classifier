# Sentiment Classifier — DistilBERT

Binary sentiment classifier (positive / negative) built on
[`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased),
fine-tuned with the HuggingFace `transformers` Trainer API and served via
FastAPI.

---

## Project layout

```
sentiment-classifier/
├── train.py          # fine-tuning script
├── server.py         # FastAPI inference server
├── requirements.txt
└── README.md
```

After training a `model/best/` directory is created:

```
model/best/
├── config.json
├── tokenizer_config.json
├── vocab.txt
└── model.safetensors
```

---

## 1 — Install dependencies

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2 — Train

### Option A: SST-2 (default, no data prep needed)

Downloads a small slice of the GLUE SST-2 dataset automatically.

```bash
python train.py
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--train-size` | `1000` | Training examples |
| `--eval-size` | `200` | Validation examples |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `16` | Per-device batch size |
| `--lr` | `2e-5` | Learning rate |
| `--output-dir` | `model` | Checkpoint directory |

Example — larger run on GPU:

```bash
python train.py --train-size 5000 --eval-size 500 --epochs 5
```

### Option B: Custom CSV dataset

Provide a CSV file with `text` and `label` columns (`label`: `0`=negative,
`1`=positive):

```csv
text,label
"The film was outstanding.",1
"Completely boring and dull.",0
```

```bash
python train.py --data my_data.csv
```

The script splits the CSV 90 % / 10 % for train / validation automatically.

---

## 3 — Run the inference server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

To point at a different model directory:

```bash
MODEL_DIR=path/to/model/best uvicorn server:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## 4 — Calling the API

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The acting was phenomenal!"}'
```

Response:

```json
{
  "text": "The acting was phenomenal!",
  "label": "positive",
  "score": 0.9971
}
```

### Batch prediction (up to 32 texts)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Absolutely loved it.", "Waste of time."]}'
```

Response:

```json
{
  "predictions": [
    {"text": "Absolutely loved it.", "label": "positive", "score": 0.9968},
    {"text": "Waste of time.",       "label": "negative", "score": 0.9903}
  ]
}
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Notes

- **GPU** — training automatically uses CUDA when available. Set `fp16=True`
  in `train.py` `TrainingArguments` for faster GPU training.
- **CPU-only** — inference runs fine on CPU; expect ~100–300 ms per request.
- The best checkpoint (highest validation accuracy) is saved to `model/best/`.
