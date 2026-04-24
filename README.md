# SentimentAI

A multi-backend sentiment analysis web app. Paste any text — review, tweet, feedback — and get instant sentiment scoring powered by three AI engines working together.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey) ![License](https://img.shields.io/badge/License-Open-green)

---

## What it does

- **3-class sentiment** — Positive / Neutral / Negative with confidence score
- **Emotion detection** — Joy, Sadness, Anger, Fear, Surprise, Disgust
- **Aspect-based analysis** — scores individual topics within a single text
- **Sarcasm detection** — flags irony with a 0–100% confidence score
- **Batch mode** — analyze up to 50 texts at once, export as CSV
- **Three backends fused** — VADER (rule-based) + SVM (ML) + optional Transformer

---

## Project structure

```
├── app.py                   # Flask web app
├── retrain.py               # Retrain on tweet_eval dataset
├── retrain_slang.py         # Retrain with slang-aware data
├── models/
│   ├── sentiment_model.pkl  # Trained SVM model
│   └── feature_extractor.pkl
├── src/
│   ├── sentiment_engine.py  # Core multi-backend engine
│   ├── train_model.py       # Model training pipeline
│   ├── preprocess.py        # Text cleaning & normalization
│   ├── features.py          # Feature extraction
│   ├── model.py             # Model wrapper
│   ├── api.py               # FastAPI alternative
│   └── analyze_text.py      # CLI text analyzer
├── templates/
│   └── index.html           # Web UI
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5001**

---

## Retrain the model

```bash
# Basic — tweet_eval dataset (~47k samples)
python retrain.py

# With slang — adds informal/profanity-aware training data
python retrain_slang.py
```

---

## Enable Transformer backend

For higher accuracy, enable the HuggingFace Transformer backend (~500 MB download):

```bash
USE_TRANSFORMER=1 python app.py
```

---

## Stack

`Flask` · `scikit-learn` · `VADER` · `NLTK` · `HuggingFace Transformers (optional)`
