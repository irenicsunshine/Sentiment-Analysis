#!/usr/bin/env python3
"""
Download tweet_eval/sentiment from HuggingFace and retrain the sentiment model.
Run: python3 retrain.py
"""
import os, sys, subprocess
import pandas as pd

print("Step 1/3 — Downloading tweet_eval/sentiment from HuggingFace...")
from datasets import load_dataset

ds = load_dataset("tweet_eval", "sentiment")

# Combine train + validation splits
train = ds["train"].to_pandas()
val   = ds["validation"].to_pandas()
df    = pd.concat([train, val], ignore_index=True)

# Labels: 0=negative, 1=neutral, 2=positive
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["sentiment"] = df["label"].map(label_map)
df = df.rename(columns={"text": "review"})[["review", "sentiment"]]

print(f"  Downloaded {len(df)} samples")
print(f"  Distribution:\n{df['sentiment'].value_counts().to_string()}\n")

os.makedirs("data", exist_ok=True)
csv_path = "data/tweet_eval_sentiment.csv"
df.to_csv(csv_path, index=False)
print(f"  Saved to {csv_path}")

print("\nStep 2/3 — Retraining model...")
result = subprocess.run(
    [sys.executable, "src/train_model.py",
     "--dataset", csv_path,
     "--text-column", "review",
     "--target-column", "sentiment",
     "--model-type", "svm",
     "--sample_size", str(len(df))],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

if result.returncode == 0:
    print("\nStep 3/3 — Done! Model retrained on 45k real samples.")
    print("Restart the app: python3 app.py")
else:
    print("\nTraining failed — check errors above.")
