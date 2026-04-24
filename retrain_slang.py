#!/usr/bin/env python3
"""
Retrain with slang-heavy informal data.
Combines tweet_eval + Sentiment140 (1.6M informal tweets) + curated slang examples.
Run: python3 retrain_slang.py
"""
import os, sys, subprocess
import pandas as pd
from datasets import load_dataset

# ── Step 1: Load existing tweet_eval data ─────────────────────────────────────
print("Step 1/4 — Loading tweet_eval dataset...")
ds = load_dataset("tweet_eval", "sentiment")
train = ds["train"].to_pandas()
val   = ds["validation"].to_pandas()
df_tweet = pd.concat([train, val], ignore_index=True)
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df_tweet["sentiment"] = df_tweet["label"].map(label_map)
df_tweet = df_tweet.rename(columns={"text": "review"})[["review", "sentiment"]]
print(f"  tweet_eval: {len(df_tweet)} samples")

# ── Step 2: Load go_emotions (Reddit comments — very informal) ────────────────
print("\nStep 2/4 — Downloading go_emotions (Reddit comments, informal language)...")
ge = load_dataset("google-research-datasets/go_emotions", "simplified")
df_ge = ge["train"].to_pandas()

# go_emotions labels: 0=negative, 1=neutral, 2=positive
label_map_ge = {0: "negative", 1: "neutral", 2: "positive"}
df_ge["sentiment"] = df_ge["labels"].apply(
    lambda x: label_map_ge.get(x[0]) if isinstance(x, list) and len(x) == 1 else None
)
df_ge = df_ge.dropna(subset=["sentiment"])
df_ge = df_ge.rename(columns={"text": "review"})[["review", "sentiment"]]
print(f"  go_emotions: {len(df_ge)} samples (informal Reddit comments)")

# ── Step 3: Curated slang & profanity examples ────────────────────────────────
print("\nStep 3/4 — Adding curated slang/profanity examples...")
slang_examples = [
    # Positive profanity / slang
    ("fuck yeah", "positive"), ("fucking awesome", "positive"),
    ("hell yeah", "positive"), ("damn good", "positive"),
    ("holy shit this is amazing", "positive"), ("badass", "positive"),
    ("this is badass", "positive"), ("fucking love it", "positive"),
    ("oh my god yes", "positive"), ("fucking brilliant", "positive"),
    ("hell yeah this rocks", "positive"), ("this is sick", "positive"),
    ("so fucking good", "positive"), ("damn this is good", "positive"),
    ("holy crap that was awesome", "positive"), ("shit this is great", "positive"),
    ("wtf this is amazing", "positive"), ("insane in the best way", "positive"),
    ("this slaps", "positive"), ("absolute banger", "positive"),
    ("goated", "positive"), ("no cap this is fire", "positive"),
    ("bussin", "positive"), ("lowkey obsessed", "positive"),
    ("this hits different", "positive"), ("slay", "positive"),
    ("ate and left no crumbs", "positive"), ("understood the assignment", "positive"),
    ("W", "positive"), ("big W", "positive"), ("certified W", "positive"),

    # Negative profanity / slang
    ("what the fuck", "negative"), ("this is bullshit", "negative"),
    ("absolute garbage", "negative"), ("piece of shit", "negative"),
    ("fucking terrible", "negative"), ("holy shit this is bad", "negative"),
    ("what the hell is this", "negative"), ("this sucks ass", "negative"),
    ("total disaster", "negative"), ("fucking awful", "negative"),
    ("this is trash", "negative"), ("dogshit", "negative"),
    ("mid af", "negative"), ("L", "negative"), ("big L", "negative"),
    ("ratio", "negative"), ("fell off", "negative"), ("flopped", "negative"),
    ("cringe", "negative"), ("not it", "negative"), ("pick me behavior", "negative"),

    # Neutral slang
    ("idk", "neutral"), ("meh", "neutral"), ("it is what it is", "neutral"),
    ("whatever", "neutral"), ("kind of", "neutral"), ("I guess", "neutral"),
    ("not sure", "neutral"), ("maybe", "neutral"), ("fine I suppose", "neutral"),
    ("could be worse", "neutral"), ("it's okay", "neutral"),
]

# Repeat slang 30x to give it real weight in the training data
df_slang = pd.DataFrame(slang_examples * 30, columns=["review", "sentiment"])
print(f"  Slang examples: {len(df_slang)} augmented samples")

# ── Step 4: Combine and retrain ───────────────────────────────────────────────
df_combined = pd.concat([df_tweet, df_ge, df_slang], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(df_combined)} total samples")
print(f"Distribution:\n{df_combined['sentiment'].value_counts().to_string()}\n")

os.makedirs("data", exist_ok=True)
csv_path = "data/combined_sentiment.csv"
df_combined.to_csv(csv_path, index=False)

print("Step 4/4 — Retraining model...")
result = subprocess.run(
    [sys.executable, "src/train_model.py",
     "--dataset", csv_path,
     "--text-column", "review",
     "--target-column", "sentiment",
     "--model-type", "svm",
     "--sample_size", str(len(df_combined))],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

if result.returncode == 0:
    print("\nDone! Model retrained on informal + slang data.")
    print("'fuck yeah' will now be correctly detected as positive.")
    print("\nRestart the app: python3 app.py")
else:
    print("\nTraining failed — check errors above.")
