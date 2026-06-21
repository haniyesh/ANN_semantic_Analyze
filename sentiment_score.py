"""
sentiment_score.py
==================
Scores sentiment using CryptoBERT only (ElKulako/cryptobert).

Design decisions vs previous version:
  - CryptoBERT only — no FinBERT / RoBERTa ensemble
  - NO neutral gate — high-neutral news is kept, not discarded
  - weight = round(confidence * 10), min=5 so all rows pass training filter
  - Runs ~3x faster than the 3-model ensemble

CryptoBERT labels: 0=Bearish  1=Neutral  2=Bullish

Usage:
  python services/sentiment_score.py news_cleaned_filtered.csv
  python services/sentiment_score.py news_cleaned_filtered.csv --output rescored.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging

hf_logging.set_verbosity_error()

BATCH_SIZE = 32
SAVE_EVERY = 500

_model_cache = {}


def load_model():
    global _model_cache
    if _model_cache:
        return _model_cache
    print("  Loading CryptoBERT...")
    tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cls = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert")
    cls.eval()
    _model_cache = {"tok": tok, "cls": cls}
    print("  ✅ CryptoBERT loaded\n")
    return _model_cache


def score_batch(titles: list[str], model: dict) -> list[dict]:
    """Score a batch of titles with CryptoBERT. Returns list of dicts."""
    titles = [str(t).strip() or "crypto news" for t in titles]
    inputs = model["tok"](
        titles, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model["cls"](**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()   # (B, 3)  0=bear 1=neu 2=bull

    results = []
    for p in probs:
        prob_neg, prob_neu, prob_pos = float(p[0]), float(p[1]), float(p[2])

        # If neutral dominates both positive and negative → neutral
        if prob_neu > max(prob_pos, prob_neg):
            score      = 0
            sentiment  = "neutral"
            confidence = prob_neu
        else:
            net  = prob_pos - prob_neg
            # Check most-extreme first so all negative tiers are reachable
            score = (
                 3 if net >  0.50 else
                 2 if net >  0.25 else
                 1 if net >  0.05 else
                -3 if net < -0.50 else
                -2 if net < -0.25 else
                -1 if net < -0.05 else 0
            )
            sentiment  = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
            # Confidence = probability of the *assigned* class
            confidence = prob_pos if score > 0 else (prob_neg if score < 0 else prob_neu)

        weight = max(5, min(10, round(confidence * 10)))  # always >= 5

        results.append({
            "sentiment":       sentiment,
            "sentiment_score": score,
            "weight":          weight,
            "confidence":      round(confidence, 4),
            "prob_positive":   round(prob_pos, 4),
            "prob_negative":   round(prob_neg, 4),
            "prob_neutral":    round(prob_neu, 4),
        })
    return results


def score_dataframe(df: pd.DataFrame, model: dict, checkpoint_path: Path) -> pd.DataFrame:
    sent_cols = ["sentiment", "sentiment_score", "weight",
                 "confidence", "prob_positive", "prob_negative", "prob_neutral"]
    for col in sent_cols:
        if col not in df.columns:
            df[col] = None

    w_col = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    needs = (df["prob_positive"].isna() | (w_col <= 5)).values
    indices = np.where(needs)[0].tolist()
    total = len(indices)

    if total == 0:
        print("  ✅ All rows already scored")
        return df

    print(f"  Rows to score : {total:,}  (already done: {len(df)-total:,})")
    print(f"  Est. time     : {total * BATCH_SIZE / 32 * 0.25 / 60:.0f} min\n")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Scoring"):
        batch_idx = indices[start:start + BATCH_SIZE]
        titles    = df["title"].iloc[batch_idx].tolist()
        results   = score_batch(titles, model)

        for i, res in zip(batch_idx, results):
            for col, val in res.items():
                df.iloc[i, df.columns.get_loc(col)] = val

        if (start // BATCH_SIZE + 1) % (SAVE_EVERY // BATCH_SIZE) == 0:
            df.to_csv(checkpoint_path, index=False)
            tqdm.write(f"  💾 checkpoint at {start + BATCH_SIZE}/{total}")

    df.to_csv(checkpoint_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"❌ Not found: {input_path}"); sys.exit(1)

    output_path     = Path(args.output) if args.output else \
                      input_path.parent / (input_path.stem + "_scored.csv")
    checkpoint_path = input_path.parent / (input_path.stem + "_score_ckpt.csv")

    print("=" * 60)
    print("  CryptoBERT SENTIMENT SCORER")
    print(f"  Input : {input_path.name}")
    print(f"  Output: {output_path.name}")
    print("=" * 60)

    if checkpoint_path.exists():
        print(f"\n⚡ Resuming from checkpoint")
        df = pd.read_csv(checkpoint_path, low_memory=False)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"  Loaded {len(df):,} rows")

    model = load_model()
    df    = score_dataframe(df, model, checkpoint_path)

    # Sort by date
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.sort_values("published").reset_index(drop=True)

    df.to_csv(output_path, index=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    total = len(df)
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    neu = (df["sentiment"] == "neutral").sum()
    print(f"\n  Sentiment distribution:")
    print(f"    positive: {pos:,}  ({pos/total:.1%})")
    print(f"    negative: {neg:,}  ({neg/total:.1%})")
    print(f"    neutral : {neu:,}  ({neu/total:.1%})")
    print(f"\n  ✅ Saved {total:,} rows → {output_path}")


if __name__ == "__main__":
    main()
