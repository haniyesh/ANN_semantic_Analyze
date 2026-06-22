"""
rescore_fix.py
==============
Re-derive sentiment, sentiment_score, confidence, and weight from the
existing prob_positive / prob_negative / prob_neutral columns using the
fixed scoring logic.

No model inference needed — just recalculates derived columns in-place.

Usage:
    python rescore_fix.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
CSV  = HERE / "news_cleaned_filtered_scored.csv"
OUT  = HERE / "news_cleaned_filtered_scored.csv"       # overwrite in-place
BAK  = HERE / "news_cleaned_filtered_scored_backup.csv" # safety backup


def fixed_score(prob_pos, prob_neg, prob_neu):
    """Fixed scoring: respects neutral dominance, correct tier ordering."""
    # If neutral dominates both positive and negative → neutral
    if prob_neu > max(prob_pos, prob_neg):
        return "neutral", 0, prob_neu

    net = prob_pos - prob_neg
    # Most-extreme negative first so all tiers are reachable
    score = (
         3 if net >  0.50 else
         2 if net >  0.25 else
         1 if net >  0.05 else
        -3 if net < -0.50 else
        -2 if net < -0.25 else
        -1 if net < -0.05 else 0
    )
    sentiment = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    # Confidence = probability of the *assigned* class
    confidence = prob_pos if score > 0 else (prob_neg if score < 0 else prob_neu)
    return sentiment, score, confidence


def main():
    print("=" * 60)
    print("  RESCORE FIX — recompute labels from existing probabilities")
    print("=" * 60)

    print(f"\n  Loading {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"  Rows: {len(df):,}")

    # Backup
    print(f"  Saving backup → {BAK.name}")
    df.to_csv(BAK, index=False)

    # Store old distribution for comparison
    old_dist = df["sentiment"].value_counts().to_dict()
    old_scores = df["sentiment_score"].value_counts().sort_index().to_dict()

    # Apply fixed scoring
    print("  Re-scoring...")
    mask = df["prob_positive"].notna() & df["prob_negative"].notna() & df["prob_neutral"].notna()
    scored = mask.sum()
    skipped = (~mask).sum()
    print(f"    Rows with probabilities: {scored:,}  (skipping {skipped:,} without)")

    sentiments = []
    scores = []
    confidences = []

    for _, row in df.iterrows():
        if pd.isna(row["prob_positive"]) or pd.isna(row["prob_negative"]) or pd.isna(row["prob_neutral"]):
            sentiments.append(row.get("sentiment", "neutral"))
            scores.append(row.get("sentiment_score", 0))
            confidences.append(row.get("confidence", 0))
        else:
            s, sc, c = fixed_score(
                float(row["prob_positive"]),
                float(row["prob_negative"]),
                float(row["prob_neutral"]),
            )
            sentiments.append(s)
            scores.append(sc)
            confidences.append(round(c, 4))

    df["sentiment"] = sentiments
    df["sentiment_score"] = scores
    df["confidence"] = confidences
    df["weight"] = df["confidence"].apply(lambda c: max(5, min(10, round(c * 10))))

    # Report
    new_dist = df["sentiment"].value_counts().to_dict()
    new_scores = df["sentiment_score"].value_counts().sort_index().to_dict()

    print(f"\n  {'Label':<12} {'OLD':>8} {'NEW':>8} {'DELTA':>8}")
    print(f"  {'-'*40}")
    for label in ["positive", "negative", "neutral"]:
        old = old_dist.get(label, 0)
        new = new_dist.get(label, 0)
        print(f"  {label:<12} {old:>8,} {new:>8,} {new-old:>+8,}")

    print(f"\n  {'Score':<8} {'OLD':>8} {'NEW':>8}")
    print(f"  {'-'*28}")
    all_keys = sorted(set(list(old_scores.keys()) + list(new_scores.keys())))
    for k in all_keys:
        old = old_scores.get(k, 0)
        new = new_scores.get(k, 0)
        print(f"  {k:<8} {old:>8,} {new:>8,}")

    # Confidence stats
    print(f"\n  Confidence stats (new):")
    print(f"    Mean:   {df['confidence'].mean():.4f}")
    print(f"    Median: {df['confidence'].median():.4f}")
    print(f"    Min:    {df['confidence'].min():.4f}")
    print(f"    Max:    {df['confidence'].max():.4f}")

    # Save
    df.to_csv(OUT, index=False)
    print(f"\n  ✅ Saved {len(df):,} rows → {OUT.name}")
    print(f"  📦 Backup at → {BAK.name}")


if __name__ == "__main__":
    main()
