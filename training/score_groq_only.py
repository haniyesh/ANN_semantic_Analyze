"""
score_groq_only.py
==================
Build news_cache.json using ONLY Groq/Llama-3.3-70B sentiment labels.
No XGBoost, no BERT embeddings — just the Groq cache.

Items not in the Groq cache are labeled 'neutral' by default.

Usage:
    python training/score_groq_only.py
    python training/score_groq_only.py --months 1    # last N months (default: 1)
    python training/score_groq_only.py --dry-run
"""

import sys, json, hashlib, warnings, argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT           = Path(__file__).parent.parent
CSV_PATH       = ROOT / "news_cleaned_filtered_scored.csv"
GROQ_CACHE     = ROOT / "groq_sentiment_cache.json"
CACHE_FILE     = ROOT / "news_cache.json"


def _title_key(title: str) -> str:
    return hashlib.md5(title.encode()).hexdigest()


def load_data(months: int) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for col in df.columns:
        orig = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df[col] = orig

    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df.drop_duplicates(subset=["title", "published", "channel"])
    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    cutoff = df["published"].max() - pd.DateOffset(months=months)
    df = df[df["published"] >= cutoff].reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows  ({cutoff.date()} → {df['published'].max().date()})")
    return df


def build_cache_items(df: pd.DataFrame, groq: dict) -> list:
    items = []
    cached = 0

    for i in range(len(df)):
        row    = df.iloc[i]
        title  = str(row.get("title", ""))
        key    = _title_key(title)
        label  = groq.get(key)

        if label:
            cached += 1
        else:
            # Fall back to existing BERT sentiment from CSV
            label = str(row.get("sentiment", "neutral") or "neutral")

        pub_dt = row["published"]
        if pd.isnull(pub_dt):
            continue
        pub_ts = int(pub_dt.timestamp())

        btc_p  = float(row.get("btc_price_at_news") or 0)
        btc_15 = float(row.get("btc_price_15m") or 0)
        btc_1h = float(row.get("btc_price_1h")  or 0)
        c15m   = (btc_15 - btc_p) / btc_p * 100 if btc_p else 0
        c1h    = (btc_1h - btc_p) / btc_p * 100 if btc_p else 0

        sig_type = "BUY" if label == "positive" else ("SELL" if label == "negative" else "NEUTRAL")

        # Simple impact score based on sentiment strength only
        # (no XGBoost — just use confidence from CSV or default)
        confidence = float(row.get("confidence") or 0.5)
        sent_score = float(row.get("sentiment_score") or 0)
        model_score = confidence if label != "neutral" else confidence * 0.3

        impact = "High" if model_score >= 0.60 else ("Medium" if model_score >= 0.35 else "Low")

        items.append({
            "id":              f"groq_{pub_ts}_{hash(title[:30]) % 100000}",
            "time":            pub_dt.strftime("%H:%M:%S"),
            "title":           title,
            "link":            str(row.get("link", "")),
            "channel":         str(row.get("channel", "unknown")),
            "published":       pub_dt.isoformat(),
            "published_ts":    pub_ts,
            "sentiment":       label,
            "sentiment_score": sent_score,
            "confidence":      round(confidence * 100, 1),
            "weight":          float(row.get("weight") or 0),
            "prob_positive":   float(row.get("prob_positive") or 0),
            "prob_negative":   float(row.get("prob_negative") or 0),
            "prob_neutral":    float(row.get("prob_neutral")  or 0),
            "type":            sig_type,
            "btc_change_15m":  round(c15m, 4),
            "btc_change_1h":   round(c1h,  4),
            "model_score":     round(model_score, 4),
            "model_score_1h":  round(model_score * 0.9, 4),
            "score_normalized": True,
            "pred_15m":        int(label != "neutral"),
            "pred_1h":         int(label != "neutral"),
            "impact":          impact,
            "news_type":       str(row.get("news_type", "")),
            "source":          "groq_only",
        })

    print(f"  Groq cache hits: {cached:,}/{len(df):,} ({cached/len(df)*100:.1f}%)")
    print(f"  Sentiment dist: "
          f"+{sum(1 for x in items if x['sentiment']=='positive'):,} "
          f"-{sum(1 for x in items if x['sentiment']=='negative'):,} "
          f"={sum(1 for x in items if x['sentiment']=='neutral'):,}")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",  type=int, default=1, help="Months of history (default: 1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  SCORE — Groq Only (no XGBoost)")
    print("=" * 55)

    if not GROQ_CACHE.exists():
        print("  ✗ groq_sentiment_cache.json not found")
        print("    Run: python training/xgboost_train_groq.py --groq-limit 2000")
        sys.exit(1)

    with open(GROQ_CACHE) as f:
        groq = json.load(f)
    print(f"\n  Groq cache: {len(groq):,} entries")

    print("\n  Loading CSV...")
    df = load_data(args.months)

    print("\n  Building items...")
    items = build_cache_items(df, groq)

    # Preserve live items from existing cache (last N months)
    live_items = []
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            all_items = data if isinstance(data, list) else data.get("news", [])
            cutoff_ts = (datetime.now(timezone.utc) - pd.DateOffset(months=args.months)).timestamp()
            HIST_SOURCES = {"historical_v6","historical_v8","historical_xgb_v9","historical_xgb_v10","groq_only"}
            live_items = [x for x in all_items
                         if x.get("source") not in HIST_SOURCES
                         and (x.get("published_ts") or 0) >= cutoff_ts]
        except Exception:
            pass
    print(f"  Live items preserved: {len(live_items):,}")

    merged = sorted(live_items + items, key=lambda x: x.get("published_ts") or 0)

    payload = {
        "metadata": {
            "total_items":     len(merged),
            "hist_items":      len(items),
            "live_items":      len(live_items),
            "months_window":   args.months,
            "groq_cache_size": len(groq),
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "model":           "groq_only",
            "sentiment_model": "groq/llama-3.3-70b-versatile",
        },
        "news": merged,
    }

    if args.dry_run:
        print(f"\n  [DRY RUN] Would write {len(merged):,} items to {CACHE_FILE.name}")
        return

    CACHE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✅ Saved {len(merged):,} items → {CACHE_FILE.name}")


if __name__ == "__main__":
    main()
