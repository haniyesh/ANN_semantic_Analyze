"""
fetch_recent_kaggle.py
======================
Downloads and processes recent 2024-2025 crypto news datasets:
  1. lukasschmidt/crypto-news-recent
     → Recent crypto headlines, closest to live bot data
  2. pratyushpuri/crypto-market-sentiment-and-price-dataset-2025
     → 2025 crypto market sentiment + price data

Why recent data matters:
  - Your bot runs in real-time 2024-2025
  - Market regime is completely different from 2021-2022
  - ETF era, institutional adoption, different volatility patterns
  - Model trained on old data won't recognize new patterns

Output: news_recent_merged.csv  (same format as news_cleaned_filtered.csv)

Run AFTER fetch_group1_kaggle.py
"""

import os
import time
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
DOWNLOAD_DIR = HERE / "kaggle_downloads"
OUTPUT_CSV   = HERE / "news_recent_merged.csv"
EXISTING_CSV = HERE / "news_cleaned_filtered.csv"
GROUP1_CSV   = HERE / "news_group1_merged.csv"
GROUP1_SCORED= HERE / "news_group1_merged_scored.csv"

DOWNLOAD_DIR.mkdir(exist_ok=True)

BINANCE_URL = "https://api.binance.com/api/v3/klines"

# ── Date filter — only keep news from 2023 onwards ────────────────
RECENT_CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")


# ══════════════════════════════════════════════════════════════════
# KAGGLE DOWNLOAD
# ══════════════════════════════════════════════════════════════════
def download_kaggle_dataset(dataset: str, dest: Path) -> bool:
    dest.mkdir(exist_ok=True)
    if list(dest.glob("*.csv")):
        print(f"  ✅ Already downloaded: {dest.name}")
        return True
    print(f"\n📥 Downloading {dataset}...")
    ret = os.system(f"kaggle datasets download {dataset} --path {dest} --unzip")
    if ret != 0:
        print(f"  ❌ Failed. Check ~/.kaggle/kaggle.json")
        return False
    print(f"  ✅ Saved to {dest}")
    return True


# ══════════════════════════════════════════════════════════════════
# DATASET R1 — lukasschmidt/crypto-news-recent
# Recent crypto headlines — no price
# ══════════════════════════════════════════════════════════════════
def parse_dataset_recent(folder: Path) -> pd.DataFrame:
    print("\n[Dataset R1] Parsing crypto-news-recent (lukasschmidt)...")

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"  ❌ No CSV in {folder}")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        print(f"  Reading {f.name}...")
        try:
            df = pd.read_csv(f, low_memory=False)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  Raw rows: {len(df)}")

    # Find columns
    title_col = next((c for c in df.columns
                      if any(x in c for x in ["title", "headline", "news", "text"])), None)
    date_col  = next((c for c in df.columns
                      if any(x in c for x in ["date", "published", "time", "created"])), None)
    src_col   = next((c for c in df.columns
                      if any(x in c for x in ["source", "channel", "publisher", "site"])), None)

    if not title_col:
        print("  ❌ No title column found")
        return pd.DataFrame()

    print(f"  Mapping: title={title_col}, date={date_col}, source={src_col}")

    rows = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title or len(title.split()) < 3:
            continue

        try:
            pub_dt = pd.to_datetime(row[date_col], utc=True) if date_col else datetime.now(timezone.utc)
            if pd.isnull(pub_dt):
                continue
            # Only keep recent news
            if pub_dt < RECENT_CUTOFF:
                continue
        except Exception:
            continue

        rows.append({
            "title":             title,
            "link":              str(row.get("link", row.get("url", ""))),
            "published":         pub_dt.isoformat(),
            "channel":           str(row.get(src_col, "kaggle_recent")) if src_col else "kaggle_recent",
            "sentiment":         "neutral",
            "sentiment_score":   0,
            "weight":            5,
            "confidence":        0.5,
            "prob_positive":     0.33,
            "prob_negative":     0.33,
            "prob_neutral":      0.34,
            "btc_price_at_news": None,
            "btc_price_15m":     None,
            "btc_price_1h":      None,
            "source":            "kaggle_recent",
        })

    result = pd.DataFrame(rows)
    print(f"  Parsed rows (2023+): {len(result)}")
    return result


# ══════════════════════════════════════════════════════════════════
# DATASET R2 — pratyushpuri/crypto-market-sentiment-and-price-dataset-2025
# 2025 sentiment + price data
# ══════════════════════════════════════════════════════════════════
def parse_dataset_2025(folder: Path) -> pd.DataFrame:
    print("\n[Dataset R2] Parsing crypto-market-sentiment-and-price-2025...")

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"  ❌ No CSV in {folder}")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        print(f"  Reading {f.name}...")
        try:
            df = pd.read_csv(f, low_memory=False)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  Raw rows: {len(df)}")

    # Find columns
    title_col = next((c for c in df.columns
                      if any(x in c for x in ["title", "headline", "news", "text", "content"])), None)
    date_col  = next((c for c in df.columns
                      if any(x in c for x in ["date", "published", "time", "timestamp"])), None)
    sent_col  = next((c for c in df.columns
                      if "sentiment" in c or "label" in c), None)

    # Check if it has price columns
    price_col = next((c for c in df.columns
                      if any(x in c for x in ["price", "close", "open"])
                      and "15m" not in c and "1h" not in c), None)
    price_15m = next((c for c in df.columns if "15m" in c or "15min" in c), None)
    price_1h  = next((c for c in df.columns if "1h" in c or "1hour" in c), None)

    if not title_col:
        print("  ❌ No title column found")
        print(f"  Available columns: {list(df.columns)}")
        return pd.DataFrame()

    print(f"  Mapping: title={title_col}, date={date_col}, "
          f"sentiment={sent_col}, price={price_col}")

    rows = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title or len(title.split()) < 3:
            continue

        try:
            pub_dt = pd.to_datetime(row[date_col], utc=True) if date_col else datetime.now(timezone.utc)
            if pd.isnull(pub_dt):
                continue
            if pub_dt < RECENT_CUTOFF:
                continue
        except Exception:
            continue

        # Map sentiment if available
        sent_raw = str(row.get(sent_col, "neutral")).lower() if sent_col else "neutral"
        if any(x in sent_raw for x in ["pos", "bull", "1"]):
            sentiment, sentiment_score = "positive", 2
            prob_pos, prob_neg, prob_neu = 0.7, 0.1, 0.2
        elif any(x in sent_raw for x in ["neg", "bear", "-1"]):
            sentiment, sentiment_score = "negative", -2
            prob_pos, prob_neg, prob_neu = 0.1, 0.7, 0.2
        else:
            sentiment, sentiment_score = "neutral", 0
            prob_pos, prob_neg, prob_neu = 0.2, 0.2, 0.6

        # Get prices if available
        try:
            btc_now = float(row[price_col]) if price_col else None
            btc_15m = float(row[price_15m]) if price_15m else None
            btc_1h  = float(row[price_1h])  if price_1h  else None
        except (TypeError, ValueError):
            btc_now = btc_15m = btc_1h = None

        rows.append({
            "title":             title,
            "link":              str(row.get("link", row.get("url", ""))),
            "published":         pub_dt.isoformat(),
            "channel":           str(row.get("source", row.get("channel", "kaggle_2025"))),
            "sentiment":         sentiment,
            "sentiment_score":   sentiment_score,
            "weight":            6,
            "confidence":        0.65,
            "prob_positive":     round(prob_pos, 4),
            "prob_negative":     round(prob_neg, 4),
            "prob_neutral":      round(prob_neu, 4),
            "btc_price_at_news": btc_now,
            "btc_price_15m":     btc_15m,
            "btc_price_1h":      btc_1h,
            "source":            "kaggle_2025",
        })

    result = pd.DataFrame(rows)
    print(f"  Parsed rows (2023+): {len(result)}")
    return result


# ══════════════════════════════════════════════════════════════════
# BTC PRICE FETCHER
# ══════════════════════════════════════════════════════════════════
def get_btc_price_at(timestamp_ms: int) -> float | None:
    try:
        resp = requests.get(BINANCE_URL, params={
            "symbol":    "BTCUSDT",
            "interval":  "1m",
            "startTime": timestamp_ms,
            "limit":     1,
        }, timeout=10)
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            return float(data[0][4])
        return None
    except Exception:
        return None


def add_btc_prices(df: pd.DataFrame) -> pd.DataFrame:
    missing_mask  = df["btc_price_at_news"].isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        print("  ✅ All rows already have BTC prices")
        return df

    print(f"\n[BTC Prices] Fetching for {missing_count} rows...")
    print(f"  Estimated time: {missing_count * 0.35 / 60:.1f} minutes")

    checkpoint = HERE / "news_recent_checkpoint.csv"
    indices    = df[missing_mask].index.tolist()

    for i, idx in enumerate(tqdm(indices, desc="Fetching BTC prices")):
        try:
            pub_dt = pd.to_datetime(df.at[idx, "published"], utc=True)
            ts_ms  = int(pub_dt.timestamp() * 1000)

            p_now = get_btc_price_at(ts_ms);              time.sleep(0.1)
            p_15m = get_btc_price_at(ts_ms + 900_000);   time.sleep(0.1)
            p_1h  = get_btc_price_at(ts_ms + 3_600_000); time.sleep(0.1)

            df.at[idx, "btc_price_at_news"] = p_now
            df.at[idx, "btc_price_15m"]     = p_15m
            df.at[idx, "btc_price_1h"]      = p_1h
        except Exception:
            continue

        if (i + 1) % 500 == 0:
            df.to_csv(checkpoint, index=False)
            tqdm.write(f"  💾 Checkpoint: {i+1}/{missing_count}")

    return df


# ══════════════════════════════════════════════════════════════════
# COMPUTE LABELS
# ══════════════════════════════════════════════════════════════════
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Labels] Computing impact labels...")

    for col in ["btc_price_at_news", "btc_price_15m", "btc_price_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df[df["btc_price_at_news"] > 0]
    print(f"  Dropped {before - len(df)} rows → {len(df)} remain")

    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["btc_change_1h"]    = (df["btc_price_1h"]  - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["abs_change_1h"]    = df["btc_change_1h"].abs()
    df["is_impactful_15m"] = (df["abs_change_15m"] >= 0.5).astype(int)
    df["is_impactful_1h"]  = (df["abs_change_1h"]  >= 0.5).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)
    df["direction_1h"]     = (df["btc_change_1h"]  > 0).astype(int)
    df["confidence_label"] = df["abs_change_15m"].clip(0, 3) / 3.0

    print(f"  Impact 15m: {df['is_impactful_15m'].mean():.1%}")
    print(f"  Impact 1h:  {df['is_impactful_1h'].mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════
# DEDUPLICATE + MERGE
# ══════════════════════════════════════════════════════════════════
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df["title_hash"] = df["title"].fillna("").str.lower().apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
    )
    df = df.drop_duplicates(subset=["title_hash"])
    print(f"  Deduplication: {before} → {len(df)} rows")
    return df


def merge_with_all_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Check against all existing CSVs to avoid duplicates."""
    existing_hashes = set()

    existing_csvs = [EXISTING_CSV, GROUP1_CSV, GROUP1_SCORED]
    for csv_path in existing_csvs:
        if csv_path.exists():
            ex = pd.read_csv(csv_path, low_memory=False)
            print(f"  Checking {csv_path.name}: {len(ex)} rows")
            if "title_hash" not in ex.columns:
                ex["title_hash"] = ex["title"].fillna("").str.lower().apply(
                    lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
                )
            existing_hashes.update(ex["title_hash"].tolist())

    new_only = new_df[~new_df["title_hash"].isin(existing_hashes)]
    print(f"  New unique rows: {len(new_only)}")
    print(f"  Duplicates skipped: {len(new_df) - len(new_only)}")

    # Align columns with existing CSV
    if EXISTING_CSV.exists():
        existing_cols = pd.read_csv(EXISTING_CSV, nrows=1).columns.tolist()
        for col in existing_cols:
            if col not in new_only.columns:
                new_only[col] = np.nan
        common = [c for c in existing_cols if c in new_only.columns]
        new_only = new_only[common]

    return new_only


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RECENT NEWS KAGGLE DATASETS FETCHER (2023-2025)")
    print("  Datasets: lukasschmidt + pratyushpuri")
    print(f"  Cutoff: {RECENT_CUTOFF.date()} onwards only")
    print("=" * 60)

    all_frames = []

    # ── Dataset R1: lukasschmidt/crypto-news-recent ───────────────
    dr1_dir = DOWNLOAD_DIR / "dataset_recent_lukasschmidt"
    download_kaggle_dataset("lukasschmidt/crypto-news-recent", dr1_dir)
    df_r1 = parse_dataset_recent(dr1_dir)
    if not df_r1.empty:
        all_frames.append(df_r1)
        print(f"  Dataset R1: {len(df_r1)} rows")

    # ── Dataset R2: pratyushpuri/crypto-market-sentiment-2025 ─────
    dr2_dir = DOWNLOAD_DIR / "dataset_2025_pratyushpuri"
    download_kaggle_dataset(
        "pratyushpuri/crypto-market-sentiment-and-price-dataset-2025", dr2_dir
    )
    df_r2 = parse_dataset_2025(dr2_dir)
    if not df_r2.empty:
        all_frames.append(df_r2)
        print(f"  Dataset R2: {len(df_r2)} rows")

    if not all_frames:
        print("\n❌ No data fetched.")
        return

    # ── Combine ───────────────────────────────────────────────────
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n[Combine] Total: {len(combined)} rows")

    # ── Deduplicate ───────────────────────────────────────────────
    combined = deduplicate(combined)

    # ── Add BTC prices ────────────────────────────────────────────
    combined = add_btc_prices(combined)

    # ── Compute labels ────────────────────────────────────────────
    combined = compute_labels(combined)

    # ── Merge (check against all existing) ────────────────────────
    print("\n[Merge] Checking against existing datasets...")
    final = merge_with_all_existing(combined)

    if final.empty:
        print("\n⚠️  No new rows to add")
        return

    # ── Sort + save ───────────────────────────────────────────────
    final["published"] = pd.to_datetime(final["published"], utc=True, errors="coerce")
    final = final.sort_values("published").reset_index(drop=True)
    final.to_csv(OUTPUT_CSV, index=False)

    # Remove checkpoint if exists
    checkpoint = HERE / "news_recent_checkpoint.csv"
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"\n{'='*60}")
    print(f"  ✅ Saved {len(final)} rows → {OUTPUT_CSV}")
    print(f"{'='*60}")
    print(f"\n  Impact 15m  : {final['is_impactful_15m'].mean():.1%}")
    print(f"  Impact 1h   : {final['is_impactful_1h'].mean():.1%}")
    print(f"  Date range  : {final['published'].min()} → {final['published'].max()}")
    print(f"\nNext steps:")
    print(f"  1. Score sentiment:  python score_sentiment.py {OUTPUT_CSV}")
    print(f"  2. Merge everything: python merge_datasets.py")
    print(f"  3. Re-train model:   python production_system_v5.py")


if __name__ == "__main__":
    main()