"""
fetch_group1_kaggle.py
======================
Downloads and processes Group 1 Kaggle datasets:
  1. aaroncbastian/crypto-news-headlines-and-market-prices-by-date
     → Already has BTC price — best match for your CSV format
  2. oliviervha/crypto-news
     → Large dataset, needs BTC price added via Binance API

Output: news_group1_merged.csv  (same format as news_cleaned_filtered.csv)

Requirements:
    pip install kaggle feedparser pandas requests tqdm

Setup (one-time):
    1. Go to https://www.kaggle.com/settings → API → Create New Token
    2. Save kaggle.json to ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import time
import hashlib
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
DOWNLOAD_DIR = HERE / "kaggle_downloads"
OUTPUT_CSV   = HERE / "news_group1_merged.csv"
EXISTING_CSV = HERE / "news_cleaned_filtered.csv"

DOWNLOAD_DIR.mkdir(exist_ok=True)

# ── Binance API ───────────────────────────────────────────────────
BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ══════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD FROM KAGGLE
# ══════════════════════════════════════════════════════════════════
def download_kaggle_dataset(dataset: str, dest: Path):
    """
    Download and unzip a Kaggle dataset.
    dataset format: 'username/dataset-name'
    """
    dest.mkdir(exist_ok=True)
    print(f"\n📥 Downloading {dataset}...")
    ret = os.system(
        f"kaggle datasets download {dataset} --path {dest} --unzip"
    )
    if ret != 0:
        print(f"  ❌ Failed. Make sure kaggle.json is set up correctly.")
        print(f"     See: https://www.kaggle.com/settings → API → Create New Token")
        return False
    print(f"  ✅ Downloaded to {dest}")
    return True


# ══════════════════════════════════════════════════════════════════
# STEP 2 — PARSE DATASET 1 (aaroncbastian)
# Already has BTC price — minimal processing needed
# ══════════════════════════════════════════════════════════════════
def parse_dataset1(folder: Path) -> pd.DataFrame:
    """
    Parse: aaroncbastian/crypto-news-headlines-and-market-prices-by-date
    Expected columns: title, date/published, price or btc_price, etc.
    """
    print("\n[Dataset 1] Parsing crypto-news-headlines-and-market-prices...")

    # Find CSV file
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"  ❌ No CSV found in {folder}")
        return pd.DataFrame()

    dfs = []
    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            print(f"  Columns: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ Error reading {csv_file}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Raw rows: {len(df)}")

    # ── Normalize column names ────────────────────────────────────
    df.columns = df.columns.str.lower().str.strip()

    # Find title column
    title_col = next((c for c in df.columns if "title" in c or "headline" in c), None)
    if not title_col:
        print("  ❌ No title/headline column found")
        return pd.DataFrame()

    # Find date column
    date_col = next((c for c in df.columns
                     if any(x in c for x in ["date", "published", "time", "timestamp"])), None)

    # Find price columns
    price_col    = next((c for c in df.columns
                         if any(x in c for x in ["btc_price", "price_at", "open", "close", "price"])
                         and "15m" not in c and "1h" not in c), None)
    price_15m    = next((c for c in df.columns if "15m" in c or "15min" in c), None)
    price_1h     = next((c for c in df.columns if "1h" in c or "1hour" in c or "60m" in c), None)

    print(f"  title={title_col}, date={date_col}, price={price_col}, "
          f"15m={price_15m}, 1h={price_1h}")

    # ── Build standardized DataFrame ─────────────────────────────
    rows = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title or len(title.split()) < 3:
            continue

        # Parse date
        try:
            if date_col:
                pub_dt = pd.to_datetime(row[date_col], utc=True)
            else:
                pub_dt = datetime.now(timezone.utc)
            if pd.isnull(pub_dt):
                continue
        except Exception:
            continue

        # Parse prices
        try:
            btc_now = float(row[price_col]) if price_col else None
            btc_15m = float(row[price_15m]) if price_15m else None
            btc_1h  = float(row[price_1h])  if price_1h  else None
        except (TypeError, ValueError):
            btc_now = btc_15m = btc_1h = None

        rows.append({
            "title":            title,
            "link":             str(row.get("link", row.get("url", ""))),
            "published":        pub_dt.isoformat(),
            "channel":          str(row.get("source", row.get("channel", "kaggle_d1"))),
            "btc_price_at_news": btc_now,
            "btc_price_15m":    btc_15m,
            "btc_price_1h":     btc_1h,
            "source":           "kaggle_d1",
        })

    result = pd.DataFrame(rows)
    print(f"  Parsed rows: {len(result)}")
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 3 — PARSE DATASET 2 (oliviervha/crypto-news)
# Large dataset — needs BTC price added
# ══════════════════════════════════════════════════════════════════
def parse_dataset2(folder: Path) -> pd.DataFrame:
    """
    Parse: oliviervha/crypto-news
    Expected columns: title, text, date, sentiment, etc.
    No BTC price — will be added via Binance API.
    """
    print("\n[Dataset 2] Parsing crypto-news+ (oliviervha)...")

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"  ❌ No CSV found in {folder}")
        return pd.DataFrame()

    dfs = []
    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            print(f"  Columns: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  Raw rows: {len(df)}")

    title_col = next((c for c in df.columns if "title" in c or "headline" in c), None)
    date_col  = next((c for c in df.columns
                      if any(x in c for x in ["date", "published", "time"])), None)

    if not title_col:
        print("  ❌ No title column found")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title or len(title.split()) < 3:
            continue
        try:
            pub_dt = pd.to_datetime(row[date_col], utc=True) if date_col else datetime.now(timezone.utc)
            if pd.isnull(pub_dt):
                continue
        except Exception:
            continue

        rows.append({
            "title":             title,
            "link":              str(row.get("link", row.get("url", ""))),
            "published":         pub_dt.isoformat(),
            "channel":           str(row.get("source", row.get("channel", "kaggle_d2"))),
            "btc_price_at_news": None,   # will be filled by add_btc_prices()
            "btc_price_15m":     None,
            "btc_price_1h":      None,
            "source":            "kaggle_d2",
        })

    result = pd.DataFrame(rows)
    print(f"  Parsed rows: {len(result)}")
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 4 — ADD BTC PRICES (for rows missing prices)
# ══════════════════════════════════════════════════════════════════
def get_btc_price_at(timestamp_ms: int) -> float | None:
    """Get BTC/USDT close price at specific timestamp from Binance."""
    try:
        resp = requests.get(BINANCE_URL, params={
            "symbol":    "BTCUSDT",
            "interval":  "1m",
            "startTime": timestamp_ms,
            "limit":     1,
        }, timeout=10)
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            return float(data[0][4])  # close price
        return None
    except Exception:
        return None


def add_btc_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing BTC prices using Binance API.
    Only fetches for rows where btc_price_at_news is None.
    """
    missing_mask = df["btc_price_at_news"].isna()
    missing_count = missing_mask.sum()

    if missing_count == 0:
        print("  ✅ All rows already have BTC prices")
        return df

    print(f"\n[BTC Prices] Fetching for {missing_count} rows missing prices...")
    print(f"  Estimated time: {missing_count * 0.35 / 60:.1f} minutes")

    indices = df[missing_mask].index.tolist()

    for i, idx in enumerate(tqdm(indices, desc="Fetching BTC prices")):
        try:
            pub_dt   = pd.to_datetime(df.at[idx, "published"], utc=True)
            ts_ms    = int(pub_dt.timestamp() * 1000)
            ts_15m   = ts_ms + 15 * 60 * 1000
            ts_1h    = ts_ms + 60 * 60 * 1000

            p_now = get_btc_price_at(ts_ms)
            time.sleep(0.1)
            p_15m = get_btc_price_at(ts_15m)
            time.sleep(0.1)
            p_1h  = get_btc_price_at(ts_1h)
            time.sleep(0.1)

            df.at[idx, "btc_price_at_news"] = p_now
            df.at[idx, "btc_price_15m"]     = p_15m
            df.at[idx, "btc_price_1h"]      = p_1h

        except Exception as e:
            continue  # leave as None, will be dropped later

        # Save checkpoint every 500 rows
        if (i + 1) % 500 == 0:
            df.to_csv(HERE / "news_group1_checkpoint.csv", index=False)
            print(f"  💾 Checkpoint saved at {i+1}/{missing_count}")

    return df


# ══════════════════════════════════════════════════════════════════
# STEP 5 — COMPUTE LABELS + CLEAN
# ══════════════════════════════════════════════════════════════════
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute btc_change_15m, is_impactful_15m, direction_15m
    Same logic as production_system_v5.py load_data()
    """
    print("\n[Labels] Computing impact labels...")

    df["btc_price_at_news"] = pd.to_numeric(df["btc_price_at_news"], errors="coerce")
    df["btc_price_15m"]     = pd.to_numeric(df["btc_price_15m"],     errors="coerce")
    df["btc_price_1h"]      = pd.to_numeric(df["btc_price_1h"],      errors="coerce")

    # Drop rows with missing prices
    before = len(df)
    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    print(f"  Dropped {before - len(df)} rows with missing prices → {len(df)} remain")

    # Drop rows with zero/invalid prices
    df = df[df["btc_price_at_news"] > 0]

    # Compute changes
    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["btc_change_1h"]    = (df["btc_price_1h"]  - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["abs_change_1h"]    = df["btc_change_1h"].abs()

    # Labels — same thresholds as production_system_v8.py
    THRESHOLD_15M = 0.5
    THRESHOLD_1H  = 0.5
    df["is_impactful_15m"] = (df["abs_change_15m"] >= THRESHOLD_15M).astype(int)
    df["is_impactful_1h"]  = (df["abs_change_1h"]  >= THRESHOLD_1H).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)
    df["direction_1h"]     = (df["btc_change_1h"]  > 0).astype(int)
    df["confidence_label"] = df["abs_change_15m"].clip(0, 3) / 3.0

    # Add default sentiment columns (will be filled by score_sentiment.py)
    for col in ["sentiment", "sentiment_score", "weight", "confidence",
                "prob_positive", "prob_negative", "prob_neutral"]:
        if col not in df.columns:
            df[col] = 0 if col not in ["sentiment"] else "neutral"

    df["weight"] = df["weight"].replace(0, 5)  # default weight=5

    impact_rate = df["is_impactful_15m"].mean()
    print(f"  Impact rate 15m: {impact_rate:.1%}")
    print(f"  Impact rate 1h:  {df['is_impactful_1h'].mean():.1%}")

    return df


# ══════════════════════════════════════════════════════════════════
# STEP 6 — DEDUPLICATE + MERGE WITH EXISTING
# ══════════════════════════════════════════════════════════════════
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate headlines by title hash."""
    before = len(df)
    df["title_hash"] = df["title"].fillna("").str.lower().apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
    )
    df = df.drop_duplicates(subset=["title_hash"])
    print(f"  Deduplication: {before} → {len(df)} rows")
    return df


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge with existing news_cleaned_filtered.csv
    Keep only NEW rows not already in existing dataset.
    """
    if not EXISTING_CSV.exists():
        print(f"  ⚠️  {EXISTING_CSV} not found — using new data only")
        return new_df

    existing = pd.read_csv(EXISTING_CSV, low_memory=False)
    print(f"\n[Merge] Existing CSV: {len(existing)} rows")

    # Build hash set of existing titles
    if "title_hash" not in existing.columns:
        existing["title_hash"] = existing["title"].fillna("").str.lower().apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
        )

    existing_hashes = set(existing["title_hash"])
    new_only = new_df[~new_df["title_hash"].isin(existing_hashes)]
    print(f"  New unique rows: {len(new_only)}")
    print(f"  Duplicates skipped: {len(new_df) - len(new_only)}")

    # Align columns with existing CSV
    for col in existing.columns:
        if col not in new_only.columns:
            new_only[col] = np.nan

    return new_only[existing.columns]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  GROUP 1 KAGGLE DATASETS FETCHER")
    print("  Datasets: aaroncbastian + oliviervha")
    print("=" * 60)

    all_frames = []

    # ── Dataset 1: aaroncbastian ──────────────────────────────────
    d1_dir = DOWNLOAD_DIR / "dataset1_aaroncbastian"
    if not d1_dir.exists() or not list(d1_dir.glob("*.csv")):
        success = download_kaggle_dataset(
            "aaroncbastian/crypto-news-headlines-and-market-prices-by-date",
            d1_dir
        )
        if not success:
            print("  Skipping dataset 1...")
    else:
        print(f"\n✅ Dataset 1 already downloaded at {d1_dir}")

    df1 = parse_dataset1(d1_dir)
    if not df1.empty:
        all_frames.append(df1)
        print(f"  Dataset 1: {len(df1)} rows added")

    # ── Dataset 2: oliviervha ─────────────────────────────────────
    d2_dir = DOWNLOAD_DIR / "dataset2_oliviervha"
    if not d2_dir.exists() or not list(d2_dir.glob("*.csv")):
        success = download_kaggle_dataset(
            "oliviervha/crypto-news",
            d2_dir
        )
        if not success:
            print("  Skipping dataset 2...")
    else:
        print(f"\n✅ Dataset 2 already downloaded at {d2_dir}")

    df2 = parse_dataset2(d2_dir)
    if not df2.empty:
        all_frames.append(df2)
        print(f"  Dataset 2: {len(df2)} rows added")

    if not all_frames:
        print("\n❌ No data fetched. Check kaggle setup.")
        return

    # ── Combine ───────────────────────────────────────────────────
    print(f"\n[Combine] Merging {len(all_frames)} datasets...")
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  Combined rows: {len(combined)}")

    # ── Deduplicate ───────────────────────────────────────────────
    combined = deduplicate(combined)

    # ── Add BTC prices for rows missing them ──────────────────────
    combined = add_btc_prices(combined)

    # ── Compute labels ────────────────────────────────────────────
    combined = compute_labels(combined)

    # ── Merge with existing CSV ───────────────────────────────────
    final = merge_with_existing(combined)

    if final.empty:
        print("\n⚠️  No new rows to add — all already in existing CSV")
        return

    # ── Sort by date ──────────────────────────────────────────────
    final["published"] = pd.to_datetime(final["published"], utc=True, errors="coerce")
    final = final.sort_values("published").reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────
    final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Saved {len(final)} new rows → {OUTPUT_CSV}")
    print(f"{'='*60}")
    print(f"\n  Impact rate 15m : {final['is_impactful_15m'].mean():.1%}")
    print(f"  Impact rate 1h  : {final['is_impactful_1h'].mean():.1%}")
    print(f"  Date range      : {final['published'].min()} → {final['published'].max()}")
    print(f"\nNext steps:")
    print(f"  1. Score sentiment:  python score_sentiment.py {OUTPUT_CSV}")
    print(f"  2. Merge with main:  python merge_datasets.py")
    print(f"  3. Re-train model:   python production_system_v5.py")


if __name__ == "__main__":
    main()