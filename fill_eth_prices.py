"""
fill_eth_prices.py
==================
Fills eth_price_at_news / eth_price_15m / eth_price_1h for rows in
news_cleaned_filtered_scored.csv that are missing ETH prices.

Strategy: bulk-download ETHUSDT 15-minute klines from Binance public API
(~165 requests for 4+ years), then join by floored timestamp — no per-row calls.

Usage:
    .venv311/bin/python fill_eth_prices.py
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

HERE      = Path(__file__).parent
CSV_PATH  = HERE / "news_cleaned_filtered_scored.csv"
CACHE_CSV = HERE / "eth_15m_klines.csv"   # local cache to avoid re-downloading

BINANCE      = "https://api.binance.com/api/v3/klines"
INTERVAL     = "15m"
INTERVAL_MS  = 15 * 60 * 1000
BATCH        = 1000
DELAY        = 0.12   # seconds between requests (Binance limit: ~10 req/s)


# ── 1. Load CSV, find missing rows ────────────────────────────────
def load_csv():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True, errors="coerce")
    missing = df["eth_price_at_news"].isna() & df["published"].notna()
    print(f"  Total rows       : {len(df):,}")
    print(f"  Missing ETH price: {missing.sum():,}")
    return df, missing


# ── 2. Bulk-download 15m klines (or load from local cache) ───────
def fetch_klines(start_ms: int, end_ms: int) -> dict:
    """Returns {open_time_ms: open_price} for ETHUSDT 15m candles."""

    if CACHE_CSV.exists():
        cached = pd.read_csv(CACHE_CSV)
        if int(cached["open_time"].min()) <= start_ms and int(cached["open_time"].max()) >= end_ms:
            print(f"  Loading klines from cache ({len(cached):,} rows)")
            return dict(zip(cached["open_time"].astype(int), cached["open_price"].astype(float)))

    print(f"  Downloading ETHUSDT 15m klines from Binance...")
    klines = []
    current = start_ms - INTERVAL_MS  # 1 candle buffer

    while current <= end_ms + INTERVAL_MS:
        try:
            resp = requests.get(BINANCE, params={
                "symbol":    "ETHUSDT",
                "interval":  INTERVAL,
                "startTime": current,
                "limit":     BATCH,
            }, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  ⚠ Request error: {e} — retrying in 2s")
            time.sleep(2)
            continue

        if not isinstance(data, list) or not data:
            break

        klines.extend(data)
        current = int(data[-1][0]) + 1
        last_dt = datetime.fromtimestamp(int(data[-1][0]) / 1000, tz=timezone.utc).date()

        if len(klines) % 10000 < BATCH:
            print(f"    {len(klines):>7,} candles  up to {last_dt}")

        time.sleep(DELAY)

    print(f"  Downloaded {len(klines):,} candles total")

    # Save cache
    cache_df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    cache_df = cache_df[["open_time","open"]].rename(columns={"open":"open_price"})
    cache_df.to_csv(CACHE_CSV, index=False)
    print(f"  Cached to {CACHE_CSV.name}")

    return dict(zip(cache_df["open_time"].astype(int), cache_df["open_price"].astype(float)))


# ── 3. Fill ETH prices using the kline map ───────────────────────
def fill_prices(df: pd.DataFrame, missing_mask, price_map: dict) -> pd.DataFrame:
    indices = df[missing_mask].index
    filled = 0

    for idx in indices:
        pub = df.at[idx, "published"]
        ts_ms = int(pub.timestamp() * 1000)

        # Floor to nearest 15m candle boundary
        candle_ts  = (ts_ms // INTERVAL_MS) * INTERVAL_MS
        p_now  = price_map.get(candle_ts)
        p_15m  = price_map.get(candle_ts + INTERVAL_MS)
        p_1h   = price_map.get(candle_ts + 4 * INTERVAL_MS)

        if p_now is None:
            continue

        df.at[idx, "eth_price_at_news"] = round(p_now, 2)
        df.at[idx, "eth_price_15m"]     = round(p_15m, 2)  if p_15m else np.nan
        df.at[idx, "eth_price_1h"]      = round(p_1h,  2)  if p_1h  else np.nan

        if p_15m:
            df.at[idx, "eth_pct_change_15m"] = round((p_15m - p_now) / p_now * 100, 6)
        if p_1h:
            df.at[idx, "eth_pct_change_1h"]  = round((p_1h  - p_now) / p_now * 100, 6)

        filled += 1
        if filled % 5000 == 0:
            print(f"    Filled {filled:,} / {len(indices):,}")

    return df, filled


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FILL ETH PRICES — Binance 15m klines bulk join")
    print("=" * 60)

    print("\n[1/4] Loading CSV...")
    df, missing_mask = load_csv()

    dates     = df[missing_mask]["published"].dropna()
    start_ms  = int(dates.min().timestamp() * 1000)
    end_ms    = int(dates.max().timestamp() * 1000)
    print(f"  Range: {dates.min().date()} → {dates.max().date()}")

    print("\n[2/4] Fetching ETH klines...")
    price_map = fetch_klines(start_ms, end_ms)
    print(f"  Price map: {len(price_map):,} candles")

    print("\n[3/4] Filling ETH prices...")
    df, filled = fill_prices(df, missing_mask, price_map)
    still_missing = df["eth_price_at_news"].isna().sum()
    print(f"  Filled   : {filled:,}")
    print(f"  Still NaN: {still_missing:,}")

    print("\n[4/4] Saving CSV...")
    df["published"] = df["published"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df.to_csv(CSV_PATH, index=False)
    print(f"  ✅ Saved {len(df):,} rows → {CSV_PATH.name}")


if __name__ == "__main__":
    main()
