"""
fetch_eth_prices.py
===================
Fetches ETH/USDT prices from Binance for all rows in
news_cleaned_filtered.csv that are missing eth_price_at_news.

Fills 3 columns per row:
  eth_price_at_news  — price at news publication time
  eth_price_15m      — price 15 minutes later
  eth_price_1h       — price 1 hour later

Then recomputes:
  eth_pct_change_15m
  eth_pct_change_1h

Strategy: one Binance call per row (62 × 1m candles covers T, T+15m, T+1h)
Estimated time: 30,814 rows × 0.22s ≈ ~1.9 hours

Usage:
  python services/fetch_eth_prices.py
  python services/fetch_eth_prices.py --dry-run
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE      = Path(__file__).parent.parent
MAIN_CSV  = HERE / "news_cleaned_filtered_scored.csv"
CKPT_CSV  = HERE / "services" / "eth_prices_checkpoint.csv"
BINANCE   = "https://api.binance.com/api/v3/klines"
DELAY     = 0.22   # seconds between calls — stays under Binance 1200 req/min limit


def fetch_eth_window(ts_ms: int) -> tuple[float | None, float | None, float | None]:
    """
    Fetch 62 × 1m candles starting at ts_ms.
    One call covers: T (index 0), T+15m (index 15), T+1h (index 61).
    """
    try:
        resp = requests.get(BINANCE, params={
            "symbol":    "ETHUSDT",
            "interval":  "1m",
            "startTime": ts_ms,
            "limit":     62,
        }, timeout=10)
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return None, None, None
        def close(idx):
            return float(data[idx][4]) if idx < len(data) else None
        return close(0), close(15), close(min(61, len(data) - 1))
    except Exception:
        return None, None, None


def main(dry_run=False, from_date=None):
    print("=" * 60)
    print("  ETH PRICE FETCHER")
    print(f"  Source: {MAIN_CSV.name}")
    if from_date:
        print(f"  From  : {from_date} onward")
    print("=" * 60)

    # ── Load — resume from checkpoint if available ────────────────
    if CKPT_CSV.exists():
        print(f"\n⚡ Resuming from checkpoint: {CKPT_CSV.name}")
        df = pd.read_csv(CKPT_CSV, low_memory=False)
    else:
        df = pd.read_csv(MAIN_CSV, low_memory=False)

    for col in ["eth_price_at_news", "eth_price_15m", "eth_price_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")

    missing_mask = df["eth_price_at_news"].isna()

    # Apply date filter if given — only fetch for rows on/after from_date
    if from_date:
        import pandas as pd2
        cutoff = pd2.Timestamp(from_date, tz="UTC")
        missing_mask = missing_mask & (df["published"] >= cutoff)

    missing_total = missing_mask.sum()

    print(f"\n  Total rows      : {len(df):,}")
    print(f"  Missing ETH     : {missing_total:,}")
    print(f"  Already filled  : {(~df['eth_price_at_news'].isna()).sum():,}")
    print(f"  Est. time       : {missing_total * DELAY / 3600:.1f} hours")

    if dry_run:
        print("\n  [dry-run] No changes written.")
        return

    if missing_total == 0:
        print("\n  ✅ All rows already have ETH prices.")
        return

    # ── Fetch ─────────────────────────────────────────────────────
    indices   = df[missing_mask].index.tolist()
    failed    = 0
    pub_col   = df["published"]  # already parsed above

    print(f"\n  Fetching {missing_total:,} rows...\n")

    for i, idx in enumerate(indices):
        ts_ms          = int(pub_col.iloc[idx].timestamp() * 1000)
        p_now, p_15m, p_1h = fetch_eth_window(ts_ms)

        if p_now is None:
            failed += 1
        else:
            df.at[idx, "eth_price_at_news"] = p_now
            df.at[idx, "eth_price_15m"]     = p_15m
            df.at[idx, "eth_price_1h"]      = p_1h

        time.sleep(DELAY)

        # Progress + checkpoint every 200 rows
        if (i + 1) % 200 == 0:
            done_pct = (i + 1) / missing_total * 100
            remaining_h = (missing_total - i - 1) * DELAY / 3600
            print(f"  [{i+1:>6}/{missing_total}]  {done_pct:.0f}%  "
                  f"failed={failed}  eta={remaining_h:.1f}h")
            df.to_csv(CKPT_CSV, index=False)

    # Final checkpoint save
    df.to_csv(CKPT_CSV, index=False)

    # ── Recompute pct change columns ──────────────────────────────
    df["eth_price_at_news"] = pd.to_numeric(df["eth_price_at_news"], errors="coerce")
    df["eth_price_15m"]     = pd.to_numeric(df["eth_price_15m"],     errors="coerce")
    df["eth_price_1h"]      = pd.to_numeric(df["eth_price_1h"],      errors="coerce")

    mask = df["eth_price_at_news"].notna() & (df["eth_price_at_news"] > 0)
    df.loc[mask, "eth_pct_change_15m"] = (
        (df.loc[mask, "eth_price_15m"] - df.loc[mask, "eth_price_at_news"])
        / df.loc[mask, "eth_price_at_news"] * 100
    ).round(6)
    df.loc[mask, "eth_pct_change_1h"] = (
        (df.loc[mask, "eth_price_1h"] - df.loc[mask, "eth_price_at_news"])
        / df.loc[mask, "eth_price_at_news"] * 100
    ).round(6)

    # ── Save to main CSV ──────────────────────────────────────────
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.sort_values("published").reset_index(drop=True)
    df.to_csv(MAIN_CSV, index=False)

    # Remove checkpoint after successful write
    if CKPT_CSV.exists():
        CKPT_CSV.unlink()

    # ── Summary ───────────────────────────────────────────────────
    filled = df["eth_price_at_news"].notna().sum()
    still_missing = df["eth_price_at_news"].isna().sum()
    print(f"\n{'='*60}")
    print(f"  ✅ Done — saved to {MAIN_CSV.name}")
    print(f"  ETH prices filled : {filled:,}")
    print(f"  Still missing     : {still_missing:,}  (Binance has no data that far back)")
    print(f"  Failed fetches    : {failed:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-date", default=None,
                        help="Only fetch rows published on/after this date (YYYY-MM-DD). "
                             "E.g. --from-date 2024-10-01")
    args = parser.parse_args()
    main(dry_run=args.dry_run, from_date=args.from_date)
