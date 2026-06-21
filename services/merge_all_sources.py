"""
merge_all_sources.py
====================
Merges three Kaggle files into news_cleaned_filtered.csv template format.

Sources:
  1. bitcoin_sentiments_21_24.csv  — headlines + exact timestamps + sentiment scores
  2. BTC.csv                       — daily OHLC + 10 article titles per day
  3. ETH.csv                       — daily OHLC + 10 article titles per day

For each new row:
  - Fetches BTC + ETH prices from Binance (one call per asset per row)
  - Maps all columns to the template schema
  - Deduplicates against existing news_cleaned_filtered.csv

Checkpoints every 200 rows — safe to interrupt and resume.

Estimated time:
  Source 1:  ~11K rows  × 2 calls × 0.22s ≈  1.4h
  Source 2+3: ~36K rows × 2 calls × 0.22s ≈  4.4h
  Total: ~6 hours (or run --source 1/2/3 separately)

Usage:
  python services/merge_all_sources.py              # all three sources
  python services/merge_all_sources.py --source 1   # bitcoin_sentiments only
  python services/merge_all_sources.py --source 2   # BTC.csv only
  python services/merge_all_sources.py --source 3   # ETH.csv only
  python services/merge_all_sources.py --dry-run    # stats only, no write
"""

import argparse
import ast
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE      = Path(__file__).parent.parent
MAIN_CSV  = HERE / "news_cleaned_filtered.csv"
CKPT_DIR  = HERE / "services"
BINANCE   = "https://api.binance.com/api/v3/klines"
DELAY     = 0.22

FOMC_DATES = [
    "2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-02-02","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
]
_FOMC_SET = set()
for d in FOMC_DATES:
    dt = pd.Timestamp(d)
    for offset in range(-3, 4):
        _FOMC_SET.add((dt + pd.Timedelta(days=offset)).date())


def _hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]


def _fomc(ts: pd.Timestamp) -> int:
    return int(ts.date() in _FOMC_SET)


def _fetch(symbol: str, ts_ms: int) -> tuple:
    """Fetch open/+15m/+1h prices for symbol from Binance. Returns (now, 15m, 1h)."""
    try:
        resp = requests.get(BINANCE, params={
            "symbol": symbol, "interval": "1m",
            "startTime": ts_ms, "limit": 62,
        }, timeout=10)
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None, None, None
        def c(i): return float(data[i][4]) if i < len(data) else None
        return c(0), c(15), c(min(61, len(data) - 1))
    except Exception:
        return None, None, None


def _macro_flags(ts: pd.Timestamp) -> dict:
    h   = ts.hour
    dow = ts.dayofweek
    return {
        "is_weekend":       int(dow >= 5),
        "is_low_liquidity": int(2 <= h <= 6),
        "is_us_hours":      int(13 <= h <= 21),
        "is_asia_hours":    int(0 <= h <= 8),
        "fomc_week":        _fomc(ts),
        "hour_utc":         h,
        "day_of_week":      dow,
        "hour_of_day":      h,
    }


def _sentiment_from_score(score: float) -> dict:
    """Map scalar sentiment score (-1..1) to sentiment columns."""
    s = float(score)
    net = s
    # Fixed: most-extreme negative first so all tiers are reachable
    disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
            -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
    pp = round(max(0, min(1, 0.5 + s * 0.4)), 4)
    pn = round(max(0, min(1, 0.5 - s * 0.4)), 4)
    pu = round(max(0, 1 - pp - pn), 4)
    # Fixed: respect neutral dominance + confidence = assigned class
    if pu > max(pp, pn):
        disc = 0
        sentiment = "neutral"
        conf = pu
    else:
        sentiment = "positive" if disc > 0 else ("negative" if disc < 0 else "neutral")
        conf = pp if disc > 0 else (pn if disc < 0 else pu)
    return {
        "sentiment":       sentiment,
        "sentiment_score": disc,
        "weight":          max(5, min(10, round(conf * 10))),
        "confidence":      round(conf, 4),
        "prob_positive":   pp,
        "prob_negative":   pn,
        "prob_neutral":    pu,
        "sentiment_binary": int(disc > 0),
    }


def _default_sentiment() -> dict:
    return {
        "sentiment": "neutral", "sentiment_score": 0,
        "weight": 5, "confidence": 0.5,
        "prob_positive": 0.33, "prob_negative": 0.33, "prob_neutral": 0.34,
        "sentiment_binary": 0,
    }


def _build_row(title, published, channel, btc_now, btc_15m, btc_1h,
               eth_now, eth_15m, eth_1h, sentiment_dict) -> dict:
    ts  = pd.to_datetime(published, utc=True)
    mac = _macro_flags(ts)

    def pct(a, b):
        try: return round((float(b) - float(a)) / float(a) * 100, 6)
        except: return np.nan

    row = {
        "title":             str(title).strip(),
        "link":              "",
        "channel":           channel,
        "published":         ts.isoformat(),
        "btc_price_at_news": btc_now,
        "btc_price_15m":     btc_15m,
        "btc_price_1h":      btc_1h,
        "eth_price_at_news": eth_now,
        "eth_price_15m":     eth_15m,
        "eth_price_1h":      eth_1h,
        "news_type":         "market_analysis",
        "word_count":        len(str(title).split()),
        "is_spam":           False,
        "is_relevant":       True,
        "_hash":             _hash(str(title)),
        "btc_pct_change_15m": pct(btc_now, btc_15m),
        "btc_pct_change_1h":  pct(btc_now, btc_1h),
        "eth_pct_change_15m": pct(eth_now, eth_15m),
        "eth_pct_change_1h":  pct(eth_now, eth_1h),
    }
    row.update(mac)
    row.update(sentiment_dict)
    return row


# ══════════════════════════════════════════════════════════════════
# SOURCE 1 — bitcoin_sentiments_21_24.csv
# ══════════════════════════════════════════════════════════════════
def parse_source1() -> pd.DataFrame:
    path = HERE / "bitcoin_sentiments_21_24.csv"
    df   = pd.read_csv(path, low_memory=False)
    df.columns = ["published", "title", "sentiment_raw"]
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published", "title"])
    df = df[df["title"].str.split().str.len() >= 4]
    df["channel"]  = "kaggle_btc_sent"
    df["source_id"] = 1
    print(f"  Source 1 (bitcoin_sentiments): {len(df):,} rows")
    return df


# ══════════════════════════════════════════════════════════════════
# SOURCE 2 — BTC.csv (explode daily articles)
# ══════════════════════════════════════════════════════════════════
def parse_source2() -> pd.DataFrame:
    path = HERE / "BTC.csv"
    df   = pd.read_csv(path, low_memory=False)
    df["begins_at"] = pd.to_datetime(df["begins_at"], utc=True, errors="coerce")
    rows = []
    for _, r in df.iterrows():
        try:
            articles = ast.literal_eval(r["articles"])
        except Exception:
            continue
        # Set timestamp to noon UTC of the date
        date_noon = r["begins_at"].normalize() + pd.Timedelta(hours=12)
        for title in articles:
            title = str(title).strip().replace("\n", " ")
            if len(title.split()) < 4:
                continue
            rows.append({
                "published":    date_noon,
                "title":        title,
                "sentiment_raw": None,
                "channel":      "kaggle_btc_daily",
                "source_id":    2,
            })
    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["title"])
    print(f"  Source 2 (BTC.csv):  {len(result):,} rows (from {len(df):,} days)")
    return result


# ══════════════════════════════════════════════════════════════════
# SOURCE 3 — ETH.csv (explode daily articles)
# ══════════════════════════════════════════════════════════════════
def parse_source3() -> pd.DataFrame:
    path = HERE / "ETH.csv"
    df   = pd.read_csv(path, low_memory=False)
    df["begins_at"] = pd.to_datetime(df["begins_at"], utc=True, errors="coerce")
    rows = []
    for _, r in df.iterrows():
        try:
            articles = ast.literal_eval(r["articles"])
        except Exception:
            continue
        date_noon = r["begins_at"].normalize() + pd.Timedelta(hours=12)
        for title in articles:
            title = str(title).strip().replace("\n", " ")
            if len(title.split()) < 4:
                continue
            rows.append({
                "published":    date_noon,
                "title":        title,
                "sentiment_raw": None,
                "channel":      "kaggle_eth_daily",
                "source_id":    3,
            })
    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["title"])
    print(f"  Source 3 (ETH.csv):  {len(result):,} rows (from {len(df):,} days)")
    return result


# ══════════════════════════════════════════════════════════════════
# FETCH PRICES + BUILD ROWS
# ══════════════════════════════════════════════════════════════════
def fetch_and_build(df: pd.DataFrame, existing_hashes: set,
                    ckpt_path: Path) -> pd.DataFrame:
    # Remove duplicates against existing CSV
    df["_h"] = df["title"].apply(lambda x: _hash(str(x)))
    new_only  = df[~df["_h"].isin(existing_hashes)].copy().reset_index(drop=True)
    print(f"  New rows after dedup: {len(new_only):,}  (skipped {len(df)-len(new_only):,} duplicates)")

    if len(new_only) == 0:
        return pd.DataFrame()

    # Resume from checkpoint if exists
    done_hashes = set()
    done_rows   = []
    if ckpt_path.exists():
        ckpt = pd.read_csv(ckpt_path, low_memory=False)
        done_hashes = set(ckpt["_hash"].dropna())
        done_rows   = ckpt.to_dict("records")
        print(f"  Checkpoint: {len(done_rows):,} rows already fetched")

    to_fetch = new_only[~new_only["_h"].isin(done_hashes)].reset_index(drop=True)
    total    = len(to_fetch)
    failed   = 0

    print(f"  Fetching prices for {total:,} rows...")

    for i, r in to_fetch.iterrows():
        ts    = pd.to_datetime(r["published"], utc=True)
        ts_ms = int(ts.timestamp() * 1000)

        btc_now, btc_15m, btc_1h = _fetch("BTCUSDT", ts_ms)
        time.sleep(DELAY)
        eth_now, eth_15m, eth_1h = _fetch("ETHUSDT", ts_ms)
        time.sleep(DELAY)

        if btc_now is None:
            failed += 1
            continue

        sent = (_sentiment_from_score(float(r["sentiment_raw"]))
                if r.get("sentiment_raw") is not None and not pd.isna(r.get("sentiment_raw", float("nan")))
                else _default_sentiment())

        row = _build_row(
            r["title"], r["published"], r["channel"],
            btc_now, btc_15m, btc_1h,
            eth_now, eth_15m, eth_1h,
            sent,
        )
        done_rows.append(row)

        if (len(done_rows)) % 200 == 0:
            pd.DataFrame(done_rows).to_csv(ckpt_path, index=False)
            pct = (i + 1) / total * 100
            eta = (total - i - 1) * DELAY * 2 / 3600
            print(f"    [{i+1:>6}/{total}]  {pct:.0f}%  failed={failed}  eta={eta:.1f}h")

    pd.DataFrame(done_rows).to_csv(ckpt_path, index=False)
    print(f"  Done. Failed fetches: {failed:,}")
    return pd.DataFrame(done_rows)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main(sources: list[int], dry_run: bool):
    print("=" * 60)
    print("  MERGE ALL SOURCES → news_cleaned_filtered.csv")
    print(f"  Sources: {sources}  |  dry_run={dry_run}")
    print("=" * 60)

    # Load existing hashes
    existing    = pd.read_csv(MAIN_CSV, low_memory=False)
    ex_hashes   = set(existing["title"].fillna("").apply(lambda x: _hash(str(x))))
    target_cols = list(existing.columns)
    print(f"\n  Existing rows: {len(existing):,}")

    all_new = []

    source_parsers = {1: parse_source1, 2: parse_source2, 3: parse_source3}
    source_ckpts   = {
        1: CKPT_DIR / "merge_s1_ckpt.csv",
        2: CKPT_DIR / "merge_s2_ckpt.csv",
        3: CKPT_DIR / "merge_s3_ckpt.csv",
    }

    for src in sources:
        print(f"\n── Source {src} ──────────────────────────────")
        df_src = source_parsers[src]()

        if dry_run:
            df_src["_h"] = df_src["title"].apply(lambda x: _hash(str(x)))
            new_count = (~df_src["_h"].isin(ex_hashes)).sum()
            print(f"  [dry-run] Would add ~{new_count:,} new rows")
            pub = pd.to_datetime(df_src["published"], utc=True, errors="coerce")
            yr  = pub.dt.year
            print(f"  Year distribution:")
            for y, c in yr.value_counts().sort_index().items():
                print(f"    {y}: {c:,}")
            continue

        result = fetch_and_build(df_src, ex_hashes, source_ckpts[src])
        if not result.empty:
            all_new.append(result)
            # Update hashes so cross-source duplicates are caught
            if "_hash" in result.columns:
                ex_hashes.update(result["_hash"].dropna())

    if dry_run:
        return

    if not all_new:
        print("\n  No new rows to add.")
        return

    # ── Align columns + append ────────────────────────────────────
    combined = pd.concat(all_new, ignore_index=True)
    for col in target_cols:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[target_cols]

    merged = pd.concat([existing, combined], ignore_index=True)
    merged["published"] = pd.to_datetime(merged["published"], utc=True, errors="coerce")
    merged = merged.sort_values("published").reset_index(drop=True)
    merged.to_csv(MAIN_CSV, index=False)

    # Remove checkpoints
    for src in sources:
        p = source_ckpts[src]
        if p.exists():
            p.unlink()

    print(f"\n{'='*60}")
    print(f"  ✅ Merged CSV saved → {MAIN_CSV.name}")
    print(f"  Before: {len(existing):,}  |  Added: {len(combined):,}  |  After: {len(merged):,}")
    yr = pd.to_datetime(merged["published"], utc=True, errors="coerce").dt.year
    print(f"\n  Year distribution:")
    for y, c in yr.value_counts().sort_index().items():
        print(f"    {y}: {c:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, choices=[1, 2, 3],
                        help="Run only one source (1=bitcoin_sentiments, 2=BTC, 3=ETH)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = [args.source] if args.source else [1, 2, 3]
    main(sources=sources, dry_run=args.dry_run)
