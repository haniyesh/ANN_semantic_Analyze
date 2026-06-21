"""
merge_bitcoin_sentiments.py
===========================
Merges bitcoin_sentiments_21_24.csv into news_cleaned_filtered.csv.

Steps:
  1. Load source file, drop the 1 duplicate with existing data
  2. Fetch BTC prices (at_news, +15m, +1h) from Binance — 1 call per row (62-candle window)
  3. Map all columns to news_cleaned_filtered.csv schema
  4. Append to news_cleaned_filtered.csv

Source columns:
  Date                → published
  Short Description   → title
  Accurate Sentiments → sentiment_score (range -1..1, real scores, weight=8)

Estimated time: ~40 minutes for 11,294 rows (1 Binance call per row at 0.2s each)

Usage:
  python services/merge_bitcoin_sentiments.py
  python services/merge_bitcoin_sentiments.py --dry-run    # show stats, no write
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE       = Path(__file__).parent.parent
SOURCE_CSV = HERE / "bitcoin_sentiments_21_24.csv"
MAIN_CSV   = HERE / "news_cleaned_filtered.csv"
CKPT_CSV   = HERE / "services" / "bitcoin_sentiments_checkpoint.csv"
BINANCE    = "https://api.binance.com/api/v3/klines"

FOMC_DATES = {
    "2023": ["2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
             "2023-09-20","2023-11-01","2023-12-13"],
    "2024": ["2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
             "2024-09-18","2024-11-07","2024-12-18"],
}
_FOMC_SET = set()
for dates in FOMC_DATES.values():
    for d in dates:
        dt = pd.Timestamp(d)
        for offset in range(-3, 4):
            _FOMC_SET.add((dt + pd.Timedelta(days=offset)).date())


def _fomc_week(ts: pd.Timestamp) -> int:
    return int(ts.date() in _FOMC_SET)


def _fetch_btc_window(ts_ms: int) -> tuple[float | None, float | None, float | None]:
    """Fetch 62 × 1m candles starting at ts_ms — covers T, T+15m, T+1h in one call."""
    try:
        resp = requests.get(BINANCE, params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": ts_ms, "limit": 62,
        }, timeout=10)
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return None, None, None
        def _close(candles, idx):
            return float(candles[idx][4]) if idx < len(candles) else None
        p_now = _close(data, 0)
        p_15m = _close(data, 15)
        p_1h  = _close(data, 61) if len(data) >= 62 else _close(data, len(data)-1)
        return p_now, p_15m, p_1h
    except Exception:
        return None, None, None


def _sentiment_to_probs(score: float) -> tuple[float, float, float]:
    """Map scalar score (-1..1) to (prob_positive, prob_negative, prob_neutral)."""
    s = float(score)
    if s > 0.1:
        pp = 0.5 + s * 0.4
        pn = (1 - pp) * 0.25
        pu = 1 - pp - pn
    elif s < -0.1:
        pn = 0.5 + (-s) * 0.4
        pp = (1 - pn) * 0.25
        pu = 1 - pp - pn
    else:
        pu = 0.6
        pp = 0.5 + s * 0.3
        pn = 1 - pp - pu
    return round(pp, 4), round(max(pn, 0), 4), round(max(pu, 0), 4)


def _title_hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]


def fetch_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch BTC prices for rows missing them. Saves checkpoint every 200 rows."""
    missing = df["btc_price_at_news"].isna()
    total   = missing.sum()
    if total == 0:
        print("  All rows already have BTC prices.")
        return df

    print(f"  Fetching BTC prices for {total:,} rows (~{total*0.22/60:.0f} min)...")
    indices = df[missing].index.tolist()

    for i, idx in enumerate(indices):
        ts_ms = int(pd.to_datetime(df.at[idx, "published"], utc=True).timestamp() * 1000)
        p_now, p_15m, p_1h = _fetch_btc_window(ts_ms)
        df.at[idx, "btc_price_at_news"] = p_now
        df.at[idx, "btc_price_15m"]     = p_15m
        df.at[idx, "btc_price_1h"]      = p_1h
        time.sleep(0.2)

        if (i + 1) % 200 == 0:
            df.to_csv(CKPT_CSV, index=False)
            pct = (i + 1) / total * 100
            print(f"    [{i+1:>5}/{total}] {pct:.0f}%  checkpoint saved")

    df.to_csv(CKPT_CSV, index=False)
    print(f"  Done. Checkpoint at {CKPT_CSV}")
    return df


def build_rows(source: pd.DataFrame) -> pd.DataFrame:
    """Convert source columns → news_cleaned_filtered.csv schema."""
    rows = []
    for _, r in source.iterrows():
        pub = pd.to_datetime(r["published"], utc=True)
        h   = pub.hour
        dow = pub.dayofweek
        score = float(r.get("sentiment_score", 0) or 0)
        pp, pn, pu = _sentiment_to_probs(score)

        sentiment = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        btc_now   = r.get("btc_price_at_news")
        btc_15m   = r.get("btc_price_15m")
        btc_1h    = r.get("btc_price_1h")

        def _pct(a, b):
            try:
                return round((float(b) - float(a)) / float(a) * 100, 6)
            except Exception:
                return np.nan

        rows.append({
            "title":             r["title"],
            "link":              "",
            "channel":           "kaggle_btc_sent",
            "published":         pub.isoformat(),
            "btc_price_at_news": btc_now,
            "btc_price_15m":     btc_15m,
            "btc_price_1h":      btc_1h,
            "eth_price_at_news": np.nan,
            "eth_price_15m":     np.nan,
            "eth_price_1h":      np.nan,
            "sentiment":         sentiment,
            "sentiment_score":   score,
            "weight":            8.0,       # real scores, better than proxy=5
            "confidence":        min(abs(score), 1.0),
            "prob_positive":     pp,
            "prob_negative":     pn,
            "prob_neutral":      pu,
            "news_type":         "market_analysis",  # will be re-classified by model
            "fomc_week":         _fomc_week(pub),
            "is_weekend":        int(dow >= 5),
            "is_low_liquidity":  int(2 <= h <= 6),
            "is_us_hours":       int(13 <= h <= 21),
            "is_asia_hours":     int(0 <= h <= 8),
            "hour_utc":          h,
            "day_of_week":       dow,
            "btc_pct_change_15m": _pct(btc_now, btc_15m),
            "btc_pct_change_1h":  _pct(btc_now, btc_1h),
            "eth_pct_change_15m": np.nan,
            "eth_pct_change_1h":  np.nan,
            "hour_of_day":       h,
            "word_count":        len(str(r["title"]).split()),
            "sentiment_binary":  int(score > 0),
            "is_spam":           False,
            "is_relevant":       True,
            "_hash":             _title_hash(str(r["title"])),
        })
    return pd.DataFrame(rows)


def main(dry_run=False):
    print("=" * 60)
    print("  MERGE bitcoin_sentiments_21_24 → news_cleaned_filtered")
    print("=" * 60)

    # ── Load source ───────────────────────────────────────────────
    raw = pd.read_csv(SOURCE_CSV, low_memory=False)
    raw.columns = ["published", "title", "sentiment_score"]
    raw["published"] = pd.to_datetime(raw["published"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["published", "title"])
    raw = raw[raw["title"].str.split().str.len() >= 4]   # skip very short titles
    print(f"  Source rows after basic filter: {len(raw):,}")

    # ── Remove duplicates with existing CSV ───────────────────────
    existing = pd.read_csv(MAIN_CSV, low_memory=False)
    existing_hashes = set(existing["title"].fillna("").str.lower().str.strip()
                          .apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:12]))
    raw["_h"] = raw["title"].fillna("").str.lower().str.strip().apply(
                    lambda x: hashlib.md5(x.encode()).hexdigest()[:12])
    new_only = raw[~raw["_h"].isin(existing_hashes)].copy().reset_index(drop=True)
    print(f"  After dedup vs existing: {len(new_only):,} new rows")

    pub = pd.to_datetime(new_only["published"], utc=True, errors="coerce")
    new_only["year"] = pub.dt.year
    print(f"  Year distribution:")
    for yr, cnt in new_only["year"].value_counts().sort_index().items():
        print(f"    {yr}: {cnt:,}")

    if dry_run:
        print("\n  [dry-run] No changes written.")
        return

    # ── Resume from checkpoint if exists ─────────────────────────
    if CKPT_CSV.exists():
        ckpt = pd.read_csv(CKPT_CSV, low_memory=False)
        ckpt_hashes = set(ckpt["_hash"].dropna())
        new_only["_h2"] = new_only["title"].fillna("").str.lower().str.strip().apply(
                              lambda x: hashlib.md5(x.encode()).hexdigest()[:12])
        already_done = new_only["_h2"].isin(ckpt_hashes).sum()
        print(f"\n  Checkpoint found: {len(ckpt):,} rows already fetched ({already_done} overlap)")
        remaining = new_only[~new_only["_h2"].isin(ckpt_hashes)].copy()
    else:
        ckpt     = pd.DataFrame()
        remaining = new_only.copy()

    # ── Build intermediate DataFrame for price fetch ──────────────
    work = remaining.copy()
    work["btc_price_at_news"] = np.nan
    work["btc_price_15m"]     = np.nan
    work["btc_price_1h"]      = np.nan

    # ── Fetch BTC prices ──────────────────────────────────────────
    print(f"\n[FETCH] {len(work):,} rows need BTC prices")
    work = fetch_prices(work)

    # ── Drop rows where fetch failed ──────────────────────────────
    before = len(work)
    work = work.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    print(f"  Dropped {before - len(work)} rows with failed price fetch → {len(work):,} remain")

    # ── Combine with checkpoint ───────────────────────────────────
    if not ckpt.empty:
        combined_source = pd.concat([ckpt, work], ignore_index=True)
    else:
        combined_source = work.copy()

    # ── Build final rows in target schema ────────────────────────
    print(f"\n[BUILD] Converting {len(combined_source):,} rows to schema...")
    final = build_rows(combined_source)

    # ── Verify column alignment ───────────────────────────────────
    target_cols = list(existing.columns)
    for c in target_cols:
        if c not in final.columns:
            final[c] = np.nan
    final = final[target_cols]

    # ── Append to main CSV ────────────────────────────────────────
    print(f"\n[MERGE]")
    print(f"  Existing rows : {len(existing):,}")
    print(f"  New rows      : {len(final):,}")

    merged = pd.concat([existing, final], ignore_index=True)
    merged["published"] = pd.to_datetime(merged["published"], utc=True, errors="coerce")
    merged = merged.sort_values("published").reset_index(drop=True)
    merged.to_csv(MAIN_CSV, index=False)

    print(f"  Total after merge: {len(merged):,}")
    print(f"\n  Year distribution of merged CSV:")
    yr = merged["published"].dt.year
    for y, cnt in yr.value_counts().sort_index().items():
        print(f"    {y}: {cnt:,}")
    print(f"\n  ✅ Saved → {MAIN_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
