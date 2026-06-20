"""
add_bot_news_to_csv.py
======================
Converts 2025+ items from crypto_news_bot_new/news_cache.json into the
training CSV format and appends them to news_cleaned_filtered_scored.csv.

Steps:
  1. Load + noise-filter bot items (2025+)
  2. Download BTC 15m klines from Binance (ETH already cached)
  3. Fill btc_price_15m / btc_price_1h / eth prices via kline lookup
  4. Derive macro timing flags from timestamp
  5. Proxy-fill ensemble sentiment cols from confidence + sentiment
  6. Append to training CSV

Usage:
    .venv311/bin/python add_bot_news_to_csv.py
"""

import hashlib
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

HERE        = Path(__file__).parent
BOT_CACHE   = Path("/home/haniye/crypto_news_bot_new/news_cache.json")
CSV_PATH    = HERE / "news_cleaned_filtered_scored.csv"
ETH_CACHE   = HERE / "eth_15m_klines.csv"
BTC_CACHE   = HERE / "btc_15m_klines.csv"

BINANCE     = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = 15 * 60 * 1000
BATCH       = 1000
DELAY       = 0.12

BLOCKED = {
    "CoinMarketCap", "whale_alert_io", "unusual_whales_TG1",
    "lookonchain", "BitcoinMagazineTelegram",
}

FOMC_DATES = [
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-10",
    "2026-01-28","2026-03-18","2026-05-06","2026-06-17",
]
_FOMC_SET = set()
for d in FOMC_DATES:
    dt = pd.Timestamp(d)
    for offset in range(-3, 4):
        _FOMC_SET.add((dt + pd.Timedelta(days=offset)).date())


# ── helpers ──────────────────────────────────────────────────────

def _hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]


def fetch_klines(symbol: str, start_ms: int, end_ms: int, cache_path: Path) -> dict:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if (int(cached["open_time"].min()) <= start_ms and
                int(cached["open_time"].max()) >= end_ms):
            print(f"  {symbol} klines from cache ({len(cached):,} rows)")
            return dict(zip(cached["open_time"].astype(int),
                            cached["open_price"].astype(float)))

    print(f"  Downloading {symbol} 15m klines from Binance...")
    klines, current = [], start_ms - INTERVAL_MS
    while current <= end_ms + INTERVAL_MS:
        try:
            resp = requests.get(BINANCE, params={
                "symbol": symbol, "interval": "15m",
                "startTime": current, "limit": BATCH,
            }, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  ⚠ {e} — retrying"); time.sleep(2); continue
        if not isinstance(data, list) or not data:
            break
        klines.extend(data)
        current = int(data[-1][0]) + 1
        if len(klines) % 20000 < BATCH:
            dt = datetime.fromtimestamp(int(data[-1][0])/1000, tz=timezone.utc).date()
            print(f"    {len(klines):>7,} candles up to {dt}")
        time.sleep(DELAY)

    print(f"  {symbol}: {len(klines):,} candles")
    cache_df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","tbbase","tbquote","ignore"
    ])[["open_time","open"]].rename(columns={"open":"open_price"})
    cache_df.to_csv(cache_path, index=False)
    return dict(zip(cache_df["open_time"].astype(int),
                    cache_df["open_price"].astype(float)))


def get_prices(ts_ms: int, price_map: dict):
    candle = (ts_ms // INTERVAL_MS) * INTERVAL_MS
    p0  = price_map.get(candle)
    p15 = price_map.get(candle + INTERVAL_MS)
    p1h = price_map.get(candle + 4 * INTERVAL_MS)
    return p0, p15, p1h


def derive_probs(sentiment: str, confidence: float):
    c = min(1.0, max(0.0, confidence / 100.0))
    if sentiment == "positive":
        pos, neg, neu = c, (1-c)*0.15, (1-c)*0.85
    elif sentiment == "negative":
        neg, pos, neu = c, (1-c)*0.15, (1-c)*0.85
    else:
        neu, pos, neg = c, (1-c)*0.5, (1-c)*0.5
    return round(pos,4), round(neg,4), round(neu,4)


# ── 1. Load + filter bot items ────────────────────────────────────
from reduce_noise import passes_news_filter

print("=" * 60)
print("  ADD BOT NEWS TO TRAINING CSV")
print("=" * 60)

print("\n[1/5] Loading bot cache...")
bot_data  = json.loads(BOT_CACHE.read_text(encoding="utf-8"))
bot_items = bot_data.get("news", bot_data) if isinstance(bot_data, dict) else bot_data

cutoff_ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
filtered  = [
    x for x in bot_items
    if x.get("channel") not in BLOCKED
    and (x.get("published_ts") or 0) >= cutoff_ts
    and passes_news_filter(x.get("title",""), x.get("channel",""))
]
print(f"  Items after filter: {len(filtered):,}")

# ── 2. Download BTC + ETH klines ─────────────────────────────────
print("\n[2/5] Fetching price klines...")
ts_list   = [int(x["published_ts"]) * 1000 for x in filtered]
start_ms  = min(ts_list) - INTERVAL_MS
end_ms    = max(ts_list) + 4 * INTERVAL_MS

btc_map = fetch_klines("BTCUSDT", start_ms, end_ms, BTC_CACHE)
eth_map = fetch_klines("ETHUSDT", start_ms, end_ms, ETH_CACHE)

# ── 3. Build rows ─────────────────────────────────────────────────
print("\n[3/5] Building training rows...")
rows = []
skipped = 0

for item in filtered:
    title   = (item.get("title") or "").strip()
    channel = item.get("channel", "")
    ts_ms   = int(item["published_ts"]) * 1000
    pub_dt  = datetime.fromtimestamp(item["published_ts"], tz=timezone.utc)

    # BTC prices
    btc_now_item = item.get("price") or 0.0
    btc0, btc15, btc1h = get_prices(ts_ms, btc_map)
    if not btc15 or not btc1h:
        skipped += 1
        continue
    btc_at = btc0 or btc_now_item or btc15

    # ETH prices
    eth0, eth15, eth1h = get_prices(ts_ms, eth_map)

    # Macro timing
    h   = pub_dt.hour
    dow = pub_dt.weekday()
    fomc = int(pub_dt.date() in _FOMC_SET)

    # Sentiment / probs
    sentiment = item.get("sentiment", "neutral")
    conf_raw  = float(item.get("confidence") or 50)
    p_pos, p_neg, p_neu = derive_probs(sentiment, conf_raw)
    sent_score = item.get("sentiment_score", 0)
    if isinstance(sent_score, str):
        sent_score = 0

    rows.append({
        "title":              title,
        "channel":            channel,
        "link":               item.get("link",""),
        "published":          pub_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "btc_price_at_news":  round(btc_at, 2),
        "btc_price_15m":      round(btc15, 2),
        "btc_price_1h":       round(btc1h, 2),
        "eth_price_at_news":  round(eth0, 2)  if eth0  else np.nan,
        "eth_price_15m":      round(eth15,2)  if eth15 else np.nan,
        "eth_price_1h":       round(eth1h,2)  if eth1h else np.nan,
        "sentiment":          sentiment,
        "sentiment_score":    round(float(sent_score), 4),
        "weight":             float(item.get("weight") or 5),
        "confidence":         round(conf_raw/100.0, 4),
        "prob_positive":      p_pos,
        "prob_negative":      p_neg,
        "prob_neutral":       p_neu,
        "news_type":          np.nan,
        "fomc_week":          fomc,
        "is_weekend":         int(dow >= 5),
        "is_low_liquidity":   int(2 <= h <= 6),
        "is_us_hours":        int(13 <= h <= 21),
        "is_asia_hours":      int(0 <= h <= 8),
        "hour_utc":           h,
        "day_of_week":        pub_dt.strftime("%A"),
        "btc_pct_change_15m": round((btc15-btc_at)/btc_at*100, 6) if btc_at else 0,
        "btc_pct_change_1h":  round((btc1h -btc_at)/btc_at*100, 6) if btc_at else 0,
        "eth_pct_change_15m": round((eth15-eth0)/eth0*100, 6) if (eth0 and eth15) else np.nan,
        "eth_pct_change_1h":  round((eth1h-eth0)/eth0*100, 6) if (eth0 and eth1h) else np.nan,
        "hour_of_day":        h,
        "word_count":         len(title.split()),
        "sentiment_binary":   1 if sentiment == "positive" else 0,
        "is_spam":            False,
        "is_relevant":        True,
        "_hash":              _hash(title),
        "cb_prob_pos":        p_pos,
        "cb_prob_neg":        p_neg,
        "cb_prob_neu":        p_neu,
        "fb_prob_pos":        p_pos,
        "fb_prob_neg":        p_neg,
        "fb_prob_neu":        p_neu,
        "rb_prob_pos":        p_pos,
        "rb_prob_neg":        p_neg,
        "rb_prob_neu":        p_neu,
        "net_agreement":      round(p_pos - p_neg, 4),
        "sentiment_reliable": True,
    })

print(f"  Rows built : {len(rows):,}")
print(f"  Skipped    : {skipped} (no forward price)")

# ── 4. Merge into training CSV ────────────────────────────────────
print("\n[4/5] Merging into training CSV...")
train = pd.read_csv(CSV_PATH, low_memory=False)
new_df = pd.DataFrame(rows)[train.columns]   # align column order

# Deduplicate by title+channel
existing_keys = set(zip(
    train["title"].str.strip().str.lower().fillna(""),
    train["channel"].fillna("")
))
new_df = new_df[~new_df.apply(
    lambda r: (r["title"].strip().lower(), r["channel"]) in existing_keys, axis=1
)]
print(f"  New unique rows: {len(new_df):,}")

merged = pd.concat([train, new_df], ignore_index=True)
merged["published"] = pd.to_datetime(merged["published"], format="mixed", utc=True, errors="coerce")
merged = merged.sort_values("published").reset_index(drop=True)
merged["published"] = merged["published"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

merged.to_csv(CSV_PATH, index=False)

# ── 5. Summary ────────────────────────────────────────────────────
print("\n[5/5] Done.")
print(f"  Before : {len(train):,} rows")
print(f"  Added  : {len(new_df):,} rows")
print(f"  After  : {len(merged):,} rows")
print(f"\n  New channels:")
from collections import Counter
for ch, cnt in Counter(new_df["channel"]).most_common():
    print(f"    {ch:<30}: {cnt:,}")
