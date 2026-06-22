"""
fetch_telegram_2026.py
======================
Fetch Telegram messages from Jan 1 2025 → now for selected channels.
Scores with XGBoost v9, merges into news_cache.json and training CSV.

Usage:
    .venv311/bin/python fetch_telegram_2026.py
"""

import asyncio
import hashlib
import json
import pickle
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

warnings.filterwarnings("ignore")

HERE       = Path(__file__).parent
sys.path.insert(0, str(HERE))

CACHE_FILE = HERE / "news_cache.json"
CSV_PATH   = HERE / "news_cleaned_filtered_scored.csv"
ETH_CACHE  = HERE / "eth_15m_klines.csv"
BTC_CACHE  = HERE / "btc_15m_klines.csv"

FETCH_FROM = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=3)

CHANNELS = [
    "the_block_crypto",
    "coindesk",
    "cointelegraph",
    "WatcherGuru",
]

BINANCE     = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = 15 * 60 * 1000
BATCH_API   = 1000
DELAY       = 0.12

FOMC_DATES = [
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-10",
    "2026-01-28","2026-03-18","2026-05-06","2026-06-17",
]
_FOMC_SET = set()
for _d in FOMC_DATES:
    _dt = pd.Timestamp(_d)
    for _off in range(-3, 4):
        _FOMC_SET.add((_dt + pd.Timedelta(days=_off)).date())

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00010000-\U0010FFFF☀-⛿✀-➿]+",
    flags=re.UNICODE,
)

from pipeline.reduce_noise import passes_news_filter


# ── helpers ──────────────────────────────────────────────────────

def clean_title(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def _hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]


# ── 1. Fetch from Telegram ────────────────────────────────────────

async def fetch_all_channels():
    from telethon import TelegramClient
    from config import TELEGRAM_API_ID, TELEGRAM_API_HASH

    client = TelegramClient(str(HERE / "telegram_session"),
                            TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()
    print("  ✅ Telegram connected")

    messages = []
    for ch in CHANNELS:
        try:
            entity = await client.get_entity(ch)
            print(f"  Fetching {ch}...", end=" ", flush=True)
            count = 0
            async for msg in client.iter_messages(entity, reverse=False, limit=None):
                if msg.date.replace(tzinfo=timezone.utc) < FETCH_FROM:
                    break
                text = (msg.text or "").strip()
                if not text:
                    continue
                title = clean_title(text.splitlines()[0][:300])
                if not passes_news_filter(title, ch):
                    continue
                messages.append({
                    "title":   title,
                    "channel": ch,
                    "pub_dt":  msg.date.replace(tzinfo=timezone.utc),
                    "link":    f"https://t.me/{ch}/{msg.id}",
                })
                count += 1
            print(f"{count}")
        except Exception as e:
            print(f"  ⚠ {ch}: {e}")

    await client.disconnect()
    return messages


# ── 2. Batch BERT embeddings + sentiment ─────────────────────────

def batch_bert(titles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns cb_emb (N,768), fb_emb (N,768), cb_probs (N,3) [neg,neu,pos]."""
    from transformers import (
        AutoTokenizer, AutoModel,
        AutoModelForSequenceClassification,
        logging as hf_logging,
    )
    hf_logging.set_verbosity_error()

    BATCH = 64

    print("  Loading CryptoBERT...")
    cb_tok  = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cb_cls  = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert").eval()
    cb_base = AutoModel.from_pretrained("ElKulako/cryptobert").eval()

    print("  Loading FinBERT...")
    fb_tok  = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_base = AutoModel.from_pretrained("ProsusAI/finbert").eval()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cb_cls  = cb_cls.to(device)
    cb_base = cb_base.to(device)
    fb_base = fb_base.to(device)

    cb_embs, cb_probs_list, fb_embs = [], [], []

    print(f"  Encoding {len(titles):,} titles...")
    for start in range(0, len(titles), BATCH):
        batch = titles[start: start + BATCH]

        # CryptoBERT
        inp_cb = cb_tok(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        inp_cb = {k: v.to(device) for k, v in inp_cb.items()}
        with torch.no_grad():
            cb_emb  = cb_base(**inp_cb).last_hidden_state[:, 0, :].cpu().numpy()
            cb_prob = torch.softmax(cb_cls(**inp_cb).logits, dim=1).cpu().numpy()
        cb_embs.append(cb_emb)
        cb_probs_list.append(cb_prob)

        # FinBERT
        inp_fb = fb_tok(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        inp_fb = {k: v.to(device) for k, v in inp_fb.items()}
        with torch.no_grad():
            fb_emb = fb_base(**inp_fb).last_hidden_state[:, 0, :].cpu().numpy()
        fb_embs.append(fb_emb)

        done = min(start + BATCH, len(titles))
        if done % 500 == 0 or done == len(titles):
            print(f"    {done}/{len(titles)}")

    return (np.vstack(cb_embs).astype(np.float32),
            np.vstack(fb_embs).astype(np.float32),
            np.vstack(cb_probs_list).astype(np.float32))


# ── 3. Build XGBoost v9 features ─────────────────────────────────

def build_features(msgs, cb_emb, fb_emb, cb_probs) -> np.ndarray:
    from xgboost_v9 import crypto_news_type_classify, NEWS_TYPE_LABELS

    n = len(msgs)

    # Proxy ensemble: all 3 models = CryptoBERT probs [neg, neu, pos]
    p_neg = cb_probs[:, 0]
    p_neu = cb_probs[:, 1]
    p_pos = cb_probs[:, 2]

    sent_arr = np.column_stack([
        p_pos, p_neg, p_neu,        # cb
        p_pos, p_neg, p_neu,        # fb proxy
        p_pos, p_neg, p_neu,        # rb proxy
        p_pos - p_neg,              # net_agreement
        p_pos - p_neg,              # sentiment_score proxy
        np.full(n, 6.0),            # weight
        np.maximum(p_pos, np.maximum(p_neg, p_neu)),  # confidence
    ]).astype(np.float32)           # 13 dims

    # News-type probs (11)
    type_probs = crypto_news_type_classify(cb_emb)

    # Macro timing (8): 5 timing + 3 price context (zeros for live)
    h   = np.array([m["pub_dt"].hour     for m in msgs], dtype=np.float32)
    dow = np.array([m["pub_dt"].weekday() for m in msgs], dtype=np.float32)
    fomc = np.array([int(m["pub_dt"].date() in _FOMC_SET) for m in msgs], dtype=np.float32)

    macro = np.column_stack([
        (dow >= 5).astype(np.float32),          # weekend
        ((h >= 2) & (h <= 6)).astype(np.float32),   # low liq
        ((h >= 13) & (h <= 21)).astype(np.float32),  # us hours
        ((h >= 0) & (h <= 8)).astype(np.float32),    # asia
        fomc,
        np.zeros(n), np.zeros(n), np.zeros(n),   # btc_vol, btc_mom, fear_greed
    ]).astype(np.float32)           # 8 dims

    # RAG = zeros (10 dims)
    rag = np.zeros((n, 10), dtype=np.float32)

    # 768+768+13+11+8+10 = 1578
    return np.hstack([cb_emb, fb_emb, sent_arr, type_probs, macro, rag]).astype(np.float32)


# ── 4. Load XGBoost v9 and score ─────────────────────────────────

def load_xgb_v9():
    clf15 = xgb.XGBClassifier(); clf15.load_model(str(HERE / "xgboost_v9_clf15m.json"))
    clf1h = xgb.XGBClassifier(); clf1h.load_model(str(HERE / "xgboost_v9_clf1h.json"))
    with open(HERE / "xgboost_v9_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    res = json.loads((HERE / "xgboost_v9_results.json").read_text())
    thr15 = res.get("threshold_15m", 0.295)
    thr1h  = res.get("threshold_1h",  0.265)
    print(f"  XGBoost v9 loaded  thresh15={thr15:.3f}  thresh1h={thr1h:.3f}")
    return clf15, clf1h, scaler, thr15, thr1h


def run_xgb(X: np.ndarray, clf15, clf1h, scaler):
    Xs = scaler.transform(X).astype(np.float32)
    p15 = clf15.predict_proba(Xs)[:, 1]
    p1h = clf1h.predict_proba(Xs)[:, 1]
    return p15.astype(np.float32), p1h.astype(np.float32)


# ── 5. Fetch BTC + ETH prices (bulk) ─────────────────────────────

def fetch_klines(symbol: str, start_ms: int, end_ms: int, cache_path: Path) -> dict:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        c_min, c_max = int(cached["open_time"].min()), int(cached["open_time"].max())
        if c_min <= start_ms and c_max >= end_ms:
            print(f"  {symbol} from cache ({len(cached):,} rows)")
            return dict(zip(cached["open_time"].astype(int),
                            cached["open_price"].astype(float)))
        print(f"  {symbol} cache insufficient — re-downloading")

    print(f"  Downloading {symbol} 15m klines...")
    klines, current = [], start_ms - INTERVAL_MS
    while current <= end_ms + INTERVAL_MS:
        try:
            resp = requests.get(BINANCE, params={
                "symbol": symbol, "interval": "15m",
                "startTime": current, "limit": BATCH_API,
            }, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  ⚠ {e} — retry"); time.sleep(2); continue
        if not isinstance(data, list) or not data:
            break
        klines.extend(data)
        current = int(data[-1][0]) + 1
        time.sleep(DELAY)

    cache_df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","trades","tbbase","tbquote","ignore"
    ])[["open_time","open"]].rename(columns={"open":"open_price"})
    cache_df.to_csv(cache_path, index=False)
    print(f"  {symbol}: {len(cache_df):,} candles cached")
    return dict(zip(cache_df["open_time"].astype(int),
                    cache_df["open_price"].astype(float)))


def get_px(ts_ms: int, pm: dict):
    c = (ts_ms // INTERVAL_MS) * INTERVAL_MS
    return pm.get(c), pm.get(c + INTERVAL_MS), pm.get(c + 4 * INTERVAL_MS)


# ── 6. Convert to cache items ─────────────────────────────────────

def to_cache_items(msgs, cb_probs, p15, p1h, thr15, thr1h):
    items = []
    for i, msg in enumerate(msgs):
        prob15 = float(p15[i])
        prob1h = float(p1h[i])
        if max(prob15, prob1h) < 0.35:
            continue

        p_neg, p_neu, p_pos = float(cb_probs[i,0]), float(cb_probs[i,1]), float(cb_probs[i,2])
        net   = p_pos - p_neg
        sent  = "positive" if net > 0.05 else "negative" if net < -0.05 else "neutral"
        sig   = "BUY" if net > 0.05 else "SELL" if net < -0.05 else "NEUTRAL"
        conf  = max(p_pos, p_neg, p_neu)

        pub_dt = msg["pub_dt"]
        items.append({
            "id":              f"tg_{int(pub_dt.timestamp())}_{abs(hash(msg['title'][:30])) % 100000}",
            "time":            pub_dt.strftime("%H:%M:%S"),
            "title":           msg["title"],
            "link":            msg["link"],
            "channel":         msg["channel"],
            "published":       pub_dt.isoformat(),
            "published_ts":    int(pub_dt.timestamp()),
            "sentiment":       sent,
            "sentiment_score": round(net, 4),
            "confidence":      round(conf * 100, 1),
            "weight":          max(5, min(9, round(conf * 10))),
            "prob_positive":   round(p_pos, 4),
            "prob_negative":   round(p_neg, 4),
            "prob_neutral":    round(p_neu, 4),
            "type":            sig,
            "model_score":     round(prob15, 4),
            "model_score_1h":  round(prob1h, 4),
            "score_normalized": True,
            "pred_15m":        int(prob15 >= thr15),
            "pred_1h":         int(prob1h >= thr1h),
            "impact":          "High" if max(prob15, prob1h) >= 0.50 else ("Medium" if max(prob15, prob1h) >= 0.25 else "Low"),
            "source":          "telegram_2025_2026",
        })
    return items


# ── 7. Convert to training CSV rows ──────────────────────────────

def to_training_rows(msgs, cb_probs, btc_map, eth_map):
    train = pd.read_csv(CSV_PATH, low_memory=False, nrows=0)
    existing_keys = set(zip(
        pd.read_csv(CSV_PATH, low_memory=False)["title"].str.strip().str.lower().fillna(""),
        pd.read_csv(CSV_PATH, low_memory=False)["channel"].fillna(""),
    ))

    rows = []
    for i, msg in enumerate(msgs):
        title   = msg["title"]
        channel = msg["channel"]
        if (title.strip().lower(), channel) in existing_keys:
            continue

        pub_dt = msg["pub_dt"]
        ts_ms  = int(pub_dt.timestamp() * 1000)

        btc0, btc15, btc1h = get_px(ts_ms, btc_map)
        eth0, eth15, eth1h = get_px(ts_ms, eth_map)
        if not btc15 or not btc1h:
            continue

        p_neg, p_neu, p_pos = float(cb_probs[i,0]), float(cb_probs[i,1]), float(cb_probs[i,2])
        net    = p_pos - p_neg
        conf   = round(max(p_pos, p_neg, p_neu), 4)
        sent   = "positive" if net > 0.05 else "negative" if net < -0.05 else "neutral"
        h, dow = pub_dt.hour, pub_dt.weekday()
        fomc   = int(pub_dt.date() in _FOMC_SET)

        rows.append({
            "title":              title,
            "channel":            channel,
            "link":               msg["link"],
            "published":          pub_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "btc_price_at_news":  round(btc0 or btc15, 2),
            "btc_price_15m":      round(btc15, 2),
            "btc_price_1h":       round(btc1h, 2),
            "eth_price_at_news":  round(eth0, 2)  if eth0  else np.nan,
            "eth_price_15m":      round(eth15, 2) if eth15 else np.nan,
            "eth_price_1h":       round(eth1h, 2) if eth1h else np.nan,
            "sentiment":          sent,
            "sentiment_score":    round(net, 4),
            "weight":             max(5, min(9, round(conf * 10))),
            "confidence":         conf,
            "prob_positive":      round(p_pos, 4),
            "prob_negative":      round(p_neg, 4),
            "prob_neutral":       round(p_neu, 4),
            "news_type":          np.nan,
            "fomc_week":          fomc,
            "is_weekend":         int(dow >= 5),
            "is_low_liquidity":   int(2 <= h <= 6),
            "is_us_hours":        int(13 <= h <= 21),
            "is_asia_hours":      int(0 <= h <= 8),
            "hour_utc":           h,
            "day_of_week":        pub_dt.strftime("%A"),
            "btc_pct_change_15m": round((btc15 - (btc0 or btc15)) / (btc0 or btc15) * 100, 6),
            "btc_pct_change_1h":  round((btc1h  - (btc0 or btc15)) / (btc0 or btc15) * 100, 6),
            "eth_pct_change_15m": round((eth15 - eth0) / eth0 * 100, 6) if (eth0 and eth15) else np.nan,
            "eth_pct_change_1h":  round((eth1h  - eth0) / eth0 * 100, 6) if (eth0 and eth1h) else np.nan,
            "hour_of_day":        h,
            "word_count":         len(title.split()),
            "sentiment_binary":   1 if sent == "positive" else 0,
            "is_spam":            False,
            "is_relevant":        True,
            "_hash":              _hash(title),
            "cb_prob_pos":        round(p_pos, 4),
            "cb_prob_neg":        round(p_neg, 4),
            "cb_prob_neu":        round(p_neu, 4),
            "fb_prob_pos":        round(p_pos, 4),
            "fb_prob_neg":        round(p_neg, 4),
            "fb_prob_neu":        round(p_neu, 4),
            "rb_prob_pos":        round(p_pos, 4),
            "rb_prob_neg":        round(p_neg, 4),
            "rb_prob_neu":        round(p_neu, 4),
            "net_agreement":      round(net, 4),
            "sentiment_reliable": True,
        })
    return rows, train.columns.tolist()


# ── Main ──────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  FETCH TELEGRAM 2025-2026 — XGBoost v9 scoring")
    print(f"  From: {FETCH_FROM.date()}  Channels: {CHANNELS}")
    print("=" * 60)

    # 1. Fetch
    print("\n[1/7] Fetching Telegram messages...")
    msgs = await fetch_all_channels()
    print(f"  Raw messages: {len(msgs):,}")

    seen, unique = set(), []
    for m in msgs:
        key = m["title"].lower()[:80]
        if key not in seen:
            seen.add(key); unique.append(m)
    msgs = unique
    print(f"  After dedup: {len(msgs):,}")
    if not msgs:
        print("No messages — check session/channels."); return

    from collections import Counter
    for ch, cnt in Counter(m["channel"] for m in msgs).most_common():
        print(f"    {ch:<30}: {cnt:,}")

    # 2. BERT
    print("\n[2/7] Computing CryptoBERT + FinBERT embeddings...")
    titles = [m["title"] for m in msgs]
    cb_emb, fb_emb, cb_probs = batch_bert(titles)

    # 3. Features
    print("\n[3/7] Building XGBoost v9 features...")
    X = build_features(msgs, cb_emb, fb_emb, cb_probs)
    print(f"  Feature matrix: {X.shape}")

    # 4. Score
    print("\n[4/7] Loading XGBoost v9 and scoring...")
    clf15, clf1h, scaler, thr15, thr1h = load_xgb_v9()
    p15, p1h = run_xgb(X, clf15, clf1h, scaler)
    print(f"  Mean prob 15m: {p15.mean():.3f}  1h: {p1h.mean():.3f}")
    print(f"  Items >= thresh: 15m={int((p15>=thr15).sum())}  1h={int((p1h>=thr1h).sum())}")

    # 5. Fetch BTC+ETH prices
    print("\n[5/7] Fetching BTC + ETH klines for price labels...")
    ts_list  = [int(m["pub_dt"].timestamp()) * 1000 for m in msgs]
    start_ms = min(ts_list) - INTERVAL_MS
    end_ms   = max(ts_list) + 4 * INTERVAL_MS
    btc_map  = fetch_klines("BTCUSDT", start_ms, end_ms, BTC_CACHE)
    eth_map  = fetch_klines("ETHUSDT", start_ms, end_ms, ETH_CACHE)

    # 6. Write to training CSV
    print("\n[6/7] Appending to training CSV...")
    train_rows, col_order = to_training_rows(msgs, cb_probs, btc_map, eth_map)
    if train_rows:
        new_df = pd.DataFrame(train_rows)
        for col in col_order:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[col_order]

        existing = pd.read_csv(CSV_PATH, low_memory=False)
        merged_csv = pd.concat([existing, new_df], ignore_index=True)
        merged_csv["published"] = pd.to_datetime(
            merged_csv["published"], format="mixed", utc=True, errors="coerce"
        )
        merged_csv = merged_csv.sort_values("published").reset_index(drop=True)
        merged_csv["published"] = merged_csv["published"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        merged_csv.to_csv(CSV_PATH, index=False)
        print(f"  Added {len(new_df):,} rows to training CSV  (total: {len(merged_csv):,})")
        for ch, cnt in Counter(new_df["channel"]).most_common():
            print(f"    {ch:<30}: {cnt:,}")
    else:
        print("  No new training rows (all already in CSV)")

    # 7. Write to cache
    print("\n[7/7] Updating news_cache.json...")
    new_cache = to_cache_items(msgs, cb_probs, p15, p1h, thr15, thr1h)

    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        existing_cache = data if isinstance(data, list) else data.get("news", [])
        existing_cache = [x for x in existing_cache if x.get("source") != "telegram_2025_2026"]
    else:
        existing_cache = []

    merged_cache = sorted(
        existing_cache + new_cache,
        key=lambda x: x.get("published_ts") or 0,
    )
    payload = {
        "metadata": {
            "total_items":  len(merged_cache),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model":        "xgboost_v9",
            "fetch_from":   FETCH_FROM.isoformat(),
        },
        "news": merged_cache,
    }
    CACHE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ✅ Cache: {len(merged_cache):,} total items")
    print(f"  New telegram items: {len(new_cache):,}")
    for ch, cnt in Counter(x["channel"] for x in new_cache).most_common():
        print(f"    {ch:<30}: {cnt:,}")


if __name__ == "__main__":
    asyncio.run(main())
