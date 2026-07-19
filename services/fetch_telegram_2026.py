"""
fetch_telegram_2026.py
======================
Fetch Telegram messages from Jan 1 2025 → now for selected channels.
Scores with XGBoost v9, merges into news_cache.json and training CSV.

Usage:
    .venv311/bin/python fetch_telegram_2026.py
"""

import asyncio
import bisect
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
ROOT       = HERE.parent
sys.path.insert(0, str(ROOT))

CACHE_FILE = ROOT / "storage" / "news_cache.json"
CSV_PATH   = ROOT / "news_cleaned_filtered_scored.csv"
ETH_CACHE  = HERE / "eth_15m_klines.csv"
BTC_CACHE  = HERE / "btc_15m_klines.csv"

def _default_fetch_from() -> "datetime":
    """Start from the last published date in the CSV minus 1 day (overlap buffer),
    or fall back to 7 days ago if CSV is missing or unparseable."""
    import datetime as _dt
    fallback = datetime.now(timezone.utc) - _dt.timedelta(days=7)
    if not CSV_PATH.exists():
        return fallback
    try:
        last = pd.read_csv(CSV_PATH, usecols=["published"], low_memory=False)["published"].dropna()
        ts = pd.to_datetime(last, utc=True, errors="coerce").max()
        if pd.isna(ts):
            return fallback
        return (ts - _dt.timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        return fallback

FETCH_FROM = _default_fetch_from()

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
from config import impact_tier as _impact_tier


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

    client = TelegramClient(str(ROOT / "telegram_session"),
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

    print("  Loading FinBERT (classifier + base)...")
    fb_tok  = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_base = AutoModel.from_pretrained("ProsusAI/finbert").eval()
    fb_cls  = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()

    print("  Loading RoBERTa-sentiment...")
    rb_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    rb_cls = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").eval()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cb_cls  = cb_cls.to(device)
    cb_base = cb_base.to(device)
    fb_base = fb_base.to(device)
    fb_cls  = fb_cls.to(device)
    rb_cls  = rb_cls.to(device)

    cb_embs, cb_probs_list, fb_embs, fb_probs_list, rb_probs_list = [], [], [], [], []

    print(f"  Encoding {len(titles):,} titles...")
    for start in range(0, len(titles), BATCH):
        batch = titles[start: start + BATCH]

        # CryptoBERT — embedding + classifier
        inp_cb = cb_tok(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        inp_cb = {k: v.to(device) for k, v in inp_cb.items()}
        with torch.no_grad():
            cb_emb  = cb_base(**inp_cb).last_hidden_state[:, 0, :].cpu().numpy()
            cb_prob = torch.softmax(cb_cls(**inp_cb).logits, dim=1).cpu().numpy()
        cb_embs.append(cb_emb)
        cb_probs_list.append(cb_prob)

        # FinBERT — embedding + classifier
        fb_batch = [f"Bitcoin crypto market: {t}" for t in batch]
        inp_fb = fb_tok(fb_batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        inp_fb = {k: v.to(device) for k, v in inp_fb.items()}
        with torch.no_grad():
            fb_emb  = fb_base(**inp_fb).last_hidden_state[:, 0, :].cpu().numpy()
            fb_prob = torch.softmax(fb_cls(**inp_fb).logits, dim=1).cpu().numpy()
        fb_embs.append(fb_emb)
        fb_probs_list.append(fb_prob)

        # RoBERTa — classifier only (no embedding needed)
        rb_batch = [f"BREAKING: {t} #Bitcoin #Crypto" for t in batch]
        inp_rb = rb_tok(rb_batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        inp_rb = {k: v.to(device) for k, v in inp_rb.items()}
        with torch.no_grad():
            rb_prob = torch.softmax(rb_cls(**inp_rb).logits, dim=1).cpu().numpy()
        rb_probs_list.append(rb_prob)

        done = min(start + BATCH, len(titles))
        if done % 500 == 0 or done == len(titles):
            print(f"    {done}/{len(titles)}")

    return (np.vstack(cb_embs).astype(np.float32),
            np.vstack(fb_embs).astype(np.float32),
            np.vstack(cb_probs_list).astype(np.float32),
            np.vstack(fb_probs_list).astype(np.float32),
            np.vstack(rb_probs_list).astype(np.float32))


# ── 3. Build XGBoost v9 features ─────────────────────────────────

def build_features(msgs, cb_emb, fb_emb, cb_probs, fb_probs, rb_probs,
                   btc_map: dict | None = None) -> np.ndarray:
    from training.xgboost_train_bert import crypto_news_type_classify, NEWS_TYPE_LABELS

    n = len(msgs)

    # CryptoBERT probs [neg, neu, pos]
    cb_neg, cb_neu, cb_pos = cb_probs[:, 0], cb_probs[:, 1], cb_probs[:, 2]

    # FinBERT probs — label order: [positive, negative, neutral]
    fb_pos, fb_neg, fb_neu = fb_probs[:, 0], fb_probs[:, 1], fb_probs[:, 2]

    # RoBERTa probs — label order: [negative, neutral, positive]
    rb_neg, rb_neu, rb_pos = rb_probs[:, 0], rb_probs[:, 1], rb_probs[:, 2]

    # Net agreement: mean of per-model (pos-neg) scaled by sign agreement
    nets = np.column_stack([cb_pos - cb_neg, fb_pos - fb_neg, rb_pos - rb_neg])
    mean_net = nets.mean(axis=1)
    avg_pos = (cb_pos + fb_pos + rb_pos) / 3
    avg_neg = (cb_neg + fb_neg + rb_neg) / 3
    avg_neu = (cb_neu + fb_neu + rb_neu) / 3
    conf = np.maximum(avg_pos, np.maximum(avg_neg, avg_neu))

    sent_arr = np.column_stack([
        cb_pos, cb_neg, cb_neu,     # cb
        fb_pos, fb_neg, fb_neu,     # fb (actual)
        rb_pos, rb_neg, rb_neu,     # rb (actual)
        mean_net,                   # net_agreement
        mean_net,                   # sentiment_score proxy
        np.full(n, 6.0),           # weight
        conf,                       # confidence
    ]).astype(np.float32)           # 13 dims

    # News-type probs (11)
    type_probs = crypto_news_type_classify(cb_emb)

    # Macro timing (8): 5 timing + 3 price context
    h   = np.array([m["pub_dt"].hour     for m in msgs], dtype=np.float32)
    dow = np.array([m["pub_dt"].weekday() for m in msgs], dtype=np.float32)
    fomc = np.array([int(m["pub_dt"].date() in _FOMC_SET) for m in msgs], dtype=np.float32)

    # Price context: btc_vol (rolling std over 20 candles), btc_mom (rolling mean over 5),
    # fear_greed. Compute from kline data when available to match training's
    # compute_price_context(); fall back to zeros only when klines weren't fetched.
    if btc_map:
        sorted_times  = sorted(btc_map.keys())
        sorted_prices = [btc_map[t] for t in sorted_times]
        btc_vol_arr   = np.zeros(n, dtype=np.float32)
        btc_mom_arr   = np.zeros(n, dtype=np.float32)
        for i, msg in enumerate(msgs):
            candle = (int(msg["pub_dt"].timestamp() * 1000) // INTERVAL_MS) * INTERVAL_MS
            idx = bisect.bisect_right(sorted_times, candle) - 1
            if idx >= 2:
                start  = max(0, idx - 20)
                prices = sorted_prices[start: idx + 1]
                if len(prices) >= 2:
                    rets = [(prices[j] - prices[j - 1]) / prices[j - 1]
                            for j in range(1, len(prices)) if prices[j - 1]]
                    if rets:
                        btc_vol_arr[i] = float(np.std(rets))
                        btc_mom_arr[i] = float(np.mean(rets[-5:])) if len(rets) >= 5 else float(np.mean(rets))
    else:
        btc_vol_arr = np.zeros(n, dtype=np.float32)
        btc_mom_arr = np.zeros(n, dtype=np.float32)

    macro = np.column_stack([
        (dow >= 5).astype(np.float32),
        ((h >= 2) & (h <= 6)).astype(np.float32),
        ((h >= 13) & (h <= 21)).astype(np.float32),
        ((h >= 0) & (h <= 8)).astype(np.float32),
        fomc,
        btc_vol_arr, btc_mom_arr, np.full(n, 0.5),  # fear_greed neutral fallback
    ]).astype(np.float32)           # 8 dims

    # RAG = zero dummy (1 dim) — model was trained with --skip-rag; must match
    rag = np.zeros((n, 1), dtype=np.float32)

    # 768+768+13+11+8+1 = 1569
    return np.hstack([cb_emb, fb_emb, sent_arr, type_probs, macro, rag]).astype(np.float32)


# ── 4. Load XGBoost v9 and score ─────────────────────────────────

def load_xgb_v9():
    # This backfill builder intentionally uses the one-dummy non-RAG layout.
    clf15 = xgb.XGBClassifier(); clf15.load_model(str(ROOT / "xgb_impact_clf_15m_bert_norag.json"))
    clf1h = xgb.XGBClassifier(); clf1h.load_model(str(ROOT / "xgb_impact_clf_1h_bert_norag.json"))
    with open(ROOT / "xgb_feature_scaler_bert_norag.pkl", "rb") as f:
        scaler = pickle.load(f)
    res = json.loads((ROOT / "xgb_bert_norag_results.json").read_text())
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
            "impact":          _impact_tier(
                prob15,
                prob1h,
                confidence=conf * 100,
                sentiment=sent,
            ),
            "source":          "telegram_2025_2026",
        })
    return items


# ── 7. Convert to training CSV rows ──────────────────────────────

def to_training_rows(msgs, cb_probs, fb_probs, rb_probs, btc_map, eth_map):
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

        # CryptoBERT: [neg, neu, pos]
        cb_neg, cb_neu, cb_pos = float(cb_probs[i,0]), float(cb_probs[i,1]), float(cb_probs[i,2])
        # FinBERT: [positive, negative, neutral]
        f_pos, f_neg, f_neu = float(fb_probs[i,0]), float(fb_probs[i,1]), float(fb_probs[i,2])
        # RoBERTa: [negative, neutral, positive]
        r_neg, r_neu, r_pos = float(rb_probs[i,0]), float(rb_probs[i,1]), float(rb_probs[i,2])

        avg_pos = (cb_pos + f_pos + r_pos) / 3
        avg_neg = (cb_neg + f_neg + r_neg) / 3
        avg_neu = (cb_neu + f_neu + r_neu) / 3
        net    = avg_pos - avg_neg
        conf   = round(max(avg_pos, avg_neg, avg_neu), 4)
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
            "prob_positive":      round(avg_pos, 4),
            "prob_negative":      round(avg_neg, 4),
            "prob_neutral":       round(avg_neu, 4),
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
            "cb_prob_pos":        round(cb_pos, 4),
            "cb_prob_neg":        round(cb_neg, 4),
            "cb_prob_neu":        round(cb_neu, 4),
            "fb_prob_pos":        round(f_pos, 4),
            "fb_prob_neg":        round(f_neg, 4),
            "fb_prob_neu":        round(f_neu, 4),
            "rb_prob_pos":        round(r_pos, 4),
            "rb_prob_neg":        round(r_neg, 4),
            "rb_prob_neu":        round(r_neu, 4),
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
    cb_emb, fb_emb, cb_probs, fb_probs, rb_probs = batch_bert(titles)

    # 3. Fetch BTC+ETH prices (before features so rolling vol/mom are real values)
    print("\n[3/7] Fetching BTC + ETH klines for price context and labels...")
    ts_list  = [int(m["pub_dt"].timestamp()) * 1000 for m in msgs]
    # Extra lookback for rolling vol/mom (20 candles × 15m = 5h before earliest msg)
    start_ms = min(ts_list) - 20 * INTERVAL_MS
    end_ms   = max(ts_list) + 4 * INTERVAL_MS
    btc_map  = fetch_klines("BTCUSDT", start_ms, end_ms, BTC_CACHE)
    eth_map  = fetch_klines("ETHUSDT", start_ms, end_ms, ETH_CACHE)

    # 4. Features — pass btc_map so rolling vol/mom match training's compute_price_context
    print("\n[4/7] Building XGBoost v9 features...")
    X = build_features(msgs, cb_emb, fb_emb, cb_probs, fb_probs, rb_probs, btc_map=btc_map)
    print(f"  Feature matrix: {X.shape}")

    # 5. Score
    print("\n[5/7] Loading XGBoost v9 and scoring...")
    clf15, clf1h, scaler, thr15, thr1h = load_xgb_v9()
    p15, p1h = run_xgb(X, clf15, clf1h, scaler)
    print(f"  Mean prob 15m: {p15.mean():.3f}  1h: {p1h.mean():.3f}")
    print(f"  Items >= thresh: 15m={int((p15>=thr15).sum())}  1h={int((p1h>=thr1h).sum())}")

    # 6. Write to training CSV
    print("\n[6/7] Appending to training CSV...")
    train_rows, col_order = to_training_rows(msgs, cb_probs, fb_probs, rb_probs, btc_map, eth_map)
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
            "model":        "xgboost_bert",
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
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--from-date", default=None,
                         help="Override start date, e.g. 2026-05-01")
    _args = _parser.parse_args()
    if _args.from_date:
        FETCH_FROM = datetime.fromisoformat(_args.from_date).replace(tzinfo=timezone.utc)
        print(f"  [override] FETCH_FROM → {FETCH_FROM.date()}")
    asyncio.run(main())
