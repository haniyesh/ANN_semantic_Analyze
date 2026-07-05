"""
main.py
=======
Crypto Intelligence Signals — Main Bot

Flow:
  1. Fetch Telegram channels
  2. Type-routed sentiment scoring:
       regulatory/macro/etf/institutional → FinBERT
       market_analysis                   → RoBERTa
       crypto-specific types             → CryptoBERT
     CryptoBERT embedding is shared — no duplicate forward pass.
  3. Run through XGBoost BERT model (DualBERT + PriceContext, 1569 features)
  4. Route — importance-tiered gate on score AND confidence (see config.py):
     - Display gate (Show): max(score_15m, score_1h) >= 0.30 AND confidence >= 0.50
     - Medium badge:        max(score_15m, score_1h) >= 0.55 AND confidence >= 0.70
     - Hot badge / alert:   max(score_15m, score_1h) >= 0.80 AND confidence >= 0.78
"""

import asyncio
import hashlib
import json
import os
import pickle
import re
import time as time_module
import warnings
warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*downstream task.*")

import numpy as np
import httpx
import torch
import xgboost as xgb
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

from config import (
    DASHBOARD_API,
    SCORE_15M_MIN, SCORE_15M_MAX,
    SCORE_1H_MIN,  SCORE_1H_MAX,
    SCORE_THRESHOLD_HOT,
    SCORE_THRESHOLD_MEDIUM,
    SCORE_THRESHOLD_SHOW,
    IMPORTANT_MIN_CONFIDENCE,
    IMPORTANT_MIN_SCORE,
    HOT_MIN_MODEL_SCORE,
    HOT_MIN_CONFIDENCE,
    HOT_MAX_AGE_MIN,
    BATCH_SIZE,
    news_importance,
    FOMC_WEEK_DATES,
)

from bot.telegram_listener import start as start_telegram_listener
from services.price_fetcher import PriceTracker, extract_coin_from_text
from storage.database import (
    create_pool, create_tables,
    is_processed, mark_processed,
    save_news, save_price_movement,
)
from log import get_logger
_log = get_logger("main")


news_queue    = deque()
price_tracker = PriceTracker()

FETCH_INTERVAL = 60
# BATCH_SIZE is imported from config — do not shadow it here


# ── SCORE NORMALIZATION ──────────────────────────────────────────────
# XGBoost v9 outputs calibrated probabilities already in [0, 1].
# min=0 / max=1 in config.py makes _normalize_score() an identity function.

def _normalize_score(raw: float, min_val: float, max_val: float) -> float:
    """Map raw model score to 0–1 range, clamped."""
    if max_val <= min_val:
        return 0.0
    return round(max(0.0, min(1.0, (raw - min_val) / (max_val - min_val))), 4)

# ── DISPLAY THRESHOLDS ───────────────────────────────────────────────
# All thresholds imported from config.py (single source of truth)
# Tiers (score/confidence): Show (≥0.30/0.62) | Medium (≥0.55/0.70) | Hot (≥0.80/0.78) | Hidden (below Show)


# ══════════════════════════════════════════════════════════════════
# SENTIMENT — type-routed scorer (shared with services/sentiment_score.py)
# ══════════════════════════════════════════════════════════════════
def _load_sentiment_models():
    """Load type-routed scorer models (one-time, cached in services/sentiment_score.py)."""
    from services.sentiment_score import load_models
    return load_models()


def score_sentiment(title: str) -> tuple[dict, np.ndarray, str]:
    """
    Type-routed sentiment scoring for a single news title.
    Returns:
      sent      — dict with sentiment_score, weight, confidence, prob_*
      embedding — CryptoBERT 768-dim vector (reused for model features)
      news_type — string label (regulatory / hack / market_analysis / ...)

    Single CryptoBERT forward pass shared for both embedding and type detection.
    """
    from services.sentiment_score import load_models
    import torch
    import torch.nn.functional as F

    m      = load_models()
    title  = str(title).strip() or "crypto news"
    inputs = m["cb_tok"](title, padding=True, truncation=True,
                         max_length=128, return_tensors="pt")
    with torch.no_grad():
        emb_out = m["cb_emb"](**inputs).last_hidden_state[:, 0, :]   # (1, 768)
        cls_out = m["cb_cls"](**inputs).logits                        # (1, 3)

    cb_probs  = torch.softmax(cls_out, dim=1).numpy()[0]              # [bear, neu, bull]
    embedding = emb_out.numpy().flatten().astype(np.float32)          # (768,)

    # News type via cosine similarity
    norm      = F.normalize(emb_out, dim=1)
    sims      = torch.mm(norm, m["proto"].T)
    from services.sentiment_score import NEWS_TYPE_LABELS, FINBERT_TYPES, ROBERTA_TYPES
    news_type = NEWS_TYPE_LABELS[sims.argmax(dim=1).item()]

    cb_neg, cb_neu, cb_pos = float(cb_probs[0]), float(cb_probs[1]), float(cb_probs[2])

    # Initial single-model result (will be overridden by ensemble in Step 1b,
    # but kept here as fallback if ensemble models fail to load)
    pp, pn, pu = cb_pos, cb_neg, cb_neu

    # Scoring from CryptoBERT only (Step 1b will override with weighted avg)
    if pu > max(pp, pn):
        disc       = 0
        sentiment  = "neutral"
        confidence = pu
    else:
        net  = pp - pn
        disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
               -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
        sentiment  = "positive" if disc > 0 else ("negative" if disc < 0 else "neutral")
        confidence = pp if disc > 0 else (pn if disc < 0 else pu)

    sent = {
        "sentiment":       sentiment,
        "sentiment_score": disc,
        "weight":          max(5, min(10, round(confidence * 10))),
        "confidence":      round(confidence, 4),
        "prob_positive":   round(pp, 4),
        "prob_negative":   round(pn, 4),
        "prob_neutral":    round(pu, 4),
    }
    return sent, embedding, news_type


# ══════════════════════════════════════════════════════════════════
# MODEL INFERENCE
# ══════════════════════════════════════════════════════════════════
ROOT_DIR = Path(__file__).parent

_model_bundle = {}

def _load_model():
    global _model_bundle
    if _model_bundle:
        return _model_bundle

    clf15_path  = ROOT_DIR / "xgb_impact_clf_15m_bert.json"
    clf1h_path  = ROOT_DIR / "xgb_impact_clf_1h_bert.json"
    scaler_path = ROOT_DIR / "xgb_feature_scaler_bert.pkl"

    if not clf15_path.exists():
        _log.warning("Impact classifier not found — scoring disabled")
        return {}

    clf_15m = xgb.XGBClassifier(); clf_15m.load_model(str(clf15_path))
    clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(str(clf1h_path))

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        _log.warning("Feature scaler not found — run training/xgboost_train_bert.py first")
        return {}

    # Load FinBERT for live embeddings — always CPU to avoid CUDA capability mismatch
    from transformers import AutoTokenizer, AutoModel
    fb_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_mdl = AutoModel.from_pretrained("ProsusAI/finbert").eval().to("cpu")

    thr15, thr1h = 0.295, 0.265
    res_path = ROOT_DIR / "xgb_bert_results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        thr15 = res.get("threshold_15m", thr15)
        thr1h = res.get("threshold_1h",  thr1h)

    _model_bundle = {
        "clf_15m": clf_15m, "clf_1h": clf_1h, "scaler": scaler,
        "fb_tok": fb_tok, "fb_mdl": fb_mdl,
        "thresh15": thr15, "thresh1h": thr1h,
    }
    _log.info("XGBoost BERT loaded (thresh15=%.3f thresh1h=%.3f)", thr15, thr1h)
    return _model_bundle


def run_model(features: np.ndarray) -> dict:
    """Run XGBoost BERT on a single 1569-dim feature vector. Returns score + prediction."""
    bundle = _load_model()
    if not bundle:
        return {"model_score": 0.0, "model_score_1h": 0.0, "pred_15m": 0, "pred_1h": 0,
                "prob_15m": 0.0, "reg_pred_15m": 0.0, "confidence_model": 0.0}

    X = bundle["scaler"].transform(features.reshape(1, -1)).astype(np.float32)
    p15 = float(bundle["clf_15m"].predict_proba(X)[0, 1])
    p1h = float(bundle["clf_1h"].predict_proba(X)[0, 1])

    return {
        "model_score":      round(p15, 4),
        "model_score_1h":   round(p1h, 4),
        "pred_15m":         int(p15 >= bundle["thresh15"]),
        "pred_1h":          int(p1h >= bundle["thresh1h"]),
        "prob_15m":         round(p15, 4),
        "reg_pred_15m":     0.0,
        "confidence_model": round((p15 + p1h) / 2, 4),
    }


# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDERS (single item, real-time)
# ══════════════════════════════════════════════════════════════════
def _get_live_fear_greed() -> float:
    """Return today's fear/greed index (0–1 scale) from cache, or 0.5 as neutral fallback."""
    try:
        fg_path = ROOT_DIR / "fear_greed_cache.json"
        if not fg_path.exists():
            return 0.5
        items = json.loads(fg_path.read_text())
        today = datetime.now(timezone.utc).date()
        for item in items:
            d = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc).date()
            if d == today:
                return float(item["value"]) / 100.0
        if items:
            return float(items[0]["value"]) / 100.0
    except Exception:
        pass
    return 0.5


def build_xgb_features(
    sent: dict,
    cb_embedding: np.ndarray,
    fb_embedding: np.ndarray,
    macro: np.ndarray,
) -> np.ndarray:
    """
    Build 1569-dim flat feature vector matching xgboost_train_bert training layout:
      CryptoBERT(768) | FinBERT(768) | sentiment(13) | type_probs(11) | macro(8) | RAG(1, zeros)
    RAG is always zeros — model trained in skip-RAG mode (np.zeros((n, 1))).
    """
    try:
        from training.xgboost_train_bert import crypto_news_type_classify
        type_probs = crypto_news_type_classify(cb_embedding.reshape(1, -1))[0]
    except Exception:
        type_probs = np.zeros(11, dtype=np.float32)

    sent_vec = np.array([
        sent.get("cb_prob_pos", 0), sent.get("cb_prob_neg", 0), sent.get("cb_prob_neu", 0),
        sent.get("fb_prob_pos", 0), sent.get("fb_prob_neg", 0), sent.get("fb_prob_neu", 0),
        sent.get("rb_prob_pos", 0), sent.get("rb_prob_neg", 0), sent.get("rb_prob_neu", 0),
        sent.get("net_agreement", 0),
        sent.get("sentiment_score", 0),
        sent.get("weight", 5),
        sent.get("confidence", 0),
    ], dtype=np.float32)

    rag = np.zeros(1, dtype=np.float32)
    return np.concatenate([cb_embedding, fb_embedding, sent_vec, type_probs, macro, rag]).astype(np.float32)


_btc_vol_mom_cache: dict = {"val": (0.0, 0.0), "ts": 0}


def _get_live_btc_vol_mom() -> tuple[float, float]:
    """Compute rolling BTC volatility and momentum from recent 15m candles.
    Returns (btc_vol, btc_mom) matching training: std(last 20 returns), mean(last 5 returns).
    Cached for 5 minutes to avoid hammering Binance.
    """
    now = time_module.time()
    if now - _btc_vol_mom_cache["ts"] < 300:
        return _btc_vol_mom_cache["val"]
    try:
        import requests
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 25},
            timeout=5,
        )
        klines = resp.json()
        closes = [float(k[4]) for k in klines]
        if len(closes) < 3:
            return 0.0, 0.0
        returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100
                   for i in range(1, len(closes))]
        vol = float(np.std(returns[-20:])) if len(returns) >= 2 else 0.0
        mom = float(np.mean(returns[-5:])) if returns else 0.0
        _btc_vol_mom_cache["val"] = (vol, mom)
        _btc_vol_mom_cache["ts"] = now
        return vol, mom
    except Exception:
        return _btc_vol_mom_cache["val"]


def build_macro_features(pub_dt: datetime) -> np.ndarray:
    """
    Build 8-dim macro feature vector: 5 timing + 3 price context.
    Matches xgboost_train_bert training layout: [weekend, low_liq, us_hours, asia_hours, fomc_week,
                                          btc_vol, btc_mom, fear_greed]
    """
    hour = pub_dt.hour
    dow  = pub_dt.weekday()
    timing = np.array([
        float(dow >= 5),                 # is_weekend
        float(2 <= hour <= 6),           # is_low_liquidity
        float(13 <= hour <= 21),         # is_us_hours
        float(0  <= hour <= 8),          # is_asia_hours
        float(pub_dt.date() in FOMC_WEEK_DATES),  # fomc_week
    ], dtype=np.float32)
    btc_vol, btc_mom = _get_live_btc_vol_mom()
    price_ctx = np.array([btc_vol, btc_mom, _get_live_fear_greed()], dtype=np.float32)
    return np.concatenate([timing, price_ctx])




# ══════════════════════════════════════════════════════════════════
# RAG QUERY (single item)
# ══════════════════════════════════════════════════════════════════
async def save_full_news(
    pool, title: str, link: str, source: str,
    coin: str, category: str, signal: str,
    impact_score: float, published_at: datetime,
) -> int | None:
    """Save news item to database. Returns news_id or None on error.

    Previously this was a stub that always returned None, so news was never
    persisted even when the DB was connected. It now delegates to the real
    save_news() and surfaces (not swallows) the error reason on failure.
    """
    if pool is None:
        return None
    try:
        return await save_news(
            pool=pool, title=title, link=link, source=source,
            coin=coin, category=category, signal=signal,
            impact_score=impact_score, published_at=published_at,
        )
    except Exception as e:
        _log.error("save_full_news failed: %s: %s", type(e).__name__, e)
        return None


def query_rag(title: str, published_ts: int, channel: str) -> tuple[np.ndarray, list]:
    """
    Query Qdrant for similar past news.
    Returns (rag_features_10dim, similar_news_list).
    """
    try:
        from pipeline.rag_news import query_single
        ch_rates = {
            "the_block_crypto":  0.062,
            "porter_news":       0.058,
            "coindesk":          0.058,
            "cryptoslatenews":   0.051,
            "cointelegraph":     0.048,
        }
        pub_dt    = datetime.fromtimestamp(published_ts, tz=timezone.utc)
        hour      = pub_dt.hour
        macro_now = {
            "is_weekend":       0.0,
            "is_low_liquidity": 0.0,
            "is_us_hours":      float(13 <= hour <= 21),
            "is_asia_hours":    float(0  <= hour <= 8),
            "fomc_week":        float(pub_dt.date() in FOMC_WEEK_DATES),
        }
        result = query_single(
            title=title,
            before_timestamp=published_ts,
            channel_impact_rates=ch_rates,
            macro_now=macro_now,
        )
        return result["features"], result.get("similar_news", [])
    except Exception as e:
        _log.warning("RAG query failed: %s", e)
        return np.zeros(10, dtype=np.float32), []


# ══════════════════════════════════════════════════════════════════
# HOT SIGNAL CHECK
# ══════════════════════════════════════════════════════════════════
def is_hot(model_score: float, model_score_1h: float,
           confidence: float, age_minutes: float) -> bool:
    """Hot tier: max(score_15m, score_1h) >= HOT_MIN_MODEL_SCORE AND
    confidence >= HOT_MIN_CONFIDENCE AND age < HOT_MAX_AGE_MIN."""
    return (
        max(abs(model_score), abs(model_score_1h)) >= HOT_MIN_MODEL_SCORE and
        confidence       >= HOT_MIN_CONFIDENCE   and
        age_minutes      <  HOT_MAX_AGE_MIN
    )


def should_display_in_all(model_score, model_score_1h, confidence, title=""):
    """Display gate: score >= IMPORTANT_MIN_SCORE AND confidence >= IMPORTANT_MIN_CONFIDENCE
    AND title >= 20 chars."""
    if len(title.strip()) < 20:
        return False
    return (
        max(abs(model_score), abs(model_score_1h)) >= IMPORTANT_MIN_SCORE and
        confidence >= IMPORTANT_MIN_CONFIDENCE
    )


# ══════════════════════════════════════════════════════════════════
# DASHBOARD ROUTING
# ══════════════════════════════════════════════════════════════════
async def send_to_dashboard(payload: dict):
    """Send signal to dashboard ALL feed."""
    try:
        headers = {}
        _key = os.getenv("INGEST_API_KEY", "")
        if _key:
            headers["X-API-Key"] = _key
        async with httpx.AsyncClient() as client:
            await client.post(f"{DASHBOARD_API}/news", json=payload, headers=headers, timeout=3)
    except Exception as e:
        _log.warning("Dashboard API error: %s", e)


async def post_hot_to_telegram(payload: dict):
    """Post hot signal to Telegram signal channel."""
    try:
        from telegram import Bot
        bot     = Bot(token=os.getenv("BOT_TOKEN"))
        chat_id = os.getenv("SIGNAL_CHANNEL_ID")
        if not chat_id:
            return

        emoji = "🟢" if payload["type"] == "BUY" else "🔴" if payload["type"] == "SELL" else "🟡"
        similar_text = ""
        for s in payload.get("similar", [])[:2]:
            similar_text += f"\n  • {s['title'][:60]} → BTC {s['change']:+.1f}%"

        text = (
            f"{emoji} *{payload['type']} SIGNAL*\n\n"
            f"📰 {payload['title']}\n\n"
            f"📡 `{payload['channel']}`  |  ⏰ `{payload['age_minutes']}m ago`\n"
            f"💪 Weight: `{payload['weight']}/10`  |  🎯 Conf: `{payload['confidence']}%`\n"
            f"🤖 Model score: `{payload['model_score']}`\n"
        )
        if similar_text:
            text += f"\n📚 *Similar past news:*{similar_text}\n"
        text += f"\n🔗 {payload.get('link', '')}"

        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception as e:
        _log.error("Telegram hot signal error: %s", e)


# ══════════════════════════════════════════════════════════════════
# INCOMING NEWS FILTERS  (delegated to pipeline.spam_filter)
# ══════════════════════════════════════════════════════════════════
from pipeline.spam_filter import passes_pre_filters as _passes_pre_filters
from pipeline.reduce_noise import passes_news_filter

MIN_WEIGHT = 5


# ══════════════════════════════════════════════════════════════════
# FULL NEWS PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════
async def process_news_item(news: dict):
    """
    Complete pipeline for one news item:
      1. Pre-filters (emoji clean, headline check)
      2. Sentiment scoring (CryptoBERT + FinBERT + RoBERTa ensemble)
      3. CryptoBERT embedding + news type classification
      4. RAG query (Qdrant)
      5. Model inference (XGBoost BERT: xgb_impact_clf_15m/1h_bert.json)
      6. Score normalization + US hours boost
      7. Dashboard + Telegram routing
    """
    title   = news.get("title", "")
    channel = news.get("source", "rss")
    pub_dt  = news.get("pub_dt", datetime.now(timezone.utc))

    if not passes_news_filter(title, channel):
        return

    if isinstance(pub_dt, str):
        pub_dt = datetime.fromisoformat(pub_dt.replace("Z", "+00:00"))
    if pub_dt.tzinfo is None:
        pub_dt = pub_dt.replace(tzinfo=timezone.utc)

    published_ts = int(pub_dt.timestamp())
    age_minutes  = (time_module.time() - published_ts) / 60

    # ── Step 0: Pre-filters ───────────────────────────────────────
    title, skip_reason = _passes_pre_filters(title)
    if skip_reason:
        _log.debug("Pre-filter [%s] | %s", skip_reason, title[:60])
        return
    news["title"] = title

    # ── Step 1: Sentiment (offloaded to thread to avoid blocking event loop) ──
    sent, embedding, news_type = await asyncio.to_thread(score_sentiment, title)

    # ── Step 1b: 3-model ensemble override ─────────────────────
    try:
        from services.sentiment_score import load_models
        from services.ensemble import (
            ensemble_probs, sentiment_from_probs, reliability,
            FB_PROMPT, RB_PROMPT,
        )
        m = load_models()
        fb_raw = m["fb"](FB_PROMPT.format(title=title), truncation=True)[0]
        fb = {s["label"].lower(): s["score"] for s in fb_raw}
        rb_raw = m["rb"](RB_PROMPT.format(title=title), truncation=True)[0]
        rb = {s["label"].lower(): s["score"] for s in rb_raw}

        cb_pos, cb_neg, cb_neu = sent["prob_positive"], sent["prob_negative"], sent["prob_neutral"]
        fb_pos = fb.get("positive", 0); fb_neg = fb.get("negative", 0); fb_neu = fb.get("neutral", 0)
        rb_pos = rb.get("positive", 0); rb_neg = rb.get("negative", 0); rb_neu = rb.get("neutral", 0)

        ens = ensemble_probs(
            (cb_pos, cb_neg, cb_neu),
            (fb_pos, fb_neg, fb_neu),
            (rb_pos, rb_neg, rb_neu),
        )
        avg_pos, avg_neg, avg_neu = ens.pop("_avg")
        label, disc, confidence = sentiment_from_probs(avg_pos, avg_neg, avg_neu)

        sent["sentiment"] = label
        sent["sentiment_score"] = disc
        sent["confidence"] = round(confidence, 4)
        sent["prob_positive"] = round(avg_pos, 4)
        sent["prob_negative"] = round(avg_neg, 4)
        sent["prob_neutral"] = round(avg_neu, 4)
        sent["weight"] = max(5, min(10, round(confidence * 10)))
        sent.update(ens)  # cb_/fb_/rb_prob_* + net_agreement

        sent["sentiment_reliable"] = reliability(
            (cb_pos, cb_neg, cb_neu),
            (fb_pos, fb_neg, fb_neu),
            (rb_pos, rb_neg, rb_neu),
            avg_pos, avg_neg,
        )
    except Exception:
        sent["sentiment_reliable"] = True  # fallback: assume reliable

    # ── Step 1c: FinBERT embedding (offloaded) ─────────────────────
    def _compute_fb_embedding():
        fb_emb = np.zeros(768, dtype=np.float32)
        bundle = _load_model()
        if bundle and "fb_tok" in bundle:
            try:
                inputs = bundle["fb_tok"](
                    title, return_tensors="pt", truncation=True, max_length=128, padding=True
                )
                with torch.no_grad():
                    fb_emb = bundle["fb_mdl"](**inputs).last_hidden_state[:, 0, :].numpy().flatten().astype(np.float32)
            except Exception:
                pass
        return fb_emb
    fb_embedding = await asyncio.to_thread(_compute_fb_embedding)

    # ── Step 2: Macro features (includes blocking Binance HTTP call) ────
    macro = await asyncio.to_thread(build_macro_features, pub_dt)
    # ── Step 3: RAG query (blocking Qdrant + fastembed) ──────────────
    rag_features, similar_news = await asyncio.to_thread(
        query_rag, title, published_ts, channel
    )

    # ── Step 4: Build flat XGBoost feature vector ─────────────────
    features = build_xgb_features(sent, embedding, fb_embedding, macro)

    # ── Step 5: Model inference (offloaded) ────────────────────────
    model_result   = await asyncio.to_thread(run_model, features)

    # XGBoost outputs calibrated probs [0,1] — normalization is identity (min=0, max=1)
    model_score    = _normalize_score(
        model_result["model_score"], SCORE_15M_MIN, SCORE_15M_MAX
    )
    model_score_1h = _normalize_score(
        model_result.get("model_score_1h", 0.0), SCORE_1H_MIN, SCORE_1H_MAX
    )

    # ── Step 5: Build payload ─────────────────────────────────────
    signal_type = (
        "BUY"  if sent["sentiment_score"] > 1  else
        "SELL" if sent["sentiment_score"] < -1 else
        "NEUTRAL"
    )
    confidence_pct = round(sent["confidence"] * 100)

    _nid = hashlib.sha1(f"{channel}|{title}|{published_ts}".encode()).hexdigest()[:16]

    payload = {
        "id":               _nid,
        "time":             pub_dt.strftime("%H:%M:%S"),
        "type":             signal_type,
        "title":            title,
        "channel":          channel,
        "confidence":       confidence_pct,
        "weight":           sent["weight"],
        "sentiment":        sent["sentiment"],
        "sentiment_score":  sent["sentiment_score"],
        "prob_neutral":     sent["prob_neutral"],
        "prob_positive":    sent["prob_positive"],
        "prob_negative":    sent["prob_negative"],
        "model_score":      model_score,
        "model_score_1h":   model_score_1h,
        "score_normalized": True,
        "pred_15m":         model_result["pred_15m"],
        "pred_1h":          model_result["pred_1h"],
        "confidence_model": model_result.get("confidence_model", 0.0),
        "impact": (
            "Hot"    if max(abs(model_score), abs(model_score_1h)) >= SCORE_THRESHOLD_HOT    else
            "Medium" if max(abs(model_score), abs(model_score_1h)) >= SCORE_THRESHOLD_MEDIUM else
            "Show"
        ),
        "age_minutes":      round(age_minutes, 1),
        "published_ts":     published_ts,
        "link":             news.get("link", ""),
        "btc_change_15m":   0.0,
        "btc_change_1h":    0.0,
        "price":            news.get("btc_price", 0.0),
        "rag_hit_rate":     float(rag_features[3]) if len(rag_features) > 3 else 0.0,
        "rag_avg_change":   float(rag_features[0]) if len(rag_features) > 0 else 0.0,
        "similarity":       float(rag_features[8]) if len(rag_features) > 8 else 0.0,
        "news_type":        news_type,
        "sentiment_reliable": sent.get("sentiment_reliable", True),
        "source":           "live_xgb_v10",
        "similar": [
            {
                "title":  s.get("title", ""),
                "change": s.get("btc_change_15m", 0.0),
                "sim":    s.get("similarity_score", 0.0),
            }
            for s in similar_news[:3]
        ],
    }

    # ── Step 6: Route ─────────────────────────────────────────────
    if should_display_in_all(model_score, model_score_1h, confidence_pct / 100, title):
        _log.info(
            "%s | w=%s | score=%.2f | conf=%s%% | %s | %s",
            signal_type, sent['weight'], model_score, confidence_pct,
            pub_dt.strftime('%Y-%m-%d %H:%M UTC'), title[:55],
        )
        await send_to_dashboard(payload)
    else:
        _log.debug(
            "Filtered | score=%.2f conf=%s%% | %s | %s",
            model_score, confidence_pct,
            pub_dt.strftime('%Y-%m-%d %H:%M UTC'), title[:50],
        )
        return

    # HOT → Telegram
    if is_hot(model_score, model_score_1h, confidence_pct / 100, age_minutes):
        _log.info("HOT SIGNAL | Posting to Telegram")
        await post_hot_to_telegram(payload)

    if similar_news:
        for s in similar_news[:3]:
            _log.debug(
                "RAG: %s → BTC %+.2f%%",
                s.get('title', '')[:60], s.get('btc_change_15m', 0),
            )

    return payload


# ══════════════════════════════════════════════════════════════════
# PROCESSOR LOOP
# ══════════════════════════════════════════════════════════════════
async def processor_loop(pool):
    while True:
        if not news_queue:
            await asyncio.sleep(2)
            continue

        batch = []
        while news_queue and len(batch) < BATCH_SIZE:
            batch.append(news_queue.popleft())

        for news in batch:
            try:
                title  = news.get("title", "")
                link   = news.get("link") or title
                pub_dt = news.get("pub_dt", datetime.now(timezone.utc))
                coin   = extract_coin_from_text(news.get("text", ""))

                # Skip items already persisted in DB (survives restarts within backfill window)
                if pool is not None and await is_processed(pool, link):
                    continue

                payload = await process_news_item(news)

                if pool is not None:
                    await mark_processed(pool, link)
                    db_id = await save_full_news(
                        pool=pool,
                        title=title,
                        link=link,
                        source=news.get("source", "rss"),
                        coin=coin or "BTC",
                        category="crypto",
                        signal=payload["type"] if payload else "FILTERED",
                        impact_score=payload.get("model_score") or 0.0 if payload else 0.0,
                        published_at=pub_dt,
                    )
                    if db_id:
                        await price_tracker.add_pending(
                            news_id=db_id,
                            symbol=(coin or "BTC")[:10],
                            news_time=pub_dt,
                        )

            except Exception as e:
                _log.error("Error processing article: %s", e)

        await asyncio.sleep(1)


# ══════════════════════════════════════════════════════════════════
# PRICE TRACKER LOOP
# ══════════════════════════════════════════════════════════════════
async def price_tracker_loop(pool):
    while True:
        try:
            if pool is None:
                await asyncio.sleep(300)
                continue
            def _mv(r, key):
                v = r.get(key)
                return v.get("change_percent") if isinstance(v, dict) else v

            results = await price_tracker.process_pending(
                lambda r: save_price_movement(
                    pool=pool,
                    news_id=r["news_id"],
                    symbol=r["symbol"],
                    price_at_news=r.get("price_at_news"),
                    price_15m=r.get("price_15m"),
                    price_1h=r.get("price_1h"),
                    price_4h=r.get("price_4h"),
                    movement_15m=_mv(r, "movement_15m"),
                    movement_1h=_mv(r, "movement_1h"),
                )
            )
            if results:
                _log.info("Tracked %d price movements", len(results))
        except Exception as e:
            _log.error("Price tracker error: %s", e)
        await asyncio.sleep(300)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
async def main():
    loop = asyncio.get_running_loop()

    def _suppress_connection_lost(loop, context):
        msg = str(context.get("exception", context.get("message", ""))).lower()
        if any(x in msg for x in ("connection_lost", "connection reset", "connection closed")):
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_suppress_connection_lost)

    # Try DB connection once — if unavailable, run in cache-only mode (news_cache.json still works)
    pool = None
    try:
        pool = await create_pool()
        await create_tables(pool)
        _log.info("Database connected")
    except Exception as db_err:
        err_name = str(db_err) or type(db_err).__name__
        _log.warning("DB unavailable (%s) — running in cache-only mode; check DATABASE_URL in .env", err_name)

    _log.info("Dashboard: %s", DASHBOARD_API)
    _log.info("Bot started")

    await asyncio.gather(
        start_telegram_listener(news_queue),
        processor_loop(pool),
        price_tracker_loop(pool),
        return_exceptions=True,
    )


if __name__ == "__main__":
    import signal
    import traceback
    from log import setup_logging

    setup_logging(log_file=os.getenv("LOG_FILE"))

    from config import validate_bot
    validate_bot()

    # SIGTERM — set a flag so the crash loop exits cleanly (systemd / Docker stop)
    _shutdown = False
    def _handle_sigterm(sig, frame):
        global _shutdown
        _shutdown = True
        _log.info("SIGTERM received — shutting down after current iteration")
    signal.signal(signal.SIGTERM, _handle_sigterm)

    _log.info("Loading models (XGBoost v9 + DualBERT)…")
    _load_sentiment_models()
    _load_model()
    _log.info("Models ready")

    _backoff = 3          # seconds; doubles on repeated non-network crashes
    _crash_streak = 0

    while not _shutdown:
        try:
            asyncio.run(main())
            _backoff = 3
            _crash_streak = 0
        except KeyboardInterrupt:
            _log.info("Bot stopped by user (KeyboardInterrupt)")
            break
        except Exception as e:
            _crash_streak += 1
            traceback.print_exc()
            msg = str(e).lower()
            if any(x in msg for x in ("connection_lost", "connection reset", "connection closed")):
                delay = 3
                _log.warning("Telegram connection dropped — reconnecting in %ds", delay)
                _backoff = 3        # reset backoff for network blips
                _crash_streak = 0
            elif isinstance(e, (TimeoutError, OSError, ConnectionRefusedError)) or "timeout" in msg or "unreachable" in msg:
                delay = min(_backoff * 2, 120)
                _log.warning("Network/DB error (%s) — retrying in %ds", type(e).__name__, delay)
                _backoff = delay
            else:
                delay = min(_backoff * 2, 300)
                _log.error("Crash streak=%d: %s — restarting in %ds", _crash_streak, e or type(e).__name__, delay)
                if _crash_streak >= 5:
                    _log.critical("5 consecutive crashes — check logs. Sleeping %ds before retry.", delay)
                _backoff = delay
            time_module.sleep(delay)

    _log.info("Bot exited cleanly")