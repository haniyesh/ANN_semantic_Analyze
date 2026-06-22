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
  3. Run through XGBoost v9 model (DualBERT + PriceContext, 1578 features)
  4. Route — display filter uses confidence only, score is for impact badges:
     - Display gate: confidence >= 50% (no score gate)
     - Medium badge: max(score_15m, score_1h) >= 0.25
     - Hot badge:    max(score_15m, score_1h) >= 0.50
"""

import asyncio
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
    MODEL_PATH,
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
)

from bot.telegram_listener import start as start_telegram_listener
from services.price_fetcher import PriceTracker, extract_coin_from_text
from storage.database import (
    create_pool, create_tables,
    is_processed, mark_processed,
    save_news, save_price_movement,
)


news_queue    = deque()
price_tracker = PriceTracker()

FETCH_INTERVAL = 60
BATCH_SIZE     = 3


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
# Tiers: Show (≥0.20/50%) | Medium (≥0.30/55%) | Hot (≥0.55/60%) | Hidden (<0.20)


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

    clf15_path  = ROOT_DIR / "xgboost_v9_clf15m.json"
    clf1h_path  = ROOT_DIR / "xgboost_v9_clf1h.json"
    scaler_path = ROOT_DIR / "xgboost_v9_scaler.pkl"

    if not clf15_path.exists():
        print(f"  ⚠️  XGBoost v9 model not found — scoring disabled")
        return {}

    clf_15m = xgb.XGBClassifier(); clf_15m.load_model(str(clf15_path))
    clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(str(clf1h_path))

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("  ⚠️  XGBoost scaler not found — run xgboost_v9.py first")
        return {}

    # Load FinBERT for live embeddings — always CPU to avoid CUDA capability mismatch
    from transformers import AutoTokenizer, AutoModel
    fb_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_mdl = AutoModel.from_pretrained("ProsusAI/finbert").eval().to("cpu")

    thr15, thr1h = 0.295, 0.265
    res_path = ROOT_DIR / "xgboost_v9_results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        thr15 = res.get("threshold_15m", thr15)
        thr1h = res.get("threshold_1h",  thr1h)

    _model_bundle = {
        "clf_15m": clf_15m, "clf_1h": clf_1h, "scaler": scaler,
        "fb_tok": fb_tok, "fb_mdl": fb_mdl,
        "thresh15": thr15, "thresh1h": thr1h,
    }
    print(f"  ✅ XGBoost v9 loaded (thresh15={thr15:.3f} thresh1h={thr1h:.3f})")
    return _model_bundle


def run_model(features: np.ndarray) -> dict:
    """Run XGBoost v9 on a single 1578-dim feature vector. Returns score + prediction."""
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
    rag: np.ndarray,
) -> np.ndarray:
    """
    Build 1578-dim flat feature vector matching xgboost_v9 training layout:
      CryptoBERT(768) | FinBERT(768) | sentiment(13) | type_probs(11) | macro(8) | RAG(10)
    """
    try:
        from training.xgboost_v9 import crypto_news_type_classify
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

    return np.concatenate([cb_embedding, fb_embedding, sent_vec, type_probs, macro, rag]).astype(np.float32)


def build_macro_features(pub_dt: datetime) -> np.ndarray:
    """
    Build 8-dim macro feature vector: 5 timing + 3 price context.
    Matches xgboost_v9 training layout: [weekend, low_liq, us_hours, asia_hours, fomc_week,
                                          btc_vol, btc_mom, fear_greed]
    btc_vol and btc_mom are unavailable for live items → zeroed.
    """
    hour = pub_dt.hour
    timing = np.array([
        0.0,                             # is_weekend       (disabled)
        0.0,                             # is_low_liquidity (disabled)
        float(13 <= hour <= 21),         # is_us_hours
        float(0  <= hour <= 8),          # is_asia_hours
        0.0,                             # fomc_week        (simplified)
    ], dtype=np.float32)
    price_ctx = np.array([0.0, 0.0, _get_live_fear_greed()], dtype=np.float32)
    return np.concatenate([timing, price_ctx])




# ══════════════════════════════════════════════════════════════════
# RAG QUERY (single item)
# ══════════════════════════════════════════════════════════════════
async def save_full_news(
    pool, title: str, link: str, source: str,
    coin: str, category: str, signal: str,
    impact_score: float, published_at: datetime,
) -> int | None:
    """Save news item to database. Returns news_id or None on error."""
    try:
        return None
    except Exception:
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
            "fomc_week":        0.0,
        }
        result = query_single(
            title=title,
            before_timestamp=published_ts,
            channel_impact_rates=ch_rates,
            macro_now=macro_now,
        )
        return result["features"], result.get("similar_news", [])
    except Exception as e:
        print(f"  ⚠️  RAG query failed: {e}")
        return np.zeros(10, dtype=np.float32), []


# ══════════════════════════════════════════════════════════════════
# HOT SIGNAL CHECK
# ══════════════════════════════════════════════════════════════════
def is_hot(model_score: float, model_score_1h: float,
           confidence: float, age_minutes: float) -> bool:
    """Hot tier: max(score_15m, score_1h) >= 0.50 AND confidence >= 50% AND age < 30min."""
    return (
        max(abs(model_score), abs(model_score_1h)) >= HOT_MIN_MODEL_SCORE and
        confidence       >= HOT_MIN_CONFIDENCE   and
        age_minutes      <  HOT_MAX_AGE_MIN
    )


def should_display_in_all(model_score, model_score_1h, confidence, title=""):
    """Display gate: confidence >= 50% AND title >= 20 chars (no score gate)."""
    if len(title.strip()) < 20:
        return False
    return confidence >= IMPORTANT_MIN_CONFIDENCE


# ══════════════════════════════════════════════════════════════════
# DASHBOARD ROUTING
# ══════════════════════════════════════════════════════════════════
async def send_to_dashboard(payload: dict):
    """Send signal to dashboard ALL feed."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{DASHBOARD_API}/news", json=payload, timeout=3)
    except Exception as e:
        print(f"  ⚠️  Dashboard API error: {e}")


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
        print(f"  ⚠️  Telegram hot signal error: {e}")


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
      5. Model inference (production_system_v8.pt)
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
        print(f"  🚫 Pre-filter [{skip_reason}] | {title[:60]}")
        return
    news["title"] = title

    # ── Step 1: Sentiment ─────────────────────────────────────────
    sent, embedding, news_type = score_sentiment(title)

    # ── Step 1b: Weighted ensemble override ─────────────────────
    # Instead of trusting one model, average all 3 models' probabilities.
    # A very confident model pulls the average proportionally.
    try:
        from services.sentiment_score import load_models
        m = load_models()
        fb_raw = m["fb"](f"Bitcoin crypto market: {title}", truncation=True)[0]
        fb = {s["label"].lower(): s["score"] for s in fb_raw}
        rb_raw = m["rb"](f"BREAKING: {title} #Bitcoin #Crypto", truncation=True)[0]
        rb = {s["label"].lower(): s["score"] for s in rb_raw}

        cb_pos, cb_neg, cb_neu = sent["prob_positive"], sent["prob_negative"], sent["prob_neutral"]
        fb_pos, fb_neg, fb_neu = fb.get("positive", 0), fb.get("negative", 0), fb.get("neutral", 0)
        rb_pos, rb_neg, rb_neu = rb.get("positive", 0), rb.get("negative", 0), rb.get("neutral", 0)

        # Weighted average across all 3 models
        avg_pos = (cb_pos + fb_pos + rb_pos) / 3
        avg_neg = (cb_neg + fb_neg + rb_neg) / 3
        avg_neu = (cb_neu + fb_neu + rb_neu) / 3

        # Derive sentiment from weighted average
        if avg_neu > max(avg_pos, avg_neg):
            disc = 0
            sentiment = "neutral"
            confidence = avg_neu
        else:
            net = avg_pos - avg_neg
            disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
                   -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
            sentiment = "positive" if disc > 0 else ("negative" if disc < 0 else "neutral")
            confidence = avg_pos if disc > 0 else (avg_neg if disc < 0 else avg_neu)

        sent["sentiment"] = sentiment
        sent["sentiment_score"] = disc
        sent["confidence"] = round(confidence, 4)
        sent["prob_positive"] = round(avg_pos, 4)
        sent["prob_negative"] = round(avg_neg, 4)
        sent["prob_neutral"] = round(avg_neu, 4)
        sent["weight"] = max(5, min(10, round(confidence * 10)))

        # Raw per-model probabilities for feature vector
        sent["cb_prob_pos"] = round(cb_pos, 4)
        sent["cb_prob_neg"] = round(cb_neg, 4)
        sent["cb_prob_neu"] = round(cb_neu, 4)
        sent["fb_prob_pos"] = round(fb_pos, 4)
        sent["fb_prob_neg"] = round(fb_neg, 4)
        sent["fb_prob_neu"] = round(fb_neu, 4)
        sent["rb_prob_pos"] = round(rb_pos, 4)
        sent["rb_prob_neg"] = round(rb_neg, 4)
        sent["rb_prob_neu"] = round(rb_neu, 4)

        # Net agreement: mean of (pos-neg) per model, penalized for disagreement
        nets = [cb_pos - cb_neg, fb_pos - fb_neg, rb_pos - rb_neg]
        mean_net = sum(nets) / 3
        signs = [1 if n > 0 else (-1 if n < 0 else 0) for n in nets]
        agreement = 1.0 if len(set(signs)) == 1 else 0.5
        sent["net_agreement"] = round(mean_net * agreement, 4)

        # Reliability: if any model diverges > 0.3 from ensemble direction
        max_spread = max(
            abs(cb_pos - cb_neg) - abs(avg_pos - avg_neg),
            abs(fb_pos - fb_neg) - abs(avg_pos - avg_neg),
            abs(rb_pos - rb_neg) - abs(avg_pos - avg_neg),
        )
        sent["sentiment_reliable"] = max_spread < 0.3
    except Exception:
        sent["sentiment_reliable"] = True  # fallback: assume reliable

    # ── Step 1c: FinBERT embedding ────────────────────────────────
    fb_embedding = np.zeros(768, dtype=np.float32)
    bundle = _load_model()
    if bundle and "fb_tok" in bundle:
        try:
            inputs = bundle["fb_tok"](
                title, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            with torch.no_grad():
                fb_embedding = bundle["fb_mdl"](**inputs).last_hidden_state[:, 0, :].numpy().flatten().astype(np.float32)
        except Exception:
            pass

    # ── Step 2: Macro + RAG features ─────────────────────────────
    macro        = build_macro_features(pub_dt)
    # ── Step 3: RAG query ─────────────────────────────────────────
    rag_features, similar_news = query_rag(title, published_ts, channel)

    # ── Step 4: Build flat XGBoost feature vector ─────────────────
    features = build_xgb_features(sent, embedding, fb_embedding, macro, rag_features)

    # ── Step 5: Model inference ───────────────────────────────────
    model_result   = run_model(features)

    # XGBoost outputs calibrated probs [0,1] — normalization is identity (min=0, max=1)
    model_score    = _normalize_score(
        model_result["model_score"], SCORE_15M_MIN, SCORE_15M_MAX
    )
    model_score_1h = _normalize_score(
        model_result.get("model_score_1h", 0.0), SCORE_1H_MIN, SCORE_1H_MAX
    )

    # US trading hours boost (+15%)
    hour = pub_dt.hour
    if 13 <= hour <= 21:
        model_score    = min(model_score    * 1.15, 1.0)
        model_score_1h = min(model_score_1h * 1.15, 1.0)

    # ── Step 5: Build payload ─────────────────────────────────────
    signal_type = (
        "BUY"  if sent["sentiment_score"] > 1  else
        "SELL" if sent["sentiment_score"] < -1 else
        "NEUTRAL"
    )
    confidence_pct = round(sent["confidence"] * 100)

    payload = {
        "id":               published_ts * 1000,
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
        "source":           "live_xgb_v9",
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
        print(
            f"📰 {signal_type} | w={sent['weight']} | "
            f"score={model_score:.2f} | conf={confidence_pct}% | "
            f"{pub_dt.strftime('%Y-%m-%d %H:%M UTC')} | {title[:55]}"
        )
        await send_to_dashboard(payload)
    else:
        print(
            f"⛔ Filtered | score={model_score:.2f} "
            f"conf={confidence_pct}% | {pub_dt.strftime('%Y-%m-%d %H:%M UTC')} | {title[:50]}"
        )
        return

    # HOT → Telegram
    if is_hot(model_score, model_score_1h, confidence_pct / 100, age_minutes):
        print(f"🔥 HOT SIGNAL | Posting to Telegram")
        await post_hot_to_telegram(payload)

    if similar_news:
        for s in similar_news[:3]:
            print(
                f"  📚 RAG: {s.get('title','')[:60]} "
                f"→ BTC {s.get('btc_change_15m', 0):+.2f}%"
            )


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

                await process_news_item(news)

                if pool is not None:
                    await mark_processed(pool, link)
                    db_id = await save_full_news(
                        pool=pool,
                        title=title,
                        link=link,
                        source=news.get("source", "rss"),
                        coin=coin or "BTC",
                        category="crypto",
                        signal="PROCESSED",
                        impact_score=0.0,
                        published_at=pub_dt,
                    )
                    if db_id:
                        await price_tracker.add_pending(
                            news_id=db_id,
                            symbol=(coin or "BTC")[:10],
                            news_time=pub_dt,
                        )

            except Exception as e:
                print(f"❌ Error processing article: {e}")

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
            results = await price_tracker.process_pending(
                lambda r: save_price_movement(
                    pool=pool,
                    news_id=r["news_id"],
                    symbol=r["symbol"],
                    price_at_news=r.get("price_at_news"),
                    price_15m=r.get("price_15m"),
                    movement_15m=(
                        r.get("movement_15m", {}).get("change_percent")
                        if isinstance(r.get("movement_15m"), dict)
                        else r.get("movement_15m")
                    ),
                )
            )
            if results:
                print(f"📊 Tracked {len(results)} price movements")
        except Exception as e:
            print(f"❌ Price tracker error: {e}")
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
        print("✅ Database connected")
    except Exception as db_err:
        err_name = str(db_err) or type(db_err).__name__
        print(f"⚠️  DB unavailable ({err_name}) — running in cache-only mode")
        print("   News will be saved to news_cache.json. Check DATABASE_URL in .env to fix.")

    print(f"📊 Dashboard: {DASHBOARD_API}")
    print("🚀 Bot started\n")

    await asyncio.gather(
        start_telegram_listener(news_queue),
        processor_loop(pool),
        price_tracker_loop(pool),
        return_exceptions=True,
    )


if __name__ == "__main__":
    # Load all models once at startup — persists across Telegram reconnects
    print("🔧 Loading models (XGBoost v9 + DualBERT)...")
    _load_sentiment_models()
    _load_model()
    print("✅ Models ready\n")

    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("👋 Bot stopped by user")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e).lower()
            if any(x in msg for x in ("connection_lost", "connection reset", "connection closed")):
                print("⚡ Telegram connection dropped, reconnecting in 3s...")
                time_module.sleep(3)
            elif isinstance(e, (TimeoutError, OSError, ConnectionRefusedError)) or "timeout" in msg or "unreachable" in msg:
                print(f"⚠️  Network/DB error ({type(e).__name__}), retrying in 20s...")
                time_module.sleep(20)
            else:
                print(f"💥 Critical error: {e or type(e).__name__}, restarting in 10s...")
                time_module.sleep(10)