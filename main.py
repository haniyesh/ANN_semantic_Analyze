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
  3. Run through the 15-minute XGBoost BERT + RAG model (1578 features)
  4. Route using the canonical impact contract (see config.py):
     - Hot:      score_15m >= 0.60, confidence >= 70%, positive/negative
     - Moderate: score_15m >= 0.40 and confidence >= 60%
       or score_15m >= 0.60 with neutral sentiment
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
# Initialize XGBoost's native/OpenMP runtime before PyTorch.  If PyTorch loads
# first, constructing the first Booster can abort the process with glibc's
# "double free detected in tcache" on Linux.  Keep the guard alive so the
# runtime is not torn down while application classifiers are in use.
import xgboost as xgb
_xgb_runtime_guard = xgb.Booster()
import torch
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
    impact_tier,
    BOT_TOKEN,
    CHANNEL_ID,
    ADMIN_ID,
)
from services.model_refs import FINBERT_MODEL, FINBERT_REVISION

from bot.telegram_listener import start as start_telegram_listener
from services.price_fetcher import PriceTracker, extract_coin_from_text, get_price_at_times, calculate_movement
from storage.database import (
    create_pool, create_tables,
    is_processed, mark_processed,
    save_news, save_price_movement,
)
from log import get_logger
from pipeline.feature_contract import SENTIMENT_KEYS, build_bert_rag_features
_log = get_logger("main")


news_queue    = deque()
price_tracker = PriceTracker()

RAG_PENDING_FILE = Path(__file__).parent / "storage" / "rag_pending_outcomes.json"
RAG_DEAD_LETTER_FILE = Path(__file__).parent / "storage" / "rag_dead_letter.json"
BOT_HEALTH_FILE = Path(__file__).parent / "storage" / "bot_health.json"
DASHBOARD_OUTBOX_FILE = Path(__file__).parent / "storage" / "dashboard_outbox.json"
DASHBOARD_DEAD_LETTER_FILE = Path(__file__).parent / "storage" / "dashboard_dead_letter.json"
BOT_INSTANCE_LOCK_FILE = Path(__file__).parent / "storage" / "bot.lock"
DASHBOARD_OUTBOX_MAX_ITEMS = 1000
DASHBOARD_OUTBOX_MAX_ATTEMPTS = 12
OPERATIONAL_ALERT_COOLDOWN_SECONDS = 30 * 60
RAG_OUTCOME_DELAY_SECONDS = 20 * 60
RAG_RETRY_SECONDS = 5 * 60
RAG_MAX_RETRIES = 12
_rag_pending_lock: asyncio.Lock | None = None
_dashboard_outbox_lock: asyncio.Lock | None = None
_dashboard_client: httpx.AsyncClient | None = None
_shutdown = False
_bot_started_at = int(time_module.time())
_last_rag_query_at = 0
_last_inference_at = 0
_last_dashboard_delivery_at = 0
_instance_lock_handle = None
_last_operational_alert: dict[str, int] = {}

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
# Tiers: Hot (score >= 0.60, confidence >= 70%, directional) |
# Moderate (score >= 0.40, confidence >= 60%; high-score neutral is Moderate)


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

    clf15_path  = ROOT_DIR / "xgb_impact_clf_15m_bert_rag.json"
    scaler_path = ROOT_DIR / "xgb_feature_scaler_bert_rag.pkl"

    if not clf15_path.exists():
        _log.warning("Impact classifier not found — scoring disabled")
        return {}

    clf_15m = xgb.XGBClassifier(); clf_15m.load_model(str(clf15_path))

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        _log.warning("Feature scaler not found — run training/xgboost_train_bert.py first")
        return {}

    # Reuse the FinBERT classifier already loaded by the sentiment pipeline.
    # Loading a second copy here wastes memory and can crash native PyTorch/CUDA
    # teardown with "double free detected" on some Linux environments.  The
    # classifier's base encoder produces the same 768-dimensional hidden state
    # required by the XGBoost feature contract.
    sentiment_models = _load_sentiment_models()
    fb_pipe = sentiment_models["fb"]
    fb_tok = fb_pipe.tokenizer
    fb_mdl = fb_pipe.model.base_model.eval().to("cpu")

    thr15 = 0.51
    res_path = ROOT_DIR / "xgb_bert_rag_results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        thr15 = res.get("threshold_15m", thr15)

    n_features = clf_15m.n_features_in_
    if scaler.n_features_in_ != n_features:
        raise RuntimeError(
            f"Scaler expects {scaler.n_features_in_} features but model expects {n_features}. "
            "Scaler and model were built separately — re-run training/xgboost_train_bert.py."
        )

    _model_bundle = {
        "clf_15m": clf_15m, "scaler": scaler,
        "fb_tok": fb_tok, "fb_mdl": fb_mdl,
        "thresh15": thr15,
        "n_features": n_features,
    }
    _log.info("15m XGBoost BERT + RAG loaded: %d features (threshold=%.3f)",
              n_features, thr15)
    return _model_bundle


def run_model(features: np.ndarray) -> dict:
    """Run XGBoost BERT on the feature vector produced by build_xgb_features."""
    bundle = _load_model()
    if not bundle:
        return {"model_score": 0.0, "pred_15m": 0,
                "prob_15m": 0.0, "reg_pred_15m": 0.0, "confidence_model": 0.0}

    n_exp = bundle["n_features"]
    if features.shape[0] != n_exp:
        raise ValueError(
            f"Feature vector is {features.shape[0]}-dim but model expects {n_exp}. "
            "Check RAG padding in build_xgb_features."
        )

    X = bundle["scaler"].transform(features.reshape(1, -1)).astype(np.float32)
    p15 = float(bundle["clf_15m"].predict_proba(X)[0, 1])

    return {
        "model_score":      round(p15, 4),
        "pred_15m":         int(p15 >= bundle["thresh15"]),
        "prob_15m":         round(p15, 4),
        "reg_pred_15m":     0.0,
        "confidence_model": round(p15, 4),
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
    rag_features: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the 1578-dim vector expected by the 15-minute RAG model:
      CryptoBERT(768) | FinBERT(768) | sentiment(13) | type_probs(11) | macro(8) | RAG(10)
    """
    try:
        from training.xgboost_train_bert import crypto_news_type_classify
        type_probs = crypto_news_type_classify(cb_embedding.reshape(1, -1))[0]
    except Exception:
        type_probs = np.zeros(11, dtype=np.float32)

    rag = rag_features if rag_features is not None else np.zeros(10, dtype=np.float32)
    missing_sentiment = [key for key in SENTIMENT_KEYS if key not in sent]
    if missing_sentiment:
        raise ValueError(f"sentiment payload missing feature columns: {missing_sentiment}")
    return build_bert_rag_features(
        sent, cb_embedding, fb_embedding, type_probs, macro, rag
    )


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
    global _last_rag_query_at
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
        _last_rag_query_at = int(time_module.time())
        return result["features"], result.get("similar_news", [])
    except Exception as e:
        _log.warning("RAG query failed: %s", e)
        return np.zeros(10, dtype=np.float32), []


# ══════════════════════════════════════════════════════════════════
# HOT SIGNAL CHECK
# ══════════════════════════════════════════════════════════════════
def is_hot(model_score: float, confidence: float, age_minutes: float, sentiment: str = "") -> bool:
    """Hot alert: high score, high confidence, and clear bullish/bearish direction."""
    directional = str(sentiment or "").lower() in {"positive", "negative"}
    return (
        age_minutes < HOT_MAX_AGE_MIN
        and abs(model_score) >= HOT_MIN_MODEL_SCORE
        and confidence >= HOT_MIN_CONFIDENCE
        and directional
    )


def should_display_in_all(model_score, confidence, title=""):
    """Display gate: score >= IMPORTANT_MIN_SCORE AND confidence >= IMPORTANT_MIN_CONFIDENCE
    AND title >= 20 chars."""
    if len(title.strip()) < 20:
        return False
    return (
        abs(model_score) >= IMPORTANT_MIN_SCORE and
        confidence >= IMPORTANT_MIN_CONFIDENCE
    )


# ══════════════════════════════════════════════════════════════════
# DASHBOARD ROUTING
# ══════════════════════════════════════════════════════════════════
async def _post_to_dashboard(payload: dict) -> bool:
    """Send signal to dashboard ALL feed."""
    global _dashboard_client, _last_dashboard_delivery_at
    try:
        headers = {}
        _key = os.getenv("INGEST_API_KEY", "")
        if _key:
            headers["X-API-Key"] = _key
        if _dashboard_client is None or _dashboard_client.is_closed:
            _dashboard_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        response = await _dashboard_client.post(
            f"{DASHBOARD_API}/news", json=payload, headers=headers
        )
        response.raise_for_status()
        _last_dashboard_delivery_at = int(time_module.time())
        return True
    except httpx.HTTPStatusError as exc:
        _log.error(
            "Dashboard rejected news | status=%d | %s",
            exc.response.status_code,
            payload.get("title", "")[:60],
        )
    except Exception as e:
        _log.warning("Dashboard API error: %s", e)
    return False


def _read_dashboard_outbox() -> list[dict]:
    if not DASHBOARD_OUTBOX_FILE.exists():
        return []
    try:
        data = json.loads(DASHBOARD_OUTBOX_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        _log.error("Cannot read dashboard outbox: %s", exc)
        return []


def _write_dashboard_outbox(items: list[dict]) -> None:
    DASHBOARD_OUTBOX_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DASHBOARD_OUTBOX_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DASHBOARD_OUTBOX_FILE)


def _append_json_list(path: Path, item: dict) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(data, list):
            data = []
    except (OSError, json.JSONDecodeError):
        data = []
    data.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data[-DASHBOARD_OUTBOX_MAX_ITEMS:], ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _get_dashboard_outbox_lock() -> asyncio.Lock:
    global _dashboard_outbox_lock
    if _dashboard_outbox_lock is None:
        _dashboard_outbox_lock = asyncio.Lock()
    return _dashboard_outbox_lock


async def _enqueue_dashboard_delivery(payload: dict) -> None:
    async with _get_dashboard_outbox_lock():
        items = await asyncio.to_thread(_read_dashboard_outbox)
        item_id = payload.get("id")
        items = [item for item in items if item.get("payload", {}).get("id") != item_id]
        items.append({
            "payload": payload,
            "attempts": 0,
            "next_attempt_at": int(time_module.time()) + 30,
        })
        while len(items) > DASHBOARD_OUTBOX_MAX_ITEMS:
            dropped = items.pop(0)
            dropped["dead_letter_reason"] = "outbox capacity exceeded"
            dropped["failed_at"] = int(time_module.time())
            await asyncio.to_thread(_append_json_list, DASHBOARD_DEAD_LETTER_FILE, dropped)
        await asyncio.to_thread(_write_dashboard_outbox, items)


async def send_to_dashboard(payload: dict) -> bool:
    """Deliver immediately or persist for retry across process restarts."""
    delivered = await _post_to_dashboard(payload)
    if not delivered:
        await _enqueue_dashboard_delivery(payload)
    return delivered


async def dashboard_delivery_worker_loop() -> None:
    while True:
        async with _get_dashboard_outbox_lock():
            items = await asyncio.to_thread(_read_dashboard_outbox)
        now = int(time_module.time())
        for queued in items:
            if int(queued.get("next_attempt_at", 0)) > now:
                continue
            payload = queued.get("payload", {})
            delivered = await _post_to_dashboard(payload)
            async with _get_dashboard_outbox_lock():
                current = await asyncio.to_thread(_read_dashboard_outbox)
                match_id = payload.get("id")
                if delivered:
                    current = [q for q in current if q.get("payload", {}).get("id") != match_id]
                else:
                    dead = None
                    for q in current:
                        if q.get("payload", {}).get("id") == match_id:
                            q["attempts"] = int(q.get("attempts", 0)) + 1
                            if q["attempts"] >= DASHBOARD_OUTBOX_MAX_ATTEMPTS:
                                q["dead_letter_reason"] = "maximum delivery attempts exceeded"
                                q["failed_at"] = int(time_module.time())
                                dead = dict(q)
                            else:
                                delay = min(30 * (2 ** min(q["attempts"], 7)), 3600)
                                q["next_attempt_at"] = int(time_module.time()) + delay
                    if dead is not None:
                        current = [q for q in current if q.get("payload", {}).get("id") != match_id]
                        await asyncio.to_thread(_append_json_list, DASHBOARD_DEAD_LETTER_FILE, dead)
                await asyncio.to_thread(_write_dashboard_outbox, current)
        await asyncio.sleep(15)


async def post_hot_to_telegram(payload: dict):
    """Post hot signal to Telegram signal channel."""
    try:
        from telegram import Bot
        if not BOT_TOKEN or not CHANNEL_ID:
            _log.error("Telegram hot alert skipped: bot token or channel ID is missing")
            return
        bot = Bot(token=BOT_TOKEN)
        chat_id = CHANNEL_ID

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
      5. Model inference (15-minute XGBoost BERT + RAG)
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
        sent.update({
            "cb_prob_pos": sent.get("prob_positive", 0.0),
            "cb_prob_neg": sent.get("prob_negative", 0.0),
            "cb_prob_neu": sent.get("prob_neutral", 0.0),
            "fb_prob_pos": sent.get("prob_positive", 0.0),
            "fb_prob_neg": sent.get("prob_negative", 0.0),
            "fb_prob_neu": sent.get("prob_neutral", 0.0),
            "rb_prob_pos": sent.get("prob_positive", 0.0),
            "rb_prob_neg": sent.get("prob_negative", 0.0),
            "rb_prob_neu": sent.get("prob_neutral", 0.0),
            "net_agreement": sent.get("prob_positive", 0.0) - sent.get("prob_negative", 0.0),
        })
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
    features = build_xgb_features(sent, embedding, fb_embedding, macro, rag_features)

    # ── Step 5: Model inference (offloaded) ────────────────────────
    global _last_inference_at
    model_result   = await asyncio.to_thread(run_model, features)
    _last_inference_at = int(time_module.time())

    # XGBoost outputs calibrated probs [0,1] — normalization is identity (min=0, max=1)
    model_score    = _normalize_score(
        model_result["model_score"], SCORE_15M_MIN, SCORE_15M_MAX
    )
    # ── Step 5: Build payload ─────────────────────────────────────
    signal_type = (
        "BUY"  if sent["sentiment_score"] > 1  else
        "SELL" if sent["sentiment_score"] < -1 else
        "NEUTRAL"
    )
    confidence_pct = round(sent["confidence"] * 100)

    _nid = hashlib.sha1(f"{channel}|{title}|{published_ts}".encode()).hexdigest()[:32]

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
        "score_normalized": True,
        "pred_15m":         model_result["pred_15m"],
        "confidence_model": model_result.get("confidence_model", 0.0),
        "impact":           impact_tier(
            model_score,
            confidence=confidence_pct,
            sentiment=sent["sentiment"],
        ),
        "age_minutes":      round(age_minutes, 1),
        "published_ts":     published_ts,
        "link":             news.get("link", ""),
        "btc_change_15m":   0.0,
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

    # Every valid scored item contributes to the future RAG corpus, regardless
    # of whether it is important enough to be displayed on the dashboard.
    try:
        await enqueue_rag_outcome(payload)
    except Exception as exc:
        _log.error("Failed to persist RAG outcome job: %s", exc)

    # ── Step 6: Persist + route ───────────────────────────────────
    # Store every valid scored item in the dashboard/API cache. The API and
    # frontend apply the Hot/Moderate display gate, so Low items remain hidden
    # from the dashboard while still preserving restart continuity and RAG data.
    await send_to_dashboard(payload)

    if should_display_in_all(model_score, confidence_pct / 100, title):
        _log.info(
            "%s | w=%s | score=%.2f | conf=%s%% | %s | %s",
            signal_type, sent['weight'], model_score, confidence_pct,
            pub_dt.strftime('%Y-%m-%d %H:%M UTC'), title[:55],
        )
    else:
        _log.debug(
            "Stored as Low (hidden from dashboard) | score=%.2f conf=%s%% | %s | %s",
            model_score, confidence_pct,
            pub_dt.strftime('%Y-%m-%d %H:%M UTC'), title[:50],
        )

    # HOT → Telegram
    if is_hot(model_score, confidence_pct / 100, age_minutes, sent["sentiment"]):
        _log.info("HOT SIGNAL | Posting to Telegram")
        await post_hot_to_telegram(payload)

    if similar_news:
        for s in similar_news[:3]:
            _log.debug(
                "RAG: %s → BTC %+.2f%%",
                s.get('title', '')[:60], s.get('btc_change_15m', 0),
            )

    return payload


def _read_rag_pending() -> list[dict]:
    if not RAG_PENDING_FILE.exists():
        return []
    try:
        data = json.loads(RAG_PENDING_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        _log.error("Cannot read persistent RAG outcome queue: %s", exc)
        return []


def _write_rag_pending(items: list[dict]) -> None:
    RAG_PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAG_PENDING_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RAG_PENDING_FILE)


def _append_rag_dead_letter(item: dict) -> None:
    try:
        existing = json.loads(RAG_DEAD_LETTER_FILE.read_text(encoding="utf-8")) \
            if RAG_DEAD_LETTER_FILE.exists() else []
        if not isinstance(existing, list):
            existing = []
    except (OSError, json.JSONDecodeError):
        existing = []
    existing.append(item)
    RAG_DEAD_LETTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAG_DEAD_LETTER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RAG_DEAD_LETTER_FILE)


def _get_rag_pending_lock() -> asyncio.Lock:
    global _rag_pending_lock
    if _rag_pending_lock is None:
        _rag_pending_lock = asyncio.Lock()
    return _rag_pending_lock


async def enqueue_rag_outcome(payload: dict) -> None:
    """Durably schedule every scored item for outcome-ready RAG indexing."""
    pub_ts = int(payload.get("published_ts") or time_module.time())
    pending = {
        "id": payload["id"],
        "title": payload.get("title", ""),
        "channel": payload.get("channel", ""),
        "link": payload.get("link", ""),
        "published_ts": pub_ts,
        "due_at": max(int(time_module.time()), pub_ts + RAG_OUTCOME_DELAY_SECONDS),
        "attempts": 0,
    }
    async with _get_rag_pending_lock():
        items = await asyncio.to_thread(_read_rag_pending)
        items = [item for item in items if item.get("id") != pending["id"]]
        items.append(pending)
        await asyncio.to_thread(_write_rag_pending, items)


async def _retry_rag_item(item: dict, error: str) -> None:
    async with _get_rag_pending_lock():
        items = await asyncio.to_thread(_read_rag_pending)
        dead_item = None
        for saved in items:
            if saved.get("id") == item.get("id"):
                saved["attempts"] = int(saved.get("attempts", 0)) + 1
                delay = min(
                    RAG_RETRY_SECONDS * (2 ** min(saved["attempts"] - 1, 6)),
                    6 * 60 * 60,
                )
                saved["due_at"] = int(time_module.time()) + delay
                saved["last_error"] = error[:200]
                if saved["attempts"] >= RAG_MAX_RETRIES:
                    saved["failed_at"] = int(time_module.time())
                    dead_item = dict(saved)
        if dead_item is not None:
            items = [saved for saved in items if saved.get("id") != dead_item["id"]]
        await asyncio.to_thread(_write_rag_pending, items)
        if dead_item is not None:
            await asyncio.to_thread(_append_rag_dead_letter, dead_item)
            _log.critical(
                "RAG job moved to dead letter after %d attempts | %s",
                dead_item["attempts"], dead_item.get("title", "")[:60],
            )


async def _complete_rag_item(item_id: str) -> None:
    async with _get_rag_pending_lock():
        items = await asyncio.to_thread(_read_rag_pending)
        items = [item for item in items if item.get("id") != item_id]
        await asyncio.to_thread(_write_rag_pending, items)


async def rag_outcome_worker_loop() -> None:
    """Resume pending Qdrant inserts/outcome updates after process restarts."""
    while True:
        async with _get_rag_pending_lock():
            items = await asyncio.to_thread(_read_rag_pending)
        now = int(time_module.time())
        due = [item for item in items if int(item.get("due_at", 0)) <= now]

        for item in due:
            try:
                from pipeline.rag_news import upsert_live_news_outcome

                pub_dt = datetime.fromtimestamp(int(item["published_ts"]), tz=timezone.utc)
                prices = await get_price_at_times("BTC", pub_dt, intervals=[15])
                p0, p15 = prices.get(0), prices.get(15)
                if not p0 or not p15:
                    await _retry_rag_item(item, "15-minute price unavailable")
                    continue

                change_15m = calculate_movement(p0, p15).get("change_percent", 0.0)
                updated = await asyncio.to_thread(
                    upsert_live_news_outcome, item, change_15m, 0.0
                )
                if not updated:
                    await _retry_rag_item(item, "Qdrant labelled upsert unavailable")
                    continue

                await _complete_rag_item(item["id"])
                _log.info("RAG outcome saved | %s | 15m=%+.2f%%",
                          item.get("title", "")[:50], change_15m)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.warning("RAG outcome retry scheduled: %s", exc)
                await _retry_rag_item(item, str(exc))

        await asyncio.sleep(30)


async def bot_heartbeat_loop() -> None:
    """Publish bot liveness/progress for the API health endpoint."""
    while True:
        try:
            pending = await asyncio.to_thread(_read_rag_pending)
            dashboard_pending = await asyncio.to_thread(_read_dashboard_outbox)
            heartbeat = {
                "timestamp": int(time_module.time()),
                "started_at": _bot_started_at,
                "news_queue_size": len(news_queue),
                "rag_pending_outcomes": len(pending),
                "dashboard_pending_deliveries": len(dashboard_pending),
                "last_rag_query_at": _last_rag_query_at,
                "last_inference_at": _last_inference_at,
                "last_dashboard_delivery_at": _last_dashboard_delivery_at,
            }
            BOT_HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = BOT_HEALTH_FILE.with_suffix(".tmp")
            await asyncio.to_thread(
                tmp.write_text, json.dumps(heartbeat), encoding="utf-8"
            )
            await asyncio.to_thread(tmp.replace, BOT_HEALTH_FILE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("Bot heartbeat write failed: %s", exc)
        await asyncio.sleep(30)


async def _send_operational_alert(key: str, message: str) -> None:
    """Send cooldown-limited administrator alerts without crashing the bot."""
    now = int(time_module.time())
    if now - _last_operational_alert.get(key, 0) < OPERATIONAL_ALERT_COOLDOWN_SECONDS:
        return
    if not BOT_TOKEN or not ADMIN_ID:
        _log.error("OPERATIONAL ALERT [%s] %s", key, message)
        _last_operational_alert[key] = now
        return
    try:
        from telegram import Bot
        await Bot(token=BOT_TOKEN).send_message(
            chat_id=ADMIN_ID, text=f"⚠️ Crypto News operational alert\n\n{message}"
        )
        _last_operational_alert[key] = now
    except Exception as exc:
        _log.error("Operational alert delivery failed: %s", exc)


async def operational_alert_loop() -> None:
    """Alert on queues/dead letters and stale delivery while respecting cooldowns."""
    while True:
        try:
            outbox = await asyncio.to_thread(_read_dashboard_outbox)
            rag = await asyncio.to_thread(_read_rag_pending)
            for key, path in (
                ("dashboard-dead-letter", DASHBOARD_DEAD_LETTER_FILE),
                ("rag-dead-letter", RAG_DEAD_LETTER_FILE),
            ):
                try:
                    dead = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
                except (OSError, json.JSONDecodeError):
                    dead = []
                if dead:
                    await _send_operational_alert(key, f"{path.name} contains {len(dead)} failed jobs.")
            if len(outbox) >= int(DASHBOARD_OUTBOX_MAX_ITEMS * 0.8):
                await _send_operational_alert(
                    "dashboard-outbox-high",
                    f"Dashboard delivery outbox is {len(outbox)}/{DASHBOARD_OUTBOX_MAX_ITEMS} full.",
                )
            if rag:
                oldest = min(int(item.get("published_ts", time_module.time())) for item in rag)
                age = int(time_module.time()) - oldest
                if age > 2 * 60 * 60:
                    await _send_operational_alert(
                        "rag-backlog-stale", f"Oldest pending RAG outcome is {age // 60} minutes old."
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.error("Operational alert check failed: %s", exc)
        await asyncio.sleep(60)


def _acquire_single_instance_lock() -> None:
    """Prevent multiple bot processes from corrupting JSON-backed queues."""
    global _instance_lock_handle
    import fcntl

    BOT_INSTANCE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = BOT_INSTANCE_LOCK_FILE.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "Another bot instance owns storage/bot.lock; JSON queues require a single bot replica"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _instance_lock_handle = handle


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

                # PostgreSQL and the dashboard JSON have different recovery
                # lifecycles. An operator may intentionally rebuild/rewind the
                # JSON while the DB still remembers a link. In that case it
                # must be rescored and delivered again, but not inserted into
                # PostgreSQL a second time.
                already_processed = pool is not None and await is_processed(pool, link)

                payload = await process_news_item(news)

                if pool is not None and not already_processed:
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

                # Commit Telegram progress only after all processing and
                # persistence work above succeeded. If the process stops before
                # this point, the message is fetched again on the next start.
                telegram_msg_id = news.get("telegram_msg_id")
                if telegram_msg_id:
                    from bot.telegram_listener import acknowledge_message
                    acknowledge_message(news.get("telegram_channel", ""), telegram_msg_id)

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
    global _shutdown, _rag_pending_lock, _dashboard_outbox_lock, _dashboard_client
    loop = asyncio.get_running_loop()
    # A crash restart creates a new event loop; do not reuse a lock that may
    # have been bound to the previous, now-closed loop.
    _rag_pending_lock = None
    _dashboard_outbox_lock = None
    shutdown_event = asyncio.Event()

    def _request_shutdown() -> None:
        global _shutdown
        _shutdown = True
        shutdown_event.set()
        _log.info("Shutdown requested — cancelling background tasks")

    try:
        import signal
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
        loop.add_signal_handler(signal.SIGINT, _request_shutdown)
    except (NotImplementedError, RuntimeError):
        # Signal handlers are unavailable on some embedded/Windows loops.
        pass

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

    tasks = {
        asyncio.create_task(start_telegram_listener(news_queue), name="telegram-listener"),
        asyncio.create_task(processor_loop(pool), name="news-processor"),
        asyncio.create_task(price_tracker_loop(pool), name="price-tracker"),
        asyncio.create_task(rag_outcome_worker_loop(), name="rag-outcome-worker"),
        asyncio.create_task(dashboard_delivery_worker_loop(), name="dashboard-delivery-worker"),
        asyncio.create_task(bot_heartbeat_loop(), name="bot-heartbeat"),
        asyncio.create_task(operational_alert_loop(), name="operational-alerts"),
    }
    stop_task = asyncio.create_task(shutdown_event.wait(), name="shutdown-waiter")

    try:
        done, _ = await asyncio.wait(tasks | {stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task not in done:
            failed = next(iter(done))
            exc = failed.exception()
            if exc is not None:
                raise RuntimeError(f"Critical task {failed.get_name()} failed") from exc
            raise RuntimeError(f"Critical task {failed.get_name()} exited unexpectedly")
    finally:
        stop_task.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(stop_task, *tasks, return_exceptions=True)
        if _dashboard_client is not None:
            await _dashboard_client.aclose()
            _dashboard_client = None
        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    import traceback
    from log import setup_logging

    setup_logging(log_file=os.getenv("LOG_FILE"))

    from config import validate_bot
    validate_bot()

    # Fail fast before loading several gigabytes of model state. Lock
    # contention is not a crash and cannot be repaired by the retry loop: it
    # means the requested bot is already running normally.
    try:
        _acquire_single_instance_lock()
    except RuntimeError as exc:
        _log.error("%s", exc)
        raise SystemExit(2) from None

    _log.info("Loading models (XGBoost v9 + DualBERT)…")
    # XGBoost must construct its native classifier before PyTorch initializes
    # the transformer models. Reversing this order can trigger a libgomp/native
    # allocator double-free inside XGBoost on Linux.
    _load_model()
    _load_sentiment_models()  # cached by _load_model; documents readiness
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
