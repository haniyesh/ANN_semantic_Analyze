import asyncio
import json
import time as time_module
import warnings
warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*downstream task.*")

import numpy as np
import httpx
import torch
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

from config import (
    DASHBOARD_API,
    MODEL_PATH,
    CHANNEL_WEIGHTS_PATH,
    SCORE_15M_MIN, SCORE_15M_MAX,
    SCORE_1H_MIN,  SCORE_1H_MAX,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
    IMPORTANT_MIN_CONFIDENCE,
    HOT_MIN_SCORE_1H,
    HOT_MAX_AGE_MIN,
    BATCH_SIZE,
)

from bot.telegram_listener import start as start_telegram_listener
from services.price_fetcher import PriceTracker, extract_coin_from_text
from services.telegram_alert import send_telegram_alert
from storage.database import (
    create_pool, create_tables,
    is_processed, mark_processed,
    save_news, save_price_movement,
)


#  GLOBALS
news_queue    = deque()
price_tracker = PriceTracker()


#  CHANNEL WEIGHTS
def _load_channel_weights() -> dict:
    path = Path(CHANNEL_WEIGHTS_PATH)
    if not path.exists():
        return {}
    try:
        data    = json.loads(path.read_text())
        weights = data.get("channels", {})
        print(f"[WEIGHTS] Channel weights loaded (next update: {data.get('next_update', '')})")
        return weights
    except Exception:
        return {}


# SCORE NORMALIZATION
def _normalize_score(raw: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return round(max(0.0, min(1.0, (raw - min_val) / (max_val - min_val))), 4)


# ==============================
# 🧠 SENTIMENT MODELS
# ==============================
_sent_models = {}

def _load_sentiment_models():
    global _sent_models
    if _sent_models:
        return _sent_models

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, logging as hf_logging,
    )
    hf_logging.set_verbosity_error()
    print("[MODELS] Loading sentiment models...")

    cb_tok   = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cb_model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert")
    cb_model.eval()

    fb_pipe = pipeline(
        "text-classification", model="ProsusAI/finbert",
        return_all_scores=True, device=-1,
    )
    rb_pipe = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True, device=-1,
    )
    _sent_models = {
        "cb_tok":   cb_tok,
        "cb_model": cb_model,
        "fb_pipe":  fb_pipe,
        "rb_pipe":  rb_pipe,
    }
    print("[MODELS] Sentiment models loaded ✅")
    return _sent_models


def score_sentiment(title: str) -> dict:
    m = _load_sentiment_models()

    # CryptoBERT gatekeeper
    inputs = m["cb_tok"](
        title, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(m["cb_model"](**inputs).logits, dim=1).squeeze().numpy()

    prob_neg, prob_neu, prob_pos = float(probs[0]), float(probs[1]), float(probs[2])

    # Neutral gate
    if prob_neu >= 0.80:
        return {
            "sentiment":        "neutral",
            "sentiment_score":  0,
            "weight":           3,
            "confidence":       round(prob_neu, 4),
            "prob_positive":    round(prob_pos, 4),
            "prob_negative":    round(prob_neg, 4),
            "prob_neutral":     round(prob_neu, 4),
            "is_neutral_gated": True,
        }

    # FinBERT
    try:
        fb = {s["label"].lower(): s["score"]
              for s in m["fb_pipe"](f"Bitcoin crypto market news: {title}", truncation=True)[0]}
    except Exception:
        fb = {"positive": 0.33, "negative": 0.33, "neutral": 0.33}

    # RoBERTa
    try:
        rb = {s["label"].lower(): s["score"]
              for s in m["rb_pipe"](f"BREAKING: {title} #Bitcoin #Crypto", truncation=True)[0]}
    except Exception:
        rb = {"positive": 0.33, "negative": 0.33, "neutral": 0.33}

    # Ensemble
    pos = 0.40 * prob_pos + 0.35 * fb.get("positive", 0) + 0.25 * rb.get("positive", 0)
    neg = 0.40 * prob_neg + 0.35 * fb.get("negative", 0) + 0.25 * rb.get("negative", 0)
    neu = 0.40 * prob_neu + 0.35 * fb.get("neutral",  0) + 0.25 * rb.get("neutral",  0)

    net = pos - neg
    if   net >  0.5:  score =  3
    elif net >  0.25: score =  2
    elif net >  0.05: score =  1
    elif net < -0.5:  score = -3
    elif net < -0.25: score = -2
    elif net < -0.05: score = -1
    else:             score =  0

    conf      = max(pos, neg, neu)
    weight    = max(5, min(9, round(conf * 10)))
    sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"

    return {
        "sentiment":        sentiment,
        "sentiment_score":  score,
        "weight":           weight,
        "confidence":       round(conf, 4),
        "prob_positive":    round(pos, 4),
        "prob_negative":    round(neg, 4),
        "prob_neutral":     round(neu, 4),
        "is_neutral_gated": False,
    }


# ==============================
# 🤖 IMPACT MODEL
# ==============================
_model_bundle = {}

def _load_model():
    global _model_bundle
    if _model_bundle:
        return _model_bundle
    if not Path(MODEL_PATH).exists():
        print(f"[MODEL] Not found at {MODEL_PATH} — using score only")
        return {}

    from production_system_v5 import CryptoImpactNetV5, EmbProjection
    ckpt     = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    scalers  = ckpt["scalers"]
    thresh15 = ckpt.get("threshold_15m", 0.40)
    thresh1h = ckpt.get("threshold_1h",  0.43)
    emb_proj = EmbProjection()
    model    = CryptoImpactNetV5(
        sem_dim=785, rag_dim=10, macro_dim=5, mem_dim=5,
        emb_proj=emb_proj,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _model_bundle = {
        "model":    model,
        "scalers":  scalers,
        "thresh15": thresh15,
        "thresh1h": thresh1h,
    }
    print(f"[MODEL] Loaded ✅ (thresh15={thresh15:.2f})")
    return _model_bundle


def run_model(sem, rag, macro, market) -> dict:
    bundle = _load_model()
    if not bundle:
        return {"model_score": 0.0, "model_score_1h": 0.0, "pred_15m": 0, "pred_1h": 0}

    model, scalers, thresh15, thresh1h = (
        bundle["model"], bundle["scalers"],
        bundle["thresh15"], bundle["thresh1h"],
    )
    sem_s, rag_s, mac_s, mem_s = [
        torch.FloatTensor(s.transform(x.reshape(1, -1)))
        for s, x in zip(scalers, [sem, rag, macro, market])
    ]
    with torch.no_grad():
        out  = model(sem_s, rag_s, mac_s, mem_s)
        p15  = float(torch.sigmoid(out["cls_15m"]).item())
        p1h  = float(torch.sigmoid(out["cls_1h"]).item())
        r15  = float(out["reg_15m"].item())
        r1h  = float(out["reg_1h"].item())
        conf = float(torch.sigmoid(out["conf"]).item())

    mod15 = p15 * (1.0 + abs(r15) * 2.0)
    mod1h = p1h * (1.0 + abs(r1h) * 2.0)

    return {
        "model_score":      round(mod15, 3),
        "model_score_1h":   round(mod1h, 3),
        "pred_15m":         int(mod15 >= thresh15),
        "pred_1h":          int(mod1h >= thresh1h),
        "confidence_model": round(conf, 3),
    }


# ==============================
# 🔢 FEATURE BUILDERS
# ==============================
_cb_emb = {}

def _get_cryptobert():
    global _cb_emb
    if _cb_emb:
        return _cb_emb
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    mdl = AutoModel.from_pretrained("ElKulako/cryptobert")
    mdl.eval()
    _cb_emb = {"tok": tok, "mdl": mdl}
    return _cb_emb


def build_semantic_features(title: str, sent: dict) -> np.ndarray:
    cb     = _get_cryptobert()
    inputs = cb["tok"](
        title, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    )
    with torch.no_grad():
        emb = cb["mdl"](**inputs).last_hidden_state[:, 0, :].numpy().flatten()

    sent_vec = np.array([
        sent.get("sentiment_score", 0),
        sent.get("weight",          5),
        sent.get("confidence",      0),
        sent.get("prob_positive",   0),
        sent.get("prob_negative",   0),
        sent.get("prob_neutral",    0),
    ], dtype=np.float32)

    type_vec = np.zeros(11, dtype=np.float32)
    return np.concatenate([emb, sent_vec, type_vec]).astype(np.float32)


def build_macro_features(pub_dt: datetime) -> np.ndarray:
    hour = pub_dt.hour
    return np.array([
        0.0,
        0.0,
        float(13 <= hour <= 21),
        float(0  <= hour <= 8),
        0.0,
    ], dtype=np.float32)


def build_market_features(btc_price: float) -> np.ndarray:
    return np.zeros(5, dtype=np.float32)


# ==============================
# 🔍 RAG QUERY
# ==============================
def query_rag(title: str, published_ts: int, channel: str) -> tuple:
    try:
        from rag_news import query_single
        pub_dt = datetime.fromtimestamp(published_ts, tz=timezone.utc)
        hour   = pub_dt.hour
        result = query_single(
            title=title,
            before_timestamp=published_ts,
            channel_impact_rates={
                "CoinMarketCap":   0.175, "the_block_crypto": 0.062,
                "porter_news":     0.058, "coindesk":         0.058,
                "cryptoslatenews": 0.051, "cointelegraph":    0.048,
            },
            macro_now={
                "is_weekend":       0.0,
                "is_low_liquidity": 0.0,
                "is_us_hours":      float(13 <= hour <= 21),
                "is_asia_hours":    float(0  <= hour <= 8),
                "fomc_week":        0.0,
            },
        )
        return result["features"], result.get("similar_news", [])
    except Exception as e:
        print(f"[RAG] Query failed: {e}")
        return np.zeros(10, dtype=np.float32), []


# ==============================
# 🚦 DISPLAY FILTERS
# ==============================
def should_display(model_score: float, model_score_1h: float,
                   confidence: float, title: str = "") -> bool:
    if len(title.strip()) < 20:
        return False
    score = abs(model_score)
    s1h   = abs(model_score_1h)
    best  = max(score, s1h)
    if best >= SCORE_THRESHOLD_HIGH:
        return True
    score_ok = score >= SCORE_THRESHOLD_MEDIUM or s1h >= SCORE_THRESHOLD_MEDIUM
    return score_ok and confidence >= IMPORTANT_MIN_CONFIDENCE


def is_hot(model_score: float, model_score_1h: float,
           confidence: float, age_minutes: float) -> bool:
    return (
        abs(model_score)    >= SCORE_THRESHOLD_HIGH     and
        abs(model_score_1h) >= HOT_MIN_SCORE_1H         and
        confidence          >= IMPORTANT_MIN_CONFIDENCE  and
        age_minutes         <  HOT_MAX_AGE_MIN
    )


# ==============================
# 📤 SEND TO DASHBOARD
# ==============================
async def send_to_dashboard(payload: dict):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{DASHBOARD_API}/news", json=payload, timeout=3)
    except Exception as e:
        print(f"[DASHBOARD] Error: {e}")


# ==============================
# ⚙️ PROCESS ONE NEWS ITEM
# ==============================
async def process_news_item(news: dict):
    from pipeline.spam_filter import passes_pre_filters

    title   = news.get("title", "")
    channel = news.get("source", "unknown")
    pub_dt  = news.get("pub_dt", datetime.now(timezone.utc))

    if isinstance(pub_dt, str):
        pub_dt = datetime.fromisoformat(pub_dt.replace("Z", "+00:00"))
    if pub_dt.tzinfo is None:
        pub_dt = pub_dt.replace(tzinfo=timezone.utc)

    published_ts = int(pub_dt.timestamp())
    age_minutes  = (time_module.time() - published_ts) / 60

    # Step 0 — Pre-filters
    title, skip_reason = passes_pre_filters(title)
    if skip_reason:
        print(f"[SKIP] {skip_reason} | {title[:60]}")
        return
    news["title"] = title

    # Step 1 — Sentiment
    sent = score_sentiment(title)
    if sent.get("is_neutral_gated"):
        print(f"[SKIP] Neutral gated | {title[:60]}")
        return
    if sent.get("weight", 10) < 5:
        print(f"[SKIP] Low weight={sent['weight']} | {title[:60]}")
        return

    # Step 2 — Features
    sem    = build_semantic_features(title, sent)
    macro  = build_macro_features(pub_dt)
    market = build_market_features(news.get("btc_price", 0.0))

    # Step 3 — RAG
    rag_features, similar_news = query_rag(title, published_ts, channel)

    # Step 4 — Model inference
    model_result   = run_model(sem, rag_features, macro, market)
    model_score    = _normalize_score(model_result["model_score"],    SCORE_15M_MIN, SCORE_15M_MAX)
    model_score_1h = _normalize_score(model_result["model_score_1h"], SCORE_1H_MIN,  SCORE_1H_MAX)

    # US hours boost
    if 13 <= pub_dt.hour <= 21:
        model_score    = min(model_score    * 1.15, 1.0)
        model_score_1h = min(model_score_1h * 1.15, 1.0)

    # Channel penalty
    ch_multiplier = _channel_weights.get(channel, 1.0)
    if ch_multiplier < 1.0:
        model_score    = round(model_score    * ch_multiplier, 4)
        model_score_1h = round(model_score_1h * ch_multiplier, 4)

    # Step 5 — Build payload
    signal_type    = (
        "BUY"  if sent["sentiment_score"] >  1 else
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
        "impact": (
            "High"   if model_score >= SCORE_THRESHOLD_HIGH   else
            "Medium" if model_score >= SCORE_THRESHOLD_MEDIUM else "Low"
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
        "similar": [
            {
                "title":  s.get("title", ""),
                "change": s.get("btc_change_15m", 0.0),
                "sim":    s.get("similarity_score", 0.0),
            }
            for s in similar_news[:3]
        ],
    }

    # Step 6 — Route
    if should_display(model_score, model_score_1h, confidence_pct / 100, title):
        print(
            f"[NEWS] {signal_type} | w={sent['weight']} | "
            f"score={model_score:.2f} | conf={confidence_pct}% | {title[:55]}"
        )
        await send_to_dashboard(payload)
    else:
        print(f"[SKIP] score={model_score:.2f} conf={confidence_pct}% | {title[:50]}")
        return

    if is_hot(model_score, model_score_1h, confidence_pct / 100, age_minutes):
        print("[HOT] Posting to Telegram...")
        send_telegram_alert(news, signal_type, model_score)

    if similar_news:
        for s in similar_news[:3]:
            print(f"  [RAG] {s.get('title','')[:60]} → BTC {s.get('btc_change_15m',0):+.2f}%")

    return published_ts, coin if 'coin' in dir() else "BTC"


# ==============================
# 🔄 PROCESSOR LOOP
# ==============================
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
                coin   = extract_coin_from_text(news.get("text", "")) or "BTC"

                await process_news_item(news)
                await mark_processed(pool, link)

                db_id = await save_news(
                    pool=pool,
                    title=title,
                    link=link,
                    source=news.get("source", "unknown"),
                    coin=coin,
                    category="crypto",
                    signal="PROCESSED",
                    impact_score=0.0,
                    published_at=pub_dt,
                )

                # ── Real-time price tracking ──────────────────────
                # Fire-and-forget — no polling loop needed
                if db_id:
                    asyncio.create_task(
                        price_tracker.track(
                            news_id=db_id,
                            symbol=coin,
                            news_time=pub_dt,
                        )
                    )

            except Exception as e:
                print(f"[ERROR] Processing failed: {e}")

        await asyncio.sleep(1)


# ==============================
# 🚀 MAIN
# ==============================
async def main():
    loop = asyncio.get_running_loop()

    def _suppress_connection_lost(loop, context):
        msg = str(context.get("exception", context.get("message", ""))).lower()
        if any(x in msg for x in ("connection_lost", "connection reset", "connection closed")):
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_suppress_connection_lost)

    pool = await create_pool()
    await create_tables(pool)

    # Give price tracker access to the pool
    price_tracker.set_pool(pool)

    print("[DB] Connected ✅")
    print(f"[DASHBOARD] {DASHBOARD_API}")
    print("[BOT] Started 🚀\n")

    await asyncio.gather(
        start_telegram_listener(news_queue),
        processor_loop(pool),
        return_exceptions=True,
    )


# ==============================
# ▶️ ENTRY POINT
# ==============================
if __name__ == "__main__":
    print("[STARTUP] Loading models...")
    _load_sentiment_models()
    _load_model()
    _get_cryptobert()
    print("[STARTUP] Models ready ✅")

    _channel_weights = _load_channel_weights()

    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("[BOT] Stopped by user")
            break
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("connection_lost", "connection reset", "connection closed")):
                print("[BOT] Connection dropped, reconnecting in 3s...")
                time_module.sleep(3)
            else:
                print(f"[BOT] Critical error: {e}, restarting in 10s...")
                time_module.sleep(10)