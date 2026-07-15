"""
API server — JSON cache mode with authenticated ingestion.
Loads news_cache.json on startup and serves it.
POST /news requires X-API-Key header matching INGEST_API_KEY env var.
"""
import os
import re
import sys
import csv
import json
import math
import time
import secrets
import aiohttp
from pathlib import Path
from typing import List
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # make project root importable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn


# ── Request models (Pydantic) ─────────────────────────────────────
class IngestNewsItem(BaseModel):
    """Validated payload from the bot's ingest pipeline (POST /news)."""
    model_config = ConfigDict(extra="allow")   # forward all fields to cache
    title: str
    channel: str = ""
    id: str | None = None
    model_score: float | None = None
    model_score_1h: float | None = None
    published_ts: int | None = None
    type: str = "NEUTRAL"
    confidence: float = 50.0
    sentiment: str = "neutral"

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be empty")
        return v


class AnalyzeRequest(BaseModel):
    """Body for POST /analyze/custom."""
    title: str = Field(..., min_length=5, max_length=500,
                       description="News headline to score")


class SimilarRequest(BaseModel):
    """Body for POST /news/similar."""
    model_config = ConfigDict(extra="allow")
    title: str
    similar: list = []


class ExplainRequest(BaseModel):
    """Body for POST /news/explain."""
    model_config = ConfigDict(extra="allow")
    title: str = ""
    sentiment: str = "neutral"
    confidence: float = 50.0
    model_score: float = 0.0
    model_score_1h: float = 0.0
    btc_change_15m: float = 0.0
    btc_change_1h: float = 0.0
    channel: str = "unknown"
    similar: list = []

from config import (
    API_HOST, API_PORT, GROQ_API_KEYS, GROQ_CLASSIFICATION_MODEL, validate_api,
    SCORE_THRESHOLD_HOT, SCORE_THRESHOLD_MEDIUM, SCORE_THRESHOLD_SHOW, CONF_SHOW,
    impact_tier as _config_impact_tier,
)
from log import setup_logging, get_logger

setup_logging(log_file=os.getenv("LOG_FILE"))
_log = get_logger("api.server")
validate_api()

# ── Security config ──────────────────────────────────────────────
# INGEST_API_KEY guards POST /news. Read by _INGEST_API_KEY below (near the endpoint).
# CORS origins are read by _allowed_origins near the middleware registration below.

# ── Rate limiting (simple in-memory) ────────────────────────────
_rate_limits: dict[str, list] = {}  # ip -> [timestamps]
RATE_LIMIT_RPM = 60  # requests per minute per IP for expensive endpoints


def _news_id(channel: str, title: str, published_ts: int) -> str:
    """Deterministic, collision-resistant news ID stable across restarts.

    Uses the first 32 hex chars of SHA-1 over channel|title|published_ts so the
    value can also be converted to the UUID used by the Qdrant live index.
    SHA-1 is fine here — this is a content-address, not a security hash.
    Avoids: Python hash() salt (different every restart), published_ts*1000
    collisions (two headlines in the same second share an ID).
    """
    import hashlib
    raw = f"{channel}|{title}|{published_ts}"
    return hashlib.sha1(raw.encode()).hexdigest()[:32]


# Only honor X-Forwarded-For when running behind a trusted reverse proxy that
# strips/overwrites the header. Set BEHIND_PROXY=1 in .env for such deployments.
# Without it, any client can spoof X-Forwarded-For to bypass per-IP rate limiting.
_BEHIND_PROXY = os.getenv("BEHIND_PROXY", "").lower() in ("1", "true", "yes")


def _client_ip(request: Request) -> str:
    """Return the real client IP.

    Reads X-Forwarded-For only when BEHIND_PROXY=1, preventing spoofing on
    deployments that don't strip the header at the proxy layer.
    """
    if _BEHIND_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_ip: str, rpm: int = RATE_LIMIT_RPM):
    """Raise 429 if client_ip exceeds rpm requests/minute."""
    now = time.time()
    window = now - 60
    timestamps = [t for t in _rate_limits.get(client_ip, []) if t > window]
    if len(timestamps) >= rpm:
        raise HTTPException(429, "Rate limit exceeded — try again in a minute")
    timestamps.append(now)
    _rate_limits[client_ip] = timestamps

CACHE_FILE   = ROOT / "storage" / "news_cache.json"
HIST_CSV     = ROOT / "news_cleaned_filtered_scored.csv"
HISTORY_WINDOW_MONTHS = 3          # only serve news from the last N months

# Canonical channel name map — None means "blocked, drop the item".
# Single source of truth; applied via _normalize_channel() below.
_CHANNEL_NORM: dict[str, str | None] = {
    "CoinTelegraph":   "cointelegraph",
    "CoinMarketCap":   None,
    "CryptoNews":      None,
    "CoingraphNews":   None,
    "cryptoslatenews": None,
}


def _normalize_channel(ch: str) -> str | None:
    """Return canonical channel name, or None if the channel is blocked."""
    if ch not in _CHANNEL_NORM:
        return ch
    return _CHANNEL_NORM[ch]   # None → blocked

from pipeline.reduce_noise import BLOCKED_CHANNELS, passes_news_filter as _passes_news_filter


def _passes_noise_filter(item: dict) -> bool:
    return _passes_news_filter(item.get("title", ""), item.get("channel", ""))

# Impact / gate thresholds — imported from config.py (single source of truth).
# Applied to the production 15-minute model_score. Confidence is 0–100 here.
SCORE_HOT  = SCORE_THRESHOLD_HOT       # 0.80
SCORE_MED  = SCORE_THRESHOLD_MEDIUM    # 0.55
SCORE_SHOW = SCORE_THRESHOLD_SHOW      # 0.30
SCORE_HIGH = SCORE_HOT                 # alias used in hot_news / explain endpoint
CONF_MIN   = CONF_SHOW * 100           # config is 0–1; server compares against 0–100 confidence


def _live_score(item: dict) -> float:
    """Absolute production 15-minute score used by every live gate."""
    return abs(float(item.get("model_score", 0) or 0))


def _recompute_impact(item: dict) -> str:
    """Impact badge recomputed live from scores — single source of truth is config.impact_tier().
    Vocabulary: Hot | Medium | Show | Low"""
    return _config_impact_tier(item.get("model_score", 0) or 0)


def _passes_display(item: dict) -> bool:
    """Display gate: confidence >= CONF_MIN. Score gate uses SCORE_SHOW for feed."""
    return float(item.get("confidence", 0) or 0) >= CONF_MIN and _live_score(item) >= SCORE_SHOW


def _gate_feed(items: list) -> list:
    """Apply the display gate and refresh the impact badge for a list of news items."""
    return [{**i, "impact": _recompute_impact(i)} for i in items if _passes_display(i)]

# ── News Importance — imported from config ──
try:
    from config import news_importance
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import news_importance

# ── Load cache on startup ──────────────────────────────────────────
def _load_cache() -> List[dict]:
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "news" in data:
                items = data["news"]
            else:
                return []
            for i in items:
                norm = _normalize_channel(i.get("channel", ""))
                i["channel"] = norm if norm is not None else "__blocked__"
            items = [i for i in items if i.get("channel") not in BLOCKED_CHANNELS and i.get("channel") != "__blocked__"]
            items = [i for i in items if _passes_noise_filter(i)]
            # Add published_ts if missing
            for item in items:
                if "published_ts" not in item:
                    if item.get("id"):
                        item["published_ts"] = int(item["id"]) // 1000
                    elif item.get("published"):
                        try:
                            from datetime import datetime
                            item["published_ts"] = int(datetime.fromisoformat(
                                item["published"].replace("Z", "+00:00")
                            ).timestamp())
                        except Exception:
                            pass
            cutoff = int(time.time()) - HISTORY_WINDOW_MONTHS * 30 * 24 * 3600
            items = [i for i in items if (i.get("published_ts") or 0) >= cutoff]
            return items
        except Exception as _e:
            _log.error("Failed to load %s: %s — dashboard will be empty until cache refreshes", CACHE_FILE, _e)
    return []

all_news: List[dict] = _load_cache()

# Hot is determined only by the deployed 15-minute score.
hot_news: List[dict] = [
    item for item in all_news
    if _live_score(item) >= SCORE_HOT
]

_log.info("Loaded %d news items from cache  (%d hot)", len(all_news), len(hot_news))


# ── Load historical CSV (last 6 months) ───────────────────────────
def _load_csv_as_news(months: int | None = None) -> List[dict]:
    """Convert news_cleaned_filtered.csv rows into the same format as cache items."""
    if not HIST_CSV.exists():
        return []

    from datetime import datetime, timezone
    items = []
    with open(HIST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ch = _normalize_channel(row.get("channel", ""))
                if ch is None:
                    continue   # blocked channel
                row["channel"] = ch
                if row.get("channel") in BLOCKED_CHANNELS:
                    continue
                if not _passes_noise_filter(row):
                    continue
                published = row.get("published", "")
                if not published:
                    continue
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                pub_ts = int(pub_dt.timestamp())

                btc_price = float(row["btc_price_at_news"])
                btc_15m   = float(row["btc_price_15m"])
                btc_1h    = float(row["btc_price_1h"])
                btc_c15m  = (btc_15m - btc_price) / btc_price * 100
                btc_c1h   = (btc_1h  - btc_price) / btc_price * 100

                confidence = float(row.get("confidence") or 0.5)
                if confidence <= 1.0:   # CSV stores 0-1; dashboard expects 0-100
                    confidence *= 100
                sentiment  = row.get("sentiment", "neutral")
                sig_type = (
                    "BUY"  if sentiment == "positive" else
                    "SELL" if sentiment == "negative" else
                    "NEUTRAL"
                )

                # realized_impact = |btc_change| normalized to 0–1.
                # This is NOT a model prediction — it is the actual price outcome
                # observed after the fact. Never expose it as model_score.
                r15 = round(min(1.0, abs(btc_c15m) / 3.0), 4)
                r1h = round(min(1.0, abs(btc_c1h)  / 3.0), 4)

                items.append({
                    "id":               _news_id(row.get("channel", "unknown"), row.get("title", ""), pub_ts),
                    "title":            row.get("title", ""),
                    "link":             row.get("link", ""),
                    "channel":          row.get("channel", "unknown"),
                    "published":        published,
                    "published_ts":     pub_ts,
                    "sentiment":        sentiment,
                    "sentiment_score":  float(row.get("sentiment_score") or 0),
                    "confidence":       confidence,
                    "weight":           float(row.get("weight") or 0),
                    "prob_positive":    float(row.get("prob_positive") or 0),
                    "prob_negative":    float(row.get("prob_negative") or 0),
                    "prob_neutral":     float(row.get("prob_neutral") or 0),
                    "type":             sig_type,
                    "btc_change_15m":   round(btc_c15m, 4),
                    "btc_change_1h":    round(btc_c1h,  4),
                    "model_score":      None,   # no model prediction for historical rows
                    "model_score_1h":   None,
                    "realized_impact":  r15,    # actual 15m outcome, not a prediction
                    "realized_impact_1h": r1h,
                    "is_realized":      True,
                    "score_normalized": True,
                    "impact":           "high" if abs(btc_c15m) >= 0.5 else "medium" if abs(btc_c15m) >= 0.3 else "low",
                    "news_type":        row.get("news_type", ""),
                    "source":           "historical",
                })
            except (ValueError, TypeError, KeyError):
                continue

    if not items:
        return []

    if months is not None:
        max_ts = max(item["published_ts"] for item in items)
        cutoff = max_ts - months * 30 * 24 * 3600
        items = [item for item in items if item["published_ts"] >= cutoff]

    return sorted(items, key=lambda x: x["published_ts"])


# Parse CSV once — derive both views without a second parse.
_all_hist: List[dict] = _load_csv_as_news(months=None)
if _all_hist:
    _max_ts  = max(i["published_ts"] for i in _all_hist)
    _cutoff6 = _max_ts - 6 * 30 * 24 * 3600
    historical_news = [i for i in _all_hist if i["published_ts"] >= _cutoff6]
else:
    historical_news = []
_hist_channels = {item["channel"] for item in historical_news}
_log.info("Loaded %d historical items (6mo), %d total", len(historical_news), len(_all_hist))

# Slim index for /news/dates and /news/by-date — derived from the already-parsed full set.
historical_dates_index: List[dict] = [
    {"published_ts": i["published_ts"], "title": i["title"],
     "channel": i["channel"], "link": i.get("link", ""),
     "model_score": i["model_score"], "sentiment": i.get("sentiment", ""),
     "confidence": i.get("confidence", 50), "impact": i.get("impact", "low"),
     "score_normalized": True}
    for i in _all_hist
]
_log.info("Dates index: %d items (full range)", len(historical_dates_index))
del _all_hist  # free the full list; both views above hold what's needed


def _compute_full_stats() -> dict:
    """Compute aggregated analyze stats from the full scored CSV (all dates)."""
    if not HIST_CSV.exists():
        return {}

    import datetime as _dt
    ch_map: dict = {}
    score_buckets = [
        {"label": "0% – 30%",   "min": 0.00, "max": 0.30, "count": 0},
        {"label": "30% – 50%",  "min": 0.30, "max": 0.50, "count": 0},
        {"label": "50% – 60%",  "min": 0.50, "max": 0.60, "count": 0},
        {"label": "60% – 70%",  "min": 0.60, "max": 0.70, "count": 0},
        {"label": "70% – 90%",  "min": 0.70, "max": 0.90, "count": 0},
        {"label": "90% – 100%", "min": 0.90, "max": 1.01, "count": 0},
    ]
    sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    total = 0
    score_sum = conf_sum = weight_sum = 0.0
    ts_min = ts_max = None

    with open(HIST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ch = _normalize_channel(row.get("channel", "") or "unknown")
                if ch is None:
                    continue
                if ch in BLOCKED_CHANNELS:
                    continue

                pub = row.get("published", "")
                if not pub:
                    continue
                pub_dt = _dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                pub_ts = int(pub_dt.timestamp())

                btc_price = float(row["btc_price_at_news"])
                btc_15m   = float(row["btc_price_15m"])
                btc_c15m  = (btc_15m - btc_price) / btc_price * 100
                impact_score = min(1.0, 0.52 + abs(btc_c15m) / 6.0)
                conf     = float(row.get("confidence") or 0.5)
                if conf <= 1.0: conf *= 100
                weight   = float(row.get("weight") or 5)
                sent     = row.get("sentiment", "neutral")

                # Score buckets
                for b in score_buckets:
                    if b["min"] <= impact_score < b["max"]:
                        b["count"] += 1
                        break

                # Sentiment
                if sent == "positive":   sentiment_counts["bullish"] += 1
                elif sent == "negative": sentiment_counts["bearish"] += 1
                else:                    sentiment_counts["neutral"] += 1

                # Channel
                if ch not in ch_map:
                    ch_map[ch] = {"count": 0, "score_sum": 0.0, "conf_sum": 0.0,
                                  "btc15_sum": 0.0, "btc15_n": 0, "buy": 0, "sell": 0}
                c = ch_map[ch]
                c["count"] += 1
                c["score_sum"] += impact_score
                c["conf_sum"]  += conf
                c["btc15_sum"] += abs(btc_c15m)
                c["btc15_n"]   += 1
                if sent == "positive":   c["buy"]  += 1
                elif sent == "negative": c["sell"] += 1

                score_sum  += impact_score
                conf_sum   += conf
                weight_sum += weight
                total += 1
                if ts_min is None or pub_ts < ts_min: ts_min = pub_ts
                if ts_max is None or pub_ts > ts_max: ts_max = pub_ts

            except (ValueError, TypeError, KeyError):
                continue

    n = total or 1
    channels = []
    for name, c in ch_map.items():
        cnt = c["count"] or 1
        channels.append({
            "name":     name,
            "count":    c["count"],
            "avgScore": round(c["score_sum"] / cnt, 4),
            "avgConf":  round(c["conf_sum"]  / cnt, 2),
            "avgBtc15": round(c["btc15_sum"] / max(c["btc15_n"], 1), 4),
            "btcCount": c["btc15_n"],
            "buyRate":  round(c["buy"]  / cnt, 4),
            "sellRate": round(c["sell"] / cnt, 4),
        })
    channels.sort(key=lambda x: -x["count"])

    def _fmt(ts):
        if ts is None: return None
        import datetime as _dt2
        return _dt2.datetime.fromtimestamp(float(ts), tz=_dt2.timezone.utc).strftime("%Y-%m-%d")

    return {
        "total":            total,
        "date_from":        _fmt(ts_min),
        "date_to":          _fmt(ts_max),
        "avg_score":        round(score_sum / n, 4),
        "avg_confidence":   round(conf_sum  / n, 2),
        "avg_weight":       round(weight_sum / n, 2),
        "score_buckets":    score_buckets,
        "sentiment":        sentiment_counts,
        "channels":         channels,
    }


_full_analyze_stats: dict = _compute_full_stats()
_log.info(
    "Full analyze stats computed: %d items (%s → %s)",
    _full_analyze_stats.get("total", 0),
    _full_analyze_stats.get("date_from"),
    _full_analyze_stats.get("date_to"),
)


import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_):
    global _last_cache_mtime
    _last_cache_mtime = CACHE_FILE.stat().st_mtime if CACHE_FILE.exists() else 0.0
    tasks = [
        asyncio.create_task(_cache_refresh_loop(), name="cache-refresh"),
        asyncio.create_task(_prune_rate_limits(), name="rate-limit-pruner"),
    ]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="Crypto News API", version="2.0", lifespan=lifespan)

# CORS: restrict to configured frontend origins. Defaults to localhost dev
# ports. Set ALLOWED_ORIGINS (comma-separated) in .env for deployment.
# Using "*" here is unsafe because the API exposes a write endpoint.
_allowed_origins = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Connected WebSocket client sets
_ws_all_clients: set = set()
_ws_hot_clients: set = set()


async def _broadcast(clients: set, item: dict):
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(item)
        except Exception:
            dead.add(ws)
    clients -= dead


# ── Background cache refresh ──────────────────────────────────────
_last_cache_mtime: float = 0.0
_cache_write_lock = asyncio.Lock()
_cache_write_version = 0

async def _prune_rate_limits():
    """Periodically remove stale IP entries to bound _rate_limits memory."""
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - 60
        stale = [ip for ip, ts_list in list(_rate_limits.items())
                 if not any(t > cutoff for t in ts_list)]
        for ip in stale:
            _rate_limits.pop(ip, None)


async def _cache_refresh_loop():
    global all_news, hot_news, _last_cache_mtime, _idf_cache, _combined_cache
    while True:
        await asyncio.sleep(60)
        try:
            if not CACHE_FILE.exists():
                continue
            mtime = CACHE_FILE.stat().st_mtime
            if mtime <= _last_cache_mtime:
                continue
            new_items = _load_cache()
            existing_ids = {i.get("id") for i in all_news}
            fresh = [i for i in new_items if i.get("id") not in existing_ids]
            # Always reload when file changed — rescoring produces same IDs with updated scores
            all_news = new_items
            hot_news = [i for i in all_news if _live_score(i) >= SCORE_HOT]
            _idf_cache = None
            _combined_cache = None
            _last_cache_mtime = mtime
            _log.info("Cache refreshed: %d items (%d new IDs)", len(all_news), len(fresh))
            for item in fresh:
                await _broadcast(_ws_all_clients, item)
                if _live_score(item) >= SCORE_HOT:
                    await _broadcast(_ws_hot_clients, item)
        except Exception as exc:
            _log.error("Cache refresh error: %s", exc)




# ── WebSocket — initial history via REST, live pushes via WS ──────
@app.websocket("/ws/all")
async def ws_all(ws: WebSocket):
    await ws.accept()
    _ws_all_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_all_clients.discard(ws)


@app.websocket("/ws/hot")
async def ws_hot(ws: WebSocket):
    await ws.accept()
    _ws_hot_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_hot_clients.discard(ws)


# ── REST — news ────────────────────────────────────────────────────
_INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")


@app.post("/news")
async def ingest_news(item: IngestNewsItem, x_api_key: str = Header(default="")):
    """Receive a scored news item from main.py and persist it to the cache.

    This is a WRITE endpoint (it mutates the cache and broadcasts to every
    connected dashboard), so it requires a shared secret. Set INGEST_API_KEY
    in .env and send it as the X-API-Key header. If INGEST_API_KEY is unset,
    the endpoint is disabled (fails closed) to avoid an open write surface.
    """
    if not _INGEST_API_KEY:
        raise HTTPException(status_code=503, detail="Ingest disabled: INGEST_API_KEY not configured")
    if not secrets.compare_digest(x_api_key, _INGEST_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    global all_news, hot_news, _idf_cache, _combined_cache, _cache_write_version

    # Convert Pydantic model to plain dict so we can freely mutate it
    data: dict = item.model_dump()

    # Normalise channel name
    norm = _normalize_channel(data.get("channel", ""))
    if norm is None:
        return {"status": "blocked"}
    data["channel"] = norm

    if data.get("channel") in BLOCKED_CHANNELS:
        return {"status": "blocked"}

    if not _passes_noise_filter(data):
        return {"status": "filtered"}

    # Ensure published_ts exists (id is now a hex string, not a numeric timestamp)
    if not data.get("published_ts"):
        data["published_ts"] = int(time.time())
    if not data.get("id"):
        data["id"] = _news_id(
            data.get("channel", ""), data["title"], int(data["published_ts"])
        )

    item = data   # type: ignore[assignment]  — work with the dict from here on

    # Deduplicate by id
    existing_ids = {i.get("id") for i in all_news}
    if item.get("id") in existing_ids:
        return {"status": "duplicate"}

    # Prepend to in-memory list and trim to MAX_CACHE_ITEMS
    MAX = 10_000
    all_news = ([item] + all_news)[:MAX]
    if _live_score(item) >= SCORE_HOT:
        hot_news = ([item] + hot_news)[:MAX]
    _idf_cache = None
    _combined_cache = None

    # Persist to cache file atomically — offloaded so json.dumps + disk I/O
    # don't block the event loop (O(N) work per ingest call).
    _snapshot = list(all_news)
    _cache_write_version += 1
    write_version = _cache_write_version
    async def _write_cache():
        global _last_cache_mtime
        async with _cache_write_lock:
            # A newer request already captured a more complete snapshot.
            if write_version != _cache_write_version:
                return
            try:
                def _do_write():
                    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    tmp = CACHE_FILE.with_suffix(".tmp")
                    tmp.write_text(json.dumps(_snapshot, default=str), encoding="utf-8")
                    tmp.replace(CACHE_FILE)
                    return CACHE_FILE.stat().st_mtime
                mtime = await asyncio.to_thread(_do_write)
                _last_cache_mtime = mtime
            except Exception as e:
                _log.error("Cache write failed: %s", e)
                raise HTTPException(status_code=500, detail="Failed to persist news") from e
    await _write_cache()

    # Broadcast to WebSocket clients
    await _broadcast(_ws_all_clients, item)
    if _live_score(item) >= SCORE_HOT:
        await _broadcast(_ws_hot_clients, item)

    return {"status": "ok"}


@app.get("/news/since")
def get_since(ts: int = 0):
    """Return items with published_ts > ts — for incremental frontend polling."""
    items = _gate_feed([
        i for i in _get_combined()
        if float(i.get("published_ts") or i.get("received_at", 0)) > ts
    ])
    return sorted(items, key=lambda x: x.get("published_ts") or 0)


@app.get("/fear-greed")
def get_fear_greed():
    """Return the latest Fear & Greed index value from cache or live API."""
    cache_path = ROOT / "fear_greed_cache.json"
    try:
        if cache_path.exists():
            items = json.loads(cache_path.read_text())
            if items:
                latest = sorted(items, key=lambda x: int(x.get("timestamp", 0)), reverse=True)[0]
                value  = int(latest["value"])
                label  = latest.get("value_classification", "")
                ts     = int(latest.get("timestamp", 0))
                return {"value": value, "label": label, "timestamp": ts, "source": "cache"}
    except Exception:
        pass
    # Live fallback
    try:
        import urllib.request
        with urllib.request.urlopen("https://api.alternative.me/fng/?limit=1&format=json", timeout=5) as r:
            data   = json.loads(r.read())["data"][0]
            value  = int(data["value"])
            label  = data["value_classification"]
            return {"value": value, "label": label, "timestamp": int(data["timestamp"]), "source": "live"}
    except Exception as e:
        return {"value": 50, "label": "Neutral", "timestamp": 0, "source": "error", "error": str(e)}


@app.get("/config")
def get_config():
    """Expose display thresholds so clients (dashboard, tests) can stay in sync.
    All values derived from config.py — this is the contract between server and frontend."""
    return {
        "score_hot":          SCORE_HOT,
        "score_medium":       SCORE_MED,
        "score_show":         SCORE_SHOW,
        "conf_min":           CONF_MIN,
        "reliable_channels":  ["the_block_crypto", "coindesk", "cointelegraph", "WatcherGuru", "google_news"],
        "impact_labels":      ["Hot", "Medium", "Show", "Low"],
    }


_artifact_health_cache: dict | None = None


def _artifact_health() -> dict:
    global _artifact_health_cache
    if _artifact_health_cache is not None:
        return _artifact_health_cache

    model_path = ROOT / "xgb_impact_clf_15m_bert_rag.json"
    scaler_path = ROOT / "xgb_feature_scaler_bert_rag.pkl"
    result = {
        "ready": False,
        "expected_features": 1578,
        "model_features": None,
        "scaler_features": None,
        "error": None,
    }
    try:
        model_json = json.loads(model_path.read_text(encoding="utf-8"))
        result["model_features"] = int(
            model_json["learner"]["learner_model_param"]["num_feature"]
        )
        import pickle
        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)
        result["scaler_features"] = int(scaler.n_features_in_)
        result["ready"] = (
            result["model_features"]
            == result["scaler_features"]
            == result["expected_features"]
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    _artifact_health_cache = result
    return result


@app.get("/health")
def health(response: Response):
    artifacts = _artifact_health()
    if not artifacts["ready"]:
        response.status_code = 503
    pending_path = ROOT / "storage" / "rag_pending_outcomes.json"
    dead_path = ROOT / "storage" / "rag_dead_letter.json"
    bot_health_path = ROOT / "storage" / "bot_health.json"
    try:
        pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else []
        if not isinstance(pending, list):
            pending = []
        pending_count = len(pending)
        oldest_pending_age = (
            max(0, int(time.time()) - min(int(i.get("published_ts", time.time())) for i in pending))
            if pending else 0
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pending_count = None
        oldest_pending_age = None
    try:
        dead = json.loads(dead_path.read_text(encoding="utf-8")) if dead_path.exists() else []
        dead_count = len(dead) if isinstance(dead, list) else 0
    except (OSError, json.JSONDecodeError):
        dead_count = None
    try:
        bot_health = json.loads(bot_health_path.read_text(encoding="utf-8")) \
            if bot_health_path.exists() else None
        bot_heartbeat_age = (
            max(0, int(time.time()) - int(bot_health.get("timestamp", 0)))
            if isinstance(bot_health, dict) else None
        )
        bot_status = "ok" if bot_heartbeat_age is not None and bot_heartbeat_age <= 90 else "stale"
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        bot_health = None
        bot_heartbeat_age = None
        bot_status = "unknown"

    return {
        "status":              "ok" if artifacts["ready"] else "degraded",
        "model":               artifacts,
        "rag_pending_outcomes": pending_count,
        "rag_oldest_pending_seconds": oldest_pending_age,
        "rag_dead_letter_count": dead_count,
        "bot_status":           bot_status,
        "bot_heartbeat_age_seconds": bot_heartbeat_age,
        "bot":                  bot_health,
        "source":              "news_cache.json + news_cleaned_filtered.csv",
        "live_news_count":     len(all_news),
        "historical_count":    len(historical_news),
        "total_news_count":    len(all_news) + len(historical_news),
        "hot_news_count":      len(hot_news),
        "historical_channels": sorted(_hist_channels),
    }


@app.get("/health/qdrant")
async def qdrant_health(response: Response):
    """Explicit Qdrant check, separated from the fast container liveness check."""
    try:
        from pipeline.rag_news import COLLECTION_NAME, get_client

        def _check():
            client = get_client()
            info = client.get_collection(COLLECTION_NAME)
            return int(info.points_count or 0)

        points = await asyncio.wait_for(asyncio.to_thread(_check), timeout=10)
        return {"status": "ok", "collection": COLLECTION_NAME, "points": points}
    except Exception as exc:
        response.status_code = 503
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/news/all")
def get_all(response: Response, limit: int = 500, offset: int = 0):
    """Return paginated news. Default page is 500 items.

    Use ?limit=N&offset=M for cursor-style pagination. Clients should
    request only what they render — avoid limit > 2000.
    """
    combined = _gate_feed(sorted(
        _get_combined(),
        key=lambda x: x.get("published_ts") or x.get("received_at") or 0,
        reverse=True,
    ))
    limit = max(1, min(limit, 5000))
    page  = combined[offset: offset + limit]
    response.headers["Cache-Control"] = "public, max-age=30"
    response.headers["X-Total-Count"]  = str(len(combined))
    return page


@app.get("/news/hot")
def get_hot():
    return [{**i, "impact": _recompute_impact(i)} for i in hot_news[:50]]


@app.get("/news/dates")
def get_dates():
    from datetime import datetime, timezone
    seen = set()
    for item in all_news + historical_dates_index:  # dates index kept separate (slim dicts)
        ts = item.get("published_ts") or item.get("received_at")
        if ts:
            d = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            seen.add(f"{d.year}-{d.month:02d}-{d.day:02d}")
    return sorted(seen)


@app.get("/news/by-date")
def get_by_date(start: int, end: int):
    return _gate_feed([
        item for item in all_news + historical_dates_index  # dates index kept separate
        if start <= float(item.get("published_ts") or item.get("received_at", 0)) <= end
    ])


# ── REST — training analytics ──────────────────────────────────────
@app.get("/training/stats")
def get_training_stats():
    """Training data statistics from news_cleaned_filtered.csv + all production_results_*.json."""
    csv_path = ROOT / "news_cleaned_filtered.csv"
    stats = {}

    # Load all model result files for comparison
    model_files = {
        "v5":           ROOT / "production_results_v5.json",
        "v6":           ROOT / "production_results_v6.json",
        "v7":           ROOT / "production_results_v7.json",
        "v8":           ROOT / "production_results_v8.json",
        "ann_bert":     ROOT / "ann_bert_results.json",
        "xgboost_groq": ROOT / "xgb_groq_results.json",
        "xgboost_bert": ROOT / "xgb_bert_results.json",
        "xgboost":      ROOT / "xgboost_results.json",
    }
    all_models: dict = {}
    for name, path in model_files.items():
        if path.exists():
            with open(path, encoding="utf-8") as f:
                all_models[name] = json.load(f)

    # Current best model = xgboost_groq → xgboost_bert → v9 → v8 → ...
    best = next((k for k in ["xgboost_groq","xgboost_bert","v9","v8","v7","v6","v5"] if k in all_models), None)
    if best:
        stats["model_performance"] = all_models[best]
    stats["all_models"] = all_models

    if csv_path.exists():
        total = 0
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        news_types: dict = {}
        channels:   dict = {}
        weights:    list = []
        confidences: list = []
        btc_15m:    list = []
        btc_1h:     list = []
        impact_15m = 0
        impact_1h  = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                sent = row.get("sentiment", "neutral")
                sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
                ntype = row.get("news_type", "unknown")
                news_types[ntype] = news_types.get(ntype, 0) + 1
                ch = row.get("channel", "unknown")
                channels[ch] = channels.get(ch, 0) + 1
                try: weights.append(float(row["weight"]))
                except (ValueError, KeyError, TypeError): pass
                try:
                    c = float(row["confidence"])
                    confidences.append(c * 100 if c <= 1.0 else c)
                except (ValueError, KeyError, TypeError): pass
                try:
                    bp   = float(row["btc_price_at_news"])
                    b15  = float(row["btc_price_15m"])
                    b1h  = float(row["btc_price_1h"])
                    c15  = (b15 - bp) / bp * 100
                    c1h  = (b1h - bp) / bp * 100
                    btc_15m.append(round(c15, 4))
                    btc_1h.append(round(c1h, 4))
                    if abs(c15) >= 0.3: impact_15m += 1
                    if abs(c1h) >= 0.5: impact_1h  += 1
                except (ValueError, KeyError, TypeError, ZeroDivisionError): pass

        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0
        def pct(n):   return round(n / total * 100, 1)     if total else 0

        stats["training_data"] = {
            "total_samples":        total,
            "sentiment_counts":     sentiment_counts,
            "sentiment_pcts":       {k: pct(v) for k, v in sentiment_counts.items()},
            "news_types":           dict(sorted(news_types.items(), key=lambda x: -x[1])[:10]),
            "channels":             dict(sorted(channels.items(), key=lambda x: -x[1])),
            "avg_weight":           avg(weights),
            "avg_confidence":       avg(confidences),
            "impact_15m_count":     impact_15m,
            "impact_15m_pct":       pct(impact_15m),
            "impact_1h_count":      impact_1h,
            "impact_1h_pct":        pct(impact_1h),
            "avg_btc_change_15m":   avg(btc_15m),
            "avg_btc_change_1h":    avg(btc_1h),
            "btc_change_histogram": [
                {"label": "< -2%",        "count": sum(1 for x in btc_15m if x < -2)},
                {"label": "-2% to -1%",   "count": sum(1 for x in btc_15m if -2  <= x < -1)},
                {"label": "-1% to -0.5%", "count": sum(1 for x in btc_15m if -1  <= x < -0.5)},
                {"label": "-0.5% to 0%",  "count": sum(1 for x in btc_15m if -0.5 <= x < 0)},
                {"label": "0% to 0.5%",   "count": sum(1 for x in btc_15m if 0   <= x < 0.5)},
                {"label": "0.5% to 1%",   "count": sum(1 for x in btc_15m if 0.5 <= x < 1)},
                {"label": "1% to 2%",     "count": sum(1 for x in btc_15m if 1   <= x < 2)},
                {"label": "> 2%",         "count": sum(1 for x in btc_15m if x >= 2)},
            ],
        }

    return stats


@app.get("/training/category-stats")
def get_category_stats():
    """Per-category accuracy from news_cleaned_filtered.csv (train) + ews_ev.csv (test)."""
    train_csv = ROOT / "news_cleaned_filtered.csv"
    test_csv  = ROOT / "ews_ev.csv"

    train_counts: dict = defaultdict(int)
    if train_csv.exists():
        with open(train_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                train_counts[row.get("news_type", "unknown")] += 1

    test_stats = defaultdict(lambda: {"count": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "scores": []})
    if test_csv.exists():
        with open(test_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nt = row.get("news_type", "unknown")
                s  = test_stats[nt]
                s["count"] += 1
                r = row.get("result_15m", "")
                if   "TP" in r: s["tp"] += 1
                elif "TN" in r: s["tn"] += 1
                elif "FP" in r: s["fp"] += 1
                elif "FN" in r: s["fn"] += 1
                try:   s["scores"].append(float(row["model_score"]))
                except (ValueError, KeyError, TypeError): pass

    result = []
    for nt in set(list(train_counts.keys()) + list(test_stats.keys())):
        v  = test_stats[nt]
        tp, tn, fp, fn = v["tp"], v["tn"], v["fp"], v["fn"]
        total     = v["count"]
        avg_score = sum(v["scores"]) / len(v["scores"]) if v["scores"] else 0
        result.append({
            "news_type":   nt,
            "train_count": train_counts.get(nt, 0),
            "test_count":  total,
            "accuracy":    round((tp + tn) / total, 4) if total else None,
            "precision":   round(tp / (tp + fp), 4)    if (tp + fp) > 0 else None,
            "recall":      round(tp / (tp + fn), 4)    if (tp + fn) > 0 else None,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "avg_score":   round(avg_score, 4),
        })
    return sorted(result, key=lambda x: -x["test_count"])


# ── Report: training CSV + cache summary ──────────────────────────
@app.get("/report/summary")
def get_report_summary():
    """Combined report: training data (news_cleaned_filtered.csv) + cache (news_cache.json)."""
    import datetime

    csv_path     = ROOT / "news_cleaned_filtered.csv"
    # The deployed signal is the 15-minute BERT + RAG model.
    results_path = next(
        (p for p in [
            ROOT / "xgb_bert_rag_results.json",
            ROOT / "xgb_bert_results.json",
            ROOT / "ann_bert_results.json",
            ROOT / "production_results_v8.json",
        ] if p.exists()),
        ROOT / "production_results_v8.json",
    )

    train = {
        "file": "news_cleaned_filtered.csv",
        "total_raw": 0,
        "total_filtered": 0,
        "date_min": None, "date_max": None,
        "impactful_15m_count": 0, "impactful_15m_pct": 0.0,
        "impactful_1h_count":  0, "impactful_1h_pct":  0.0,
        "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
        "channel_counts": {},
        "news_type_counts": {},
        "split": {"train_pct": 70, "val_pct": 15, "test_pct": 15, "seed": 43},
    }
    if csv_path.exists():
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                train["total_raw"] += 1
                try:
                    w = float(row.get("weight", 0))
                    float(row.get("btc_price_at_news", ""))
                    float(row.get("btc_price_15m", ""))
                    float(row.get("btc_price_1h", ""))
                except (ValueError, TypeError):
                    continue
                if w < 5:
                    continue
                rows.append(row)
                train["total_filtered"] += 1
                sent = row.get("sentiment", "neutral")
                train["sentiment_counts"][sent] = train["sentiment_counts"].get(sent, 0) + 1
                ch = row.get("channel", "unknown")
                train["channel_counts"][ch] = train["channel_counts"].get(ch, 0) + 1
                nt = row.get("news_type", "unknown")
                train["news_type_counts"][nt] = train["news_type_counts"].get(nt, 0) + 1
                try:
                    pub = row.get("published", "")
                    if pub:
                        if train["date_min"] is None or pub < train["date_min"]: train["date_min"] = pub[:10]
                        if train["date_max"] is None or pub > train["date_max"]: train["date_max"] = pub[:10]
                    c15 = (float(row["btc_price_15m"]) - float(row["btc_price_at_news"])) / float(row["btc_price_at_news"]) * 100
                    c1h = (float(row["btc_price_1h"])  - float(row["btc_price_at_news"])) / float(row["btc_price_at_news"]) * 100
                    if abs(c15) >= 0.3: train["impactful_15m_count"] += 1
                    if abs(c1h) >= 0.5: train["impactful_1h_count"]  += 1
                except (ValueError, KeyError, TypeError, ZeroDivisionError): pass
        n = train["total_filtered"] or 1
        train["impactful_15m_pct"] = round(train["impactful_15m_count"] / n * 100, 1)
        train["impactful_1h_pct"]  = round(train["impactful_1h_count"]  / n * 100, 1)
        train["channel_counts"] = dict(sorted(train["channel_counts"].items(), key=lambda x: -x[1])[:10])
        train["news_type_counts"] = dict(sorted(train["news_type_counts"].items(), key=lambda x: -x[1]))
        n = train["total_filtered"]
        train["split"]["train_n"] = int(n * 0.70)
        train["split"]["val_n"]   = int(n * 0.15)
        train["split"]["test_n"]  = n - int(n * 0.70) - int(n * 0.15)

    cache_channels: dict = {}
    cache_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    cache_signals    = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
    score_high = score_med = score_low = 0
    with_btc = pred15_pos = pred1h_pos = 0
    ts_min = ts_max = None

    for item in all_news:
        sc = _live_score(item)
        if sc >= SCORE_HOT:   score_high += 1
        elif sc >= SCORE_MED: score_med  += 1
        else:                 score_low  += 1

        sent = item.get("sentiment", "neutral")
        cache_sentiments[sent] = cache_sentiments.get(sent, 0) + 1
        sig = item.get("type", "NEUTRAL")
        cache_signals[sig] = cache_signals.get(sig, 0) + 1
        ch = item.get("channel", "unknown")
        cache_channels[ch] = cache_channels.get(ch, 0) + 1

        if abs(float(item.get("btc_change_15m", 0))) > 0: with_btc += 1
        if item.get("pred_15m") == 1:  pred15_pos += 1
        if item.get("pred_1h")  == 1:  pred1h_pos += 1

        ts = item.get("published_ts") or item.get("received_at")
        if ts:
            if ts_min is None or ts < ts_min: ts_min = ts
            if ts_max is None or ts > ts_max: ts_max = ts

    def _fmt_date(ts):
        if ts is None: return None
        return datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    n_cache = len(all_news) or 1
    cache = {
        "file": "news_cache.json",
        "total": len(all_news),
        "date_min": _fmt_date(ts_min),
        "date_max": _fmt_date(ts_max),
        "score_high":  score_high,
        "score_medium": score_med,
        "score_low":   score_low,
        "score_high_pct":   round(score_high / n_cache * 100, 1),
        "score_medium_pct": round(score_med  / n_cache * 100, 1),
        "sentiment_counts": cache_sentiments,
        "signal_counts":    cache_signals,
        "channel_counts":   dict(sorted(cache_channels.items(), key=lambda x: -x[1])[:10]),
        "with_btc_data":    with_btc,
        "with_btc_pct":     round(with_btc / n_cache * 100, 1),
        "pred_15m_positive":  pred15_pos,
        "pred_15m_positive_pct": round(pred15_pos / n_cache * 100, 1),
        "pred_1h_positive":   pred1h_pos,
        "pred_1h_positive_pct": round(pred1h_pos / n_cache * 100, 1),
    }

    model_results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            model_results = json.load(f)

    # Derive metrics from the results JSON — never hardcode them here.
    _r15 = model_results.get("15_minute", {})
    _r1h = model_results.get("1_hour", {})
    _src = results_path.name if results_path.exists() else "not found"
    _is_bert = "bert" in _src
    architecture = {
        "name":       ("15-minute XGBoost (DualBERT + 3-BERT Ensemble Sentiment + RAG)"
                       if _is_bert else
                       "XGBoost (DualBERT + Groq/Llama-3.3-70B Sentiment)"),
        "type":       "Gradient Boosted Trees — CPU (device=cpu, tree_method=approx)",
        "file":       "xgboost_train_bert.py" if _is_bert else "xgboost_train_groq.py",
        "results_source": _src,
        "feature_dim": model_results.get("feature_dim",
                        1578 if _is_bert else 1562),
        "feature_layout": [
            {"name": "CryptoBERT embedding",    "dims": 768},
            {"name": "FinBERT embedding",        "dims": 768},
            {"name": "3-BERT ensemble sentiment (9 probs + net_agreement + 3 scalar)" if _is_bert
                     else "Groq/Llama-3.3-70B sentiment (3 one-hot + 3 scalar)", "dims": 13 if _is_bert else 6},
            {"name": "News-type probs",          "dims": 11},
            {"name": "Macro timing (5) + price context (3)", "dims": 8},
            {"name": "RAG features (Qdrant macro-reweighted)", "dims": 10},
        ],
        "embeddings": ["ElKulako/cryptobert (768)", "ProsusAI/finbert (768)"],
        "price_context": ["btc_vol (rolling std 20)", "btc_mom (rolling mean 5)", "fear_greed (Alternative.me)"],
        "params": {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "early_stopping_rounds": 20,
        },
        "threshold_15m": model_results.get("threshold_15m"),
        "threshold_1h":  model_results.get("threshold_1h"),
        "min_precision": 0.20,
        "monthly_seed":  43,
        # Metrics read from results JSON — regenerate by running training/xgboost_train_bert.py
        "roc_auc_15m": _r15.get("ROC_AUC"),
        "roc_auc_1h":  _r1h.get("ROC_AUC"),
        "f1_15m":      _r15.get("F1"),
        "f1_1h":       _r1h.get("F1"),
        "precision_15m": _r15.get("Precision"),
        "recall_15m":  _r15.get("Recall"),
        "precision_1h":  _r1h.get("Precision"),
        "recall_1h":   _r1h.get("Recall"),
    }

    eval_csv = ROOT / "ews_ev.csv"
    cat_stats: dict = defaultdict(lambda: {"train_count": 0, "test_count": 0,
                                            "tp15": 0, "tn15": 0, "fp15": 0, "fn15": 0,
                                            "tp1h": 0, "tn1h": 0, "fp1h": 0, "fn1h": 0})
    for nt, cnt in train["news_type_counts"].items():
        cat_stats[nt]["train_count"] = cnt
    if eval_csv.exists():
        with open(eval_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nt = row.get("news_type", "unknown")
                cat_stats[nt]["test_count"] += 1
                r15 = row.get("result_15m", "")
                r1h = row.get("result_1h",  "")
                if "TP" in r15:  cat_stats[nt]["tp15"] += 1
                elif "TN" in r15: cat_stats[nt]["tn15"] += 1
                elif "FP" in r15: cat_stats[nt]["fp15"] += 1
                elif "FN" in r15: cat_stats[nt]["fn15"] += 1
                if "TP" in r1h:  cat_stats[nt]["tp1h"] += 1
                elif "TN" in r1h: cat_stats[nt]["tn1h"] += 1
                elif "FP" in r1h: cat_stats[nt]["fp1h"] += 1
                elif "FN" in r1h: cat_stats[nt]["fn1h"] += 1

    def _cat_metrics(tp, tn, fp, fn):
        total = tp + tn + fp + fn or 1
        acc   = round((tp + tn) / total, 4)
        prec  = round(tp / (tp + fp), 4) if (tp + fp) else None
        rec   = round(tp / (tp + fn), 4) if (tp + fn) else None
        f1    = round(2 * prec * rec / (prec + rec), 4) if (prec and rec and prec + rec) else None
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total}

    category_results = []
    for nt, v in cat_stats.items():
        m15 = _cat_metrics(v["tp15"], v["tn15"], v["fp15"], v["fn15"])
        m1h = _cat_metrics(v["tp1h"], v["tn1h"], v["fp1h"], v["fn1h"])
        category_results.append({
            "news_type":   nt,
            "train_count": v["train_count"],
            "test_count":  v["test_count"],
            "15m":         m15,
            "1h":          m1h,
        })
    category_results.sort(key=lambda x: -x["test_count"])

    return {"training": train, "cache": cache, "results": model_results,
            "architecture": architecture, "category_results": category_results}


# ── RAG-style similarity search (TF-IDF cosine, no external model) ──
_STOP = {
    "the","and","for","are","was","not","but","with","its","has","had",
    "have","will","from","that","this","into","than","more","over","about",
    "after","before","says","said","new","now","get","can","all","one","top",
    "just","also","amid","per","via","out","off",
}
_SIM_THRESHOLD = 0.12  # minimum cosine similarity for /news/similar results

def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP]

def _tfidf_vec(tokens: List[str], idf: dict) -> dict:
    tf = Counter(tokens)
    return {w: (1 + math.log(c)) * idf.get(w, 1.0) for w, c in tf.items()}

def _cosine(a: dict, b: dict) -> float:
    dot = sum(a.get(w, 0) * v for w, v in b.items())
    na  = math.sqrt(sum(v * v for v in a.values())) or 1
    nb  = math.sqrt(sum(v * v for v in b.values())) or 1
    return dot / (na * nb)

# Build IDF once at startup over all available news titles
def _build_idf(items: List[dict]) -> dict:
    df: Counter = Counter()
    for n in items:
        df.update(set(_tokens(n.get("title", ""))))
    N = max(len(items), 1)
    return {w: math.log(N / (c + 1)) for w, c in df.items()}

# IDF is built lazily on first call so startup isn't delayed
_idf_cache: dict | None = None
_combined_cache: list | None = None  # cached all_news + historical_news


def _get_combined() -> list:
    global _combined_cache
    if _combined_cache is None:
        _combined_cache = all_news + historical_news
    return _combined_cache

def _get_idf() -> dict:
    global _idf_cache
    if _idf_cache is None:
        _idf_cache = _build_idf(_get_combined())
    return _idf_cache


@app.post("/news/similar")
async def find_similar(request: Request, item: SimilarRequest):
    _check_rate_limit(_client_ip(request), rpm=30)
    title = item.title.strip()
    if not title:
        return {"similar": []}

    # Items that came from live ingestion already carry pre-computed similar list
    if item.similar:
        return {"similar": item.similar}

    idf      = _get_idf()
    q_tokens = _tokens(title)
    if not q_tokens:
        return {"similar": []}
    q_vec = _tfidf_vec(q_tokens, idf)

    pool    = _get_combined()
    results = []
    for n in pool:
        t = (n.get("title") or "").strip()
        if not t or t == title:
            continue
        sim = _cosine(q_vec, _tfidf_vec(_tokens(t), idf))
        if sim >= _SIM_THRESHOLD:
            results.append({
                "title":  t,
                "sim":    round(sim, 3),
                "change": float(n.get("btc_change_15m") or 0),
            })

    results.sort(key=lambda x: -x["sim"])
    return {"similar": results[:5]}


# ── AI explanation (Groq) ──────────────────────────────────────────
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.post("/news/explain")
async def explain_news(request: Request, item: ExplainRequest):
    # Rate limit: 10 req/min for this expensive Groq-backed endpoint
    _check_rate_limit(_client_ip(request), rpm=10)
    title     = item.title
    sentiment = item.sentiment
    confidence= item.confidence
    score     = abs(item.model_score)
    score_1h  = abs(item.model_score_1h)
    btc_15m   = item.btc_change_15m
    btc_1h    = item.btc_change_1h
    channel   = item.channel
    similar   = item.similar
    max_score = score
    impact = _config_impact_tier(score)

    sim_block = "No similar historical news found."
    if similar:
        lines = [
            f'  - "{s.get("title","")[:80]}" → BTC {s.get("change",0):+.2f}% '
            f'(similarity {s.get("sim",0)*100:.0f}%)'
            for s in similar[:3]
        ]
        sim_block = "Similar historical news:\n" + "\n".join(lines)

    btc_block = (
        f"Actual BTC reaction: {btc_15m:+.2f}% in 15m, {btc_1h:+.2f}% in 1h"
        if btc_15m != 0 or btc_1h != 0
        else "BTC reaction: data available in cache"
    )

    prompt = f"""You are a crypto trading signal analyst explaining a model's prediction to a trader.

News headline: "{title}"
Source channel: {channel}

Model output:
  - Sentiment: {sentiment} (confidence {confidence}%)
  - Impact score (15m): {score*100:.0f}%  → tier: {impact}
  - Impact score (1h):  {score_1h*100:.0f}%
  - Signal: {"BUY" if sentiment == "positive" else "SELL" if sentiment == "negative" else "NEUTRAL"}

{sim_block}

{btc_block}

Explain step by step IN 4 SHORT BULLET POINTS why the model gave these scores.
Be specific to this headline. Use plain language a trader can act on.
Format: each bullet starts with an emoji, max 2 sentences per bullet.
Do NOT repeat the numbers — explain the REASONING behind them."""

    payload = {
        "model":       GROQ_CLASSIFICATION_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  300,
        "temperature": 0.4,
    }
    for key in GROQ_API_KEYS:
        if not key:
            continue
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_URL, json=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data  = await resp.json()
                        text  = data["choices"][0]["message"]["content"].strip()
                        lines = [l.strip() for l in text.split("\n") if l.strip()]
                        steps = [l for l in lines if l[0] in "•-→✅⚠📈📉💡🔴🟡🟢⚡🧠📊🏦🔥❗💰🌍🎯🔵"] or lines
                        return {"explanation": text, "steps": steps[:5]}
        except Exception:
            continue

    return {"error": "Groq unavailable", "explanation": "", "steps": []}


# ── Binance proxy (chart data) ─────────────────────────────────────
# Strict allowlists prevent this proxy from being used as an open relay /
# SSRF vector: symbol and interval are interpolated into outbound URLs, so
# only known-good values are permitted.
_ALLOWED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}
_ALLOWED_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


@app.get("/proxy/klines")
async def proxy_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                       limit: int = 200, startTime: int = None):
    symbol = symbol.upper()
    if symbol not in _ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"symbol not allowed: {symbol}")
    if interval not in _ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval not allowed: {interval}")
    limit = max(1, min(int(limit), 1000))
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval={interval}&limit={limit}")
    if startTime:
        url += f"&startTime={int(startTime)}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                return await resp.json()
    except Exception as e:
        return {"error": str(e)}


# Fan-out: one upstream Binance connection per stream key shared by all subscribers.
# Without this, N browser tabs = N upstream sockets; Binance throttles at ~20.
_stream_subs:     dict[str, set[WebSocket]] = {}  # key -> subscriber set
_stream_tasks:    dict[str, asyncio.Task]   = {}  # key -> upstream task


async def _binance_upstream(key: str, url: str):
    """Maintain one upstream connection and broadcast to all subscribers."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as bws:
                async for msg in bws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        subs = list(_stream_subs.get(key, []))
                        if not subs:
                            return  # no subscribers left — stop upstream
                        dead = []
                        for sub in subs:
                            try:
                                await sub.send_text(msg.data)
                            except Exception:
                                dead.append(sub)
                        for sub in dead:
                            _stream_subs[key].discard(sub)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
    except Exception as e:
        _log.error("Binance upstream error [%s]: %s", key, e)
    finally:
        _stream_tasks.pop(key, None)


@app.websocket("/proxy/stream/{symbol}/{interval}")
async def proxy_stream(ws: WebSocket, symbol: str, interval: str):
    if symbol.upper() not in _ALLOWED_SYMBOLS or interval not in _ALLOWED_INTERVALS:
        await ws.close(code=1008)
        return
    await ws.accept()
    key = f"{symbol.lower()}@kline_{interval}"
    _stream_subs.setdefault(key, set()).add(ws)
    if key not in _stream_tasks or _stream_tasks[key].done():
        url = f"wss://stream.binance.com:9443/ws/{key}"
        _stream_tasks[key] = asyncio.create_task(_binance_upstream(key, url))
    try:
        # Hold the socket open until the client disconnects.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _stream_subs.get(key, set()).discard(ws)
        try:
            await ws.close()
        except Exception:
            pass


# ── Custom news analyzer ──────────────────────────────────────────
_analyzer_models = {}   # lazy-loaded on first call
_analyze_semaphore = asyncio.Semaphore(1)

def _get_analyzer_models():
    if _analyzer_models:
        return _analyzer_models
    from training.create_sample_cache import _load_bert_models, _load_xgb
    _log.info("Loading BERT + XGBoost models for custom analyzer (CPU)...")
    bert = _load_bert_models(force_cpu=True)
    clf15, _, scaler, thr15, _ = _load_xgb()
    if scaler.n_features_in_ != clf15.n_features_in_:
        raise RuntimeError(
            f"Analyzer model expects {clf15.n_features_in_} features but "
            f"its scaler expects {scaler.n_features_in_}"
        )
    _analyzer_models.update({"bert": bert, "clf15": clf15, "scaler": scaler, "thr15": thr15})
    _log.info("Analyzer models ready")
    return _analyzer_models

@app.get("/analyze/full-stats")
def get_full_stats():
    """Pre-computed aggregated stats from the entire scored CSV."""
    return _full_analyze_stats


def _analyze_custom_sync(title: str) -> dict:
    """Synchronous ML inference for /analyze/custom — run via asyncio.to_thread."""
    from datetime import datetime, timezone as _tz
    from training.create_sample_cache import _encode, _build_sentiment, _build_features
    import numpy as np

    try:
        mdl    = _get_analyzer_models()
        bert   = mdl["bert"]
        clf15  = mdl["clf15"]
        scaler = mdl["scaler"]
        thr15  = mdl["thr15"]

        cb_emb, cb_probs, fb_emb, fb_probs = _encode(bert, title)

        # Initialize per-model variables so bert_scores block always has them
        cb_pos = float(cb_probs[2]); cb_neg = float(cb_probs[0])
        fb_pos = fb_neg = rb_pos = rb_neg = 0.0

        # 3-model ensemble — identical to main.py Step 1b and training pipeline.
        try:
            from services.sentiment_score import load_models as _load_sent
            from services.ensemble import (
                ensemble_probs as _ens_probs, sentiment_from_probs as _sent_from_probs,
                FB_PROMPT as _FB_PROMPT, RB_PROMPT as _RB_PROMPT,
            )
            sm = _load_sent()
            fb_raw = sm["fb"](_FB_PROMPT.format(title=title), truncation=True)[0]
            fb = {s["label"].lower(): s["score"] for s in fb_raw}
            rb_raw = sm["rb"](_RB_PROMPT.format(title=title), truncation=True)[0]
            rb = {s["label"].lower(): s["score"] for s in rb_raw}
            cb_neu = max(0.0, 1 - cb_pos - cb_neg)
            fb_pos, fb_neg, fb_neu = fb.get("positive", 0), fb.get("negative", 0), fb.get("neutral", 0)
            rb_pos, rb_neg, rb_neu = rb.get("positive", 0), rb.get("negative", 0), rb.get("neutral", 0)
            ens = _ens_probs(
                (cb_pos, cb_neg, cb_neu),
                (fb_pos, fb_neg, fb_neu),
                (rb_pos, rb_neg, rb_neu),
            )
            avg_pos, avg_neg, avg_neu = ens.pop("_avg")
            ens_sent, ens_disc, ens_conf = _sent_from_probs(avg_pos, avg_neg, avg_neu)
            sent = _build_sentiment(cb_probs, fb_probs)
            sent.update(ens)
            sent["sentiment"] = ens_sent
            sent["sentiment_score"] = ens_disc
            sent["confidence"] = round(ens_conf * 100, 2)
            sent["prob_positive"] = round(avg_pos, 4)
            sent["prob_negative"] = round(avg_neg, 4)
            sent["prob_neutral"]  = round(avg_neu, 4)
        except Exception:
            # Fallback: CryptoBERT only with neutral-wins rule
            cb_neu_fb = max(0.0, 1 - cb_pos - cb_neg)
            if cb_neu_fb > max(cb_pos, cb_neg):
                ens_sent, ens_disc, ens_conf = "neutral", 0, cb_neu_fb
            elif cb_pos > cb_neg:
                net = cb_pos - cb_neg
                ens_disc = 3 if net > 0.50 else 2 if net > 0.25 else 1
                ens_sent, ens_conf = "positive", cb_pos
            else:
                net = cb_neg - cb_pos
                ens_disc = -(3 if net > 0.50 else 2 if net > 0.25 else 1)
                ens_sent, ens_conf = "negative", cb_neg
            sent = _build_sentiment(cb_probs, fb_probs)
            sent["sentiment"] = ens_sent
            sent["sentiment_score"] = ens_disc
            sent["confidence"] = round(ens_conf * 100, 2)
            sent.update({
                "rb_prob_pos": 0.0,
                "rb_prob_neg": 0.0,
                "rb_prob_neu": 0.0,
                "net_agreement": cb_pos - cb_neg,
            })
            fb_pos = fb_neg = rb_pos = rb_neg = 0.0

        pub_dt = datetime.now(tz=_tz.utc)
        now_ts = int(pub_dt.timestamp())

        # Query RAG before 15-minute inference; its 10 features are part of the
        # 1578-dimensional input and its neighbors are also shown in the UI.
        similar = []
        rag_feats = np.zeros(10, dtype=np.float32)
        try:
            from pipeline.rag_news import query_single
            rag_result = query_single(title=title, before_timestamp=now_ts,
                                      channel_impact_rates={}, macro_now=None)
            rag_feats = rag_result["features"].astype(np.float32)
            similar = [
                {"title": s.get("title",""), "change": s.get("btc_change_15m", 0.0), "sim": s.get("similarity_score", 0.0)}
                for s in rag_result.get("similar_news", [])[:3]
            ]
        except Exception:
            pass

        features = _build_features(cb_emb, fb_emb, sent, pub_dt, rag_feats)

        if features.shape[0] != clf15.n_features_in_:
            raise ValueError(
                f"Analyzer built {features.shape[0]} features, but the deployed "
                f"model expects {clf15.n_features_in_}"
            )

        X    = scaler.transform(features.reshape(1, -1)).astype(np.float32)
        p15  = float(clf15.predict_proba(X)[0, 1])
        pred = int(p15 >= thr15)

        impact = _config_impact_tier(p15)
        signal = "BUY" if sent["sentiment"] == "positive" else ("SELL" if sent["sentiment"] == "negative" else "NEUTRAL")

        from training.xgboost_train_groq import crypto_news_type_classify
        type_probs = crypto_news_type_classify(cb_emb.reshape(1, -1))[0]
        TYPE_LABELS = ["regulatory","partnership","product","hack_security","market_move",
                       "macro","adoption","exchange","defi","nft","other"]
        top_type = TYPE_LABELS[int(np.argmax(type_probs))]

        # Explanation with per-model breakdown
        sent_word = "bullish" if sent["sentiment"] == "positive" else ("bearish" if sent["sentiment"] == "negative" else "neutral")
        type_label = top_type.replace("_", " ")
        try:
            def _vote(p, n): return "bullish" if p > n else ("bearish" if n > p else "neutral")
            votes = [f"CryptoBERT→{_vote(cb_pos,cb_neg)}"]
            if fb_pos > 0 or fb_neg > 0: votes.append(f"FinBERT→{_vote(fb_pos,fb_neg)}")
            if rb_pos > 0 or rb_neg > 0: votes.append(f"RoBERTa→{_vote(rb_pos,rb_neg)}")
            model_votes = ", ".join(votes)
        except Exception:
            model_votes = "CryptoBERT only"
        explanation = (
            f"Classified as {type_label} news. "
            f"Model votes: {model_votes}. Final: {sent_word} (equal 1/3 ensemble). "
            f"{'Short-term price impact predicted.' if impact in ('Hot','Medium') else 'No strong short-term price impact predicted.'}"
        )

        try:
            bert_scores = [{"name": "CryptoBERT", "pos": round(cb_pos*100,1), "neg": round(cb_neg*100,1),
                             "neu": round(max(0,(1-cb_pos-cb_neg))*100,1), "weight": 33}]
            if fb_pos > 0 or fb_neg > 0:
                bert_scores.append({"name": "FinBERT", "pos": round(fb_pos*100,1), "neg": round(fb_neg*100,1),
                                    "neu": round(max(0,(1-fb_pos-fb_neg))*100,1), "weight": 33})
            if rb_pos > 0 or rb_neg > 0:
                bert_scores.append({"name": "RoBERTa", "pos": round(rb_pos*100,1), "neg": round(rb_neg*100,1),
                                    "neu": round(max(0,(1-rb_pos-rb_neg))*100,1), "weight": 33})
        except Exception:
            bert_scores = []

        return {
            "title":          title,
            "sentiment":      sent["sentiment"],
            "confidence":     round(sent["confidence"], 1),
            "sentiment_score": sent["sentiment_score"],
            "prob_positive":  round(sent["prob_positive"] * 100, 1),
            "prob_negative":  round(sent["prob_negative"] * 100, 1),
            "prob_neutral":   round(sent["prob_neutral"]  * 100, 1),
            "model_score":    round(p15, 4),
            "model_score_pct": round(p15 * 100, 1),
            "pred_15m":       pred,
            "impact":         impact,
            "news_importance": news_importance({"title": title, "confidence": sent["confidence"] * 100,
                              "prob_positive": sent["prob_positive"], "prob_negative": sent["prob_negative"],
                              "prob_neutral": sent["prob_neutral"], "channel": "live"}),
            "signal":         signal,
            "news_type":      top_type,
            "similar":        similar,
            "explanation":    explanation,
            "bert_scores":    bert_scores,
        }
    except Exception as e:
        raise RuntimeError(str(e))


@app.post("/analyze/custom")
async def analyze_custom(request: Request, body: AnalyzeRequest):
    _check_rate_limit(_client_ip(request), rpm=20)
    try:
        await asyncio.wait_for(_analyze_semaphore.acquire(), timeout=0.1)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Analyzer is busy; try again shortly")
    try:
        return await asyncio.to_thread(_analyze_custom_sync, body.title)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _analyze_semaphore.release()


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)
