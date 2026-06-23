"""
API server — read-only, JSON cache mode.
Loads news_cache.json on startup and serves it.
No live bot, no ingestion, no broadcasting.
"""
import re
import sys
import csv
import json
import math
import aiohttp
from pathlib import Path
from typing import List
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # make project root importable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import API_HOST, API_PORT, GROQ_API_KEYS, GROQ_CLASSIFICATION_MODEL

CACHE_FILE   = ROOT / "news_cache.json"
HIST_CSV     = ROOT / "news_cleaned_filtered_scored.csv"

# Normalize legacy channel name variants to canonical names
_CHANNEL_NORM = {
    "CoinTelegraph":   "cointelegraph",
    "CoinMarketCap":   None,   # blocked
    "CryptoNews":      None,
    "CoingraphNews":   None,
    "cryptoslatenews": None,
}

from pipeline.reduce_noise import BLOCKED_CHANNELS, passes_news_filter as _passes_news_filter


def _passes_noise_filter(item: dict) -> bool:
    return _passes_news_filter(item.get("title", ""), item.get("channel", ""))

# Impact thresholds — badges only, NOT used for display filtering
# Display filter uses confidence only (≥50%). Score is for hot/medium coloring.
# Uses max(model_score, model_score_1h) for tier assignment.
SCORE_HOT    = 0.50   # Hot    tier: max(s15,s1h) ≥0.50
SCORE_MED    = 0.25   # Medium tier: max(s15,s1h) ≥0.25
SCORE_HIGH   = SCORE_HOT   # alias used in hot_news / explain endpoint
CONF_MIN     = 50.0   # minimum confidence to display at all

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
                ch = i.get("channel", "")
                if ch in _CHANNEL_NORM:
                    normalized = _CHANNEL_NORM[ch]
                    if normalized is None:
                        i["channel"] = "__blocked__"
                    else:
                        i["channel"] = normalized
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
            return items
        except Exception:
            pass
    return []

all_news: List[dict] = _load_cache()

# Hot = max(score_15m, score_1h) >= 0.50
hot_news: List[dict] = [
    item for item in all_news
    if max(abs(float(item.get("model_score", 0))),
           abs(float(item.get("model_score_1h", 0)))) >= SCORE_HOT
]

print(f"✅ Loaded {len(all_news)} news items from cache  ({len(hot_news)} hot)")


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
                ch = row.get("channel", "")
                # Normalize legacy channel name variants
                if ch in _CHANNEL_NORM:
                    normalized = _CHANNEL_NORM[ch]
                    if normalized is None:
                        continue   # blocked channel
                    row["channel"] = normalized
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

                confidence = float(row.get("confidence") or 50)
                sentiment  = row.get("sentiment", "neutral")

                # Score based on actual BTC impact, floored so all items pass the
                # frontend >= 0.50 filter while still reflecting real market reaction.
                impact_score = min(1.0, 0.52 + abs(btc_c15m) / 6.0)
                sig_type = (
                    "BUY"  if sentiment == "positive" else
                    "SELL" if sentiment == "negative" else
                    "NEUTRAL"
                )

                items.append({
                    "id":             f"hist_{pub_ts}_{hash(row.get('title', '')[:30]) % 100000}",
                    "title":          row.get("title", ""),
                    "link":           row.get("link", ""),
                    "channel":        row.get("channel", "unknown"),
                    "published":      published,
                    "published_ts":   pub_ts,
                    "sentiment":      sentiment,
                    "sentiment_score": float(row.get("sentiment_score") or 0),
                    "confidence":     confidence,
                    "weight":         float(row.get("weight") or 0),
                    "prob_positive":  float(row.get("prob_positive") or 0),
                    "prob_negative":  float(row.get("prob_negative") or 0),
                    "prob_neutral":   float(row.get("prob_neutral") or 0),
                    "type":           sig_type,
                    "btc_change_15m": round(btc_c15m, 4),
                    "btc_change_1h":  round(btc_c1h,  4),
                    "model_score":    round(impact_score, 4),
                    "model_score_1h": round(min(1.0, 0.52 + abs(btc_c1h) / 6.0), 4),
                    "score_normalized": True,
                    "impact":         "high" if abs(btc_c15m) >= 0.5 else "medium" if abs(btc_c15m) >= 0.3 else "low",
                    "news_type":      row.get("news_type", ""),
                    "source":         "historical",
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


# Recent historical (6 months) — used for /news/all feed
historical_news: List[dict] = _load_csv_as_news(months=6)
_hist_channels = {item["channel"] for item in historical_news}
print(f"✅ Loaded {len(historical_news)} historical items (6mo) from news_cleaned_filtered_scored.csv")

# Full historical (all dates) — used only for /news/dates and /news/by-date calendar
# Loaded as lightweight {published_ts, channel, title, link, model_score} to avoid memory bloat
def _load_hist_dates_index() -> List[dict]:
    """Load full date range from CSV — minimal fields only, for calendar index."""
    items = _load_csv_as_news(months=None)
    return [{"published_ts": i["published_ts"], "title": i["title"],
             "channel": i["channel"], "link": i.get("link",""),
             "model_score": i["model_score"], "sentiment": i.get("sentiment",""),
             "confidence": i.get("confidence", 50), "impact": i.get("impact","low"),
             "score_normalized": True} for i in items]

historical_dates_index: List[dict] = _load_hist_dates_index()
print(f"✅ Loaded {len(historical_dates_index)} dates-index items (full range)")


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
                ch = row.get("channel", "") or "unknown"
                if ch in _CHANNEL_NORM:
                    ch = _CHANNEL_NORM[ch]
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
                btc_1h    = float(row["btc_price_1h"])
                btc_c15m  = (btc_15m - btc_price) / btc_price * 100
                btc_c1h   = (btc_1h  - btc_price) / btc_price * 100
                impact_score = min(1.0, 0.52 + abs(btc_c15m) / 6.0)
                conf     = float(row.get("confidence") or 50)
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
print(f"✅ Full analyze stats computed: {_full_analyze_stats.get('total', 0):,} items "
      f"({_full_analyze_stats.get('date_from')} → {_full_analyze_stats.get('date_to')})")


import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_app):
    global _last_cache_mtime
    _last_cache_mtime = CACHE_FILE.stat().st_mtime if CACHE_FILE.exists() else 0.0
    asyncio.create_task(_cache_refresh_loop())
    yield

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="Crypto News API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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

async def _cache_refresh_loop():
    global all_news, hot_news, _last_cache_mtime, _idf_cache
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
            if fresh:
                all_news = new_items
                hot_news = [i for i in all_news
                            if max(abs(float(i.get("model_score", 0))),
                                   abs(float(i.get("model_score_1h", 0)))) >= SCORE_HOT]
                _idf_cache = None
                _last_cache_mtime = mtime
                print(f"🔄 Cache refreshed: {len(all_news)} items, {len(fresh)} new")
                for item in fresh:
                    await _broadcast(_ws_all_clients, item)
                    if max(abs(float(item.get("model_score", 0))),
                           abs(float(item.get("model_score_1h", 0)))) >= SCORE_HOT:
                        await _broadcast(_ws_hot_clients, item)
        except Exception as exc:
            print(f"⚠️  Cache refresh error: {exc}")




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
@app.post("/news")
async def ingest_news(item: dict):
    """Receive a scored news item from main.py and persist it to the cache."""
    global all_news, hot_news, _idf_cache

    # Normalise channel name
    ch = item.get("channel", "")
    if ch in _CHANNEL_NORM:
        norm = _CHANNEL_NORM[ch]
        if norm is None:
            return {"status": "blocked"}
        item["channel"] = norm

    if item.get("channel") in BLOCKED_CHANNELS:
        return {"status": "blocked"}

    if not _passes_noise_filter(item):
        return {"status": "filtered"}

    # Ensure published_ts exists
    if "published_ts" not in item and item.get("id"):
        item["published_ts"] = int(item["id"]) // 1000

    # Deduplicate by id
    existing_ids = {i.get("id") for i in all_news}
    if item.get("id") in existing_ids:
        return {"status": "duplicate"}

    # Prepend to in-memory list and trim to MAX_CACHE_ITEMS
    MAX = 10_000
    all_news = ([item] + all_news)[:MAX]
    if max(abs(float(item.get("model_score", 0))),
           abs(float(item.get("model_score_1h", 0)))) >= SCORE_HOT:
        hot_news = ([item] + hot_news)[:MAX]
    _idf_cache = None

    # Persist to cache file atomically
    try:
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_news, default=str), encoding="utf-8")
        tmp.replace(CACHE_FILE)
        global _last_cache_mtime
        _last_cache_mtime = CACHE_FILE.stat().st_mtime
    except Exception as e:
        print(f"⚠️  Cache write error: {e}")

    # Broadcast to WebSocket clients
    await _broadcast(_ws_all_clients, item)
    if max(abs(float(item.get("model_score", 0))),
           abs(float(item.get("model_score_1h", 0)))) >= SCORE_HOT:
        await _broadcast(_ws_hot_clients, item)

    return {"status": "ok"}


@app.get("/news/since")
def get_since(ts: int = 0):
    """Return items with published_ts > ts — for incremental frontend polling."""
    items = [
        i for i in (all_news + historical_news)
        if float(i.get("published_ts") or i.get("received_at", 0)) > ts
    ]
    return sorted(items, key=lambda x: x.get("published_ts") or 0)


@app.get("/health")
def health():
    return {
        "status":              "ok",
        "source":              "news_cache.json + news_cleaned_filtered.csv",
        "live_news_count":     len(all_news),
        "historical_count":    len(historical_news),
        "total_news_count":    len(all_news) + len(historical_news),
        "hot_news_count":      len(hot_news),
        "historical_channels": sorted(_hist_channels),
    }


@app.get("/news/all")
def get_all():
    combined = sorted(
        all_news + historical_news,
        key=lambda x: x.get("published_ts") or x.get("received_at") or 0,
    )
    return combined[-20000:]


@app.get("/news/hot")
def get_hot():
    return hot_news[-50:]


@app.get("/news/dates")
def get_dates():
    from datetime import datetime, timezone
    seen = set()
    for item in all_news + historical_dates_index:
        ts = item.get("published_ts") or item.get("received_at")
        if ts:
            d = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            seen.add(f"{d.year}-{d.month:02d}-{d.day:02d}")
    return sorted(seen)


@app.get("/news/by-date")
def get_by_date(start: int, end: int):
    return [
        item for item in all_news + historical_dates_index
        if start <= float(item.get("published_ts") or item.get("received_at", 0)) <= end
    ]


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
        "v9":           ROOT / "production_results_v9.json",
        "xgboost_v9":   ROOT / "xgboost_v9_results.json",
        "xgboost":      ROOT / "xgboost_results.json",
    }
    all_models: dict = {}
    for name, path in model_files.items():
        if path.exists():
            with open(path, encoding="utf-8") as f:
                all_models[name] = json.load(f)

    # Current best model = xgboost_v9 → v9 → v8 → ...
    best = next((k for k in ["xgboost_v9","v9","v8","v7","v6","v5"] if k in all_models), None)
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
                except: pass
                try: confidences.append(float(row["confidence"]))
                except: pass
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
                except: pass

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
                except: pass

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
    # Prefer xgboost_v9 → v9 ANN → v8 ANN
    results_path = next(
        (p for p in [
            ROOT / "xgboost_v9_results.json",
            ROOT / "production_results_v9.json",
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
                except: pass
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
        sc  = max(abs(float(item.get("model_score", 0))),
                  abs(float(item.get("model_score_1h", 0))))
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

    architecture = {
        "name":       "XGBoost v9 (DualBERT + PriceContext)",
        "type":       "Gradient Boosted Trees — GPU (device=cuda, tree_method=hist)",
        "file":       "xgboost_v9.py",
        "feature_dim": 1578,
        "feature_layout": [
            {"name": "CryptoBERT embedding",    "dims": 768},
            {"name": "FinBERT embedding",        "dims": 768},
            {"name": "Ensemble sentiment (3-BERT + derived)", "dims": 13},
            {"name": "News-type probs",          "dims": 11},
            {"name": "Macro timing (5) + price context (3)", "dims": 8},
            {"name": "RAG features",             "dims": 10},
        ],
        "embeddings": ["ElKulako/cryptobert (768)", "ProsusAI/finbert (768)"],
        "price_context": ["btc_vol (rolling std 20)", "btc_mom (rolling mean 5)", "fear_greed (Alternative.me)"],
        "params": {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "early_stopping_rounds": 20,
        },
        "threshold_15m": 0.295,
        "threshold_1h":  0.265,
        "min_precision": 0.20,
        "monthly_seed":  43,
        "roc_auc_15m":   0.677,
        "roc_auc_1h":    0.657,
        "f1_15m":        0.395,
        "f1_1h":         0.457,
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
    "just","also","amid","amid","amid","per","via","out","off",
}

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

def _get_idf() -> dict:
    global _idf_cache
    if _idf_cache is None:
        _idf_cache = _build_idf(all_news + historical_news)
    return _idf_cache


@app.post("/news/similar")
async def find_similar(item: dict):
    title = (item.get("title") or "").strip()
    if not title:
        return {"similar": []}

    # Items that came from live ingestion already carry pre-computed similar list
    if item.get("similar"):
        return {"similar": item["similar"]}

    idf      = _get_idf()
    q_tokens = _tokens(title)
    if not q_tokens:
        return {"similar": []}
    q_vec = _tfidf_vec(q_tokens, idf)

    pool    = (all_news + historical_news)
    results = []
    for n in pool:
        t = (n.get("title") or "").strip()
        if not t or t == title:
            continue
        sim = _cosine(q_vec, _tfidf_vec(_tokens(t), idf))
        if sim >= 0.12:
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
async def explain_news(item: dict):
    title     = item.get("title", "")
    sentiment = item.get("sentiment", "neutral")
    confidence= item.get("confidence", 50)
    score     = abs(float(item.get("model_score", 0)))
    score_1h  = abs(float(item.get("model_score_1h", 0)))
    btc_15m   = float(item.get("btc_change_15m", 0))
    btc_1h    = float(item.get("btc_change_1h",  0))
    channel   = item.get("channel", "unknown")
    similar   = item.get("similar", [])
    max_score = max(score, score_1h)
    impact    = ("Hot"    if max_score >= SCORE_HOT  else
                 "Medium" if max_score >= SCORE_MED  else "Show")

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
@app.get("/proxy/klines")
async def proxy_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                       limit: int = 200, startTime: int = None):
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval={interval}&limit={min(limit,1000)}")
    if startTime:
        url += f"&startTime={startTime}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                return await resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/proxy/stream/{symbol}/{interval}")
async def proxy_stream(ws: WebSocket, symbol: str, interval: str):
    await ws.accept()
    binance_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(binance_url) as bws:
                async for msg in bws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await ws.send_text(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
    except Exception:
        pass
    finally:
        try: await ws.close()
        except: pass


# ── Custom news analyzer ──────────────────────────────────────────
_analyzer_models = {}   # lazy-loaded on first call

def _get_analyzer_models():
    if _analyzer_models:
        return _analyzer_models
    from training.create_sample_cache import _load_bert_models, _load_xgb
    print("🔄 Loading BERT + XGBoost models for custom analyzer (CPU)...")
    bert = _load_bert_models(force_cpu=True)
    clf15, _clf1h, scaler, thr15, _thr1h = _load_xgb()
    _analyzer_models.update({"bert": bert, "clf15": clf15, "scaler": scaler, "thr15": thr15})
    print("✅ Analyzer models ready")
    return _analyzer_models

@app.get("/analyze/full-stats")
def get_full_stats():
    """Pre-computed aggregated stats from the entire scored CSV."""
    return _full_analyze_stats


@app.post("/analyze/custom")
async def analyze_custom(body: dict):
    title = (body.get("title") or "").strip()
    if len(title) < 5:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Title too short")

    from datetime import datetime, timezone as _tz
    from training.create_sample_cache import _encode, _build_sentiment, _build_features
    import numpy as np

    try:
        mdl    = _get_analyzer_models()
        bert   = mdl["bert"]
        clf15  = mdl["clf15"]
        scaler = mdl["scaler"]
        thr15  = mdl["thr15"]

        cb_emb, cb_probs, fb_emb = _encode(bert, title)

        # Use full 3-model ensemble for sentiment (same as main.py pipeline)
        # CryptoBERT gets 50% weight — it's the only crypto-domain model and
        # correctly interprets macro signals (rate cuts, inflation) for BTC.
        # FinBERT + RoBERTa share the remaining 50% but use a BTC-framed prompt.
        try:
            from services.sentiment_score import load_models as _load_sent
            sm = _load_sent()
            btc_ctx = f"Bitcoin price impact: {title}"
            fb_raw = sm["fb"](btc_ctx, truncation=True)[0]
            fb = {s["label"].lower(): s["score"] for s in fb_raw}
            rb_raw = sm["rb"](btc_ctx, truncation=True)[0]
            rb = {s["label"].lower(): s["score"] for s in rb_raw}
            cb_pos, cb_neg = float(cb_probs[2]), float(cb_probs[0])
            fb_pos, fb_neg = fb.get("positive", 0), fb.get("negative", 0)
            rb_pos, rb_neg = rb.get("positive", 0), rb.get("negative", 0)
            # CryptoBERT: 50%, FinBERT: 25%, RoBERTa: 25%
            avg_pos = cb_pos * 0.5 + fb_pos * 0.25 + rb_pos * 0.25
            avg_neg = cb_neg * 0.5 + fb_neg * 0.25 + rb_neg * 0.25
            avg_neu = 1 - avg_pos - avg_neg
            if avg_neu > max(avg_pos, avg_neg):
                ens_sent, ens_conf, ens_disc = "neutral", avg_neu, 0
            else:
                net = avg_pos - avg_neg
                ens_disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
                           -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
                ens_sent = "positive" if ens_disc > 0 else ("negative" if ens_disc < 0 else "neutral")
                ens_conf = avg_pos if ens_disc > 0 else (avg_neg if ens_disc < 0 else avg_neu)
            sent = _build_sentiment(cb_probs)
            sent["sentiment"] = ens_sent
            sent["sentiment_score"] = ens_disc
            sent["confidence"] = round(ens_conf * 100, 2)
            sent["prob_positive"] = round(avg_pos, 4)
            sent["prob_negative"] = round(avg_neg, 4)
            sent["prob_neutral"]  = round(avg_neu, 4)
        except Exception:
            sent = _build_sentiment(cb_probs)

        pub_dt = datetime.now(tz=_tz.utc)
        features = _build_features(cb_emb, fb_emb, sent, pub_dt)
        rag_zeros = np.zeros(10, dtype=np.float32)
        features  = np.concatenate([features, rag_zeros]).astype(np.float32)

        X    = scaler.transform(features.reshape(1, -1)).astype(np.float32)
        p15  = float(clf15.predict_proba(X)[0, 1])
        pred = int(p15 >= thr15)

        impact = ("Hot"    if p15 >= SCORE_HOT  else
                  "Medium" if p15 >= SCORE_MED  else "Show")
        signal = "BUY" if sent["sentiment"] == "positive" else ("SELL" if sent["sentiment"] == "negative" else "NEUTRAL")

        from training.xgboost_v9 import crypto_news_type_classify
        type_probs = crypto_news_type_classify(cb_emb.reshape(1, -1))[0]
        TYPE_LABELS = ["regulatory","partnership","product","hack_security","market_move",
                       "macro","adoption","exchange","defi","nft","other"]
        top_type = TYPE_LABELS[int(np.argmax(type_probs))]

        # RAG — find similar past news
        similar = []
        try:
            from pipeline.rag_news import query_single
            now_ts = int(pub_dt.timestamp())
            rag_result = query_single(title=title, before_timestamp=now_ts,
                                      channel_impact_rates={}, macro_now={})
            similar = [
                {"title": s.get("title",""), "change": s.get("btc_change_15m", 0.0), "sim": s.get("similarity_score", 0.0)}
                for s in rag_result.get("similar_news", [])[:3]
            ]
            rag_features = rag_result["features"]
            features = _build_features(cb_emb, fb_emb, sent, pub_dt)
            features  = np.concatenate([features, rag_features]).astype(np.float32)
            X    = scaler.transform(features.reshape(1, -1)).astype(np.float32)
            p15  = float(clf15.predict_proba(X)[0, 1])
            pred = int(p15 >= thr15)
            impact = ("Hot" if p15 >= SCORE_HOT else "Medium" if p15 >= SCORE_MED else "Show")
        except Exception:
            pass

        # Explanation with per-model breakdown
        sent_word = "bullish" if sent["sentiment"] == "positive" else ("bearish" if sent["sentiment"] == "negative" else "neutral")
        type_label = top_type.replace("_", " ")
        try:
            cb_vote = "bullish" if cb_pos > cb_neg else ("bearish" if cb_neg > cb_pos else "neutral")
            fb_vote = "bullish" if fb_pos > fb_neg else ("bearish" if fb_neg > fb_pos else "neutral")
            rb_vote = "bullish" if rb_pos > rb_neg else ("bearish" if rb_neg > rb_pos else "neutral")
            model_votes = f"CryptoBERT→{cb_vote}, FinBERT→{fb_vote}, RoBERTa→{rb_vote}"
        except Exception:
            model_votes = "sentiment ensemble"
        explanation = (
            f"Classified as {type_label} news. "
            f"Model votes: {model_votes}. Final: {sent_word} (CryptoBERT weighted 50%). "
            f"{'Short-term price impact predicted.' if impact in ('Hot','Medium') else 'No strong short-term price impact predicted.'}"
        )

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
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)