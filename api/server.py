"""
API server — read-only, JSON cache mode.
Loads news_cache.json on startup and serves it.
No live bot, no ingestion, no broadcasting.
"""
import sys
import csv
import json
import aiohttp
from pathlib import Path
from typing import List
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # make project root importable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import API_HOST, API_PORT, GROQ_API_KEYS, GROQ_CLASSIFICATION_MODEL

CACHE_FILE = ROOT / "news_cache.json"

# Score thresholds (must match frontend App.jsx)
SCORE_HIGH   = 0.67   # "High impact" tier
SCORE_MEDIUM = 0.50   # "Medium impact" tier

# ── Load cache on startup ──────────────────────────────────────────
def _load_cache() -> List[dict]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

all_news: List[dict] = _load_cache()

# Hot = high-score items from cache (score ≥ 0.67)
hot_news: List[dict] = [
    item for item in all_news
    if abs(float(item.get("model_score", 0))) >= SCORE_HIGH
]

print(f"✅ Loaded {len(all_news)} news items from cache  ({len(hot_news)} hot)")


# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="Crypto News API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket — send cache once, then keep connection open ─────────
@app.websocket("/ws/all")
async def ws_all(ws: WebSocket):
    await ws.accept()
    for item in all_news[-200:]:
        await ws.send_json(item)
    try:
        while True:
            await ws.receive_text()   # keep-alive, ignore any incoming text
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/hot")
async def ws_hot(ws: WebSocket):
    await ws.accept()
    for item in hot_news[-50:]:
        await ws.send_json(item)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass


# ── REST — news ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "source":         "news_cache.json",
        "all_news_count": len(all_news),
        "hot_news_count": len(hot_news),
    }


@app.get("/news/all")
def get_all():
    return all_news[-2000:]


@app.get("/news/hot")
def get_hot():
    return hot_news[-50:]


@app.get("/news/dates")
def get_dates():
    import datetime
    seen = set()
    for item in all_news:
        ts = item.get("published_ts") or item.get("received_at")
        if ts:
            d = datetime.datetime.utcfromtimestamp(float(ts))
            seen.add(f"{d.year}-{d.month:02d}-{d.day:02d}")
    return sorted(seen)


@app.get("/news/by-date")
def get_by_date(start: int, end: int):
    return [
        item for item in all_news
        if start <= float(item.get("published_ts") or item.get("received_at", 0)) <= end
    ]


# ── REST — training analytics ──────────────────────────────────────
@app.get("/training/stats")
def get_training_stats():
    """Training data statistics from news_cleaned.csv + production_results_v5.json."""
    csv_path     = ROOT / "news_cleaned.csv"
    results_path = ROOT / "production_results_v5.json"
    stats = {}

    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            stats["model_performance"] = json.load(f)

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
    """Per-category accuracy from news_cleaned.csv (train) + ews_ev.csv (test)."""
    train_csv = ROOT / "news_cleaned.csv"
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
    """Combined report: training data (news_cleaned.csv) + cache (news_cache.json)."""
    import datetime

    csv_path     = ROOT / "news_cleaned.csv"
    results_path = ROOT / "production_results_v5.json"

    # ── Training CSV stats ────────────────────────────────────────
    train = {
        "file": "news_cleaned.csv",
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
                    w   = float(row.get("weight", 0))
                    bp  = float(row.get("btc_price_at_news", ""))
                    b15 = float(row.get("btc_price_15m", ""))
                    b1h = float(row.get("btc_price_1h", ""))
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
        # estimate split counts
        n = train["total_filtered"]
        train["split"]["train_n"] = int(n * 0.70)
        train["split"]["val_n"]   = int(n * 0.15)
        train["split"]["test_n"]  = n - int(n * 0.70) - int(n * 0.15)

    # ── Cache stats (all_news in memory) ──────────────────────────
    cache_channels: dict = {}
    cache_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    cache_signals    = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
    score_high = score_med = score_low = 0
    with_btc = pred15_pos = pred1h_pos = 0
    ts_min = ts_max = None

    for item in all_news:
        sc  = abs(float(item.get("model_score", 0)))
        if sc >= 0.67:  score_high += 1
        elif sc >= 0.50: score_med += 1
        else:            score_low += 1

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
        return datetime.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")

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

    # ── Model performance results ─────────────────────────────────
    model_results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            model_results = json.load(f)

    # ── Model architecture (hardcoded from production_system_v5.py) ──
    architecture = {
        "name":       "CryptoImpactNetV5",
        "type":       "4-Tower Gated Fusion Neural Network",
        "file":       "production_system_v5.py",
        "towers": [
            {"name": "Semantic", "input": "CryptoBERT 768-dim + 6 sentiment + 11 news-type probs", "output": 16},
            {"name": "RAG",      "input": "Qdrant vector DB — macro-conditioned re-weighting",     "output":  8},
            {"name": "Macro",    "input": "5-dim: weekend / low-liq / US hours / Asia / FOMC",     "output":  8},
            {"name": "Market",   "input": "5-dim: BTC 1h/4h/24h return + 1h/4h volatility",       "output":  8},
        ],
        "fusion":     "40 → 24 → 12, gated softmax weighting",
        "heads":      ["cls_15m", "cls_1h", "reg_15m", "reg_1h", "confidence", "direction"],
        "loss":       "FocalLoss (cls) + MSE (reg) + CrossEntropy (direction)",
        "optimizer":  "AdamW  lr=3e-4  weight_decay=1e-3",
        "epochs":     200,
        "patience":   20,
        "batch_size": 64,
        "threshold_15m": 0.39,
        "threshold_1h":  0.40,
        "embedding":  "ElKulako/cryptobert",
    }

    # ── Per-category stats from ews_ev.csv ───────────────────────────
    eval_csv = ROOT / "ews_ev.csv"
    cat_stats: dict = defaultdict(lambda: {"train_count": 0, "test_count": 0,
                                            "tp15": 0, "tn15": 0, "fp15": 0, "fn15": 0,
                                            "tp1h": 0, "tn1h": 0, "fp1h": 0, "fn1h": 0})
    # fill train counts
    for nt, cnt in train["news_type_counts"].items():
        cat_stats[nt]["train_count"] = cnt
    # fill test results
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
    impact    = "High" if score >= SCORE_HIGH else "Medium" if score >= SCORE_MEDIUM else "Low"

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


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)
