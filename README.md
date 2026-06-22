# Crypto News Impact — Live Sentiment Dashboard & Prediction System

> Real-time cryptocurrency news scoring pipeline with a live dashboard, backed by a multi-stream neural architecture for short-term price impact prediction.

---

## What This Does

This system monitors crypto news channels in real time, scores each headline for market impact using a trained ML model, and displays the results on a live dashboard with BTC/ETH price charts and sentiment analysis.

**Two layers:**
1. **Live pipeline** — Telegram listener → NLP scoring → WebSocket broadcast → dashboard
2. **Research model** — Multi-stream neural network trained to predict whether a headline will move BTC price within 15 minutes or 1 hour

---

## Live Dashboard

The dashboard (`dashboard2/`) is a React app that connects to the FastAPI backend via WebSocket and REST.

**Features:**
- Real-time BTC & ETH candlestick charts (Binance data)
- News markers on chart — hover to see headline
- News cards sorted by sentiment impact tier (Hot / Medium / Show)
- BTC 15-minute momentum gauge (live price data)
- News & Sentiment tab with coin filter
- Analyze tab — per-channel statistics, model performance, training data explorer

**Impact tiers:**
| Tier | Score threshold | Confidence |
|------|----------------|------------|
| Hot | ≥ 0.50 | ≥ 60% |
| Medium | ≥ 0.25 | ≥ 55% |
| Show | ≥ 0.20 | ≥ 50% |
| Hidden | below threshold | — |

---

## How to Run

### Prerequisites

```bash
python -m venv .venv311
source .venv311/bin/activate      # Windows: .venv311\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:
```
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_CHANNELS=channel1,channel2
BOT_TOKEN=...
```

### 1. Start the API server

```bash
cd api
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the news pipeline

```bash
.venv311/bin/python main.py
```

This connects to Telegram, backfills the last 5 days of history, then listens for new messages in real time. Each news item is scored and posted to the API.

### 3. Start the dashboard

```bash
cd dashboard2
npm install
npm run dev        # development
npm run build      # production build → dist/
```

The dashboard serves from `http://localhost:5173` in dev mode, or from `dist/` via the API's static file serving in production.

---

## Project Structure

```
├── main.py                     # Main pipeline: score news → POST to API
├── config.py                   # Env config
├── api/
│   └── server.py               # FastAPI: REST + WebSocket + Binance proxy
├── bot/
│   └── telegram_listener.py    # Telegram backfill + real-time listener
├── dashboard2/
│   ├── src/App.jsx             # Full React dashboard
│   └── public/                 # Static assets
├── pipeline/                   # NLP scoring modules
├── services/                   # Data collection utilities
├── models/                     # Saved model weights
├── xgboost_v9_*.json           # XGBoost ensemble (15m + 1h classifiers)
└── requirements.txt
```

---

## Model Architecture

The prediction model fuses **four parallel input streams**:

| Stream | Input |
|--------|-------|
| Semantic | CryptoBERT headline embedding (frozen) |
| Chain-of-Thought | Encoded financial reasoning trace |
| Historical Context | RAG retrieval summary from similar past events |
| Contextual | Time of day, volatility, sentiment/category features |

Fused through a shared MLP with six output heads:
- 2× binary classification (15-min and 1-hour impact)
- 2× regression (price change magnitude)
- 1× direction (up / down)
- 1× confidence (self-estimated reliability)

**XGBoost ensemble (v9)** runs on top of the neural embeddings for production scoring.

---

## Key Results

| Metric | 15-Minute | 1-Hour |
|--------|-----------|--------|
| ROC-AUC | 0.659 | 0.632 |
| F1 Score | 0.318 | 0.421 |
| Precision | 25.0% | 28.7% |
| Recall | 43.6% | 79.0% |
| Accuracy | 66.8% | 49.3% |

> Majority-class baseline (always "no impact") achieves 81.8% / 76.6% accuracy with F1 = 0. ROC-AUC and F1 are the meaningful metrics.

**Dataset:** ~31,500 BTC headlines with paired USDT price data, split chronologically (70 / 15 / 15).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | PyTorch, XGBoost, HuggingFace Transformers, CryptoBERT |
| Backend | FastAPI, Uvicorn, WebSocket |
| Frontend | React, Vite, lightweight-charts (Binance charts) |
| Data | Telegram (Telethon), Binance REST API |
| Embeddings | fastembed, Qdrant |

---

## Limitations

- Trained on Bitcoin only — transferability to other assets is unverified
- Headlines only; full article body is not used
- Reaction speed: algorithmic traders act in milliseconds; pipeline latency means initial moves may have concluded before scoring
- Temporal drift: crypto dynamics evolve rapidly, periodic retraining is needed

---

## Citation

```bibtex
@article{shakibayi2026multistream,
  title     = {A Multi-Stream Neural Architecture for Short-Term Cryptocurrency Price Impact Prediction from News},
  author    = {Shakibayi Senobari, Haniye},
  year      = {2026},
  institution = {Department of Artificial Intelligence, Bahçeşehir University, Istanbul, Turkey}
}
```

---

*Department of Artificial Intelligence, Bahçeşehir University, Istanbul, Turkey — 2026*
