# Crypto News Sentiment & Market Impact — Live Dashboard

> A real-time system that monitors cryptocurrency news, scores each headline for market impact, and displays the results on a live dashboard with BTC/ETH price charts.

---

## What This Does

This system listens to crypto news channels in real time, analyzes each headline using a multi-model NLP pipeline, predicts whether the news will move the market, and streams the results to a live dashboard.

**Two layers:**
1. **Live pipeline** — Telegram listener → sentiment analysis → impact scoring → live dashboard
2. **Prediction model** — trained to classify whether a news headline will cause a short-term BTC/ETH price movement

---

## Live Dashboard

A React-based dashboard that connects to the backend in real time.

**Features:**
- Live BTC & ETH candlestick charts with news markers
- Hover over chart markers to see the headline
- News cards ranked by predicted market impact
- Real-time BTC momentum gauge
- News & Sentiment tab with coin filter
- Channel analysis and model performance overview

**Impact tiers:**

| Tier | Meaning |
|------|---------|
| Hot | High confidence, strong predicted impact |
| Medium | Moderate predicted impact |
| Show | Low but notable signal |
| Hidden | Filtered out — low signal |

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
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the news pipeline

```bash
.venv311/bin/python main.py
```

Connects to Telegram, backfills the last 5 days of history, then listens for new messages in real time. Each headline is scored and pushed to the dashboard instantly.

### 3. Start the dashboard

```bash
cd dashboard2
npm install
npm run dev        # development
npm run build      # production build
```

---

## Project Structure

```
├── main.py                     # Entry point: Telegram → score → API
├── config.py                   # Config and thresholds
├── requirements.txt
│
├── api/
│   └── server.py               # FastAPI backend: REST + WebSocket + Binance proxy
│
├── bot/
│   ├── telegram_listener.py    # Telegram backfill + real-time listener
│   └── telegram_alert.py
│
├── pipeline/
│   ├── spam_filter.py          # Pre-filters for incoming news
│   ├── rag_news.py             # Similar news retrieval (Qdrant)
│   ├── processor.py
│   └── reduce_noise.py         # Channel and noise filters
│
├── services/
│   ├── sentiment_score.py      # Multi-model sentiment ensemble
│   ├── price_fetcher.py        # Live BTC/ETH price tracking
│   └── ...                     # Data collection scripts
│
├── models/
│   └── ...                     # Model architecture classes
│
├── storage/
│   ├── database.py             # Database connection
│   └── cache.py                # JSON cache fallback
│
├── training/
│   ├── xgboost_v9.py           # Train the scoring model (run once)
│   └── ...                     # Historical scoring and data prep scripts
│
├── archive/
│   └── ...                     # Earlier model versions (kept for reference)
│
└── dashboard2/
    ├── src/App.jsx             # Full React dashboard
    ├── public/                 # Static assets
    └── package.json
```

> **Note:** Model files are not committed to the repo. Run `python training/xgboost_v9.py` once to generate them locally.

---

## Model Architecture

### Production Model — XGBoost

The live scoring pipeline uses a gradient-boosted tree ensemble. Each headline is converted into a rich feature vector combining:

- **Semantic embeddings** — two frozen language models (crypto-domain + financial domain)
- **Sentiment ensemble** — three independent NLP models averaged for stability
- **News-type classification** — detects the category of the headline
- **Market context** — trading session, fear & greed index
- **Historical similarity** — retrieves similar past news and their market outcomes

Two classifiers run in parallel — one for short-term impact, one for longer-term impact.

### Earlier Model — Neural Network (ANN)

An earlier version used a custom 3-tower neural network with cross-attention fusion between the semantic, retrieval, and market context streams. It is preserved in `archive/` for reference.

---

## ANN vs XGBoost — Comparison

| | Neural Network (ANN) | XGBoost |
|---|---|---|
| **Architecture** | 3-tower MLP + cross-attention | Gradient-boosted trees |
| **Text encoding** | Single language model | Two language models combined |
| **Sentiment** | Single model | Ensemble of three models |
| **RAG integration** | Cross-attention fusion | Feature vector lookup |
| **Training** | Focal loss + early stopping | Threshold sweep on validation set |
| **Inference speed** | Slower | Fast, no GPU needed |
| **Interpretability** | Low | Medium (feature importance) |
| **Performance** | Baseline | Better across all metrics |
| **Status** | Archived | **Production** |

**Why XGBoost replaced the ANN:**
- Dual language model features capture both crypto-domain and financial signals
- Three-model sentiment ensemble reduces noise from any single model
- More robust with limited training data, less prone to overfitting
- Faster inference with no GPU dependency

---

## Dataset

Over **75,000 crypto news headlines** paired with real BTC/ETH price data, split chronologically into train, validation, and test sets (no lookahead bias).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | XGBoost, PyTorch, HuggingFace Transformers |
| NLP Models | CryptoBERT, FinBERT, RoBERTa |
| Backend | FastAPI, Uvicorn, WebSocket |
| Frontend | React, Vite, Lightweight Charts |
| Data | Telegram (Telethon), Binance API |
| Vector DB | Qdrant |

---

## Limitations

- Trained primarily on Bitcoin news — transferability to other assets is unverified
- Headlines only; full article body is not used
- Algorithmic traders react in milliseconds; some initial price moves may conclude before the pipeline scores the news
- Crypto market dynamics evolve rapidly — periodic retraining is recommended

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
