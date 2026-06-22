# Crypto News Impact — Live Sentiment Dashboard & Prediction System

> Real-time cryptocurrency news scoring pipeline with a live dashboard, backed by a multi-stream neural architecture for short-term price impact prediction.

---

## What This Does

This system monitors crypto news channels in real time, scores each headline for market impact using a trained ML model, and displays the results on a live dashboard with BTC/ETH price charts and sentiment analysis.

**Two layers:**
1. **Live pipeline** — Telegram listener → 3-model NLP ensemble → XGBoost scoring → WebSocket broadcast → dashboard
2. **Research model** — XGBoost v9 trained on DualBERT features (CryptoBERT + FinBERT, 1578-dim) to predict BTC price impact within 15 minutes and 1 hour

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
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
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
├── main.py                     # Entry point: Telegram → score → API
├── config.py                   # Env config and thresholds
├── requirements.txt
│
├── api/
│   └── server.py               # FastAPI: REST + WebSocket + Binance proxy
│
├── bot/
│   ├── telegram_listener.py    # Telegram backfill + real-time listener
│   └── telegram_alert.py
│
├── pipeline/
│   ├── spam_filter.py          # Pre-filters for incoming news
│   ├── rag_news.py             # RAG query against Qdrant
│   ├── processor.py
│   └── reduce_noise.py         # Noise/channel filters (shared across modules)
│
├── services/
│   ├── sentiment_score.py      # CryptoBERT + FinBERT + RoBERTa ensemble
│   ├── price_fetcher.py        # Live BTC/ETH price tracking
│   └── ...                     # Data collection scripts
│
├── models/
│   └── ...                     # Model architecture classes
│
├── storage/
│   ├── database.py             # PostgreSQL async pool
│   └── cache.py                # JSON cache fallback
│
├── training/
│   ├── xgboost_v9.py           # Train XGBoost v9 (run once to generate model files)
│   ├── score_historical*.py    # Historical scoring utilities
│   └── create_sample_cache.py
│
├── archive/
│   └── ...                     # Old ANN (v8) and XGBoost v6 code (kept for reference)
│
└── dashboard2/
    ├── src/App.jsx             # Full React dashboard (single file)
    ├── public/                 # Static assets
    └── package.json
```

> **Note:** XGBoost model files (`xgboost_v9_clf15m.json`, `xgboost_v9_clf1h.json`, `xgboost_v9_scaler.pkl`) are not committed. Run `python training/xgboost_v9.py` once to generate them.

---

## Model Architecture

### Production: XGBoost v9

The live scoring pipeline uses an **XGBoost ensemble** trained on a 1578-dimensional feature vector:

| Feature group | Dimensions | Source |
|---------------|-----------|--------|
| CryptoBERT embedding | 768 | Frozen `ElKulako/cryptobert` |
| FinBERT embedding | 768 | Frozen `ProsusAI/finbert` |
| Sentiment (3-model ensemble) | 13 | CryptoBERT + FinBERT + RoBERTa |
| News-type probabilities | 11 | Cosine similarity to 11 prototypes |
| Macro / timing features | 8 | US/Asia hours, fear & greed index |
| RAG context | 10 | Qdrant nearest-neighbor lookup |

Two XGBoost classifiers are trained independently:
- `xgboost_v9_clf15m` — 15-minute impact probability
- `xgboost_v9_clf1h` — 1-hour impact probability

### Research: ANN (CryptoImpactNetV5)

An earlier 3-tower neural architecture (semantic + RAG + macro towers with cross-attention fusion, 6 output heads) is preserved in `archive/` for reference. It was superseded by XGBoost v9.

---

## ANN vs XGBoost v9 — Comparison

| | ANN (CryptoImpactNetV5) | XGBoost v9 |
|---|---|---|
| **Architecture** | 3-tower MLP + CrossAttention | Gradient-boosted trees (×2 classifiers) |
| **Text encoder** | CryptoBERT only (768d) | CryptoBERT + FinBERT (768 + 768d) |
| **Sentiment** | Single-model (CryptoBERT) | 3-model ensemble (CB + FB + RoBERTa) |
| **Feature size** | ~800 dim | 1578 dim |
| **News-type gating** | Learned NewsTypeGating layer | 11-dim type probability vector |
| **RAG integration** | Cross-attention (sem ↔ rag) | 10-dim RAG feature vector |
| **Outputs** | cls_15m, cls_1h, reg_15m, reg_1h, conf, direction | prob_15m, prob_1h |
| **Training** | Focal loss, AdamW, early stopping | XGBoost with threshold sweep on val set |
| **Inference speed** | Slower (transformer forward passes + MLP) | Fast (tree traversal) |
| **Interpretability** | Low (black box MLP) | Medium (feature importance available) |
| **15m ROC-AUC** | ~0.63 | **0.659** |
| **1h ROC-AUC** | ~0.61 | **0.632** |
| **Status** | Archived (`archive/production_system_v8.py`) | **Production** |

**Why XGBoost v9 replaced the ANN:**
- DualBERT features (CryptoBERT + FinBERT together) capture both crypto-domain and financial sentiment signals better than CryptoBERT alone
- 3-model sentiment ensemble reduces single-model noise
- XGBoost is more robust to small dataset size and less prone to overfitting than deep MLP heads
- Faster inference with no GPU dependency

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
