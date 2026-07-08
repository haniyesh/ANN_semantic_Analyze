# Sentiment Analysis — Crypto News Market Impact

> A real-time system that monitors cryptocurrency news, scores each headline for predicted BTC price impact, and streams results to a live dashboard with candlestick charts.

> **Status: Research prototype / academic thesis.** See [Known Issues & Limitations](#known-issues--limitations) before drawing conclusions from reported metrics.

---

## What This Does

The system listens to crypto news Telegram channels in real time, passes each headline through a multi-model NLP pipeline, predicts whether it will move BTC price within 15 minutes or 1 hour, and pushes scored results to a live dashboard.

**Two layers:**
1. **Live pipeline** — Telegram listener → sentiment analysis → feature extraction → impact scoring → live dashboard
2. **Prediction models** — XGBoost and ANN classifiers trained to predict whether a BTC headline causes a ≥0.3% price move within 15 minutes or ≥0.5% within 1 hour

---

## Dataset

| Property | Value |
|---|---|
| Total headlines | 76,214 |
| Date range | Aug 2021 – Jul 2026 |
| Sources | 7 Telegram channels |
| Impactful (15-min, ≥0.3% BTC) | 17.7% — 13,498 items |
| Impactful (1-hour, ≥0.5% BTC) | 22.8% — 17,395 items |
| Split | Chronological 70 / 15 / 15 |

**Channels:** CoinTelegraph (22,608) · The Block (11,631) · CryptoNews (10,382) · Google News (9,659) · CoinDesk (9,429) · CryptoPotato (7,565) · WatcherGuru (4,940)

**Split (chronological — no temporal leakage):**

| Subset | Items | Period |
|---|---|---|
| Train | 53,349 | Aug 2021 → Jul 2025 |
| Validation | 11,432 | Jul 2025 → Dec 2025 |
| Test | 11,433 | Dec 2025 → Jul 2026 |

> Nominal chronological split of 76,214 (70/15/15). The test set is evaluated on 11,365 rows (68 dropped from the nominal 11,433 for missing forward prices).

---

## Model Architecture

### Feature Vectors

The two sentiment branches share all blocks except sentiment dimensionality:

| Group | BERT branch | Groq branch | Description |
|---|---|---|---|
| CryptoBERT embedding | 768 | 768 | Domain-adapted BERT for crypto text |
| FinBERT embedding | 768 | 768 | Finance-domain BERT |
| Sentiment | **13** | **6** | BERT: cb+fb+rb probs (9) + net\_agreement + score + weight + conf; Groq: 3-class label + score + weight + conf |
| News-type classification | 11 | 11 | Cosine similarity to 11 category prototypes |
| Macro timing features | 8 | 8 | Weekend, low-liquidity hours, US/Asia hours, FOMC week, price context |
| RAG / historical context | 1 | 1 | Zeroed in skip-RAG training mode |
| **Total** | **1,569** | **1,562** | |

### Model Variants

Four classifiers — two architectures × two sentiment backbones:

| Model | Params | Feature dims | Sentiment source |
|---|---|---|---|
| **XGBoost + CryptoBERT** | n\_estimators=500 | 1,569 | CryptoBERT + FinBERT + RoBERTa-Twitter ensemble |
| **XGBoost + Groq/Llama** | n\_estimators=500 | 1,562 | Llama-3.3-70B via Groq API |
| **ANN + CryptoBERT** | 54,615 | 1,569 | CryptoBERT + FinBERT + RoBERTa-Twitter ensemble |
| **ANN + Groq/Llama** | 54,447 | 1,562 | Llama-3.3-70B via Groq API |

---

## Results

### 15-Minute Horizon (test set, 11,365 rows, 18.2% positive)

| Model | Decision thr. | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **XGBoost + CryptoBERT** | 0.50 | **76.1%** | **40.6%** | **67.6%** | **50.7%** | 80.7% |
| XGBoost + Groq/Llama | 0.465 | 73.9% | 38.2% | 70.7% | 49.6% | **81.0%** |
| ANN + CryptoBERT | 0.57 | 70.5% | 33.7% | 64.1% | 44.1% | 73.9% |
| ANN + Groq/Llama | 0.59 | 74.1% | 35.8% | 53.3% | 42.8% | 73.4% |

### 1-Hour Horizon (test set, 11,365 rows, 22.2% positive)

| Model | Decision thr. | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **XGBoost + CryptoBERT** | 0.39 | 61.2% | 33.0% | **72.8%** | **45.4%** | **71.2%** |
| XGBoost + Groq/Llama | 0.42 | **63.6%** | **33.8%** | 66.5% | 44.8% | 70.9% |
| ANN + CryptoBERT | 0.495 | 63.7% | 33.6% | 64.8% | 44.3% | 69.0% |
| ANN + Groq/Llama | 0.49 | 64.2% | **33.8%** | 64.0% | 44.3% | 69.1% |

### Reference points (15-min, test set)

| | F1 | Accuracy | Note |
|---|---|---|---|
| Majority class (always non-impactful) | 0.000 | 81.8% | Causal baseline |
| Random classifier | 18.1% | 70.0% | Causal baseline |
| Always predict impactful | 30.8% | 18.2% | Causal baseline |
| Realized-return oracle | 53.5% | 68.3% | **Non-causal** — thresholds the same realized return that defines the label; upper-bound reference only |
| **XGBoost + CryptoBERT** | **50.7%** | **76.1%** | Best causal model |

---

## Live Dashboard

A React dashboard that connects to the backend via WebSocket and REST.

**Features:**
- Live BTC/ETH candlestick chart with news event markers
- Hover over chart dots to see the headline and impact score
- News cards ranked by predicted market impact
- Real-time BTC momentum gauge
- Calendar view — click any date to browse historical news and price markers
- 3-month rolling history window (live cache) + 6-month historical CSV

**Impact tiers** (derived from `config.py` thresholds, fetched at runtime):

| Tier | Score gate | Meaning |
|------|-----------|---------|
| Hot | ≥ 0.80 | High confidence strong signal |
| Medium | ≥ 0.55 | Moderate predicted impact |
| Show | ≥ 0.30 | Low but notable signal |
| Hidden | < 0.30 | Filtered from main feed |

---

## How to Run

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Copy `.env.example` to `.env` and fill in credentials:

```
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_CHANNELS=channel1,channel2
TELEGRAM_CHANNEL_ID=...          # Telegram channel ID for alerts
BOT_TOKEN=...                    # Telegram bot token
GROQ_API_KEY=...                 # Primary Groq key (Llama-3.3-70B sentiment)
GROQ_API_KEY_2=...               # Optional second key for rotation
INGEST_API_KEY=...               # Shared secret: POST /news (bot → API)
ALLOWED_ORIGINS=http://localhost:5173
DATABASE_URL=...                 # Optional PostgreSQL; runs in cache-only mode without it
```

> `INGEST_API_KEY` must be set identically in `.env` for both `main.py` and `server.py`. Without it, `POST /news` returns `503` (fails closed).

### 1. Start the API server

```bash
cd api
python server.py
# or: uvicorn api.server:app --host 0.0.0.0 --port 8000
```

On startup the server loads the last **3 months** of news from `storage/news_cache.json` and the last 6 months from the historical CSV.

### 2. Start the news pipeline

```bash
python main.py
```

On first run: prompts for a Telegram login code (creates `telegram_session` file).  
On subsequent starts: **resumes from the last stored Telegram message ID** per channel — no duplicate re-ingestion, no fixed look-back window. Falls back to a 30-day window for channels with no stored history.

### 3. Start the dashboard

```bash
cd dashboard2
npm install
npm run dev        # development (hot-reload)
npm run build      # production build → dist/
```

---

## Project Structure

```
├── main.py                           # Entry point: Telegram → score → API
├── config.py                         # All thresholds and impact tiers (single source of truth)
├── requirements.txt
├── .env.example
│
├── api/
│   └── server.py                     # FastAPI: REST + WebSocket + Binance proxy
│
├── bot/
│   ├── telegram_listener.py          # Cursor-based backfill + real-time listener
│   └── telegram_alert.py             # Pushes high-impact alerts to Telegram
│
├── pipeline/
│   ├── spam_filter.py                # Pre-filters incoming headlines
│   ├── rag_news.py                   # Similar-news retrieval (Qdrant)
│   ├── processor.py
│   └── reduce_noise.py               # Channel and noise filters
│
├── services/
│   ├── sentiment_score.py            # Multi-model sentiment ensemble (BERT + Groq)
│   ├── price_fetcher.py              # Live BTC/ETH price tracking
│   └── ...
│
├── storage/
│   ├── news_cache.json               # Live rolling 3-month news cache
│   ├── database.py
│   └── cache.py
│
├── training/
│   ├── xgboost_train_bert.py         # Train XGBoost + CryptoBERT model
│   ├── xgboost_train_groq.py         # Train XGBoost + Groq/Llama model
│   ├── ann_train.py                  # Train ANN (--sentiment bert or --sentiment groq)
│   ├── score_historical_xgb.py       # Score historical CSV with BERT model
│   ├── score_historical_xgb_groq.py  # Score historical CSV with Groq model
│   ├── build_groq_csv.py             # Build Groq sentiment cache from CSV
│   ├── create_sample_cache.py        # Generate a sample cache for testing
│   └── score_groq_only.py            # Re-score cached items with Groq sentiment
│
└── dashboard2/
    ├── src/App.tsx                   # React dashboard (single-file)
    ├── public/
    └── package.json
```

> **Note:** Model files, embedding caches, and training data are not committed. They must be generated locally before training (see [Reproducibility](#reproducibility)).

---

## Reproducibility

Reproducing results from a clean clone is **not yet fully possible** — several inputs are not committed:

- `news_cleaned_filtered_scored.csv` — training table (BERT branch); built from Telegram-collected data
- `news_cleaned_filtered_scored_groq.csv` — training table (Groq branch)
- `cryptobert_embeddings_cache.npy`, `finbert_embeddings_cache.npy` — generated by training scripts
- `xgb_feature_scaler_bert.pkl`, `xgb_feature_scaler_groq.pkl` — fitted scalers
- `xgb_impact_clf_15m_bert.json`, `xgb_impact_clf_15m_groq.json`, etc. — XGBoost model files
- `ann_bert_weights.pt`, `ann_groq_weights.pt` — ANN weights

A Qdrant instance (`QDRANT_URL` + `QDRANT_API_KEY`) is required for RAG retrieval. Train without it using `--skip-rag` (RAG feature is zeroed to 1 dim; all four reported models were trained in this mode).

---

## Known Issues & Limitations

1. ~~**Random split**~~ **Fixed.** Split is now strictly chronological (train ends before val starts, val ends before test starts).

2. ~~**RAG index leakage**~~ **Fixed.** Qdrant index is built from training rows only; val/test rows are excluded.

3. ~~**No baselines**~~ **Fixed.** Majority-class, random, always-impactful baselines and a realized-return oracle reference are evaluated on the test set.

4. ~~**Train/serve skew**~~ **Largely fixed.** Sentiment, price context, and RAG similarity are consistent between training and serving.

5. **BTC only.** Both classifiers are trained on Bitcoin price moves. ETH headlines are scored with the BTC model — treat ETH predictions as illustrative.

6. **Label noise at short horizons.** The 15-min label threshold (0.3%) is aggressive — BTC regularly moves ≥0.3% from normal intraday volatility alone. A portion of positive-class labels may be noise rather than news-driven signal.

7. **Realized-return oracle is not a causal baseline.** The oracle row in the results table thresholds the same realized return that defines the label, so it has perfect look-ahead. It is an upper-bound reference, not a comparable predictor.

8. **Latency.** The pipeline processes headlines after they appear on Telegram. Fast price reactions may complete before scoring finishes.

9. **Training artifacts not committed.** Embedding caches and model files are not in the repo.

10. **Headlines only.** Full article body is not used; only the headline/lead sentence from Telegram.

11. **Crypto market dynamics shift rapidly.** Periodic retraining is recommended as market conditions and relevant news types evolve.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | XGBoost, PyTorch |
| NLP Models | CryptoBERT, FinBERT, RoBERTa-Twitter, Llama-3.3-70B (Groq) |
| Backend | FastAPI, Uvicorn, WebSocket |
| Frontend | React 18, Vite, Lightweight Charts |
| Data collection | Telegram (Telethon), Binance API |
| Vector DB | Qdrant (optional, for RAG similarity search) |
| Database | PostgreSQL (optional; falls back to JSON cache) |

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
