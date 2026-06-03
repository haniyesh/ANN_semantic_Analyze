# Multi-Stream Neural Architecture for Cryptocurrency News Impact Prediction
> Predicting whether a Bitcoin news headline will move the market — within 15 minutes and 1 hour — using a multi-stream neural network with chain-of-thought reasoning.
---
## Overview
This project explores a core question in computational finance: **can a neural network read cryptocurrency news and infer short-term price impact?**
Given a news headline, the model predicts binary impact labels at two horizons:
- **15-minute window** (threshold: ±0.3% price change)
- **1-hour window** (threshold: ±0.5% price change)
Rather than relying on a single signal, the architecture fuses multiple input streams — semantic text embeddings, chain-of-thought financial reasoning traces, historical-context retrieval, and market-state features — into a shared representation mapped to several output tasks.

---

## How to Run

### 1. Dashboard (Frontend)

```bash
cd dashboard
npm run dev
```

### 2. API (Backend)

```bash
cd api
python server.py
```

---

## Key Results
| Metric | 15-Minute | 1-Hour |
|---|---|---|
| ROC-AUC | 0.659 | 0.632 |
| F1 Score | 0.318 | 0.421 |
| Precision | 25.0% | 28.7% |
| Recall | 43.6% | 79.0% |
| Accuracy | 66.8% | 49.3% |
> **Note:** The majority-class baseline (always predicting "no impact") achieves 81.8% / 76.6% accuracy while catching **zero** impactful events (F1 = 0). ROC-AUC and F1 are the meaningful metrics here.

---

## Architecture
The model consists of **four parallel processing streams** fused through a learned aggregation step:
| Stream | Description |
|---|---|
| **Semantic stream** | Dense headline embedding from a frozen pre-trained language model (CryptoBERT) |
| **Chain-of-thought stream** | Encoded reasoning trace articulating the causal news-to-price mechanism |
| **Historical-context stream** | Retrieval summary from similar past news events (RAG-inspired) |
| **Contextual stream** | Market-state features: time of day, recent volatility, sentiment/category descriptors |
Fused representations feed into a shared MLP, which branches into:
- 2× binary classification heads (15-min and 1-hour impact)
- 2× regression heads (continuous price change magnitude)
- 1× direction head (up / down)
- 1× confidence head (self-estimated prediction reliability)
Total trainable parameters: **tens of thousands** (text encoder is frozen).

---

## Dataset
- **Source:** Cryptocurrency news headlines from major outlets, paired with BTC/USDT market data
- **Raw items:** ~70,000 headlines
- **After filtering:** 31,549 samples
- **Split:** Chronological (no lookahead bias) — 70 / 15 / 15
| Split | Samples |
|---|---|
| Train | 22,084 |
| Validation | 4,732 |
| Test | 4,733 |
**Class distribution (test set):**
- 15-min: 18.2% impactful, 81.8% non-impactful
- 1-hour: 23.4% impactful, 76.6% non-impactful

---

## Training
- **Loss:** Focal Loss (γ=2) for classification heads; MSE for regression; cross-entropy for direction
- **Optimizer:** AdamW (lr=3e-4, weight decay=1e-3)
- **Scheduler:** ReduceLROnPlateau (halves LR on plateau)
- **Early stopping:** patience=20 epochs on validation F1 (15-min head)
- **Max epochs:** 200
- **Decision thresholds:** Swept on validation set; optimal range 0.39–0.40 (below 0.5 due to class imbalance)

---

## Per-Category Performance (15-Minute Horizon)
| Category | N | Accuracy |
|---|---|---|
| Exchange | 12 | 83% |
| DeFi | 77 | 77% |
| Macroeconomic | 148 | 76% |
| ETF | 137 | 74% |
| Regulatory | 37 | 73% |
| Partnership | 99 | 73% |
| Market Analysis | 94 | 72% |
| Institutional | 116 | 72% |
| Mining | 9 | 67% |
Mining news produced **zero true positives** — likely because mining-related events affect Bitcoin through slow structural mechanisms rather than sharp short-term reactions.

---

## Chain-of-Thought Reasoning
A distinctive feature of this system is the integration of **chain-of-thought (CoT) financial reasoning**. For each headline, the model receives an explicit causal trace linking the news event to an expected market reaction. Example:
> *"ETF approval increases institutional access to Bitcoin, expanding the buyer base and reducing friction for large capital inflows — likely generating upward price pressure within minutes to hours."*
CoT reasoning serves two roles:
1. **Regularization** — grounds predictions in causal mechanisms, reducing false positives on emotionally charged but financially irrelevant language
2. **Interpretability** — human analysts can inspect the reasoning trace to audit model decisions

---

## Limitations
- **Single asset:** Trained and evaluated on Bitcoin only; transferability to other assets is unverified
- **Headlines only:** Full article body is not used; important context sometimes lives below the headline
- **Temporal drift:** Crypto market dynamics evolve rapidly; periodic retraining is likely necessary
- **Label noise:** Price movements are attributed to the most recent headline, which may not be the true cause
- **Reaction speed:** Algorithmic traders act in milliseconds; by pipeline ingestion time, initial market reactions may have already concluded

---

## Future Work
- Multi-asset extension (Ethereum, major altcoins)
- Full article text and social-media signal integration
- Post-hoc confidence calibration (temperature scaling)
- Fine-tuned CoT reasoning on expert trader annotations

---

## References
A full reference list is provided in the accompanying paper. Key works include:
- Focal Loss — Lin et al., ICCV 2017
- Chain-of-Thought Prompting — Wei et al., NeurIPS 2022
- Retrieval-Augmented Generation — Lewis et al., NeurIPS 2020
- CryptoBERT — ElKulako, Hugging Face 2022
- Multitask Learning — Caruana, Machine Learning 1997

---

## Citation
If you use this work, please cite:
```
@article{shakibayi2026multistream,
  title     = {A Multi-Stream Neural Architecture for Short-Term Cryptocurrency Price Impact Prediction from News},
  author    = {Shakibayi Senobari, Haniye},
  year      = {2026},
  institution = {Department of Artificial Intelligence, Bahçeşehir University, Istanbul, Turkey}
}
```
---
*Department of Artificial Intelligence, Bahçeşehir University, Istanbul, Turkey — 2026*
