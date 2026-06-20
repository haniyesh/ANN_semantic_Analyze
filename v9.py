"""
Production Multi-Tower Fusion System — v9
==========================================
Fixes vs previous versions:
  - No macro_features.py dependency (all inline)
  - THRESHOLD_15M = 0.3  (18% positive — more learnable)
  - THRESHOLD_1H  = 0.5  (20% positive)
  - MONTHLY_SEED  = 43
  - Macro tower: 5 dims (weekend/low_liq/us_hours/asia_hours/fomc)
  - Import from rag_news (not rag_qdrant)
  - Progress bar for CryptoBERT embedding

v5 UPGRADE — Cross-Attention Fusion + Macro Injection + NewsTypeGating + Anti-Overfitting:
  - CrossAttentionFusion module replaces simple gating
  - sem ↔ rag   bidirectional cross-attention
  - sem ↔ market bidirectional cross-attention
  - macro injected INTO sem BEFORE cross-attention (not after)
  - NewsTypeGating: type_probs gate sem_tower output per news type
  - Anti-overfitting: dropout=0.3, EMB_PROJ_DIMS=16
  - POS_WEIGHT_15M=3.0 (below balanced 4.23), POS_WEIGHT_1H=2.5 (below balanced 2.85)
  - MIN_PRECISION=0.20, fallback threshold=0.50
"""

import os
import json
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, r2_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

from pipeline.rag_news import build_rag_features_qdrant, query_single

load_dotenv()

HERE          = Path(__file__).parent
SENTIMENT_CSV = HERE / "news_cleaned_filtered_scored.csv"
CRYPTOBERT_CACHE = HERE / "cryptobert_v8_pipeline.npy"
FINBERT_CACHE    = HERE / "finbert_v9_pipeline.npy"
FEAR_GREED_CACHE = HERE / "fear_greed_cache.json"
FINBERT_MODEL    = "ProsusAI/finbert"
DUAL_EMB_DIM     = 768 + 768   # CryptoBERT + FinBERT concatenated
MODEL_PATH    = HERE / "production_system_v9.pt"
RESULTS_PATH  = HERE / "production_results_v9.json"

THRESHOLD_15M = 0.3
THRESHOLD_1H  = 0.5
REG_SCALE     = 2.0
LEARNING_RATE = 0.0003
WEIGHT_DECAY  = 0.001
EPOCHS        = 200
PATIENCE      = 20
BATCH_SIZE    = 64
GRAD_CLIP     = 1.0
REPLAY_CAP    = 5000
MIN_PRECISION  = 0.20
POS_WEIGHT_15M = 3.0   # below balanced (4.23) → optimal constant 0.41 < fallback 0.50
POS_WEIGHT_1H  = 2.5   # below balanced (2.85) → optimal constant 0.47 < fallback 0.50
EMB_PROJ_DIMS = 16
MONTHLY_SEED  = 43

device = torch.device("cpu")
print(f"  Device: {device}")

NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis",
]

_NEWS_TYPE_PROTOTYPES_TEXT = {
    "regulatory": [
        "SEC charges crypto exchange securities violations",
        "government bans cryptocurrency trading country",
        "court rules against crypto company lawsuit",
        "regulatory compliance required exchange",
    ],
    "etf": [
        "Bitcoin ETF approved by SEC trading",
        "spot bitcoin fund launches stock exchange",
        "BlackRock files ETF application approval",
        "grayscale bitcoin trust conversion",
    ],
    "hack": [
        "crypto exchange hacked millions stolen",
        "DeFi protocol exploited flash loan attack",
        "security breach drains user funds wallet",
    ],
    "macro_economic": [
        "Federal Reserve raises interest rates decision",
        "inflation data CPI report released",
        "GDP growth recession fears mount",
        "FOMC rate decision monetary policy",
    ],
    "exchange": [
        "Binance lists new cryptocurrency token",
        "Coinbase delists token regulatory concerns",
        "exchange trading volume record high",
        "new trading pair launched users",
    ],
    "defi": [
        "DeFi protocol TVL record liquidity",
        "Uniswap launches new version features",
        "yield farming liquidity pool AMM",
        "decentralized exchange volume surpasses",
    ],
    "mining": [
        "Bitcoin mining difficulty adjusts record",
        "miner capitulation hashrate drops significantly",
        "Bitcoin halving block reward reduced",
        "mining company orders ASIC machines",
    ],
    "institutional": [
        "MicroStrategy purchases Bitcoin treasury reserve",
        "hedge fund allocates Bitcoin portfolio",
        "corporate treasury adds BTC asset",
        "institutional investors increase holdings",
    ],
    "technical": [
        "Bitcoin network upgrade soft fork activates",
        "Ethereum developers confirm upgrade date",
        "layer two scaling solution launches mainnet",
        "protocol implements new consensus mechanism",
    ],
    "partnership": [
        "crypto company partnership bank deal",
        "blockchain firm integrates payment processor",
        "exchange signs deal financial institution",
        "technology company acquires crypto startup",
    ],
    "market_analysis": [
        "Bitcoin price analysis bullish breakout target",
        "technical analysis support level tested",
        "market sentiment fear greed index extreme",
        "on chain data accumulation long term",
    ],
}

_proto_matrix = None


def _build_proto_matrix():
    global _proto_matrix
    if _proto_matrix is not None:
        return
    print("  Building news type prototype embeddings (one-time)...")
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model     = transformers.AutoModel.from_pretrained("ElKulako/cryptobert").eval()
    vecs = []
    with torch.no_grad():
        for label in NEWS_TYPE_LABELS:
            texts  = _NEWS_TYPE_PROTOTYPES_TEXT[label]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            out    = model(**inputs).last_hidden_state[:, 0, :].mean(dim=0)
            vecs.append(out)
    _proto_matrix = F.normalize(torch.stack(vecs), dim=1)


def crypto_news_type_classify(
    embeddings: np.ndarray,
    return_probs: bool = False,
):
    _build_proto_matrix()
    emb_t   = F.normalize(torch.FloatTensor(embeddings), dim=1)
    sims    = torch.mm(emb_t, _proto_matrix.T)
    probs   = torch.softmax(sims * 5.0, dim=1).numpy().astype(np.float32)
    indices = probs.argmax(axis=1)
    if return_probs:
        return indices, probs
    return indices


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def load_data() -> pd.DataFrame:
    print("[1/8] LOAD DATA")
    df = pd.read_csv(SENTIMENT_CSV, low_memory=False)
    for col in df.columns:
        orig = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df[col] = orig

    orig = len(df)
    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df.drop_duplicates(subset=["title", "published", "channel"])

    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["is_impactful_15m"] = (df["abs_change_15m"] > THRESHOLD_15M).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)

    df["btc_change_1h"]   = (df["btc_price_1h"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_1h"]   = df["btc_change_1h"].abs()
    df["is_impactful_1h"] = (df["abs_change_1h"] > THRESHOLD_1H).astype(int)
    df["direction_1h"]    = (df["btc_change_1h"] > 0).astype(int)

    df["confidence_label"] = (3.0 * df["abs_change_15m"]).clip(0, 3)

    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    print(f"  Rows: {len(df):,} (from {orig:,})")
    print(f"  Impactful 15m: {df['is_impactful_15m'].mean():.1f}%  1h: {df['is_impactful_1h'].mean():.1f}%")
    return df


def compute_cryptobert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if CRYPTOBERT_CACHE.exists():
        emb = np.load(CRYPTOBERT_CACHE).astype(np.float32)
        if len(emb) == len(df):
            print("  CryptoBERT cache hit")
            return emb

    print(f"  Computing CryptoBERT embeddings for {len(df):,} rows...")
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model     = transformers.AutoModel.from_pretrained("ElKulako/cryptobert").eval()
    titles = df["title"].fillna("").tolist()
    embs   = []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)} ({i * 100 // len(titles)}%)...")
            batch  = titles[i : i + 32]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out    = model(**inputs).last_hidden_state[:, 0, :].numpy()
            embs.append(out)
    emb = np.vstack(embs).astype(np.float32)
    np.save(CRYPTOBERT_CACHE, emb)
    print(f"  CryptoBERT done: {emb.shape}")
    return emb


def compute_finbert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if FINBERT_CACHE.exists():
        emb = np.load(FINBERT_CACHE).astype(np.float32)
        if len(emb) == len(df):
            print("  FinBERT cache hit")
            return emb

    print(f"  Computing FinBERT embeddings for {len(df):,} rows...")
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = transformers.AutoModel.from_pretrained(FINBERT_MODEL).eval()
    titles    = df["title"].fillna("").tolist()
    embs      = []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)} ({i * 100 // len(titles)}%)...")
            batch  = titles[i : i + 32]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out    = model(**inputs).last_hidden_state[:, 0, :].numpy()
            embs.append(out)
    emb = np.vstack(embs).astype(np.float32)
    np.save(FINBERT_CACHE, emb)
    print(f"  FinBERT done: {emb.shape}")
    return emb


def fetch_fear_greed_index(published_series: pd.Series) -> np.ndarray:
    """Map each news timestamp to Alternative.me fear/greed index (0–1)."""
    fg_map = {}
    if FEAR_GREED_CACHE.exists():
        try:
            with open(FEAR_GREED_CACHE) as f:
                items = json.load(f)
            for item in items:
                date = pd.Timestamp(int(item["timestamp"]), unit="s").date()
                fg_map[date] = float(item["value"]) / 100.0
        except Exception:
            pass

    if not fg_map:
        try:
            import urllib.request
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            print("  Fetching Fear/Greed index from Alternative.me...")
            with urllib.request.urlopen(url, timeout=15) as r:
                items = json.loads(r.read())["data"]
            with open(FEAR_GREED_CACHE, "w") as f:
                json.dump(items, f)
            for item in items:
                date = pd.Timestamp(int(item["timestamp"]), unit="s").date()
                fg_map[date] = float(item["value"]) / 100.0
            print(f"  Fear/Greed: {len(fg_map)} days cached")
        except Exception as e:
            print(f"  Fear/Greed unavailable ({e}), using neutral 0.5")

    return np.array(
        [fg_map.get(pd.Timestamp(ts).date(), 0.5) for ts in published_series],
        dtype=np.float32,
    )


def compute_price_context(df: pd.DataFrame) -> np.ndarray:
    """
    3 price-context features (df must be sorted by published time):
      btc_vol   — rolling std of |btc_change_15m| over prior 20 items (prior-hour vol proxy)
      btc_mom   — rolling mean of btc_change_15m over prior 5 items  (short momentum)
      fear_greed — daily Alternative.me fear/greed index (0=extreme fear, 1=extreme greed)
    """
    changes = pd.Series(df["btc_change_15m"].values.astype(np.float32))
    shifted = changes.shift(1)

    btc_vol = shifted.rolling(window=20, min_periods=2).std().fillna(0).values.astype(np.float32)
    btc_mom = shifted.rolling(window=5,  min_periods=1).mean().fillna(0).values.astype(np.float32)
    fg      = fetch_fear_greed_index(df["published"])

    return np.column_stack([btc_vol, btc_mom, fg]).astype(np.float32)


def compute_channel_impact_rates(df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    train_df  = df.iloc[train_idx]
    rates     = train_df.groupby("channel")["is_impactful_15m"].mean().to_dict()
    mean_rate = float(df["is_impactful_15m"].mean())
    print(f"  Channel impact rates (training data only):")
    for ch, rate in sorted(rates.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        n   = int((df["channel"] == ch).sum())
        tag = "<30" if n < 30 else "   "
        print(f"    {ch:<30} {tag}: {rate:.1%}  {bar}")
    print(f"  (dataset mean)")
    return rates


def build_macro_features(df: pd.DataFrame) -> np.ndarray:
    hour = df["published"].dt.hour.values
    dow  = df["published"].dt.dayofweek.values

    def _col(arr):
        return arr.astype(float)

    is_weekend       = _col(dow >= 5)
    is_low_liquidity = _col((hour >= 2) & (hour <= 6))
    is_us_hours      = _col((hour >= 13) & (hour <= 21))
    is_asia_hours    = _col((hour >= 0) & (hour <= 8))
    fomc_week        = np.zeros(len(df))
    if "fomc_week" in df.columns:
        fomc_week = df["fomc_week"].fillna(0).values.astype(float)

    macro = np.column_stack([
        is_weekend, is_low_liquidity, is_us_hours, is_asia_hours, fomc_week,
    ]).astype(np.float32)
    print(f"  Macro : {macro.shape[1]} dims (weekend/low_liq/us/asia/fomc)")
    return macro


NEWS_TYPE_LOSS_WEIGHTS = {
    "regulatory":    2.0,
    "etf":           2.0,
    "hack":          2.0,
    "macro_economic": 2.0,
    "institutional": 1.8,
    "mining":        1.5,
    "exchange":      1.3,
    "defi":          1.2,
    "technical":     1.1,
    "partnership":   1.0,
    "market_analysis": 0.8,
}


def compute_type_weights(type_indices: np.ndarray) -> np.ndarray:
    """Map news type index → loss weight per sample."""
    weights = np.ones(len(type_indices), dtype=np.float32)
    for i, idx in enumerate(type_indices):
        label = NEWS_TYPE_LABELS[int(idx)]
        weights[i] = NEWS_TYPE_LOSS_WEIGHTS.get(label, 1.0)
    return weights


def build_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    emb=None,
) -> tuple:
    print("[2/8] FEATURE ENGINEERING")
    cb_emb = compute_cryptobert_embeddings(df)   # (N, 768)
    fb_emb = compute_finbert_embeddings(df)       # (N, 768)

    # Type classification uses CryptoBERT only (domain-specific similarity)
    type_indices, type_probs = crypto_news_type_classify(cb_emb, return_probs=True)

    # Auto-detect ensemble columns (9 probs from 3 models + agreement)
    ensemble_cols = [
        "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
        "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
        "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
        "net_agreement",
    ]
    legacy_cols = ["sentiment_score", "weight", "confidence",
                   "prob_positive", "prob_negative", "prob_neutral"]

    if all(c in df.columns for c in ensemble_cols):
        extra = ["sentiment_score", "weight", "confidence"]
        if "sentiment_reliable" in df.columns:
            # Convert bool to float for the neural network
            df["sentiment_reliable"] = df["sentiment_reliable"].astype(float)
            extra.append("sentiment_reliable")
        sent_cols = ensemble_cols + extra
        print(f"  Using ensemble sentiment features ({len(sent_cols)} dims)")
    else:
        sent_cols = legacy_cols
        print(f"  Using legacy sentiment features ({len(sent_cols)} dims)")

    sent_df  = df[sent_cols].fillna(0).values.astype(np.float32)
    semantic = np.hstack([cb_emb, fb_emb, sent_df, type_probs]).astype(np.float32)
    print(f"  Semantic: {semantic.shape[1]} dims (CryptoBERT+FinBERT {DUAL_EMB_DIM} + sent{len(sent_cols)} + type11)")

    # Per-sample type loss weights
    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        csv_types = df["news_type"].fillna("market_analysis").values
        type_w    = np.array(
            [NEWS_TYPE_LOSS_WEIGHTS.get(t, 1.0) for t in csv_types],
            dtype=np.float32,
        )
        type_counts = {}
        for t in csv_types:
            type_counts[t] = type_counts.get(t, 0) + 1
    else:
        type_w      = compute_type_weights(type_indices)
        type_counts = {}
        for i, idx in enumerate(type_indices):
            label = NEWS_TYPE_LABELS[int(idx)]
            type_counts[label] = type_counts.get(label, 0) + 1

    print("  News type distribution (loss weight):")
    for label, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        w = NEWS_TYPE_LOSS_WEIGHTS.get(label, 1.0)
        print(f"    {label:<18}: {cnt:>6,}  weight={w:.1f}x")

    timing_macro = build_macro_features(df)            # (N, 5)
    price_ctx    = compute_price_context(df)            # (N, 3) — vol/momentum/fear-greed
    macro        = np.hstack([timing_macro, price_ctx]).astype(np.float32)
    print(f"  Macro: {macro.shape[1]} dims (5 timing + 3 price context: vol/momentum/fear-greed)")

    ch_rates = compute_channel_impact_rates(df, train_idx)

    print("  RAG: Qdrant Cloud + 5-feature macro-conditioned re-weighting")
    rag, rag_details = build_rag_features_qdrant(
        df,
        channel_impact_rates=ch_rates,
    )
    print(f"  RAG features: {rag.shape[1]}")

    return semantic, macro, rag, rag_details, type_w, type_indices, type_probs


# ── Neural Network Modules ─────────────────────────────────────────────────────

class EmbProjection(nn.Module):
    def __init__(self, in_dim: int = DUAL_EMB_DIM, out_dim: int = EMB_PROJ_DIMS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class SmallTower(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between two feature towers.

    Direction 1 — a queries b:
        "What part of b is relevant to what a is saying?"
        e.g. semantic asks RAG: "which past similar news matters for this headline?"

    Direction 2 — b queries a:
        "What part of a is relevant to what b is saying?"
        e.g. RAG asks semantic: "which aspect of this headline matches what I've seen before?"
    """

    def __init__(self, dim_a: int, dim_b: int, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.proj_a   = nn.Linear(dim_a, d_model)
        self.proj_b   = nn.Linear(dim_b, d_model)
        self.attn_a2b = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.attn_b2a = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.norm_a  = nn.LayerNorm(d_model)
        self.norm_b  = nn.LayerNorm(d_model)
        self.out_dim = d_model * 2

    def forward(self, a, b):
        """
        a: (B, dim_a)
        b: (B, dim_b)
        returns: (B, d_model * 2)
        """
        a_proj = self.proj_a(a).unsqueeze(1)  # (B, 1, d_model)
        b_proj = self.proj_b(b).unsqueeze(1)  # (B, 1, d_model)

        a_enriched, _ = self.attn_a2b(query=a_proj, key=b_proj, value=b_proj)
        a_out = self.norm_a((a_proj + a_enriched).squeeze(1))

        b_enriched, _ = self.attn_b2a(query=b_proj, key=a_proj, value=a_proj)
        b_out = self.norm_b((b_proj + b_enriched).squeeze(1))

        return torch.cat([a_out, b_out], dim=1)


class NewsTypeGating(nn.Module):
    """
    Applies a learned soft gate to the semantic embedding
    conditioned on news type probabilities.

    Problem it solves:
        Without this, CryptoBERT embedding is treated identically for ALL
        news types. A regulatory headline and a market analysis headline look
        different in embedding space but the model can't exploit that structure.
        This gate learns to amplify/suppress dimensions of the semantic vector
        based on what type of news it is.
    """

    def __init__(self, type_dim: int, sem_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(type_dim, sem_dim),
            nn.Sigmoid(),
        )

    def forward(self, sem_out, type_probs):
        """
        sem_out    : (B, sem_dim) — output of sem_tower
        type_probs : (B, 11)     — news type probability distribution
        returns    : (B, sem_dim) — gated semantic features
        """
        gate = self.gate_net(type_probs)
        return sem_out * gate


class CryptoImpactNetV5(nn.Module):
    """
    3-Tower architecture with Cross-Attention Fusion + Macro Injection + NewsTypeGating.
    Market momentum/volatility removed — model is sentiment-focused.

    Towers:
        sem_tower   (semantic + sentiment + type → 24)
        rag_tower   (RAG context → 6)
        mac_tower   (macro timing → 6)

    Flow:
        sem_tower(sem) → s (24)
        news_type_gate(s, type_probs) → s (24)    [type-aware gating]
        s_ctx = cat(s, mac_tower(macro))  → 30    [time-aware query]
        cross_sem_rag(s_ctx, rag_tower(rag)) → 24 [cross-attention]
        fusion(24) → 12
        heads → cls_15m, cls_1h, reg_15m, reg_1h, conf, direction
    """

    def __init__(
        self,
        sem_dim:   int,
        rag_dim:   int = 10,
        macro_dim: int = 5,
        emb_proj:  "EmbProjection | None" = None,
    ):
        super().__init__()
        self.emb_proj = emb_proj

        sem_in = sem_dim - DUAL_EMB_DIM + EMB_PROJ_DIMS if emb_proj else sem_dim

        self.sem_tower     = SmallTower(sem_in,    24, dropout=0.3)
        self.rag_tower     = SmallTower(rag_dim,    6, dropout=0.4)
        self.mac_tower     = SmallTower(macro_dim,  6, dropout=0.4)

        self.news_type_gate = NewsTypeGating(type_dim=11, sem_dim=24)

        # s_ctx (24+6=30) ↔ rag (6) → 24
        self.cross_sem_rag = CrossAttentionFusion(dim_a=30, dim_b=6, d_model=12, num_heads=2)

        self.fusion = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 12),
            nn.ReLU(),
        )

        self.head_cls_15m = nn.Linear(12, 1)
        self.head_cls_1h  = nn.Linear(12, 1)
        self.head_reg_15m = nn.Linear(12, 1)
        self.head_reg_1h  = nn.Linear(12, 1)
        self.head_conf    = nn.Linear(12, 1)
        self.head_dir     = nn.Linear(12, 2)

    def forward(self, sem, rag, macro):
        if self.emb_proj:
            sem = torch.cat([
                self.emb_proj(sem[:, :DUAL_EMB_DIM]),   # CryptoBERT+FinBERT → 16
                sem[:, DUAL_EMB_DIM:],                   # sentiment + type probs
            ], dim=1)

        s = self.sem_tower(sem)
        r = self.rag_tower(rag)
        m = self.mac_tower(macro)

        type_probs = sem[:, -11:]
        s = self.news_type_gate(s, type_probs)

        s_ctx   = torch.cat([s, m], dim=1)      # 30 (24+6)
        sem_rag = self.cross_sem_rag(s_ctx, r)  # 24

        fused = self.fusion(sem_rag)             # 12

        return {
            "cls_15m":   self.head_cls_15m(fused).squeeze(1),
            "cls_1h":    self.head_cls_1h(fused).squeeze(1),
            "reg_15m":   self.head_reg_15m(fused).squeeze(1),
            "reg_1h":    self.head_reg_1h(fused).squeeze(1),
            "conf":      self.head_conf(fused).squeeze(1),
            "direction": self.head_dir(fused),
        }


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, sem, rag, macro, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf):
        self.buffer.append((sem, rag, macro, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf))

    def sample(self, n):
        idx   = np.random.choice(len(self.buffer), n)
        batch = [self.buffer[i] for i in range(len(self.buffer))]
        batch = [batch[i] for i in idx]
        return [np.array([b[j] for b in batch]) for j in range(9)]

    def __len__(self):
        return len(self.buffer)


class Trainer:
    def __init__(self, model, _=None, sem_dim=785):
        self.model     = model.to(device)
        self.sem_dim   = sem_dim
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS,
        )
        self.scalers        = [StandardScaler() for _ in range(3)]
        self.best_val_loss  = float("inf")
        self.patience_counter = 0
        self.threshold_15m  = 0.5
        self.threshold_1h   = 0.5
        self.replay         = ReplayBuffer(REPLAY_CAP)

    def fit_scalers(self, sem, rag, macro):
        for s, x in zip(self.scalers, [sem, rag, macro]):
            s.fit(x)

    def scale(self, sem, rag, macro):
        return tuple(s.transform(x) for s, x in zip(self.scalers, [sem, rag, macro]))

    def _loss(self, out, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf, type_w=None):
        """
        type_w: (B,) per-sample weight based on news type.
        High-impact types (regulatory/etf/hack/macro/institutional) get 2x weight
        so the model learns from them more than generic market analysis.
        """
        pw15     = torch.tensor([POS_WEIGHT_15M]).to(device)
        pw1h     = torch.tensor([POS_WEIGHT_1H]).to(device)
        dir_mask = (y_r15.abs() > THRESHOLD_15M).float()

        if type_w is not None:
            bce_15 = nn.BCEWithLogitsLoss(pos_weight=pw15, reduction="none")
            bce_1h = nn.BCEWithLogitsLoss(pos_weight=pw1h, reduction="none")
            loss_c15  = (bce_15(out["cls_15m"], y_c15) * type_w).mean()
            loss_c1h  = (bce_1h(out["cls_1h"],  y_c1h) * type_w).mean()
            ce_fn     = nn.CrossEntropyLoss(reduction="none")
            loss_dir  = (ce_fn(out["direction"], y_d) * type_w * dir_mask).sum() / (dir_mask.sum() + 1e-8)
        else:
            loss_c15  = nn.BCEWithLogitsLoss(pos_weight=pw15)(out["cls_15m"], y_c15)
            loss_c1h  = nn.BCEWithLogitsLoss(pos_weight=pw1h)(out["cls_1h"],  y_c1h)
            ce_fn     = nn.CrossEntropyLoss(reduction="none")
            loss_dir  = (ce_fn(out["direction"], y_d) * dir_mask).sum() / (dir_mask.sum() + 1e-8)

        return (
            loss_c15 * 2.0
            + loss_c1h  * 2.0
            + loss_dir  * 1.0
            + nn.MSELoss()(out["reg_15m"], y_r15) * 0.3
            + nn.MSELoss()(out["reg_1h"],  y_r1h) * 0.3
            + nn.MSELoss()(out["conf"],    y_cf)  * 0.2
        )

    def _to_device(self, batch):
        sem, rag, mac, yr15, yc15, yr1h, yc1h, yd, ycf, tw = batch
        return (
            sem.to(device), rag.to(device), mac.to(device),
            yr15.to(device), yc15.to(device),
            yr1h.to(device), yc1h.to(device),
            yd.to(device), ycf.to(device), tw.to(device),
        )

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            sem, rag, mac, yr15, yc15, yr1h, yc1h, yd, ycf, tw = self._to_device(batch)
            self.optimizer.zero_grad()
            out  = self.model(sem, rag, mac)
            loss = self._loss(out, yr15, yc15, yr1h, yc1h, yd, ycf, tw)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            total += loss.item()
            n     += 1
        return total / n

    def validate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                sem, rag, mac, yr15, yc15, yr1h, yc1h, yd, ycf, tw = self._to_device(batch)
                out  = self.model(sem, rag, mac)
                loss = self._loss(out, yr15, yc15, yr1h, yc1h, yd, ycf, tw)
                total += loss.item()
                n     += 1
        return total / n

    def _modulate(self, p, r):
        return p * (1.0 + np.abs(r) * REG_SCALE)

    def find_threshold(self, val_loader):
        self.model.eval()
        y_c15, y_c1h = [], []
        p15,   p1h   = [], []
        r15,   r1h   = [], []
        with torch.no_grad():
            for batch in val_loader:
                b   = batch
                sem = b[0].to(device)
                rag = b[1].to(device)
                mac = b[2].to(device)
                out = self.model(sem, rag, mac)
                p15.append(torch.sigmoid(out["cls_15m"]).cpu().numpy())
                p1h.append(torch.sigmoid(out["cls_1h"]).cpu().numpy())
                r15.append(out["reg_15m"].cpu().numpy())
                r1h.append(out["reg_1h"].cpu().numpy())
                y_c15.append(b[4].numpy())
                y_c1h.append(b[6].numpy())

        p15  = np.concatenate(p15)
        p1h  = np.concatenate(p1h)
        r15  = np.concatenate(r15)
        r1h  = np.concatenate(r1h)
        y_c15 = np.concatenate(y_c15)
        y_c1h = np.concatenate(y_c1h)

        mod15 = self._modulate(p15, r15)
        mod1h = self._modulate(p1h, r1h)

        print(f"\n  Threshold search (min_precision={MIN_PRECISION}):")
        for label, mod, y_cls, attr in [
            ("15m", mod15, y_c15, "threshold_15m"),
            ("1h",  mod1h, y_c1h, "threshold_1h"),
        ]:
            best_f1 = 0.0
            best_t  = 0.50
            found   = False
            for t in np.linspace(0.02, 0.9, 177):
                preds = (mod >= t).astype(int)
                prec  = precision_score(y_cls, preds, zero_division=0)
                f1    = f1_score(y_cls, preds, zero_division=0)
                if prec >= MIN_PRECISION and f1 > best_f1:
                    best_f1 = f1
                    best_t  = t
                    found   = True
            if not found:
                print(f"    {label}: WARNING — fallback threshold=0.50")
            else:
                print(f"    {label}: threshold={best_t:.2f}  val F1={best_f1:.3f}")
            setattr(self, attr, best_t)

    def train(self, train_loader, val_loader, y_c15, y_c1h):
        print(f"\n  Training {EPOCHS} epochs (patience={PATIENCE})...")
        print(f"  {'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} Status")
        print(f"  {'────────────────────────────────────────────────────'}")

        for epoch in range(1, EPOCHS + 1):
            tr = self.train_epoch(train_loader)
            vl = self.validate(val_loader)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            if vl < self.best_val_loss:
                self.best_val_loss   = vl
                self.patience_counter = 0
                status = "✅ best"
                torch.save({
                    "model_state":  self.model.state_dict(),
                    "scalers":      self.scalers,
                    "threshold_15m": self.threshold_15m,
                    "threshold_1h":  self.threshold_1h,
                    "sem_dim":       self.sem_dim,
                }, MODEL_PATH)
            else:
                self.patience_counter += 1
                status = f"⏳ ({self.patience_counter}/{PATIENCE})"

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>6} {tr:>10.4f} {vl:>10.4f} {lr:>10.2e} {status}")

            if self.patience_counter >= PATIENCE:
                print(f"\n  ⚡ Early stopping at epoch {epoch}")
                break

        print(f"\n  ✅ Best val loss: {self.best_val_loss:.6f}")
        ckpt = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.find_threshold(val_loader)
        torch.save({
            "model_state":   self.model.state_dict(),
            "scalers":       self.scalers,
            "threshold_15m": self.threshold_15m,
            "threshold_1h":  self.threshold_1h,
            "sem_dim":       self.sem_dim,
        }, MODEL_PATH)

    def predict(self, sem, rag, macro):
        self.model.eval()
        sem_s, rag_s, mac_s = self.scale(sem, rag, macro)
        with torch.no_grad():
            out = self.model(
                torch.FloatTensor(sem_s).to(device),
                torch.FloatTensor(rag_s).to(device),
                torch.FloatTensor(mac_s).to(device),
            )
        p15  = torch.sigmoid(out["cls_15m"]).cpu().numpy()
        p1h  = torch.sigmoid(out["cls_1h"]).cpu().numpy()
        r15  = out["reg_15m"].cpu().numpy()
        r1h  = out["reg_1h"].cpu().numpy()
        m15  = self._modulate(p15, r15)
        m1h  = self._modulate(p1h, r1h)
        return {
            "prob_15m":       p15,
            "prob_1h":        p1h,
            "modulated_15m":  m15,
            "modulated_1h":   m1h,
            "reg_pred_15m":   r15,
            "reg_pred_1h":    r1h,
            "pred_15m":       (m15 >= self.threshold_15m).astype(int),
            "pred_1h":        (m1h >= self.threshold_1h).astype(int),
            "conf":           torch.sigmoid(out["conf"]).cpu().numpy() * 1.0,
            "direction":      out["direction"].argmax(dim=1).cpu().numpy(),
        }

    def online_update(self, sem, rag, macro, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf, steps=1, lr=1e-6):
        self.replay.add(
            sem.flatten(), rag.flatten(), macro.flatten(),
            y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf,
        )
        if len(self.replay) < 32:
            return
        sample = self.replay.sample(32)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sems, rags, macs, yr15, yc15, yr1h, yc1h, yds, ycfs = sample
        sem_s, rag_s, mac_s = self.scale(sems, rags, macs)
        for s, x, t in zip(self.scalers, [sems, rags, macs], [sem_s, rag_s, mac_s]):
            pass
        tgts = (
            torch.FloatTensor(sem_s).to(device),
            torch.FloatTensor(rag_s).to(device),
            torch.FloatTensor(mac_s).to(device),
            torch.FloatTensor(yr15).to(device),
            torch.FloatTensor(yc15).to(device),
            torch.FloatTensor(yr1h).to(device),
            torch.FloatTensor(yc1h).to(device),
            torch.LongTensor(yds).to(device),
            torch.FloatTensor(ycfs).to(device),
        )
        for _ in range(steps):
            opt.zero_grad()
            out  = self.model(tgts[0], tgts[1], tgts[2])
            loss = self._loss(out, tgts[3], tgts[4], tgts[5], tgts[6], tgts[7], tgts[8])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            opt.step()
        self.model.eval()


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_horizon(label, probs, modulated, threshold, y_cls, reg_pred, y_reg):
    preds = (modulated >= threshold).astype(int)
    f1    = f1_score(y_cls, preds, zero_division=0)
    prec  = precision_score(y_cls, preds, zero_division=0)
    rec   = recall_score(y_cls, preds, zero_division=0)
    acc   = accuracy_score(y_cls, preds)

    if len(np.unique(y_cls)) > 1:
        auc = roc_auc_score(y_cls, probs)
    else:
        auc = 0.0

    mae    = mean_absolute_error(y_reg, reg_pred)
    r2     = r2_score(y_reg, reg_pred)
    dir_acc = float(np.sign(reg_pred) == np.sign(y_reg)) if len(y_reg) == 1 else np.mean(np.sign(reg_pred) == np.sign(y_reg))
    cm     = confusion_matrix(y_cls, preds)
    tn, fp, fn, tp = cm.ravel().tolist() if cm.shape == (2, 2) else (0, 0, 0, 0)

    horizon = "15-minute" if label == "15" else "1-hour" if label == "1h" else label
    print(f"\n  ── {label} ({horizon}) ──")
    print(f"    Threshold : {threshold:.2f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}  AUC: {auc:.3f}")
    print(f"    Acc: {acc:.1%}  DirAcc: {dir_acc:.1%}  MAE: {mae:.4f}%  R²: {r2:.4f}")
    print(f"    Confusion: NO[TN={tn:>5} FP={fp:>5}]  YES[FN={fn:>5} TP={tp:>5}]")
    if preds.sum() == len(preds):
        print(f"    ⚠️  All predicted YES")
    elif preds.sum() == 0:
        print(f"    ⚠️  All predicted NO")

    return {
        "F1": f1, "Precision": prec, "Recall": rec, "Accuracy": acc,
        "ROC_AUC": auc, "MAE": mae, "R2": r2, "DirAcc": dir_acc,
        "Threshold": threshold,
        "CM": [[tn, fp], [fn, tp]],
    }


def evaluate(trainer, sem, rag, macro, y_r15, y_c15, y_r1h, y_c1h, y_dir, y_cf):
    print("\n" + "=" * 65)
    print("\n  TEST SET EVALUATION — Sentiment-Focused + Cross-Attention\n")

    preds = trainer.predict(sem, rag, macro)

    r15 = _eval_horizon(
        "15m", preds["prob_15m"], preds["modulated_15m"], trainer.threshold_15m,
        y_c15, preds["reg_pred_15m"], y_r15,
    )
    r1h = _eval_horizon(
        "1h", preds["prob_1h"], preds["modulated_1h"], trainer.threshold_1h,
        y_c1h, preds["reg_pred_1h"], y_r1h,
    )

    big_mask  = np.abs(y_r15) > THRESHOLD_15M
    big_count = int(big_mask.sum())
    dir_acc   = float(accuracy_score(y_dir[big_mask], preds["direction"][big_mask])) if big_count > 0 else 0.0
    dir_f1    = float(f1_score(y_dir[big_mask], preds["direction"][big_mask], zero_division=0)) if big_count > 0 else 0.0
    conf_mae  = float(mean_absolute_error(y_cf, preds["conf"]))

    print(f"\n  Direction (|BTC|>{THRESHOLD_15M}%, n={big_count},): Acc={dir_acc:.1%}  F1={dir_f1:.3f}")
    print(f"  Confidence MAE: {conf_mae:.4f}")
    print("─" * 65)

    print("\n  COMPARISON\n")
    for m, v15, v1h in [("F1", r15["F1"], r1h["F1"]), ("AUC", r15["ROC_AUC"], r1h["ROC_AUC"]),
                         ("MAE", r15["MAE"], r1h["MAE"])]:
        win = "15m" if (v15 > v1h if m != "MAE" else v15 < v1h) else "1h"
        bar = "█" * int(max(v15, v1h) * 100 / 2)
        print(f"  {m:<12}: 15m={v15:.3f}  1h={v1h:.3f}  winner={win}")

    results = {
        "15_minute": r15,
        "1_hour":    r1h,
        "direction": {"Acc": dir_acc, "F1": dir_f1},
        "confidence": {"MAE": conf_mae},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_PATH}")
    return results


def generate_explanation(df, idx, preds, rag_row, rag_detail, threshold_15m, preds_idx=None):
    if preds_idx is None:
        preds_idx = idx
    row   = df.iloc[idx]
    p15   = float(preds["prob_15m"][preds_idx])
    r15   = float(preds["reg_pred_15m"][preds_idx])
    sc15  = REG_SCALE
    mod15 = float(preds["modulated_15m"][preds_idx])
    pred15 = int(preds["pred_15m"][preds_idx])

    hour = int(pd.Timestamp(row["published"]).hour)
    dow  = int(pd.Timestamp(row["published"]).dayofweek)

    ctx = []
    if dow >= 5:    ctx.append("Weekend")
    if 2 <= hour <= 6:  ctx.append("Low-liquidity (02-06 UTC)")
    if 13 <= hour <= 21: ctx.append("US hours (13-21 UTC)")
    if hour <= 8:   ctx.append("Asia hours")
    if row.get("fomc_week"):    ctx.append("FOMC week")

    lines = [
        f'Headline: "{str(row.get("title", ""))[:80]}"',
        f'Channel:  {row.get("channel", "")} | {" | ".join(ctx) or "No special context"}',
        "\n--- Classification ---",
        f"Raw cls prob   : {p15:.3f}",
        f"Reg prediction : {r15:+.3f}% (15m BTC)",
        f"Reg scale      : {sc15:.2f}x",
        f"Modulated score: {mod15:.3f} vs threshold {threshold_15m:.2f}",
        f"Decision       : {'✅ YES (IMPACTFUL)' if pred15 else '❌ NO'}",
        "\n--- RAG Context (Qdrant + 5-feature macro-reweighted) ---",
    ]

    if rag_row is not None:
        lines += [
            f"Weighted avg BTC: {rag_row[0]:.3f}%",
            f"Max BTC change  : {rag_row[1]:.3f}%",
            f"Weighted hit rt : {rag_row[3]:.1%}",
            f"Avg similarity  : {rag_row[4]:.3f}",
            f"Impactful/10    : {int(rag_row[9])}",
        ]

    if rag_detail:
        lines.append("\n  Top 3 retrieved:")
        similar = rag_detail.get("similar_news", [])
        for s in similar[:3]:
            tags = []
            if s.get("weekend_match"):   tags.append("📅WKD")
            if s.get("low_liq_match"):   tags.append("🌙LOW-LIQ")
            if s.get("us_match"):        tags.append("🇺🇸US")
            tag_str = " ".join(tags)
            lines.append(
                f"    [{s.get('similarity_score', 0):.3f}|w={s.get('macro_weight', 1):.2f}] "
                f"BTC:{s.get('btc_change_15m', 0):+.2f}% | "
                f"{tag_str} {str(s.get('title', ''))[:55]}"
            )

    lines += [
        "\n--- Ground Truth ---",
        f"Actual BTC 15m: {row.get('btc_change_15m', 0):+.3f}%  "
        f"({'IMPACTFUL' if row.get('is_impactful_15m') else 'not impactful'})",
    ]
    return "\n".join(lines) + "\n"


def make_loader(sem_s, rag_s, mac_s, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf, type_w, shuffle):
    ds = TensorDataset(
        torch.FloatTensor(sem_s),
        torch.FloatTensor(rag_s),
        torch.FloatTensor(mac_s),
        torch.FloatTensor(y_r15),
        torch.FloatTensor(y_c15),
        torch.FloatTensor(y_r1h),
        torch.FloatTensor(y_c1h),
        torch.LongTensor(y_d),
        torch.FloatTensor(y_cf),
        torch.FloatTensor(type_w),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PRODUCTION MULTI-TOWER FUSION — v9 + Cross-Attention + Macro Injection")
    print(f"  seed={MONTHLY_SEED} | threshold_15m={THRESHOLD_15M} | 5-dim macro")

    df = load_data()

    print(f"\n[3/8] MONTHLY RANDOM SPLIT (seed={MONTHLY_SEED})")
    df["_ym"] = df["published"].dt.to_period("M")
    months    = sorted(df["_ym"].unique())
    rng       = np.random.default_rng(MONTHLY_SEED)
    rng.shuffle(months)
    n_train = int(len(months) * 0.70)
    n_val   = int(len(months) * 0.15)
    train_months = set(months[:n_train])
    val_months   = set(months[n_train : n_train + n_val])
    test_months  = set(months[n_train + n_val :])

    train_idx = np.where(df["_ym"].isin(train_months))[0]
    val_idx   = np.where(df["_ym"].isin(val_months))[0]
    test_idx  = np.where(df["_ym"].isin(test_months))[0]
    df = df.drop(columns=["_ym"])

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    print("\n  Impact rate check (should be similar):")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        sub = df.iloc[idx]
        print(
            f"  {name:<5}: n={len(idx):5d}  15m={sub['is_impactful_15m'].mean():.1%}"
            f"  1h={sub['is_impactful_1h'].mean():.1%}"
            f"  avg|BTC|={sub['abs_change_15m'].mean():.3f}%"
        )

    print("\n[4/8] BUILD MODEL")
    semantic, macro, rag, rag_details, type_w, type_indices, type_probs = \
        build_features(df, train_idx)

    sem_dim   = semantic.shape[1]
    rag_dim   = rag.shape[1]
    macro_dim = macro.shape[1]
    emb_proj  = EmbProjection()
    model     = CryptoImpactNetV5(sem_dim=sem_dim, rag_dim=rag_dim, macro_dim=macro_dim, emb_proj=emb_proj)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio     = n_params / len(df)
    sign      = "✅" if ratio < 1.0 else "⚠️ add more data"
    print(f"  CryptoImpactNetV5 + CrossAttention + NewsTypeGating + DualBERT + PriceCtx: {n_params:,} params | params/row: {ratio:.2f} {sign}")
    print(f"    Semantic    : {sem_dim} → (CryptoBERT+FinBERT {DUAL_EMB_DIM}→{EMB_PROJ_DIMS}) → 24")
    print(f"    NewsTypeGate: type_probs(11) → gate(24) → gated sem(24)")
    print(f"    RAG         : {rag_dim} (Qdrant macro-reweighted) → 6")
    print(f"    Macro       : {macro_dim} (5 timing + 3 price ctx) → 6  [injected before attn]")
    print(f"    s_ctx       : gated_sem(24) + macro(6) = 30  [time+price-aware query]")
    print(f"    CrossAttn   : s_ctx(30) ↔ rag(6) → 24")
    print(f"    Fusion      : 24 → 16 → 12")

    # Split arrays
    sem_train = semantic[train_idx];  sem_val = semantic[val_idx];  sem_test = semantic[test_idx]
    rag_train = rag[train_idx];       rag_val = rag[val_idx];       rag_test = rag[test_idx]
    mac_train = macro[train_idx];     mac_val = macro[val_idx];     mac_test = macro[test_idx]
    tw_train  = type_w[train_idx];    tw_val  = type_w[val_idx];    tw_test  = type_w[test_idx]

    y_r15_tr  = df.iloc[train_idx]["btc_change_15m"].values.astype(np.float32)
    y_c15_tr  = df.iloc[train_idx]["is_impactful_15m"].values.astype(np.float32)
    y_r1h_tr  = df.iloc[train_idx]["btc_change_1h"].values.astype(np.float32)
    y_c1h_tr  = df.iloc[train_idx]["is_impactful_1h"].values.astype(np.float32)
    y_dir_tr  = df.iloc[train_idx]["direction_15m"].values.astype(int)
    y_cf_tr   = df.iloc[train_idx]["confidence_label"].values.astype(np.float32)

    y_r15_va  = df.iloc[val_idx]["btc_change_15m"].values.astype(np.float32)
    y_c15_va  = df.iloc[val_idx]["is_impactful_15m"].values.astype(np.float32)
    y_r1h_va  = df.iloc[val_idx]["btc_change_1h"].values.astype(np.float32)
    y_c1h_va  = df.iloc[val_idx]["is_impactful_1h"].values.astype(np.float32)
    y_dir_va  = df.iloc[val_idx]["direction_15m"].values.astype(int)
    y_cf_va   = df.iloc[val_idx]["confidence_label"].values.astype(np.float32)

    y_r15_te  = df.iloc[test_idx]["btc_change_15m"].values.astype(np.float32)
    y_c15_te  = df.iloc[test_idx]["is_impactful_15m"].values.astype(np.float32)
    y_r1h_te  = df.iloc[test_idx]["btc_change_1h"].values.astype(np.float32)
    y_c1h_te  = df.iloc[test_idx]["is_impactful_1h"].values.astype(np.float32)
    y_dir_te  = df.iloc[test_idx]["direction_15m"].values.astype(int)
    y_cf_te   = df.iloc[test_idx]["confidence_label"].values.astype(np.float32)

    trainer = Trainer(model, sem_dim=sem_dim)
    trainer.fit_scalers(sem_train, rag_train, mac_train)
    sem_s_tr, rag_s_tr, mac_s_tr = trainer.scale(sem_train, rag_train, mac_train)
    sem_s_va, rag_s_va, mac_s_va = trainer.scale(sem_val,   rag_val,   mac_val)
    sem_s_te, rag_s_te, mac_s_te = trainer.scale(sem_test,  rag_test,  mac_test)

    train_loader = make_loader(sem_s_tr, rag_s_tr, mac_s_tr,
                               y_r15_tr, y_c15_tr, y_r1h_tr, y_c1h_tr,
                               y_dir_tr, y_cf_tr, tw_train, shuffle=True)
    val_loader   = make_loader(sem_s_va, rag_s_va, mac_s_va,
                               y_r15_va, y_c15_va, y_r1h_va, y_c1h_va,
                               y_dir_va, y_cf_va, tw_val, shuffle=False)
    test_loader  = make_loader(sem_s_te, rag_s_te, mac_s_te,
                               y_r15_te, y_c15_te, y_r1h_te, y_c1h_te,
                               y_dir_te, y_cf_te, tw_test, shuffle=False)

    print(f"\n[5/8] TRAINING  (MIN_PRECISION={MIN_PRECISION}")
    trainer.train(train_loader, val_loader, y_c15_tr, y_c1h_tr)

    print(f"\n[6/8] EVALUATION")
    results = evaluate(
        trainer,
        sem_s_te, rag_s_te, mac_s_te,
        y_r15_te, y_c15_te, y_r1h_te, y_c1h_te, y_dir_te, y_cf_te,
    )

    print(f"\n[7/8] QUALITATIVE EXAMPLES")
    sample_idx = np.random.choice(test_idx, min(3, len(test_idx)), replace=False)
    preds_full = trainer.predict(sem_s_te, rag_s_te, mac_s_te)
    for k, abs_idx in enumerate(sample_idx):
        rel_idx = list(test_idx).index(abs_idx)
        rag_row    = rag_test[rel_idx]
        rag_detail = rag_details[abs_idx] if rag_details and abs_idx < len(rag_details) else None
        preds_single = {k2: v[rel_idx:rel_idx+1] for k2, v in preds_full.items()}
        print(f"\n  {'='*55}")
        print(f"\n  Example {k+1}:")
        expl = generate_explanation(df, abs_idx, preds_full, rag_row, rag_detail, trainer.threshold_15m, preds_idx=rel_idx)
        print(f"\n  {expl}")

    print(f"\n[8/8] ONLINE LEARNING DEMO")
    demo_idx = np.random.choice(test_idx, min(5, len(test_idx)), replace=False)
    for abs_idx in demo_idx:
        rel_idx = list(test_idx).index(abs_idx)
        trainer.online_update(
            sem_test[rel_idx:rel_idx+1], rag_test[rel_idx:rel_idx+1], mac_test[rel_idx:rel_idx+1],
            y_r15_te[rel_idx:rel_idx+1], y_c15_te[rel_idx:rel_idx+1],
            y_r1h_te[rel_idx:rel_idx+1], y_c1h_te[rel_idx:rel_idx+1],
            y_dir_te[rel_idx:rel_idx+1], y_cf_te[rel_idx:rel_idx+1],
        )
    print(f"  Replay buffer: {len(trainer.replay)} samples")
    print()

    print("\n")
    print(f"\n  SUMMARY — v9 + Cross-Attention\n")
    print(f"  Params        : {n_params:,} | params/row: {ratio:.2f}")
    print(f"  Architecture  : 3-Tower (sem+rag+macro) + NewsTypeGating + MacroInjection + CrossAttention")
    print(f"  NewsTypeGate  : type_probs(11) gates sem(12) → type-aware embedding")
    print(f"  Macro inject  : gated_sem(12) + macro(6) → s_ctx(18) BEFORE attention")
    print(f"  CrossAttn     : s_ctx ↔ rag (time+type-aware news queries historical context)")
    print(f"  RAG           : Qdrant Cloud (bge-small, cosine, top_k=10)")
    print(f"  Macro reweight: 5 features (weekend/low_liq/us/asia/fomc)")
    print(f"  Semantic      : CryptoBERT 768-dim")
    print(f"  Split         : monthly random (seed={MONTHLY_SEED})")
    print(f"  THRESHOLD_15M : {THRESHOLD_15M}")
    print(f"  MIN_PRECISION : {MIN_PRECISION}")
    print(f"  Threshold 15m : {trainer.threshold_15m:.2f}")
    print(f"  Threshold 1h  : {trainer.threshold_1h:.2f}")
    print(f"  Model         → {MODEL_PATH}")
    print(f"  Results       → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
