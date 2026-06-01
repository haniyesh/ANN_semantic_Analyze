"""
Production Multi-Tower Fusion System — v5
==========================================
Fixes vs previous versions:
  - No macro_features.py dependency (all inline)
  - THRESHOLD_15M = 0.3
  - MONTHLY_SEED  = 43
  - Macro tower: 5 dims (weekend/low_liq/us_hours/asia_hours/fomc)
  - Import from rag_news (not rag_qdrant)
  - Progress bar for CryptoBERT embedding
"""

import os, json, warnings
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

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
HERE             = Path(__file__).parent
SENTIMENT_CSV    = HERE / "news_cleaned.csv"
CRYPTOBERT_CACHE = HERE / "cryptobert_your_pipeline.npy"
MODEL_PATH       = HERE / "production_system_v5.pt"
RESULTS_PATH     = HERE / "production_results_v5.json"

THRESHOLD_15M = 0.3
THRESHOLD_1H  = 0.5
REG_SCALE     = 2.0
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-3
EPOCHS        = 200
PATIENCE      = 20
BATCH_SIZE    = 64
GRAD_CLIP     = 1.0
REPLAY_CAP    = 5000
MIN_PRECISION = 0.25
EMB_PROJ_DIMS = 24
MONTHLY_SEED  = 43

device = torch.device("cpu")
print(f"  Device: {device}")

# ══════════════════════════════════════════════════════════════════
# NEWS TYPE CLASSIFIER — inline, no macro_features.py needed
# Uses CryptoBERT embeddings + cosine similarity to prototypes
# ══════════════════════════════════════════════════════════════════
NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis"
]

_NEWS_TYPE_PROTOTYPES_TEXT = {
    "regulatory":     ["SEC charges crypto exchange securities violations", "government bans cryptocurrency trading country", "court rules against crypto company lawsuit", "regulatory compliance required exchange"],
    "etf":            ["Bitcoin ETF approved by SEC trading", "spot bitcoin fund launches stock exchange", "BlackRock files ETF application approval", "grayscale bitcoin trust conversion"],
    "hack":           ["crypto exchange hacked millions stolen", "DeFi protocol exploited flash loan attack", "security breach drains user funds wallet"],
    "macro_economic": ["Federal Reserve raises interest rates decision", "inflation data CPI report released", "GDP growth recession fears mount", "FOMC rate decision monetary policy"],
    "exchange":       ["Binance lists new cryptocurrency token", "Coinbase delists token regulatory concerns", "exchange trading volume record high", "new trading pair launched users"],
    "defi":           ["DeFi protocol TVL record liquidity", "Uniswap launches new version features", "yield farming liquidity pool AMM", "decentralized exchange volume surpasses"],
    "mining":         ["Bitcoin mining difficulty adjusts record", "miner capitulation hashrate drops significantly", "Bitcoin halving block reward reduced", "mining company orders ASIC machines"],
    "institutional":  ["MicroStrategy purchases Bitcoin treasury reserve", "hedge fund allocates Bitcoin portfolio", "corporate treasury adds BTC asset", "institutional investors increase holdings"],
    "technical":      ["Bitcoin network upgrade soft fork activates", "Ethereum developers confirm upgrade date", "layer two scaling solution launches mainnet", "protocol implements new consensus mechanism"],
    "partnership":    ["crypto company partnership bank deal", "blockchain firm integrates payment processor", "exchange signs deal financial institution", "technology company acquires crypto startup"],
    "market_analysis":["Bitcoin price analysis bullish breakout target", "technical analysis support level tested", "market sentiment fear greed index extreme", "on chain data accumulation long term"],
}

_proto_matrix = None

#use for news classification using cryptobert embeddings and cosine similarity to prototypes. The prototypes are defined as the average embedding of a few representative sentences for each news type label. The function builds the prototype matrix if it hasn't been built yet, and then computes the cosine similarity between the input embeddings and the prototype matrix to classify the news type. It can also return the probabilities for each news type if requested.
def _build_proto_matrix():
    global _proto_matrix
    if _proto_matrix is not None:
        return _proto_matrix

    from transformers import AutoTokenizer, AutoModel
    print("  Building news type prototype embeddings (one-time)...")
    tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    mdl = AutoModel.from_pretrained("ElKulako/cryptobert")
    mdl.eval()

    proto_embs = []
    for label in NEWS_TYPE_LABELS:
        sentences = _NEWS_TYPE_PROTOTYPES_TEXT[label]
        inputs    = tok(sentences, padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
        with torch.no_grad():
            emb = mdl(**inputs).last_hidden_state[:, 0, :]
        proto_embs.append(emb.mean(dim=0))

    mat = torch.stack(proto_embs)            # (11, 768)
    _proto_matrix = F.normalize(mat, dim=1)
    return _proto_matrix


def crypto_news_type_classify(embeddings: np.ndarray, return_probs: bool = False):
    proto   = _build_proto_matrix()
    emb_t   = F.normalize(torch.FloatTensor(embeddings), dim=1)
    sims    = torch.mm(emb_t, proto.T)              # (N, 11)
    probs   = F.softmax(sims * 5.0, dim=1).numpy().astype(np.float32)
    indices = probs.argmax(axis=1)
    if return_probs:
        return indices, probs
    return indices

# focal loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
    
#start stages of pipeline: data loading, feature engineering, model definition, replay buffer, trainer, evaluation
# 1. DATA

def load_data() -> pd.DataFrame:
    print("[1/8] LOAD DATA")
    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["published"], low_memory=False)

    for col in ["btc_price_at_news", "btc_price_15m", "btc_price_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sentiment_score", "weight", "confidence",
                "prob_positive", "prob_negative", "prob_neutral"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    orig = len(df)
    df = df[df["weight"] >= 5]
    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df.drop_duplicates(subset=["title", "published", "channel"])

    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["is_impactful_15m"] = (df["abs_change_15m"] >= THRESHOLD_15M).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)

    df["btc_change_1h"]   = (df["btc_price_1h"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_1h"]   = df["btc_change_1h"].abs()
    df["is_impactful_1h"] = (df["abs_change_1h"] >= THRESHOLD_1H).astype(int)
    df["direction_1h"]    = (df["btc_change_1h"] > 0).astype(int)

    df["confidence_label"] = df["abs_change_15m"].clip(0, 3) / 3.0
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    print(f"  Rows: {len(df):,} (from {orig:,})")
    print(f"  Impactful 15m: {df['is_impactful_15m'].mean()*100:.1f}%  "
          f"1h: {df['is_impactful_1h'].mean()*100:.1f}%")
    return df

# 2.urns every news title in your table into a CryptoBERT embedding.
def compute_cryptobert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if CRYPTOBERT_CACHE.exists():
        emb = np.load(CRYPTOBERT_CACHE)
        if len(emb) == len(df):
            print("  CryptoBERT cache hit")
            return emb.astype(np.float32)

    print(f"  Computing CryptoBERT embeddings for {len(df):,} rows...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model     = AutoModel.from_pretrained("ElKulako/cryptobert")
    model.eval()
    titles, embs = df["title"].fillna("").tolist(), []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)} ({i*100//len(titles)}%)...")
            inputs = tokenizer(titles[i:i+32], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            embs.append(model(**inputs).last_hidden_state[:, 0, :].numpy())
    emb = np.vstack(embs).astype(np.float32)
    np.save(CRYPTOBERT_CACHE, emb)
    print(f"  CryptoBERT done: {emb.shape}")
    return emb

#compute channel impact rates by grouping the training data by channel and calculating the mean of the "is_impactful_15m" column for each channel. It also prints out the impact rates for each channel in a sorted manner, along with a visual bar representation. The function returns a dictionary containing the impact rates for each channel.
def compute_channel_impact_rates(df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    train_df  = df.iloc[train_idx]
    rates     = train_df.groupby("channel")["is_impactful_15m"].mean().to_dict()
    mean_rate = float(train_df["is_impactful_15m"].mean())
    print("  Channel impact rates (training data only):")
    for ch, rate in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"    {ch:<30}: {rate:.1%}  {'█' * int(rate * 20)}")
    print(f"    {'(dataset mean)':<30}: {mean_rate:.1%}")
    return rates


def build_market_features(df: pd.DataFrame):
    """
    Returns:
        macro  : (N, 5) — time-of-day context features
        market : (N, 5) — BTC momentum + volatility
    """
    price   = df["btc_price_at_news"].values
    ret_1h  = pd.Series(price).pct_change(4).shift(1).fillna(0).values  * 100
    ret_4h  = pd.Series(price).pct_change(16).shift(1).fillna(0).values * 100
    ret_24h = pd.Series(price).pct_change(96).shift(1).fillna(0).values * 100
    vol_1h  = pd.Series(price).pct_change().rolling(4,  min_periods=1).std().shift(1).fillna(0).values * 100
    vol_4h  = pd.Series(price).pct_change().rolling(16, min_periods=1).std().shift(1).fillna(0).values * 100

    hour = df["published"].dt.hour.values
    dow  = df["published"].dt.dayofweek.values

    def _col(name, fallback):
        if name in df.columns:
            return df[name].fillna(0).values.astype(np.float32)
        return fallback.astype(np.float32)

    is_weekend       = _col("is_weekend",       (dow >= 5).astype(float))
    is_low_liquidity = _col("is_low_liquidity", ((hour >= 2) & (hour <= 6)).astype(float))
    is_us_hours      = _col("is_us_hours",      ((hour >= 13) & (hour <= 21)).astype(float))
    is_asia_hours    = _col("is_asia_hours",    ((hour >= 0)  & (hour <= 8)).astype(float))
    fomc_week        = _col("fomc_week",        np.zeros(len(df), dtype=float))

    df["fomc_week"] = fomc_week   # rag_news upload_vectors reads this

    macro  = np.column_stack([
        is_weekend, is_low_liquidity, is_us_hours, is_asia_hours, fomc_week
    ]).astype(np.float32)

    market = np.column_stack([
        ret_1h, ret_4h, ret_24h, vol_1h, vol_4h
    ]).astype(np.float32)

    print(f"  Macro : {macro.shape[1]} dims (weekend/low_liq/us/asia/fomc)")
    print(f"  Market: {market.shape[1]} dims (BTC momentum/vol)")
    return macro, market


def build_features(df: pd.DataFrame, train_idx: np.ndarray | None = None):
    print("[2/8] FEATURE ENGINEERING")

    emb = compute_cryptobert_embeddings(df)
    _, type_probs = crypto_news_type_classify(emb, return_probs=True)

    sent_cols = ["sentiment_score", "weight", "confidence",
                 "prob_positive", "prob_negative", "prob_neutral"]
    sent_df  = df[sent_cols].fillna(0).values.astype(np.float32)
    semantic = np.hstack([emb, sent_df, type_probs]).astype(np.float32)
    print(f"  Semantic: {semantic.shape[1]} dims (CryptoBERT 768 + sent6 + type11)")

    macro, market = build_market_features(df)

    if train_idx is not None:
        ch_rates = compute_channel_impact_rates(df, train_idx)
    else:
        ch_rates = df.groupby("channel")["is_impactful_15m"].mean().to_dict()

    print("  RAG: Qdrant Cloud + 5-feature macro-conditioned re-weighting")
    rag, rag_details = build_rag_features_qdrant(df, channel_impact_rates=ch_rates)
    print(f"  RAG features: {rag.shape}")

    return (
        semantic, macro, rag, rag_details, market, ch_rates,
        df["btc_change_15m"].values.astype(np.float32),
        df["is_impactful_15m"].values.astype(np.float32),
        df["btc_change_1h"].values.astype(np.float32),
        df["is_impactful_1h"].values.astype(np.float32),
        df["direction_15m"].values.astype(np.int64),
        df["confidence_label"].values.astype(np.float32),
        df,
    )


# 4. MODEL
class EmbProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=EMB_PROJ_DIMS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 48), nn.LayerNorm(48), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(48, out_dim),
        )
    def forward(self, x): return self.proj(x)


class SmallTower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class CryptoImpactNetV5(nn.Module):
    """4-Tower Gated Fusion + Regression-Modulated Classification."""
    def __init__(self, sem_dim, rag_dim, macro_dim, mem_dim, emb_proj=None):
        super().__init__()
        self.emb_proj  = emb_proj
        sem_in = (sem_dim - 768 + EMB_PROJ_DIMS) if emb_proj else sem_dim

        self.sem_tower = SmallTower(sem_in,    16, dropout=0.4)
        self.rag_tower = SmallTower(rag_dim,    8, dropout=0.3)
        self.mac_tower = SmallTower(macro_dim,  8, dropout=0.3)
        self.mem_tower = SmallTower(mem_dim,    8, dropout=0.3)

        fused       = 16 + 8 + 8 + 8
        self.gate   = nn.Linear(fused, 4)
        self.fusion = nn.Sequential(
            nn.Linear(fused, 24), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(24, 12),   nn.ReLU(),
        )
        self.head_cls_15m = nn.Linear(12, 1)
        self.head_cls_1h  = nn.Linear(12, 1)
        self.head_reg_15m = nn.Linear(12, 1)
        self.head_reg_1h  = nn.Linear(12, 1)
        self.head_conf    = nn.Linear(12, 1)
        self.head_dir     = nn.Linear(12, 2)

    def forward(self, sem, rag, macro, mem):
        if self.emb_proj is not None:
            sem = torch.cat([self.emb_proj(sem[:, :768]), sem[:, 768:]], dim=1)
        s, r, m, p = (self.sem_tower(sem), self.rag_tower(rag),
                      self.mac_tower(macro), self.mem_tower(mem))
        cat   = torch.cat([s, r, m, p], dim=1)
        gates = torch.softmax(self.gate(cat), dim=1)
        s = s * gates[:, 0:1]; r = r * gates[:, 1:2]
        m = m * gates[:, 2:3]; p = p * gates[:, 3:4]
        fused = self.fusion(torch.cat([s, r, m, p], dim=1))
        return {
            "cls_15m":   self.head_cls_15m(fused).squeeze(1),
            "cls_1h":    self.head_cls_1h(fused).squeeze(1),
            "reg_15m":   self.head_reg_15m(fused).squeeze(1),
            "reg_1h":    self.head_reg_1h(fused).squeeze(1),
            "conf":      self.head_conf(fused).squeeze(1),
            "direction": self.head_dir(fused),
        }


# ══════════════════════════════════════════════════════════════════
# 5. REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAP):
        self.buffer = deque(maxlen=capacity)

    def add(self, sem, rag, macro, mem, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf):
        self.buffer.append((sem, rag, macro, mem, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf))

    def sample(self, n=32):
        if len(self.buffer) < n: return None
        idx   = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in idx]
        return [np.array([b[j] for b in batch]) for j in range(10)]

    def __len__(self): return len(self.buffer)


# ══════════════════════════════════════════════════════════════════
# 6. TRAINER
# ══════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model):
        self.model            = model.to(device)
        self.optimizer        = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=1e-6)
        self.scalers          = [StandardScaler() for _ in range(4)]
        self.best_val_loss    = float("inf")
        self.patience_counter = 0
        self.threshold_15m    = 0.30
        self.threshold_1h     = 0.30
        self.replay           = ReplayBuffer()

    def fit_scalers(self, sem, rag, macro, mem):
        for s, x in zip(self.scalers, [sem, rag, macro, mem]): s.fit(x)

    def scale(self, sem, rag, macro, mem):
        return tuple(s.transform(x) for s, x in zip(self.scalers, [sem, rag, macro, mem]))

    def _loss(self, out, y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf):
        return (
            FocalLoss()(out["cls_15m"],  y_c15) * 2.0 +
            FocalLoss()(out["cls_1h"],   y_c1h) * 2.0 +
            nn.CrossEntropyLoss()(out["direction"], y_d) * 1.0 +
            nn.MSELoss()(out["reg_15m"], y_r15) * 0.3 +
            nn.MSELoss()(out["reg_1h"],  y_r1h) * 0.3 +
            nn.MSELoss()(out["conf"],    y_cf)  * 0.2
        )

    def _to_device(self, batch):
        sem, rag, mac, mem, yr15, yc15, yr1h, yc1h, yd, ycf = batch
        return (sem.to(device), rag.to(device), mac.to(device), mem.to(device),
                yr15.to(device), yc15.to(device), yr1h.to(device), yc1h.to(device),
                yd.to(device), ycf.to(device))

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            sem, rag, mac, mem, yr15, yc15, yr1h, yc1h, yd, ycf = self._to_device(batch)
            self.optimizer.zero_grad()
            out  = self.model(sem, rag, mac, mem)
            loss = self._loss(out, yr15, yc15, yr1h, yc1h, yd, ycf)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / n

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            sem, rag, mac, mem, yr15, yc15, yr1h, yc1h, yd, ycf = self._to_device(batch)
            out  = self.model(sem, rag, mac, mem)
            loss = self._loss(out, yr15, yc15, yr1h, yc1h, yd, ycf)
            total += loss.item(); n += 1
        return total / n

    def _modulate(self, p, r):
        return p * (1.0 + np.abs(r) * REG_SCALE)

    def find_threshold(self, val_loader, y_c15, y_c1h):
        self.model.eval()
        p15, p1h, r15, r1h = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                sem, rag, mac, mem = [b.to(device) for b in batch[:4]]
                out = self.model(sem, rag, mac, mem)
                p15.append(torch.sigmoid(out["cls_15m"]).cpu().numpy())
                p1h.append(torch.sigmoid(out["cls_1h"]).cpu().numpy())
                r15.append(out["reg_15m"].cpu().numpy())
                r1h.append(out["reg_1h"].cpu().numpy())

        p15, p1h = np.concatenate(p15), np.concatenate(p1h)
        r15, r1h = np.concatenate(r15), np.concatenate(r1h)
        mod15    = self._modulate(p15, r15)
        mod1h    = self._modulate(p1h, r1h)

        print(f"\n  Threshold search (min_precision={MIN_PRECISION}):")
        for label, mod, y_cls, attr in [
            ("15m", mod15, y_c15, "threshold_15m"),
            ("1h",  mod1h, y_c1h, "threshold_1h"),
        ]:
            best_f1, best_t, found = 0, 0.30, False
            for t in np.linspace(0.02, 0.90, 177):
                preds = (mod >= t).astype(int)
                prec  = precision_score(y_cls, preds, zero_division=0)
                if prec < MIN_PRECISION: continue
                f1 = f1_score(y_cls, preds, zero_division=0)
                if f1 > best_f1: best_f1, best_t, found = f1, t, True
            if not found:
                print(f"    {label}: WARNING — fallback threshold=0.30")
            else:
                print(f"    {label}: threshold={best_t:.2f}  val F1={best_f1:.3f}")
            setattr(self, attr, best_t)

    def train(self, train_loader, val_loader, y_c15, y_c1h):
        print(f"\n  Training {EPOCHS} epochs (patience={PATIENCE})...")
        print(f"  {'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} Status")
        print(f"  {'─'*52}")

        for epoch in range(1, EPOCHS + 1):
            tr = self.train_epoch(train_loader)
            vl = self.validate(val_loader)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            if vl < self.best_val_loss:
                self.best_val_loss    = vl
                self.patience_counter = 0
                torch.save({
                    "model_state":   self.model.state_dict(),
                    "scalers":       self.scalers,
                    "threshold_15m": self.threshold_15m,
                    "threshold_1h":  self.threshold_1h,
                }, MODEL_PATH)
                status = "✅ best"
            else:
                self.patience_counter += 1
                status = f"⏳ ({self.patience_counter}/{PATIENCE})"

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>6} {tr:>10.4f} {vl:>10.4f} {lr:>10.2e} {status}")
            if self.patience_counter >= PATIENCE:
                print(f"\n  ⚡ Early stopping at epoch {epoch}")
                break

        print(f"\n  ✅ Best val loss: {self.best_val_loss:.6f}")
        ckpt = torch.load(MODEL_PATH, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.scalers = ckpt["scalers"]
        self.find_threshold(val_loader, y_c15, y_c1h)
        torch.save({
            "model_state":   self.model.state_dict(),
            "scalers":       self.scalers,
            "threshold_15m": self.threshold_15m,
            "threshold_1h":  self.threshold_1h,
        }, MODEL_PATH)
        return self.model

    @torch.no_grad()
    def predict(self, sem, rag, macro, mem):
        self.model.eval()
        t = [torch.FloatTensor(s.transform(x)).to(device)
             for s, x in zip(self.scalers, [sem, rag, macro, mem])]
        out = self.model(*t)
        p15 = torch.sigmoid(out["cls_15m"]).cpu().numpy()
        p1h = torch.sigmoid(out["cls_1h"]).cpu().numpy()
        r15 = out["reg_15m"].cpu().numpy()
        r1h = out["reg_1h"].cpu().numpy()
        m15 = self._modulate(p15, r15)
        m1h = self._modulate(p1h, r1h)
        return {
            "prob_15m":       p15,
            "prob_1h":        p1h,
            "reg_pred_15m":   r15,
            "reg_pred_1h":    r1h,
            "reg_scale_15m":  1.0 + np.abs(r15) * REG_SCALE,
            "reg_scale_1h":   1.0 + np.abs(r1h) * REG_SCALE,
            "modulated_15m":  m15,
            "modulated_1h":   m1h,
            "pred_15m":       (m15 >= self.threshold_15m).astype(int),
            "pred_1h":        (m1h >= self.threshold_1h).astype(int),
            "confidence":     out["conf"].cpu().numpy(),
            "direction_pred": out["direction"].argmax(dim=1).cpu().numpy(),
        }

    def online_update(self, sem, rag, macro, mem,
                      y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf,
                      steps=2, lr=1e-5):
        self.replay.add(sem.flatten(), rag.flatten(), macro.flatten(), mem.flatten(),
                        y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf)
        sample = self.replay.sample(32)
        if sample is None: return
        self.model.train()
        opt  = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sems, rags, macs, mems, yr15, yc15, yr1h, yc1h, yds, ycfs = sample
        t    = [torch.FloatTensor(s.transform(x)).to(device)
                for s, x in zip(self.scalers, [sems, rags, macs, mems])]
        tgts = [torch.FloatTensor(yr15).to(device), torch.FloatTensor(yc15).to(device),
                torch.FloatTensor(yr1h).to(device), torch.FloatTensor(yc1h).to(device),
                torch.LongTensor(yds).to(device),   torch.FloatTensor(ycfs).to(device)]
        for _ in range(steps):
            opt.zero_grad()
            loss = self._loss(self.model(*t), *tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            opt.step()
        self.model.eval()


# ══════════════════════════════════════════════════════════════════
# 7. EVALUATION
# ══════════════════════════════════════════════════════════════════
def _eval_horizon(label, probs, modulated, threshold, y_cls, reg_pred, y_reg):
    preds   = (modulated >= threshold).astype(int)
    f1      = f1_score(y_cls, preds, zero_division=0)
    prec    = precision_score(y_cls, preds, zero_division=0)
    rec     = recall_score(y_cls, preds, zero_division=0)
    acc     = accuracy_score(y_cls, preds)
    auc     = roc_auc_score(y_cls, probs) if len(np.unique(y_cls)) > 1 else 0.0
    mae     = mean_absolute_error(y_reg, reg_pred)
    r2      = r2_score(y_reg, reg_pred)
    dir_acc = (np.sign(y_reg) == np.sign(reg_pred)).mean()
    cm      = confusion_matrix(y_cls, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ── {label} ({'15-minute' if '15' in label else '1-hour'}) ──")
    print(f"    Threshold : {threshold:.2f}  F1: {f1:.3f}  Prec: {prec:.3f}  "
          f"Rec: {rec:.3f}  AUC: {auc:.3f}")
    print(f"    Acc: {acc:.3f}  DirAcc: {dir_acc:.1%}  MAE: {mae:.4f}%  R²: {r2:.3f}")
    print(f"    Confusion: NO[{tn:>5} {fp:>5}]  YES[{fn:>5} {tp:>5}]")
    if tn == 0: print("    ⚠️  All predicted YES")
    if tp == 0: print("    ⚠️  All predicted NO")

    return {
        "F1": float(f1), "Precision": float(prec), "Recall": float(rec),
        "Accuracy": float(acc), "ROC_AUC": float(auc),
        "MAE": float(mae), "R2": float(r2), "DirAcc": float(dir_acc),
        "Threshold": float(threshold), "CM": cm.tolist(),
    }


def evaluate(trainer, sem, rag, macro, mem,
             y_r15, y_c15, y_r1h, y_c1h, y_dir, y_cf):
    print(f"\n{'='*65}\n  TEST SET EVALUATION — V5\n{'='*65}")
    preds = trainer.predict(sem, rag, macro, mem)

    r15 = _eval_horizon("15m", preds["prob_15m"], preds["modulated_15m"],
                         trainer.threshold_15m, y_c15, preds["reg_pred_15m"], y_r15)
    r1h = _eval_horizon("1h",  preds["prob_1h"],  preds["modulated_1h"],
                         trainer.threshold_1h,  y_c1h, preds["reg_pred_1h"],  y_r1h)

    dir_acc  = accuracy_score(y_dir, preds["direction_pred"])
    dir_f1   = f1_score(y_dir, preds["direction_pred"], zero_division=0)
    conf_mae = mean_absolute_error(y_cf, preds["confidence"])
    print(f"\n  Direction: Acc={dir_acc:.1%}  F1={dir_f1:.3f}")
    print(f"  Confidence MAE: {conf_mae:.4f}")

    print(f"\n{'─'*65}\n  COMPARISON\n{'─'*65}")
    for m in ["F1", "ROC_AUC", "Precision", "DirAcc"]:
        v15, v1h = r15[m], r1h[m]
        win = "1h" if v1h > v15 else "15m"
        print(f"  {m:<12}: 15m={v15:.3f}  1h={v1h:.3f}  winner={win}  "
              f"{'█' * int(abs(v1h - v15) * 100)}")

    results = {
        "15_minute":  r15, "1_hour": r1h,
        "direction":  {"Acc": float(dir_acc), "F1": float(dir_f1)},
        "confidence": {"MAE": float(conf_mae)},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_PATH}")
    return results


# ══════════════════════════════════════════════════════════════════
# 8. QUALITATIVE EXPLANATION
# ══════════════════════════════════════════════════════════════════
def generate_explanation(df, idx, preds, rag_row, rag_detail, threshold_15m):
    row    = df.iloc[idx]
    p15    = float(preds["prob_15m"][0])
    r15    = float(preds["reg_pred_15m"][0])
    sc15   = float(preds["reg_scale_15m"][0])
    mod15  = float(preds["modulated_15m"][0])
    pred15 = int(preds["pred_15m"][0])
    hour   = pd.Timestamp(row["published"]).hour
    dow    = pd.Timestamp(row["published"]).dayofweek

    ctx = []
    if dow >= 5:                ctx.append("Weekend")
    if 2 <= hour <= 6:          ctx.append("Low-liquidity (02-06 UTC)")
    if 13 <= hour <= 21:        ctx.append("US hours (13-21 UTC)")
    if hour <= 8:               ctx.append("Asia hours")
    if row.get("fomc_week", 0): ctx.append("FOMC week")

    lines = [
        f"Headline: \"{row['title'][:80]}\"",
        f"Channel:  {row['channel']} | {row['published']}",
        f"Context:  {' | '.join(ctx) or 'No special context'}",
        f"\n--- Classification ---",
        f"Raw cls prob   : {p15:.3f}",
        f"Reg prediction : {r15:+.3f}% (15m BTC)",
        f"Reg scale      : {sc15:.2f}x",
        f"Modulated score: {mod15:.3f} vs threshold {threshold_15m:.2f}",
        f"Decision       : {'✅ YES (IMPACTFUL)' if pred15 else '❌ NO'}",
        f"\n--- RAG Context (Qdrant + 5-feature macro-reweighted) ---",
        f"Weighted avg BTC: {float(rag_row[0]):+.3f}%",
        f"Max BTC change  : {float(rag_row[1]):+.3f}%",
        f"Weighted hit rt : {float(rag_row[3]):.1%}",
        f"Avg similarity  : {float(rag_row[8]):.3f}",
        f"Impactful/10    : {int(rag_row[9])}",
    ]

    similar = rag_detail.get("similar_news", [])[:3]
    if similar:
        lines.append("\n  Top 3 retrieved:")
        for s in similar:
            tags = []
            if s.get("weekend_match"): tags.append("📅WKD")
            if s.get("low_liq_match"): tags.append("🌙LOW-LIQ")
            if s.get("us_match"):      tags.append("🇺🇸US")
            t = " " + " ".join(tags) if tags else ""
            lines.append(
                f"    [{s['similarity_score']:.3f}|w={s.get('macro_weight',0):.3f}{t}] "
                f"BTC:{s['btc_change_15m']:+.2f}% | {s['title'][:55]}"
            )

    lines += [
        f"\n--- Ground Truth ---",
        f"Actual BTC 15m: {float(row['btc_change_15m']):+.3f}%  "
        f"({'IMPACTFUL' if row['is_impactful_15m'] else 'not impactful'})",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════
def make_loader(sem_s, rag_s, mac_s, mem_s,
                y_r15, y_c15, y_r1h, y_c1h, y_d, y_cf, shuffle=False):
    ds = TensorDataset(
        torch.FloatTensor(sem_s), torch.FloatTensor(rag_s),
        torch.FloatTensor(mac_s), torch.FloatTensor(mem_s),
        torch.FloatTensor(y_r15), torch.FloatTensor(y_c15),
        torch.FloatTensor(y_r1h), torch.FloatTensor(y_c1h),
        torch.LongTensor(y_d),    torch.FloatTensor(y_cf),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def main():
    print("=" * 65)
    print("  PRODUCTION MULTI-TOWER FUSION — v5")
    print(f"  seed={MONTHLY_SEED} | threshold_15m={THRESHOLD_15M} | 5-dim macro")
    print("=" * 65)

    df = load_data()

    # Split BEFORE features so channel rates use training data only
    print(f"\n[3/8] MONTHLY RANDOM SPLIT (seed={MONTHLY_SEED})")
    df["_ym"]  = df["published"].dt.to_period("M")
    all_months = sorted(df["_ym"].unique())
    rng        = np.random.default_rng(MONTHLY_SEED)
    shuffled   = np.array(all_months, dtype=object)
    rng.shuffle(shuffled)
    n_tr  = int(len(all_months) * 0.70)
    n_val = int(len(all_months) * 0.15)
    train_m = set(shuffled[:n_tr])
    val_m   = set(shuffled[n_tr:n_tr + n_val])
    test_m  = set(shuffled[n_tr + n_val:])
    tri_pre = np.where(df["_ym"].isin(train_m))[0]
    vi_pre  = np.where(df["_ym"].isin(val_m))[0]
    te_pre  = np.where(df["_ym"].isin(test_m))[0]
    df.drop(columns=["_ym"], inplace=True)
    print(f"  Train: {len(tri_pre):,} | Val: {len(vi_pre):,} | Test: {len(te_pre):,}")

    # Build features
    (sem, macro, rag, rag_details, market, ch_rates,
     y_r15, y_c15, y_r1h, y_c1h, y_dir, y_cf, _) = build_features(df, train_idx=tri_pre)

    tri    = tri_pre[np.argsort(df["published"].iloc[tri_pre].values)]
    vi     = vi_pre[np.argsort(df["published"].iloc[vi_pre].values)]
    te_idx = te_pre[np.argsort(df["published"].iloc[te_pre].values)]

    print(f"\n  Impact rate check (should be similar):")
    for name, idx in [("Train", tri), ("Val", vi), ("Test", te_idx)]:
        print(f"  {name:<5}: n={len(idx):5d}  "
              f"15m={y_c15[idx].mean():.1%}  "
              f"1h={y_c1h[idx].mean():.1%}  "
              f"avg|BTC|={np.abs(y_r15[idx]).mean():.3f}%")

    # Build model
    print(f"\n[4/8] BUILD MODEL")
    emb_proj = EmbProjection().to(device)
    model    = CryptoImpactNetV5(
        sem_dim=sem.shape[1], rag_dim=rag.shape[1],
        macro_dim=macro.shape[1], mem_dim=market.shape[1],
        emb_proj=emb_proj,
    )
    total = sum(p.numel() for p in model.parameters())
    ratio = total / len(tri)
    print(f"  CryptoImpactNetV5: {total:,} params | params/row: {ratio:.2f} "
          f"{'✅' if ratio < 1.0 else '⚠️ add more data'}")
    print(f"    Semantic : {sem.shape[1]} → (768→{EMB_PROJ_DIMS}) → 16")
    print(f"    RAG      : {rag.shape[1]} (Qdrant macro-reweighted) → 8")
    print(f"    Macro    : {macro.shape[1]} (5 features) → 8")
    print(f"    Market   : {market.shape[1]} (BTC momentum/vol) → 8")
    print(f"    Fusion   : 40 → 24 → 12")

    # Train
    print(f"\n[5/8] TRAINING  (MIN_PRECISION={MIN_PRECISION})")
    trainer = Trainer(model)
    trainer.fit_scalers(sem[tri], rag[tri], macro[tri], market[tri])

    sem_tr, rag_tr, mac_tr, mem_tr = trainer.scale(sem[tri], rag[tri], macro[tri], market[tri])
    sem_vl, rag_vl, mac_vl, mem_vl = trainer.scale(sem[vi],  rag[vi],  macro[vi],  market[vi])

    train_loader = make_loader(sem_tr, rag_tr, mac_tr, mem_tr,
                               y_r15[tri], y_c15[tri], y_r1h[tri], y_c1h[tri],
                               y_dir[tri], y_cf[tri], shuffle=True)
    val_loader   = make_loader(sem_vl, rag_vl, mac_vl, mem_vl,
                               y_r15[vi], y_c15[vi], y_r1h[vi], y_c1h[vi],
                               y_dir[vi], y_cf[vi])

    trainer.train(train_loader, val_loader, y_c15[vi], y_c1h[vi])

    # Evaluate
    print(f"\n[6/8] EVALUATION")
    evaluate(trainer,
             sem[te_idx], rag[te_idx], macro[te_idx], market[te_idx],
             y_r15[te_idx], y_c15[te_idx],
             y_r1h[te_idx], y_c1h[te_idx],
             y_dir[te_idx], y_cf[te_idx])

    # Qualitative examples
    print(f"\n[7/8] QUALITATIVE EXAMPLES")
    preds   = trainer.predict(sem[te_idx], rag[te_idx], macro[te_idx], market[te_idx])
    offsets = np.random.choice(len(te_idx), size=min(3, len(te_idx)), replace=False)
    for i, off in enumerate(offsets):
        single = {k: v[off:off+1] for k, v in preds.items() if hasattr(v, "__len__")}
        print(f"\n  {'='*55}\n  Example {i+1}:")
        print(generate_explanation(
            df, te_idx[off], single,
            rag[te_idx[off]], rag_details[te_idx[off]],
            trainer.threshold_15m,
        ))

    # Online learning demo
    print(f"\n[8/8] ONLINE LEARNING DEMO")
    for i in range(min(5, len(te_idx))):
        idx = te_idx[i]
        trainer.online_update(
            sem[idx:idx+1], rag[idx:idx+1], macro[idx:idx+1], market[idx:idx+1],
            y_r15[idx], y_c15[idx], y_r1h[idx], y_c1h[idx], y_dir[idx], y_cf[idx],
        )
    print(f"  Replay buffer: {len(trainer.replay)} samples")

    # Summary
    print(f"\n{'='*65}\n  SUMMARY — v5\n{'='*65}")
    print(f"  Params        : {total:,} | params/row: {ratio:.2f}")
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