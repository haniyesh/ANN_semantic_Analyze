"""
XGBoost Comparison System — v9
================================
Mirrors v9.py (ANN pipeline) exactly:
  - Same dual CryptoBERT+FinBERT embeddings (DUAL_EMB_DIM=1536)
  - Same ensemble sentiment features (3-BERT: cb/fb/rb probs + net_agreement)
  - Same price-context macro features (btc_vol, btc_mom, fear_greed)
  - Same RAG features (Qdrant macro-conditioned)
  - Same monthly split (seed=43)
  - Same threshold search (MIN_PRECISION=0.20, fallback=0.50)

Usage:
    python xgboost_v9.py                 # train + evaluate
    python xgboost_v9.py --compare       # also show ANN vs XGBoost table
    python xgboost_v9.py --skip-rag      # skip RAG (faster debug run)
    python xgboost_v9.py --load-only     # load saved models, skip training
"""

import os, json, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, mean_absolute_error, r2_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG  — mirrors v9.py
# ══════════════════════════════════════════════════════════════════
HERE             = Path(__file__).parent
SENTIMENT_CSV    = HERE / "news_cleaned_filtered_scored.csv"
CRYPTOBERT_CACHE = HERE / "cryptobert_v8_pipeline.npy"
FINBERT_CACHE    = HERE / "finbert_v9_pipeline.npy"
FEAR_GREED_CACHE = HERE / "fear_greed_cache.json"
FINBERT_MODEL    = "ProsusAI/finbert"
DUAL_EMB_DIM     = 768 + 768

XGB_MODEL_BASE   = HERE / "xgboost_v9"
XGB_RESULTS_PATH = HERE / "xgboost_v9_results.json"
ANN_RESULTS_PATH = HERE / "production_results_v9.json"

THRESHOLD_15M = 0.3
THRESHOLD_1H  = 0.5
MONTHLY_SEED  = 43
MIN_PRECISION = 0.20   # same as v9.py

# ══════════════════════════════════════════════════════════════════
# NEWS TYPE PROTOTYPES  — same as v9
# ══════════════════════════════════════════════════════════════════
NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis",
]

_NEWS_TYPE_PROTOTYPES_TEXT = {
    "regulatory":     ["SEC charges crypto exchange securities violations", "government bans cryptocurrency trading country"],
    "etf":            ["Bitcoin ETF approved by SEC trading", "spot bitcoin fund launches stock exchange"],
    "hack":           ["crypto exchange hacked millions stolen", "DeFi protocol exploited flash loan attack"],
    "macro_economic": ["Federal Reserve raises interest rates decision", "inflation data CPI report released"],
    "exchange":       ["Binance lists new cryptocurrency token", "Coinbase delists token regulatory concerns"],
    "defi":           ["DeFi protocol TVL record liquidity", "Uniswap launches new version features"],
    "mining":         ["Bitcoin mining difficulty adjusts record", "miner capitulation hashrate drops significantly"],
    "institutional":  ["MicroStrategy purchases Bitcoin treasury reserve", "hedge fund allocates Bitcoin portfolio"],
    "technical":      ["Bitcoin network upgrade soft fork activates", "Ethereum developers confirm upgrade date"],
    "partnership":    ["crypto company partnership bank deal", "blockchain firm integrates payment processor"],
    "market_analysis":["Bitcoin price analysis bullish breakout target", "technical analysis support level tested"],
}

_proto_matrix = None

def _build_proto_matrix():
    global _proto_matrix
    if _proto_matrix is not None:
        return _proto_matrix
    from transformers import AutoTokenizer, AutoModel
    print("  Building news type prototype embeddings (one-time)...")
    tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    mdl = AutoModel.from_pretrained("ElKulako/cryptobert").eval()
    proto_embs = []
    for label in NEWS_TYPE_LABELS:
        sentences = _NEWS_TYPE_PROTOTYPES_TEXT[label]
        inputs    = tok(sentences, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            emb = mdl(**inputs).last_hidden_state[:, 0, :]
        proto_embs.append(emb.mean(dim=0))
    mat = torch.stack(proto_embs)
    _proto_matrix = F.normalize(mat, dim=1)
    return _proto_matrix


def crypto_news_type_classify(embeddings: np.ndarray) -> np.ndarray:
    proto = _build_proto_matrix()
    emb_t = F.normalize(torch.FloatTensor(embeddings), dim=1)
    sims  = torch.mm(emb_t, proto.T)
    return F.softmax(sims * 5.0, dim=1).numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("[1/7] LOAD DATA")
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

    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    print(f"  Rows: {len(df):,} (from {orig:,})")
    print(f"  Impactful 15m: {df['is_impactful_15m'].mean()*100:.1f}%  "
          f"1h: {df['is_impactful_1h'].mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. EMBEDDINGS
# ══════════════════════════════════════════════════════════════════
def compute_cryptobert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if CRYPTOBERT_CACHE.exists():
        emb = np.load(CRYPTOBERT_CACHE).astype(np.float32)
        if len(emb) == len(df):
            print("  CryptoBERT cache hit")
            return emb

    print(f"  Computing CryptoBERT embeddings for {len(df):,} rows...")
    from transformers import AutoTokenizer, AutoModel
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model     = AutoModel.from_pretrained("ElKulako/cryptobert").eval().to(device)
    titles, embs = df["title"].fillna("").tolist(), []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)} ({i*100//len(titles)}%)...")
            inputs = tokenizer(titles[i:i+32], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embs.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    emb = np.vstack(embs).astype(np.float32)
    np.save(CRYPTOBERT_CACHE, emb)
    return emb


def compute_finbert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if FINBERT_CACHE.exists():
        emb = np.load(FINBERT_CACHE).astype(np.float32)
        if len(emb) == len(df):
            print("  FinBERT cache hit")
            return emb

    print(f"  Computing FinBERT embeddings for {len(df):,} rows...")
    from transformers import AutoTokenizer, AutoModel
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModel.from_pretrained(FINBERT_MODEL).eval().to(device)
    titles, embs = df["title"].fillna("").tolist(), []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)} ({i*100//len(titles)}%)...")
            inputs = tokenizer(titles[i:i+32], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embs.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    emb = np.vstack(embs).astype(np.float32)
    np.save(FINBERT_CACHE, emb)
    print(f"  FinBERT done: {emb.shape}")
    return emb


# ══════════════════════════════════════════════════════════════════
# 3. PRICE CONTEXT FEATURES
# ══════════════════════════════════════════════════════════════════
def fetch_fear_greed_index(published_series: pd.Series) -> np.ndarray:
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
    changes = pd.Series(df["btc_change_15m"].values.astype(np.float32))
    shifted = changes.shift(1)
    btc_vol = shifted.rolling(window=20, min_periods=2).std().fillna(0).values.astype(np.float32)
    btc_mom = shifted.rolling(window=5,  min_periods=1).mean().fillna(0).values.astype(np.float32)
    fg      = fetch_fear_greed_index(df["published"])
    return np.column_stack([btc_vol, btc_mom, fg]).astype(np.float32)


def build_macro_features(df: pd.DataFrame) -> np.ndarray:
    hour = df["published"].dt.hour.values
    dow  = df["published"].dt.dayofweek.values
    is_weekend       = (dow >= 5).astype(np.float32)
    is_low_liquidity = ((hour >= 2) & (hour <= 6)).astype(np.float32)
    is_us_hours      = ((hour >= 13) & (hour <= 21)).astype(np.float32)
    is_asia_hours    = ((hour >= 0) & (hour <= 8)).astype(np.float32)
    fomc_week        = df["fomc_week"].fillna(0).values.astype(np.float32) if "fomc_week" in df.columns else np.zeros(len(df), dtype=np.float32)
    return np.column_stack([is_weekend, is_low_liquidity, is_us_hours, is_asia_hours, fomc_week]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING  — flat matrix for XGBoost
# ══════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, train_idx: np.ndarray, skip_rag: bool = False):
    print("[2/7] FEATURE ENGINEERING")

    cb_emb = compute_cryptobert_embeddings(df)   # (N, 768)
    fb_emb = compute_finbert_embeddings(df)       # (N, 768)

    # Type classification uses CryptoBERT only
    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        type_onehot = pd.get_dummies(df["news_type"].fillna("market_analysis"))
        for label in NEWS_TYPE_LABELS:
            if label not in type_onehot.columns:
                type_onehot[label] = 0
        type_probs = type_onehot[NEWS_TYPE_LABELS].values.astype(np.float32)
    else:
        type_probs = crypto_news_type_classify(cb_emb)

    # Ensemble sentiment (3-BERT) or legacy
    ensemble_cols = [
        "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
        "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
        "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
        "net_agreement",
    ]
    if all(c in df.columns for c in ensemble_cols):
        extra     = ["sentiment_score", "weight", "confidence"]
        sent_cols = ensemble_cols + extra
        print(f"  Using ensemble sentiment features ({len(sent_cols)} dims)")
    else:
        sent_cols = ["sentiment_score", "weight", "confidence",
                     "prob_positive", "prob_negative", "prob_neutral"]
        print(f"  Using legacy sentiment features ({len(sent_cols)} dims)")

    sent_df = df[sent_cols].fillna(0).values.astype(np.float32)

    # Macro: 5 timing + 3 price context
    timing_mac = build_macro_features(df)
    price_ctx  = compute_price_context(df)
    macro      = np.hstack([timing_mac, price_ctx]).astype(np.float32)

    print(f"  Semantic : {DUAL_EMB_DIM} emb + {len(sent_cols)} sent + 11 type = {DUAL_EMB_DIM + len(sent_cols) + 11} dims")
    print(f"  Macro    : {macro.shape[1]} dims (5 timing + 3 price ctx)")

    if skip_rag:
        print("  RAG      : SKIPPED (--skip-rag flag)")
        rag = np.zeros((len(df), 1), dtype=np.float32)
    else:
        from pipeline.rag_news import build_rag_features_qdrant
        ch_rates = df.iloc[train_idx].groupby("channel")["is_impactful_15m"].mean().to_dict()
        rag, _   = build_rag_features_qdrant(df, channel_impact_rates=ch_rates)
        print(f"  RAG      : {rag.shape[1]} dims")

    X = np.hstack([cb_emb, fb_emb, sent_df, type_probs, macro, rag]).astype(np.float32)
    print(f"  Total features: {X.shape[1]}")

    feat_names = (
        [f"cb_{i}"  for i in range(768)] +
        [f"fb_{i}"  for i in range(768)] +
        list(sent_cols) +
        [f"type_{l}" for l in NEWS_TYPE_LABELS] +
        ["is_weekend", "is_low_liq", "is_us_hours", "is_asia_hours", "fomc_week",
         "btc_vol", "btc_mom", "fear_greed"] +
        ([f"rag_{i}" for i in range(rag.shape[1])] if not skip_rag else ["rag_dummy"])
    )

    return (
        X, feat_names,
        df["btc_change_15m"].values.astype(np.float32),
        df["is_impactful_15m"].values.astype(np.float32),
        df["btc_change_1h"].values.astype(np.float32),
        df["is_impactful_1h"].values.astype(np.float32),
        df["direction_15m"].values.astype(np.int64),
    )


# ══════════════════════════════════════════════════════════════════
# 5. XGBOOST MODELS
# ══════════════════════════════════════════════════════════════════
def train_xgboost_models(X_tr, X_vl,
                          y_c15_tr, y_c15_vl,
                          y_c1h_tr, y_c1h_vl,
                          y_r15_tr, y_r1h_tr):
    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost not found. Installing...")
        os.system("pip install xgboost --break-system-packages -q")
        import xgboost as xgb

    print(f"\n[4/7] TRAINING XGBOOST MODELS")

    neg_15 = int((y_c15_tr == 0).sum())
    pos_15 = int((y_c15_tr == 1).sum())
    neg_1h = int((y_c1h_tr == 0).sum())
    pos_1h = int((y_c1h_tr == 1).sum())
    print(f"  15m — neg: {neg_15:,}  pos: {pos_15:,}  scale_pos_weight: {neg_15/pos_15:.2f}")
    print(f"  1h  — neg: {neg_1h:,}  pos: {pos_1h:,}  scale_pos_weight: {neg_1h/pos_1h:.2f}")

    base_params = dict(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.6,
        min_child_weight      = 5,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        random_state          = MONTHLY_SEED,
        device                = "cuda",
        tree_method           = "hist",
        early_stopping_rounds = 20,
        eval_metric           = "logloss",
    )

    print("  Training clf_15m...")
    clf_15m = xgb.XGBClassifier(
        **base_params,
        objective        = "binary:logistic",
        scale_pos_weight = neg_15 / pos_15,
    )
    clf_15m.fit(X_tr, y_c15_tr, eval_set=[(X_vl, y_c15_vl)], verbose=False)
    print(f"    Best iteration: {clf_15m.best_iteration}")

    print("  Training clf_1h...")
    clf_1h = xgb.XGBClassifier(
        **base_params,
        objective        = "binary:logistic",
        scale_pos_weight = neg_1h / pos_1h,
    )
    clf_1h.fit(X_tr, y_c1h_tr, eval_set=[(X_vl, y_c1h_vl)], verbose=False)
    print(f"    Best iteration: {clf_1h.best_iteration}")

    reg_params = {k: v for k, v in base_params.items()
                  if k not in ["scale_pos_weight", "eval_metric", "n_jobs"]}
    reg_params["eval_metric"] = "rmse"
    print("  Training reg_15m...")
    reg_15m = xgb.XGBRegressor(**reg_params, objective="reg:squarederror")
    reg_15m.fit(X_tr, y_r15_tr, eval_set=[(X_vl, y_r15_tr[:len(X_vl)])], verbose=False)

    return clf_15m, clf_1h, reg_15m


# ══════════════════════════════════════════════════════════════════
# 6. THRESHOLD SEARCH  — same as v9.py (fallback=0.50)
# ══════════════════════════════════════════════════════════════════
def find_threshold(probs: np.ndarray, y_cls: np.ndarray, label: str = "15m") -> float:
    best_f1, best_t, found = 0.0, 0.50, False
    for t in np.linspace(0.02, 0.90, 177):
        preds = (probs >= t).astype(int)
        prec  = precision_score(y_cls, preds, zero_division=0)
        if prec < MIN_PRECISION:
            continue
        f1 = f1_score(y_cls, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, found = f1, t, True
    if not found:
        print(f"    {label}: WARNING — fallback threshold=0.50")
    else:
        print(f"    {label}: threshold={best_t:.2f}  val F1={best_f1:.3f}")
    return best_t


# ══════════════════════════════════════════════════════════════════
# 7. EVALUATION  — same metrics as v9 _eval_horizon
# ══════════════════════════════════════════════════════════════════
def eval_horizon(label, probs, threshold, y_cls, reg_pred, y_reg):
    preds  = (probs >= threshold).astype(int)
    f1     = f1_score(y_cls, preds, zero_division=0)
    prec   = precision_score(y_cls, preds, zero_division=0)
    rec    = recall_score(y_cls, preds, zero_division=0)
    acc    = accuracy_score(y_cls, preds)
    auc    = roc_auc_score(y_cls, probs) if len(np.unique(y_cls)) > 1 else 0.0
    mae    = mean_absolute_error(y_reg, reg_pred)
    r2     = r2_score(y_reg, reg_pred)
    dir_acc = (np.sign(y_reg) == np.sign(reg_pred)).mean()
    cm     = confusion_matrix(y_cls, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ── {label} ──")
    print(f"    Threshold : {threshold:.2f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}  AUC: {auc:.3f}")
    print(f"    Acc: {acc:.3f}  DirAcc: {dir_acc:.1%}  MAE: {mae:.4f}%  R²: {r2:.3f}")
    print(f"    Confusion: NO[TN={tn:>5} FP={fp:>5}]  YES[FN={fn:>5} TP={tp:>5}]")
    if tn == 0: print("    ⚠️  All predicted YES")
    if tp == 0: print("    ⚠️  All predicted NO")

    return {
        "F1": float(f1), "Precision": float(prec), "Recall": float(rec),
        "Accuracy": float(acc), "ROC_AUC": float(auc),
        "MAE": float(mae), "R2": float(r2), "DirAcc": float(dir_acc),
        "Threshold": float(threshold), "CM": cm.tolist(),
    }


# ══════════════════════════════════════════════════════════════════
# ANN vs XGBoost COMPARISON
# ══════════════════════════════════════════════════════════════════
def print_comparison(xgb_results: dict):
    if not ANN_RESULTS_PATH.exists():
        print(f"\n  ⚠️  v9 ANN results not found at {ANN_RESULTS_PATH}")
        return

    with open(ANN_RESULTS_PATH) as f:
        ann = json.load(f)

    metrics = ["F1", "Precision", "Recall", "Accuracy", "ROC_AUC"]

    print(f"\n{'='*70}")
    print(f"  v9 ANN  vs  XGBoost v9 — HEAD TO HEAD")
    print(f"{'='*70}")

    for horizon, key in [("15-minute", "15_minute"), ("1-hour", "1_hour")]:
        print(f"\n  ── {horizon} ──")
        print(f"  {'Metric':<14} {'ANN':>10} {'XGBoost':>10} {'Winner':>12}  {'Δ':>8}")
        print(f"  {'─'*58}")
        ann_h = ann.get(key, {})
        xgb_h = xgb_results.get(key, {})
        for m in metrics:
            a = ann_h.get(m, 0.0)
            x = xgb_h.get(m, 0.0)
            diff   = x - a
            winner = "XGBoost ✅" if x > a else ("ANN ✅" if a > x else "TIE")
            bar    = "▲" * min(int(abs(diff) * 50), 10) if diff > 0 else "▼" * min(int(abs(diff) * 50), 10)
            print(f"  {m:<14} {a:>10.3f} {x:>10.3f} {winner:>12}  {diff:>+7.3f} {bar}")

    ann_avg = (ann.get("15_minute", {}).get("F1", 0) + ann.get("1_hour", {}).get("F1", 0)) / 2
    xgb_avg = (xgb_results.get("15_minute", {}).get("F1", 0) + xgb_results.get("1_hour", {}).get("F1", 0)) / 2
    print(f"\n  ANN avg F1     : {ann_avg:.3f}")
    print(f"  XGBoost avg F1 : {xgb_avg:.3f}")
    if xgb_avg > ann_avg:
        print(f"  🏆 XGBoost wins by {xgb_avg - ann_avg:+.3f} F1")
    elif ann_avg > xgb_avg:
        print(f"  🏆 ANN wins by {ann_avg - xgb_avg:+.3f} F1")
    else:
        print(f"  🤝 TIE")


def print_feature_importance(clf_15m, feat_names: list, top_n: int = 20):
    importances = clf_15m.feature_importances_
    pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    meaningful = [(n, v) for n, v in pairs if not n.startswith("cb_") and not n.startswith("fb_")]

    print(f"\n{'─'*50}")
    print(f"  TOP {top_n} FEATURES (non-embedding) — clf_15m")
    print(f"{'─'*50}")
    for name, val in meaningful[:top_n]:
        bar = "█" * int(val * 500)
        print(f"  {name:<30} {val:.4f}  {bar}")

    emb_total = sum(v for n, v in zip(feat_names, importances)
                    if n.startswith("cb_") or n.startswith("fb_"))
    print(f"\n  Dual embedding block ({DUAL_EMB_DIM} dims) total importance: {emb_total:.4f}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare",   action="store_true", help="Print ANN vs XGBoost table")
    parser.add_argument("--skip-rag",  action="store_true", help="Skip RAG (faster debug run)")
    parser.add_argument("--load-only", action="store_true", help="Load saved models, skip training")
    args = parser.parse_args()

    print("=" * 65)
    print("  XGBOOST v9 — Crypto News Impact  (DualBERT + PriceCtx + GPU)")
    print(f"  seed={MONTHLY_SEED}  threshold_15m={THRESHOLD_15M}  min_precision={MIN_PRECISION}")
    print("=" * 65)

    df = load_data()

    print(f"\n[3/7] MONTHLY RANDOM SPLIT (seed={MONTHLY_SEED})")
    df["_ym"]  = df["published"].dt.to_period("M")
    months     = sorted(df["_ym"].unique())
    rng        = np.random.default_rng(MONTHLY_SEED)
    shuffled   = np.array(months, dtype=object)
    rng.shuffle(shuffled)
    n_tr  = int(len(months) * 0.70)
    n_val = int(len(months) * 0.15)
    train_m = set(shuffled[:n_tr])
    val_m   = set(shuffled[n_tr:n_tr + n_val])
    test_m  = set(shuffled[n_tr + n_val:])
    tri     = np.where(df["_ym"].isin(train_m))[0]
    vi      = np.where(df["_ym"].isin(val_m))[0]
    te_idx  = np.where(df["_ym"].isin(test_m))[0]
    df.drop(columns=["_ym"], inplace=True)
    print(f"  Train: {len(tri):,} | Val: {len(vi):,} | Test: {len(te_idx):,}")

    (X, feat_names,
     y_r15, y_c15, y_r1h, y_c1h, y_dir) = build_features(df, tri, skip_rag=args.skip_rag)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[tri]).astype(np.float32)
    X_vl   = scaler.transform(X[vi]).astype(np.float32)
    X_te   = scaler.transform(X[te_idx]).astype(np.float32)

    clf15_path = str(XGB_MODEL_BASE) + "_clf15m.json"
    clf1h_path = str(XGB_MODEL_BASE) + "_clf1h.json"
    reg15_path = str(XGB_MODEL_BASE) + "_reg15m.json"

    if args.load_only and Path(clf15_path).exists():
        import xgboost as xgb
        print(f"\n[4/7] LOADING saved models")
        clf_15m = xgb.XGBClassifier(); clf_15m.load_model(clf15_path)
        clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(clf1h_path)
        reg_15m = xgb.XGBRegressor();  reg_15m.load_model(reg15_path)
    else:
        clf_15m, clf_1h, reg_15m = train_xgboost_models(
            X_tr, X_vl,
            y_c15[tri], y_c15[vi],
            y_c1h[tri], y_c1h[vi],
            y_r15[tri], y_r1h[tri],
        )
        clf_15m.save_model(clf15_path)
        clf_1h.save_model(clf1h_path)
        reg_15m.save_model(reg15_path)
        import pickle
        scaler_path = str(XGB_MODEL_BASE) + "_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Models + scaler saved → {XGB_MODEL_BASE}*")

    print(f"\n[5/7] THRESHOLD SEARCH (min_precision={MIN_PRECISION})")
    p15_vl  = clf_15m.predict_proba(X_vl)[:, 1]
    p1h_vl  = clf_1h.predict_proba(X_vl)[:, 1]
    thr_15m = find_threshold(p15_vl, y_c15[vi], "15m")
    thr_1h  = find_threshold(p1h_vl, y_c1h[vi], "1h")

    print(f"\n[6/7] TEST SET EVALUATION")
    print(f"{'='*65}\n  XGBoost v9 TEST RESULTS\n{'='*65}")

    p15_te   = clf_15m.predict_proba(X_te)[:, 1]
    p1h_te   = clf_1h.predict_proba(X_te)[:, 1]
    r15_te   = reg_15m.predict(X_te)
    dir_pred = (p15_te >= 0.5).astype(int)

    r15 = eval_horizon("15-minute", p15_te, thr_15m, y_c15[te_idx], r15_te, y_r15[te_idx])
    r1h = eval_horizon("1-hour",    p1h_te, thr_1h,  y_c1h[te_idx], r15_te, y_r1h[te_idx])

    dir_acc = accuracy_score(y_dir[te_idx], dir_pred)
    dir_f1  = f1_score(y_dir[te_idx], dir_pred, zero_division=0)
    print(f"\n  Direction: Acc={dir_acc:.1%}  F1={dir_f1:.3f}")

    xgb_results = {
        "15_minute": r15,
        "1_hour":    r1h,
        "direction": {"Acc": float(dir_acc), "F1": float(dir_f1)},
        "threshold_15m": float(thr_15m),
        "threshold_1h":  float(thr_1h),
    }
    with open(XGB_RESULTS_PATH, "w") as f:
        json.dump(xgb_results, f, indent=2)
    print(f"\n  Saved → {XGB_RESULTS_PATH}")

    print_feature_importance(clf_15m, feat_names)

    if args.compare or ANN_RESULTS_PATH.exists():
        print_comparison(xgb_results)

    print(f"\n{'='*65}\n  SUMMARY — XGBoost v9\n{'='*65}")
    print(f"  Features      : {X.shape[1]} total")
    print(f"  Train/Val/Test: {len(tri):,} / {len(vi):,} / {len(te_idx):,}")
    print(f"  Threshold 15m : {thr_15m:.2f}")
    print(f"  Threshold 1h  : {thr_1h:.2f}")
    print(f"  clf_15m F1    : {r15['F1']:.3f}")
    print(f"  clf_1h  F1    : {r1h['F1']:.3f}")
    print(f"  Results       → {XGB_RESULTS_PATH}")


if __name__ == "__main__":
    main()
