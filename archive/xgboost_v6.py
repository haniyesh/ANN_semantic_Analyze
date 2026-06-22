"""
XGBoost Comparison System — Crypto News Impact Prediction
==========================================================
Mirrors production_system_v6.py (ANN pipeline) exactly:
  - Same data loading & filtering
  - Same CryptoBERT embeddings (reuses cryptobert_v6_pipeline.npy cache)
  - Same feature engineering (semantic + macro + RAG + market)
  - Same train/val/test monthly split (seed=43)
  - Same evaluation metrics for fair comparison

Usage:
    python xgboost_comparison.py                  # train + evaluate
    python xgboost_comparison.py --compare         # also loads ANN results
    python xgboost_comparison.py --skip-rag        # skip RAG (faster debug run)
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
# CONFIG  — must match production_system_v6.py exactly
# ══════════════════════════════════════════════════════════════════
HERE             = Path(__file__).parent
SENTIMENT_CSV    = HERE / "news_cleaned_filtered_scored.csv"
CRYPTOBERT_CACHE = HERE / "cryptobert_v6_pipeline.npy"   # shared cache
XGB_MODEL_PATH   = HERE / "xgboost_model.json"
XGB_RESULTS_PATH = HERE / "xgboost_results.json"
ANN_RESULTS_PATH = HERE / "production_results_v6.json"   # from ANN run

THRESHOLD_15M = 0.3
THRESHOLD_1H  = 0.5
MONTHLY_SEED  = 43          # same split as ANN
MIN_PRECISION = 0.30

# ══════════════════════════════════════════════════════════════════
# NEWS TYPE PROTOTYPES  — same as v6
# ══════════════════════════════════════════════════════════════════
NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis"
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
    mat          = torch.stack(proto_embs)
    _proto_matrix = F.normalize(mat, dim=1)
    return _proto_matrix


def crypto_news_type_classify(embeddings: np.ndarray):
    proto   = _build_proto_matrix()
    emb_t   = F.normalize(torch.FloatTensor(embeddings), dim=1)
    sims    = torch.mm(emb_t, proto.T)
    probs   = F.softmax(sims * 5.0, dim=1).numpy().astype(np.float32)
    return probs


# ══════════════════════════════════════════════════════════════════
# 1. DATA  — identical to v6
# ══════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("[1/7] LOAD DATA")
    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["published"], low_memory=False)

    for col in ["btc_price_at_news", "btc_price_15m", "btc_price_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sentiment_score", "weight", "confidence",
                "prob_positive", "prob_negative", "prob_neutral"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    orig = len(df)
    df   = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df   = df.drop_duplicates(subset=["title", "published", "channel"])

    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["is_impactful_15m"] = (df["abs_change_15m"] >= THRESHOLD_15M).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)

    df["btc_change_1h"]   = (df["btc_price_1h"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_1h"]   = df["btc_change_1h"].abs()
    df["is_impactful_1h"] = (df["abs_change_1h"] >= THRESHOLD_1H).astype(int)
    df["direction_1h"]    = (df["btc_change_1h"] > 0).astype(int)

    df["confidence_label"] = df["abs_change_15m"].clip(0, 3) / 3.0
    df["published"]        = pd.to_datetime(df["published"], utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    print(f"  Rows: {len(df):,} (from {orig:,})")
    print(f"  Impactful 15m: {df['is_impactful_15m'].mean()*100:.1f}%  "
          f"1h: {df['is_impactful_1h'].mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. EMBEDDINGS  — reuses the same .npy cache as ANN
# ══════════════════════════════════════════════════════════════════
def compute_cryptobert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if CRYPTOBERT_CACHE.exists():
        emb = np.load(CRYPTOBERT_CACHE)
        if len(emb) == len(df):
            print("  CryptoBERT cache hit (shared with ANN)")
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
    return emb


# ══════════════════════════════════════════════════════════════════
# 3. FEATURES  — same towers as ANN, flattened for XGBoost
# ══════════════════════════════════════════════════════════════════
def build_macro_features(df: pd.DataFrame):

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
    df["fomc_week"]  = fomc_week

    macro  = np.column_stack([
        is_weekend, is_low_liquidity, is_us_hours, is_asia_hours, fomc_week
    ]).astype(np.float32)

    return macro


def build_features(df: pd.DataFrame, train_idx: np.ndarray, skip_rag: bool = False):
    print("[2/7] FEATURE ENGINEERING")

    emb = compute_cryptobert_embeddings(df)

    # Use pre-computed news_type from scored CSV if available,
    # otherwise fall back to cosine-similarity classification
    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        type_onehot = pd.get_dummies(df["news_type"].fillna("market_analysis"),
                                     columns=NEWS_TYPE_LABELS)
        for label in NEWS_TYPE_LABELS:
            if label not in type_onehot.columns:
                type_onehot[label] = 0
        type_probs = type_onehot[NEWS_TYPE_LABELS].values.astype(np.float32)
    else:
        type_probs = crypto_news_type_classify(emb)       # (N, 11)

    sent_cols = ["sentiment_score", "weight", "confidence",
                 "prob_positive", "prob_negative", "prob_neutral"]
    sent_df   = df[sent_cols].fillna(0).values.astype(np.float32)

    # ── Semantic block: CryptoBERT 768 + sentiment 6 + type_probs 11 = 785
    semantic  = np.hstack([emb, sent_df, type_probs]).astype(np.float32)
    print(f"  Semantic : {semantic.shape[1]} dims")

    macro = build_macro_features(df)
    print(f"  Macro    : {macro.shape[1]} dims")

    if skip_rag:
        print("  RAG      : SKIPPED (--skip-rag flag)")
        rag = np.zeros((len(df), 1), dtype=np.float32)
    else:
        from pipeline.rag_news import build_rag_features_qdrant
        ch_rates = df.iloc[train_idx].groupby("channel")["is_impactful_15m"].mean().to_dict()
        rag, _   = build_rag_features_qdrant(df, channel_impact_rates=ch_rates)
        print(f"  RAG      : {rag.shape[1]} dims")

    # ── Concatenate all towers into one flat feature matrix for XGBoost
    X = np.hstack([semantic, macro, rag]).astype(np.float32)
    print(f"  Total features: {X.shape[1]}")

    feat_names = (
        [f"emb_{i}"   for i in range(768)] +
        sent_cols +
        [f"type_{l}"  for l in NEWS_TYPE_LABELS] +
        ["is_weekend", "is_low_liq", "is_us_hours", "is_asia_hours", "fomc_week"] +
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
# 4. XGBOOST MODELS
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

    # Class imbalance ratio — same logic as ANN pos_weight
    neg_count = int((y_c15_tr == 0).sum())
    pos_count = int((y_c15_tr == 1).sum())
    scale_pos = neg_count / max(pos_count, 1)
    print(f"  Class balance — neg: {neg_count:,}  pos: {pos_count:,}  "
          f"scale_pos_weight: {scale_pos:.2f}")

    # ── Shared hyperparameters
    base_params = dict(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.6,
        min_child_weight  = 5,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = MONTHLY_SEED,
        n_jobs            = -1,
        early_stopping_rounds = 20,
        eval_metric       = "logloss",
    )

    # ── 15-minute classifier
    print("  Training clf_15m...")
    clf_15m = xgb.XGBClassifier(
        **base_params,
        objective         = "binary:logistic",
        scale_pos_weight  = scale_pos,
    )
    clf_15m.fit(
        X_tr, y_c15_tr,
        eval_set=[(X_vl, y_c15_vl)],
        verbose=False,
    )
    print(f"    Best iteration: {clf_15m.best_iteration}")

    # ── 1-hour classifier
    neg_1h   = int((y_c1h_tr == 0).sum())
    pos_1h   = int((y_c1h_tr == 1).sum())
    scale_1h = neg_1h / max(pos_1h, 1)
    print("  Training clf_1h...")
    clf_1h = xgb.XGBClassifier(
        **base_params,
        objective         = "binary:logistic",
        scale_pos_weight  = scale_1h,
    )
    clf_1h.fit(
        X_tr, y_c1h_tr,
        eval_set=[(X_vl, y_c1h_vl)],
        verbose=False,
    )
    print(f"    Best iteration: {clf_1h.best_iteration}")

    # ── Regression model (15m BTC % change)
    print("  Training reg_15m...")
    reg_params = {k: v for k, v in base_params.items()
                  if k not in ["scale_pos_weight", "eval_metric"]}
    reg_params["eval_metric"] = "rmse"
    reg_15m = xgb.XGBRegressor(
        **reg_params,
        objective = "reg:squarederror",
    )
    reg_15m.fit(
        X_tr, y_r15_tr,
        eval_set=[(X_vl, y_r15_tr[:len(X_vl)])],   # val reg target
        verbose=False,
    )

    return clf_15m, clf_1h, reg_15m


# ══════════════════════════════════════════════════════════════════
# 5. THRESHOLD SEARCH  — same logic as ANN trainer.find_threshold
# ══════════════════════════════════════════════════════════════════
def find_threshold(probs: np.ndarray, y_cls: np.ndarray,
                   label: str = "15m") -> float:
    best_f1, best_t, found = 0, 0.30, False
    for t in np.linspace(0.02, 0.90, 177):
        preds = (probs >= t).astype(int)
        prec  = precision_score(y_cls, preds, zero_division=0)
        if prec < MIN_PRECISION:
            continue
        f1 = f1_score(y_cls, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, found = f1, t, True
    if not found:
        print(f"    {label}: WARNING — fallback threshold=0.30")
    else:
        print(f"    {label}: threshold={best_t:.2f}  val F1={best_f1:.3f}")
    return best_t


# ══════════════════════════════════════════════════════════════════
# 6. EVALUATION  — same metrics as ANN _eval_horizon
# ══════════════════════════════════════════════════════════════════
def eval_horizon(label, probs, threshold, y_cls, reg_pred, y_reg):
    preds   = (probs >= threshold).astype(int)
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

    print(f"\n  ── {label} ──")
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


# ══════════════════════════════════════════════════════════════════
# 7. ANN vs XGBoost COMPARISON PRINTER
# ══════════════════════════════════════════════════════════════════
def print_comparison(xgb_results: dict):
    if not ANN_RESULTS_PATH.exists():
        print(f"\n  ⚠️  ANN results not found at {ANN_RESULTS_PATH}")
        print("  Run production_system_v6.py first, then re-run with --compare")
        return

    with open(ANN_RESULTS_PATH) as f:
        ann = json.load(f)

    metrics = ["F1", "Precision", "Recall", "Accuracy", "ROC_AUC", "DirAcc"]

    print(f"\n{'='*70}")
    print(f"  ANN  vs  XGBoost — HEAD TO HEAD COMPARISON")
    print(f"{'='*70}")

    for horizon, ann_key, xgb_key in [
        ("15-minute", "15_minute", "15_minute"),
        ("1-hour",    "1_hour",    "1_hour"),
    ]:
        print(f"\n  ── {horizon} ──")
        print(f"  {'Metric':<14} {'ANN':>10} {'XGBoost':>10} {'Winner':>10}  {'Δ':>8}")
        print(f"  {'─'*56}")
        ann_h = ann.get(ann_key, {})
        xgb_h = xgb_results.get(xgb_key, {})
        for m in metrics:
            a = ann_h.get(m, 0.0)
            x = xgb_h.get(m, 0.0)
            diff   = x - a
            winner = "XGBoost ✅" if x > a else ("ANN ✅" if a > x else "TIE")
            bar    = "▲" * min(int(abs(diff) * 50), 10) if diff > 0 else "▼" * min(int(abs(diff) * 50), 10)
            print(f"  {m:<14} {a:>10.3f} {x:>10.3f} {winner:>10}  {diff:>+7.3f} {bar}")

    print(f"\n{'─'*70}")
    print(f"  VERDICT")
    print(f"{'─'*70}")

    ann_avg_f1 = (ann.get("15_minute", {}).get("F1", 0) + ann.get("1_hour", {}).get("F1", 0)) / 2
    xgb_avg_f1 = (xgb_results.get("15_minute", {}).get("F1", 0) + xgb_results.get("1_hour", {}).get("F1", 0)) / 2

    print(f"  ANN     avg F1: {ann_avg_f1:.3f}")
    print(f"  XGBoost avg F1: {xgb_avg_f1:.3f}")
    if xgb_avg_f1 > ann_avg_f1:
        print(f"  🏆 XGBoost wins by {xgb_avg_f1 - ann_avg_f1:+.3f} F1")
    elif ann_avg_f1 > xgb_avg_f1:
        print(f"  🏆 ANN wins by {ann_avg_f1 - xgb_avg_f1:+.3f} F1")
    else:
        print(f"  🤝 TIE")


# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — top 20 non-embedding features
# ══════════════════════════════════════════════════════════════════
def print_feature_importance(clf_15m, feat_names: list, top_n: int = 20):
    import xgboost as xgb
    importances = clf_15m.feature_importances_
    pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])

    # Filter out raw embedding dims (emb_0 .. emb_767) for readability
    meaningful = [(n, v) for n, v in pairs if not n.startswith("emb_")]

    print(f"\n{'─'*50}")
    print(f"  TOP {top_n} FEATURES (non-embedding) — clf_15m")
    print(f"{'─'*50}")
    for name, val in meaningful[:top_n]:
        bar = "█" * int(val * 500)
        print(f"  {name:<30} {val:.4f}  {bar}")

    # Aggregate embedding block importance
    emb_total = sum(v for n, v in zip(feat_names, importances) if n.startswith("emb_"))
    print(f"\n  Embedding block (768 dims) total importance: {emb_total:.4f}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare",   action="store_true",
                        help="Print ANN vs XGBoost comparison table")
    parser.add_argument("--skip-rag",  action="store_true",
                        help="Skip RAG features (faster debug run)")
    parser.add_argument("--load-only", action="store_true",
                        help="Load saved model and only evaluate (skip training)")
    args = parser.parse_args()

    print("=" * 65)
    print("  XGBOOST COMPARISON SYSTEM — Crypto News Impact")
    print(f"  seed={MONTHLY_SEED} | threshold_15m={THRESHOLD_15M}")
    print("=" * 65)

    # ── 1. Data
    df = load_data()

    # ── 2. Same monthly split as ANN
    print(f"\n[3/7] MONTHLY RANDOM SPLIT (seed={MONTHLY_SEED})")
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
    tri     = np.where(df["_ym"].isin(train_m))[0]
    vi      = np.where(df["_ym"].isin(val_m))[0]
    te_idx  = np.where(df["_ym"].isin(test_m))[0]
    df.drop(columns=["_ym"], inplace=True)
    print(f"  Train: {len(tri):,} | Val: {len(vi):,} | Test: {len(te_idx):,}")

    # ── 3. Features
    (X, feat_names,
     y_r15, y_c15, y_r1h, y_c1h, y_dir) = build_features(df, tri, skip_rag=args.skip_rag)

    # Scale features
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[tri]).astype(np.float32)
    X_vl   = scaler.transform(X[vi]).astype(np.float32)
    X_te   = scaler.transform(X[te_idx]).astype(np.float32)

    # ── 4. Train or load
    if args.load_only and XGB_MODEL_PATH.exists():
        import xgboost as xgb
        print(f"\n[4/7] LOADING saved models from {XGB_MODEL_PATH}")
        clf_15m = xgb.XGBClassifier(); clf_15m.load_model(str(XGB_MODEL_PATH).replace(".json", "_clf15m.json"))
        clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(str(XGB_MODEL_PATH).replace(".json", "_clf1h.json"))
        reg_15m = xgb.XGBRegressor();  reg_15m.load_model(str(XGB_MODEL_PATH).replace(".json", "_reg15m.json"))
    else:
        clf_15m, clf_1h, reg_15m = train_xgboost_models(
            X_tr, X_vl,
            y_c15[tri], y_c15[vi],
            y_c1h[tri], y_c1h[vi],
            y_r15[tri], y_r1h[tri],
        )
        # Save models
        clf_15m.save_model(str(XGB_MODEL_PATH).replace(".json", "_clf15m.json"))
        clf_1h.save_model(str(XGB_MODEL_PATH).replace(".json", "_clf1h.json"))
        reg_15m.save_model(str(XGB_MODEL_PATH).replace(".json", "_reg15m.json"))
        print(f"  Models saved → {XGB_MODEL_PATH.parent}")

    # ── 5. Threshold search on validation set
    print(f"\n[5/7] THRESHOLD SEARCH (min_precision={MIN_PRECISION})")
    p15_vl  = clf_15m.predict_proba(X_vl)[:, 1]
    p1h_vl  = clf_1h.predict_proba(X_vl)[:, 1]
    thr_15m = find_threshold(p15_vl, y_c15[vi], "15m")
    thr_1h  = find_threshold(p1h_vl, y_c1h[vi], "1h")

    # ── 6. Test set evaluation
    print(f"\n[6/7] TEST SET EVALUATION")
    print(f"{'='*65}\n  XGBoost TEST RESULTS\n{'='*65}")

    p15_te   = clf_15m.predict_proba(X_te)[:, 1]
    p1h_te   = clf_1h.predict_proba(X_te)[:, 1]
    r15_te   = reg_15m.predict(X_te)
    dir_pred = (p15_te >= 0.5).astype(int)   # direction from 15m prob

    r15 = eval_horizon("15-minute", p15_te, thr_15m, y_c15[te_idx], r15_te, y_r15[te_idx])
    r1h = eval_horizon("1-hour",    p1h_te, thr_1h,  y_c1h[te_idx], r15_te, y_r1h[te_idx])

    dir_acc = accuracy_score(y_dir[te_idx], dir_pred)
    dir_f1  = f1_score(y_dir[te_idx], dir_pred, zero_division=0)
    print(f"\n  Direction: Acc={dir_acc:.1%}  F1={dir_f1:.3f}")

    xgb_results = {
        "15_minute":  r15,
        "1_hour":     r1h,
        "direction":  {"Acc": float(dir_acc), "F1": float(dir_f1)},
        "threshold_15m": float(thr_15m),
        "threshold_1h":  float(thr_1h),
    }
    with open(XGB_RESULTS_PATH, "w") as f:
        json.dump(xgb_results, f, indent=2)
    print(f"\n  Saved → {XGB_RESULTS_PATH}")

    # ── 7. Feature importance
    print_feature_importance(clf_15m, feat_names)

    # ── 8. Comparison (if --compare or ANN results already exist)
    if args.compare or ANN_RESULTS_PATH.exists():
        print_comparison(xgb_results)

    # ── Summary
    print(f"\n{'='*65}\n  SUMMARY — XGBoost\n{'='*65}")
    print(f"  Features      : {X.shape[1]} total")
    print(f"  Train/Val/Test: {len(tri):,} / {len(vi):,} / {len(te_idx):,}")
    print(f"  Threshold 15m : {thr_15m:.2f}")
    print(f"  Threshold 1h  : {thr_1h:.2f}")
    print(f"  clf_15m F1    : {r15['F1']:.3f}")
    print(f"  clf_1h  F1    : {r1h['F1']:.3f}")
    print(f"  Results       → {XGB_RESULTS_PATH}")


if __name__ == "__main__":
    main()