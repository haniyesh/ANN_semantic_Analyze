"""
xgboost_train.py
================
Train XGBoost models on the same data/split as production_system_v8.pt
and produce a side-by-side comparison.

Features fed to XGBoost:
  PCA-50  : CryptoBERT 768-dim → PCA(50)   (fitted on train only)
  sent-6   : sentiment_score, weight, confidence,
             prob_positive, prob_negative, prob_neutral
  type-11  : cosine-sim news-type probabilities
  macro-5  : weekend, low_liq, us_hours, asia_hours, fomc_week
  ch-rate  : per-channel historical impact rate (train-only)
  rag-10   : Qdrant similarity features (zeros if Qdrant unavailable)
  ──────────────────────────────────────────────────────────────────
  Total    : 83 features

Models:
  xgb_cls15  binary  |btc_change_15m| ≥ 0.2%
  xgb_cls1h  binary  |btc_change_1h|  ≥ 0.4%
  xgb_dir    binary  btc_change_15m > 0  (eval on |change| > 0.5%)

Usage:
  .venv/bin/python xgboost_train.py
"""

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE          = Path(__file__).parent
SENTIMENT_CSV = HERE / "news_cleaned_filtered_scored.csv"
EMB_CACHE     = HERE / "cryptobert_v8_pipeline.npy"
MODEL_OUT     = HERE / "xgboost_model.json"
RESULTS_OUT   = HERE / "xgboost_results.json"

THRESHOLD_15M = 0.2
THRESHOLD_1H  = 0.4
MONTHLY_SEED  = 43       # same seed as v8 for identical train/val/test split
PCA_DIMS      = 50
OPTUNA_TRIALS = 40


# ── copy exact proto definitions from production_system_v8 ────────
sys.path.insert(0, str(HERE))
from production_system_v8 import (
    crypto_news_type_classify,
    NEWS_TYPE_LOSS_WEIGHTS,
    NEWS_TYPE_LABELS,
)

try:
    from pipeline.rag_news import build_rag_features_qdrant
    _HAS_RAG = True
except Exception:
    _HAS_RAG = False


# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA — identical to v8
# ══════════════════════════════════════════════════════════════════
def load_data():
    print("[1/6] Loading data...")
    df = pd.read_csv(SENTIMENT_CSV, low_memory=False)
    for col in ["btc_price_at_news", "btc_price_15m", "btc_price_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sentiment_score", "weight", "confidence",
                "prob_positive", "prob_negative", "prob_neutral"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df.drop_duplicates(subset=["title", "published", "channel"])

    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["is_impactful_15m"] = (df["abs_change_15m"] >= THRESHOLD_15M).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)

    df["btc_change_1h"]   = (df["btc_price_1h"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_1h"]   = df["btc_change_1h"].abs()
    df["is_impactful_1h"] = (df["abs_change_1h"] >= THRESHOLD_1H).astype(int)

    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    print(f"  Rows: {len(df):,}")
    print(f"  Impactful 15m: {df['is_impactful_15m'].mean()*100:.1f}%  "
          f"1h: {df['is_impactful_1h'].mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. SAME MONTHLY SPLIT AS v8 (seed=43)
# ══════════════════════════════════════════════════════════════════
def monthly_split(df):
    print(f"[2/6] Monthly split (seed={MONTHLY_SEED}, 70/15/15)...")
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

    tr_idx  = np.where(df["_ym"].isin(train_m))[0]
    val_idx = np.where(df["_ym"].isin(val_m))[0]
    te_idx  = np.where(df["_ym"].isin(test_m))[0]

    print(f"  Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}  "
          f"({len(train_m)}/{len(val_m)}/{len(test_m)} months)")
    return tr_idx, val_idx, te_idx


# ══════════════════════════════════════════════════════════════════
# 3. BUILD FEATURES
# ══════════════════════════════════════════════════════════════════
def build_features(df, tr_idx, te_idx):
    print("[3/6] Building features...")

    # CryptoBERT embeddings
    emb = np.load(EMB_CACHE).astype(np.float32)
    if len(emb) != len(df):
        n = min(len(emb), len(df))
        emb, df = emb[:n], df.iloc[:n].reset_index(drop=True)
    print(f"  Embeddings: {emb.shape}")

    # PCA on embeddings — fit ONLY on training rows
    print(f"  PCA({PCA_DIMS}) on embeddings...")
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    pca.fit(emb[tr_idx])
    emb_pca = pca.transform(emb).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance explained: {explained:.1%}")

    # News-type probabilities (11-dim) — same as v8
    print("  Computing news-type probs...")
    _, type_probs = crypto_news_type_classify(emb, return_probs=True)

    # Sentiment features (6-dim)
    sent_cols = ["sentiment_score", "weight", "confidence",
                 "prob_positive", "prob_negative", "prob_neutral"]
    sent_arr = df[sent_cols].fillna(0).values.astype(np.float32)

    # Macro features (5-dim) — identical formula to v8
    hour = df["published"].dt.hour.values
    dow  = df["published"].dt.dayofweek.values
    fomc = df["fomc_week"].fillna(0).values.astype(np.float32) if "fomc_week" in df.columns \
           else np.zeros(len(df), dtype=np.float32)
    macro = np.column_stack([
        (dow >= 5).astype(np.float32),
        ((hour >= 2) & (hour <= 6)).astype(np.float32),
        ((hour >= 13) & (hour <= 21)).astype(np.float32),
        ((hour >= 0) & (hour <= 8)).astype(np.float32),
        fomc,
    ]).astype(np.float32)

    # Channel impact rate (1-dim) — compute from training rows only
    ch_rates   = df.iloc[tr_idx].groupby("channel")["is_impactful_15m"].mean().to_dict()
    mean_rate  = float(df.iloc[tr_idx]["is_impactful_15m"].mean())
    ch_rate_col = np.array([ch_rates.get(c, mean_rate)
                            for c in df["channel"]], dtype=np.float32).reshape(-1, 1)

    # RAG features (10-dim) — from Qdrant, zeros if unavailable
    if _HAS_RAG:
        print("  Querying Qdrant for RAG features...")
        try:
            rag, _ = build_rag_features_qdrant(df, channel_impact_rates=ch_rates)
            print(f"  RAG: {rag.shape}")
        except Exception as e:
            print(f"  ⚠ RAG failed ({e}), using zeros")
            rag = np.zeros((len(df), 10), dtype=np.float32)
    else:
        print("  RAG: skipped (rag_news not available)")
        rag = np.zeros((len(df), 10), dtype=np.float32)

    # Concatenate all features
    X = np.hstack([emb_pca, sent_arr, type_probs, macro, ch_rate_col, rag]).astype(np.float32)
    print(f"  Feature matrix: {X.shape}  "
          f"(pca{PCA_DIMS}+sent6+type11+macro5+ch1+rag10)")

    # Normalise (StandardScaler on train only)
    scaler = StandardScaler()
    X[tr_idx] = scaler.fit_transform(X[tr_idx])
    mask_other = np.ones(len(df), dtype=bool)
    mask_other[tr_idx] = False
    X[mask_other] = scaler.transform(X[mask_other])

    return X, scaler, pca


# ══════════════════════════════════════════════════════════════════
# 4. OPTUNA TUNING + TRAIN
# ══════════════════════════════════════════════════════════════════
def _pos_weight(y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return neg / max(pos, 1)


def tune_and_train(X_tr, y_tr, X_val, y_val, task="cls15", n_trials=OPTUNA_TRIALS):
    pw = _pos_weight(y_tr)
    print(f"  class balance → pos_weight={pw:.2f}")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": pw,
            "objective":        "binary:logistic",
            "eval_metric":      "aucpr",
            "tree_method":      "hist",
            "device":           "cpu",
            "random_state":     42,
            "verbosity":        0,
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False)
        prob = clf.predict_proba(X_val)[:, 1]
        return f1_score(y_val, (prob >= 0.5).astype(int), zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "scale_pos_weight": pw,
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "tree_method":      "hist",
        "device":           "cpu",
        "random_state":     42,
        "verbosity":        0,
    })
    print(f"  Best trial F1={study.best_value:.4f}  params: {best}")

    clf = xgb.XGBClassifier(**best)
    clf.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False)
    return clf


# ══════════════════════════════════════════════════════════════════
# 5. FIND BEST THRESHOLD (Youden's J on validation set)
# ══════════════════════════════════════════════════════════════════
def find_threshold(clf, X_val, y_val):
    prob = clf.predict_proba(X_val)[:, 1]
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.20, 0.85, 0.01):
        f1 = f1_score(y_val, (prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return round(best_thr, 2)


# ══════════════════════════════════════════════════════════════════
# 6. EVALUATE
# ══════════════════════════════════════════════════════════════════
def evaluate_clf(clf, X_te, y_te, y_change, label, threshold):
    prob  = clf.predict_proba(X_te)[:, 1]
    pred  = (prob >= threshold).astype(int)
    f1    = f1_score(y_te, pred, zero_division=0)
    prec  = precision_score(y_te, pred, zero_division=0)
    rec   = recall_score(y_te, pred, zero_division=0)
    acc   = accuracy_score(y_te, pred)
    auc   = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else 0.0
    cm    = confusion_matrix(y_te, pred)
    tn, fp, fn, tp = cm.ravel()

    # Direction accuracy: sign of actual change vs predicted probability > 0.5
    dir_acc = (np.sign(y_change) == np.sign(prob - 0.5)).mean()

    print(f"\n  ── {label} ──")
    print(f"    Threshold: {threshold:.2f}  F1: {f1:.3f}  Prec: {prec:.3f}  "
          f"Rec: {rec:.3f}  AUC: {auc:.3f}")
    print(f"    Acc: {acc:.3f}  DirAcc: {dir_acc:.1%}")
    print(f"    Confusion: NO[{tn:>5} {fp:>5}]  YES[{fn:>5} {tp:>5}]")

    return {
        "F1": float(f1), "Precision": float(prec), "Recall": float(rec),
        "Accuracy": float(acc), "ROC_AUC": float(auc),
        "DirAcc": float(dir_acc), "Threshold": float(threshold),
        "CM": cm.tolist(),
    }


def evaluate_direction(clf_dir, X_te, y_dir, y_change):
    big  = np.abs(y_change) > 0.5
    n    = big.sum()
    if n == 0:
        print("  Direction: no big moves in test set")
        return {"Acc": 0.0, "F1": 0.0}
    prob = clf_dir.predict_proba(X_te[big])[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc  = accuracy_score(y_dir[big], pred)
    f1   = f1_score(y_dir[big], pred, zero_division=0)
    print(f"\n  ── Direction (|BTC|>0.5%, n={n:,}) ──")
    print(f"    Acc: {acc:.1%}  F1: {f1:.3f}")
    return {"Acc": float(acc), "F1": float(f1)}


# ══════════════════════════════════════════════════════════════════
# 7. COMPARE WITH v8
# ══════════════════════════════════════════════════════════════════
def compare_with_v8(xgb_res):
    v8_path = HERE / "production_results_v8.json"
    if not v8_path.exists():
        print("\n  (no production_results_v8.json found for comparison)")
        return

    with open(v8_path) as f:
        v8 = json.load(f)

    print(f"\n{'='*65}")
    print("  HEAD-TO-HEAD: XGBoost vs v8 Neural Net")
    print(f"{'='*65}")
    metrics = ["F1", "ROC_AUC", "Precision", "Recall", "Accuracy", "DirAcc"]

    for horizon, label in [("15_minute", "15m"), ("1_hour", "1h")]:
        print(f"\n  [{label.upper()}]")
        print(f"  {'Metric':<14} {'XGBoost':>10} {'v8 Net':>10} {'Winner':>10}")
        print(f"  {'─'*46}")
        for m in metrics:
            xv = xgb_res[horizon].get(m, 0)
            nv = v8[horizon].get(m, 0)
            winner = "XGBoost ✓" if xv > nv else "v8 Net  ✓" if nv > xv else "Tie"
            print(f"  {m:<14} {xv:>10.4f} {nv:>10.4f} {winner:>10}")

    print(f"\n  [Direction]")
    xd = xgb_res["direction"]["Acc"]
    nd = v8["direction"]["Acc"]
    winner = "XGBoost ✓" if xd > nd else "v8 Net  ✓" if nd > xd else "Tie"
    print(f"  {'DirAcc':<14} {xd:>10.4f} {nd:>10.4f} {winner:>10}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  XGBoost Training — same split/features as v8")
    print(f"  PCA_DIMS={PCA_DIMS}  OPTUNA_TRIALS={OPTUNA_TRIALS}")
    print("=" * 65)

    df = load_data()
    tr_idx, val_idx, te_idx = monthly_split(df)
    X, scaler, pca = build_features(df, tr_idx, te_idx)

    y_c15 = df["is_impactful_15m"].values
    y_c1h = df["is_impactful_1h"].values
    y_dir  = df["direction_15m"].values
    y_r15  = df["btc_change_15m"].values

    # Training subsets
    X_tr, X_val, X_te = X[tr_idx], X[val_idx], X[te_idx]
    splits = {
        "cls15": (y_c15[tr_idx], y_c15[val_idx], y_c15[te_idx]),
        "cls1h":  (y_c1h[tr_idx], y_c1h[val_idx], y_c1h[te_idx]),
        "dir":   (y_dir[tr_idx],  y_dir[val_idx],  y_dir[te_idx]),
    }

    print(f"\n[4/6] Tuning + training xgb_cls15 ({OPTUNA_TRIALS} trials)...")
    clf15 = tune_and_train(X_tr, splits["cls15"][0], X_val, splits["cls15"][1], "cls15")
    thr15 = find_threshold(clf15, X_val, splits["cls15"][1])
    print(f"  Best threshold: {thr15}")

    print(f"\n[5/6] Tuning + training xgb_cls1h ({OPTUNA_TRIALS} trials)...")
    clf1h = tune_and_train(X_tr, splits["cls1h"][0], X_val, splits["cls1h"][1], "cls1h")
    thr1h = find_threshold(clf1h, X_val, splits["cls1h"][1])
    print(f"  Best threshold: {thr1h}")

    print(f"\n[6/6] Training xgb_dir (direction) on big moves only...")
    big_tr  = np.abs(y_r15[tr_idx]) > 0.5
    big_val = np.abs(y_r15[val_idx]) > 0.5
    if big_tr.sum() > 100:
        clf_dir = tune_and_train(
            X_tr[big_tr], splits["dir"][0][big_tr],
            X_val[big_val] if big_val.sum() > 10 else X_val,
            splits["dir"][1][big_val] if big_val.sum() > 10 else splits["dir"][1],
            "dir", n_trials=20,
        )
    else:
        print("  Not enough big-move samples, training on full set")
        clf_dir = tune_and_train(X_tr, splits["dir"][0], X_val, splits["dir"][1], "dir", n_trials=20)

    # ── Evaluation ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TEST SET EVALUATION")
    print(f"{'='*65}")

    r15 = evaluate_clf(clf15, X_te, splits["cls15"][2], y_r15[te_idx], "15-minute", thr15)
    r1h = evaluate_clf(clf1h, X_te, splits["cls1h"][2], y_r15[te_idx], "1-hour",   thr1h)
    rdir = evaluate_direction(clf_dir, X_te, splits["dir"][2], y_r15[te_idx])

    results = {
        "15_minute": r15,
        "1_hour":    r1h,
        "direction": rdir,
        "features":  {"pca_dims": PCA_DIMS, "total": X.shape[1]},
        "model":     "XGBoost",
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_OUT}")

    # ── Feature importance (top 20) ───────────────────────────────
    feat_names = (
        [f"pca_{i}" for i in range(PCA_DIMS)] +
        ["sentiment_score", "weight", "confidence",
         "prob_positive", "prob_negative", "prob_neutral"] +
        [f"type_{i}" for i in range(11)] +
        ["is_weekend", "is_low_liq", "is_us_hours", "is_asia_hours", "fomc"] +
        ["ch_impact_rate"] +
        [f"rag_{i}" for i in range(10)]
    )
    imp = clf15.feature_importances_
    top20 = np.argsort(imp)[-20:][::-1]
    print(f"\n  Top 20 features (cls15):")
    for i in top20:
        name = feat_names[i] if i < len(feat_names) else f"f{i}"
        print(f"    {name:<25}: {imp[i]:.4f}  {'█' * int(imp[i] * 500)}")

    # ── Save model ────────────────────────────────────────────────
    clf15.save_model(HERE / "xgboost_cls15.json")
    clf1h.save_model(HERE / "xgboost_cls1h.json")
    clf_dir.save_model(HERE / "xgboost_dir.json")
    import pickle
    with open(HERE / "xgboost_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "thr15": thr15, "thr1h": thr1h}, f)
    print(f"\n  Models saved: xgboost_cls15.json / xgboost_cls1h.json / xgboost_dir.json")
    print(f"  Scaler/PCA:  xgboost_scaler.pkl")

    compare_with_v8(results)
    print(f"\n{'='*65}")
    print("  Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
