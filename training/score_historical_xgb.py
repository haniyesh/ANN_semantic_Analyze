"""
score_historical_xgb.py
========================
Score news_cleaned_filtered_scored.csv using XGBoost v9 models
and write results to news_cache.json (live items preserved).

Features mirror xgboost_v9.py exactly:
  - Dual BERT embeddings (CryptoBERT + FinBERT, 1536 dims)
  - Ensemble 3-BERT sentiment (13 cols)
  - News-type probs (11 dims)
  - Macro timing (5) + price context (3) = 8 dims
  - RAG = zeros (same convention as score_historical.py)

Usage:
    .venv311/bin/python score_historical_xgb.py
"""

import re
import json, pickle, warnings, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from xgboost_v9 import (
    compute_cryptobert_embeddings,
    compute_finbert_embeddings,
    build_macro_features,
    compute_price_context,
    crypto_news_type_classify,
    NEWS_TYPE_LABELS,
    DUAL_EMB_DIM,
    XGB_MODEL_BASE,
    THRESHOLD_15M, THRESHOLD_1H,
    MONTHLY_SEED,
)

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

CSV_PATH   = HERE / "news_cleaned_filtered_scored_pre_ensemble.csv"
CACHE_FILE = HERE / "news_cache.json"

from pipeline.reduce_noise import (
    BLOCKED_CHANNELS, NOISE_TITLE_RE, CRYPTO_KW_RE,
    CRYPTO_FILTERED_CHANNELS, passes_news_filter,
)

MONTHS_WINDOW = 21   # cover all google_news (Oct 2024 → Jun 2026)

CLF15_PATH   = str(XGB_MODEL_BASE) + "_clf15m.json"
CLF1H_PATH   = str(XGB_MODEL_BASE) + "_clf1h.json"
SCALER_PATH  = str(XGB_MODEL_BASE) + "_scaler.pkl"
RESULTS_PATH = HERE / "xgboost_v9_results.json"


# ── 1. Load CSV (same filters as xgboost_v9.load_data) ───────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for col in df.columns:
        orig = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df[col] = orig

    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df[~df["channel"].isin(BLOCKED_CHANNELS)]

    # Noise filter
    noise_mask   = df["title"].str.contains(NOISE_TITLE_RE, na=False)
    short_mask   = df["title"].str.len() < 30
    crypto_ch    = df["channel"].isin(CRYPTO_FILTERED_CHANNELS)
    no_kw_mask   = ~df["title"].str.contains(CRYPTO_KW_RE, na=False)
    off_topic    = crypto_ch & no_kw_mask
    drop_mask    = noise_mask | short_mask | off_topic
    removed      = drop_mask.sum()
    df = df[~drop_mask].reset_index(drop=True)
    print(f"  Noise filter removed {removed:,} rows "
          f"({noise_mask.sum()} noise titles, {off_topic.sum()} off-topic, {short_mask.sum()} short)")

    df = df.drop_duplicates(subset=["title", "published", "channel"])
    df["btc_change_15m"] = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["btc_change_1h"]  = (df["btc_price_1h"]  - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    # Keep only the most recent MONTHS_WINDOW months
    cutoff = df["published"].max() - pd.DateOffset(months=MONTHS_WINDOW)
    df = df[df["published"] >= cutoff].reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows  (last {MONTHS_WINDOW} months: {cutoff.date()} → {df['published'].max().date()})")
    return df


# ── 2. Build feature matrix (matches xgboost_v9.build_features) ──
def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list, int]:
    cb_emb = compute_cryptobert_embeddings(df)
    fb_emb = compute_finbert_embeddings(df)

    # Type probs from CryptoBERT only
    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        type_onehot = pd.get_dummies(df["news_type"].fillna("market_analysis"))
        for label in NEWS_TYPE_LABELS:
            if label not in type_onehot.columns:
                type_onehot[label] = 0
        type_probs = type_onehot[NEWS_TYPE_LABELS].values.astype(np.float32)
    else:
        type_probs = crypto_news_type_classify(cb_emb)

    # Ensemble or legacy sentiment
    ensemble_cols = [
        "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
        "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
        "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
        "net_agreement",
    ]
    if all(c in df.columns for c in ensemble_cols):
        sent_cols = ensemble_cols + ["sentiment_score", "weight", "confidence"]
    else:
        sent_cols = ["sentiment_score", "weight", "confidence",
                     "prob_positive", "prob_negative", "prob_neutral"]
    sent_df = df[sent_cols].fillna(0).values.astype(np.float32)

    timing_mac = build_macro_features(df)
    price_ctx  = compute_price_context(df)
    macro      = np.hstack([timing_mac, price_ctx]).astype(np.float32)

    # RAG = zeros; actual dim determined from model after loading
    return np.hstack([cb_emb, fb_emb, sent_df, type_probs, macro]).astype(np.float32), sent_cols, type_probs.shape[1]


# ── 3. Load models & scaler ───────────────────────────────────────
def load_models():
    clf_15m = xgb.XGBClassifier(); clf_15m.load_model(CLF15_PATH)
    clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(CLF1H_PATH)

    # Derive RAG dims from expected feature count
    n_expected = clf_15m.n_features_in_
    print(f"  Model expects {n_expected} features")

    if Path(SCALER_PATH).exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("  Scaler loaded from file")
    else:
        scaler = None
        print("  ⚠  Scaler not found — will rebuild from training split")

    thr_15m, thr_1h = 0.295, 0.265
    if RESULTS_PATH.exists():
        res = json.loads(RESULTS_PATH.read_text())
        thr_15m = res.get("threshold_15m", thr_15m)
        thr_1h  = res.get("threshold_1h",  thr_1h)
    print(f"  Thresholds — 15m: {thr_15m:.3f}  1h: {thr_1h:.3f}")

    return clf_15m, clf_1h, scaler, n_expected, thr_15m, thr_1h


# ── 4. Rebuild scaler from training split if not saved ───────────
def rebuild_scaler(X_all: np.ndarray, df: pd.DataFrame) -> StandardScaler:
    print("  Rebuilding scaler from training split (seed=43, 70%)...")
    df["_ym"] = df["published"].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    rng = np.random.default_rng(MONTHLY_SEED)
    shuffled = np.array(months, dtype=object)
    rng.shuffle(shuffled)
    n_tr = int(len(months) * 0.70)
    train_months = set(shuffled[:n_tr])
    train_idx = np.where(df["_ym"].isin(train_months))[0]
    df.drop(columns=["_ym"], inplace=True)
    scaler = StandardScaler()
    scaler.fit(X_all[train_idx])
    print(f"  Scaler fit on {len(train_idx):,} training rows")
    return scaler


# ── 5. Convert predictions to cache items ────────────────────────
def to_cache_items(df: pd.DataFrame, p15: np.ndarray, p1h: np.ndarray,
                   thr_15m: float, thr_1h: float) -> list:
    items = []
    for i in range(len(df)):
        row = df.iloc[i]

        prob15 = float(p15[i])
        prob1h = float(p1h[i])

        pub_dt = row["published"]
        if pd.isnull(pub_dt):
            continue
        pub_ts = int(pub_dt.timestamp())

        btc_p  = float(row.get("btc_price_at_news") or 0)
        btc_15 = float(row.get("btc_price_15m") or 0)
        btc_1h = float(row.get("btc_price_1h")  or 0)
        c15m   = (btc_15 - btc_p) / btc_p * 100 if btc_p else 0
        c1h    = (btc_1h - btc_p) / btc_p * 100 if btc_p else 0

        sentiment = str(row.get("sentiment", "neutral") or "neutral")
        sig_type  = "BUY" if sentiment == "positive" else "SELL" if sentiment == "negative" else "NEUTRAL"

        impact = "High" if max(prob15, prob1h) >= 0.50 else ("Medium" if max(prob15, prob1h) >= 0.25 else "Low")

        items.append({
            "id":              f"hist_{pub_ts}_{hash(str(row.get('title', ''))[:30]) % 100000}",
            "time":            pub_dt.strftime("%H:%M:%S"),
            "title":           str(row.get("title", "")),
            "link":            str(row.get("link", "")),
            "channel":         str(row.get("channel", "unknown")),
            "published":       pub_dt.isoformat(),
            "published_ts":    pub_ts,
            "sentiment":       sentiment,
            "sentiment_score": float(row.get("sentiment_score") or 0),
            "confidence":      round(float(row.get("confidence") or 0) * 100, 1),
            "weight":          float(row.get("weight") or 0),
            "prob_positive":   float(row.get("prob_positive") or 0),
            "prob_negative":   float(row.get("prob_negative") or 0),
            "prob_neutral":    float(row.get("prob_neutral")  or 0),
            "type":            sig_type,
            "btc_change_15m":  round(c15m, 4),
            "btc_change_1h":   round(c1h,  4),
            "model_score":     round(prob15, 4),
            "model_score_1h":  round(prob1h, 4),
            "score_normalized": True,
            "pred_15m":        int(prob15 >= thr_15m),
            "pred_1h":         int(prob1h >= thr_1h),
            "direction":       int(float(row.get("btc_change_15m") or 0) > 0),
            "impact":          impact,
            "news_type":       str(row.get("news_type", "")),
            "source":          "historical_xgb_v9",
        })
    return items


# ── 6. Preserve live items from existing cache ───────────────────
def load_live_cache() -> list:
    if not CACHE_FILE.exists():
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("news", [])
        # Keep only live items (not any historical scored run or old live_xgb sources)
        HIST_SOURCES = {"historical_v6", "historical_v8", "historical_xgb_v9"}
        return [x for x in items if x.get("source") not in HIST_SOURCES]
    except Exception:
        return []


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SCORE HISTORICAL — XGBoost v9  (DualBERT + PriceCtx)")
    print("=" * 60)

    print("\n[1/5] Loading CSV...")
    df = load_data()

    print("\n[2/5] Building features (dual BERT + price ctx)...")
    X_no_rag, sent_cols, type_dim = build_features(df)
    print(f"  Features (no RAG): {X_no_rag.shape[1]}")

    print("\n[3/5] Loading models...")
    clf_15m, clf_1h, scaler, n_expected, thr_15m, thr_1h = load_models()

    # Pad with zero RAG features to match model's expected input
    rag_dim = n_expected - X_no_rag.shape[1]
    if rag_dim < 0:
        raise ValueError(f"Feature count mismatch: built {X_no_rag.shape[1]}, model expects {n_expected}")
    rag = np.zeros((len(df), rag_dim), dtype=np.float32)
    X   = np.hstack([X_no_rag, rag]).astype(np.float32)
    print(f"  RAG padding: {rag_dim} zero dims  →  total: {X.shape[1]}")

    if scaler is None:
        scaler = rebuild_scaler(X, df.copy())
        # Save for next time
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler saved → {SCALER_PATH}")

    X_scaled = scaler.transform(X).astype(np.float32)

    print(f"\n[4/5] Running inference on {len(df):,} rows...")
    p15 = clf_15m.predict_proba(X_scaled)[:, 1]
    p1h = clf_1h.predict_proba(X_scaled)[:, 1]
    print(f"  Mean prob 15m: {p15.mean():.3f}  1h: {p1h.mean():.3f}")
    print(f"  Pred impactful 15m: {(p15 >= thr_15m).mean()*100:.1f}%  "
          f"1h: {(p1h >= thr_1h).mean()*100:.1f}%")

    print("\n[5/5] Writing news_cache.json...")
    hist_items = to_cache_items(df, p15, p1h, thr_15m, thr_1h)
    print(f"  Scored items (prob >= 0.35): {len(hist_items):,}")

    live_items = load_live_cache()
    print(f"  Live items preserved: {len(live_items):,}")

    merged = sorted(
        live_items + hist_items,
        key=lambda x: x.get("published_ts") or 0,
    )

    payload = {
        "metadata": {
            "total_items":  len(merged),
            "live_items":   len(live_items),
            "hist_items":   len(hist_items),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model":        "xgboost_v9",
        },
        "news": merged,
    }
    CACHE_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    channels = {}
    for item in merged:
        ch = item.get("channel", "unknown")
        channels[ch] = channels.get(ch, 0) + 1

    print(f"\n{'='*60}")
    print(f"  ✅ Saved {len(merged):,} items → {CACHE_FILE.name}")
    print(f"  Channels:")
    for ch, cnt in sorted(channels.items(), key=lambda x: -x[1]):
        print(f"    {ch:<35}: {cnt:,}")


if __name__ == "__main__":
    main()
