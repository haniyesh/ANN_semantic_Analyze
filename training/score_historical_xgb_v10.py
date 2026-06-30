"""
score_historical_xgb_v10.py
============================
Score the last 3 months of news_cleaned_filtered_scored.csv using
XGBoost v10 (Groq/Llama-3.3-70B sentiment) and write to news_cache.json.

Changes from v9:
  - Groq LLM sentiment features instead of 3-BERT ensemble
  - MONTHS_WINDOW = 3 (last 3 months only, dashboard-optimised)
  - source field: "historical_xgb_v10"

Usage:
    cd /project_root
    .venv311/bin/python training/score_historical_xgb_v10.py
    .venv311/bin/python training/score_historical_xgb_v10.py --dry-run   # no write
"""

import sys, json, pickle, warnings, argparse, hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE   = Path(__file__).parent
ROOT   = HERE.parent
sys.path.insert(0, str(ROOT))

from training.xgboost_v10_groq import (
    compute_cryptobert_embeddings,
    compute_finbert_embeddings,
    build_macro_features,
    compute_price_context,
    crypto_news_type_classify,
    build_groq_features,
    NEWS_TYPE_LABELS,
    DUAL_EMB_DIM,
    XGB_MODEL_BASE,
    GROQ_CACHE,
)

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from pipeline.reduce_noise import (
    BLOCKED_CHANNELS, NOISE_TITLE_RE,
    CRYPTO_KW_RE, CRYPTO_FILTERED_CHANNELS,
)

CSV_PATH     = ROOT / "news_cleaned_filtered_scored.csv"
CACHE_FILE   = ROOT / "news_cache.json"
CLF15_PATH   = str(XGB_MODEL_BASE) + "_clf15m.json"
CLF1H_PATH   = str(XGB_MODEL_BASE) + "_clf1h.json"
SCALER_PATH  = str(XGB_MODEL_BASE) + "_scaler.pkl"
RESULTS_PATH = ROOT / "xgboost_v10_groq_results.json"
MONTHLY_SEED = 43
MONTHS_WINDOW = 1   # last 1 month only


# ── 1. Load CSV ──────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for col in df.columns:
        orig = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df[col] = orig

    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df[~df["channel"].isin(BLOCKED_CHANNELS)]

    noise_mask = df["title"].str.contains(NOISE_TITLE_RE, na=False)
    short_mask = df["title"].str.len() < 30
    crypto_ch  = df["channel"].isin(CRYPTO_FILTERED_CHANNELS)
    no_kw_mask = ~df["title"].str.contains(CRYPTO_KW_RE, na=False)
    drop_mask  = noise_mask | short_mask | (crypto_ch & no_kw_mask)
    df = df[~drop_mask].reset_index(drop=True)

    df = df.drop_duplicates(subset=["title", "published", "channel"])
    df["btc_change_15m"] = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["btc_change_1h"]  = (df["btc_price_1h"]  - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    cutoff = df["published"].max() - pd.DateOffset(months=MONTHS_WINDOW)
    df = df[df["published"] >= cutoff].reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows  (last {MONTHS_WINDOW}m: {cutoff.date()} → {df['published'].max().date()})")
    return df


# ── 2. Build features (v10: Groq sentiment) ──────────────────────
def build_features(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    cb_emb = compute_cryptobert_embeddings(df)
    fb_emb = compute_finbert_embeddings(df)

    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        type_onehot = pd.get_dummies(df["news_type"].fillna("market_analysis"))
        for label in NEWS_TYPE_LABELS:
            if label not in type_onehot.columns:
                type_onehot[label] = 0
        type_probs = type_onehot[NEWS_TYPE_LABELS].values.astype(np.float32)
    else:
        type_probs = crypto_news_type_classify(cb_emb)

    # Groq sentiment (3 one-hot dims) + scalar features from CSV
    print("  Loading Groq sentiment from cache...")
    groq_feats  = build_groq_features(df)          # (N, 3)
    scalar_cols = ["sentiment_score", "weight", "confidence"]
    scalar_df   = df[scalar_cols].fillna(0).values.astype(np.float32)
    sent_feats  = np.hstack([groq_feats, scalar_df])   # (N, 6)

    timing_mac = build_macro_features(df)
    price_ctx  = compute_price_context(df)
    macro      = np.hstack([timing_mac, price_ctx]).astype(np.float32)

    X = np.hstack([cb_emb, fb_emb, sent_feats, type_probs, macro]).astype(np.float32)
    print(f"  Features (no RAG): {X.shape[1]}")
    return X, type_probs.shape[1]


# ── 3. Load models ───────────────────────────────────────────────
def load_models():
    if not Path(CLF15_PATH).exists():
        print(f"  ✗ Model not found: {CLF15_PATH}")
        print(f"    Run training/xgboost_v10_groq.py first.")
        sys.exit(1)

    clf_15m = xgb.XGBClassifier(); clf_15m.load_model(CLF15_PATH)
    clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(CLF1H_PATH)
    n_expected = clf_15m.n_features_in_
    print(f"  Model expects {n_expected} features")

    scaler = None
    if Path(SCALER_PATH).exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("  Scaler loaded")
    else:
        print("  ⚠ Scaler not found — will rebuild from data")

    thr_15m, thr_1h = 0.30, 0.50
    if RESULTS_PATH.exists():
        res = json.loads(RESULTS_PATH.read_text())
        thr_15m = res.get("threshold_15m", thr_15m)
        thr_1h  = res.get("threshold_1h",  thr_1h)
    print(f"  Thresholds — 15m: {thr_15m:.3f}  1h: {thr_1h:.3f}")

    return clf_15m, clf_1h, scaler, n_expected, thr_15m, thr_1h


# ── 4. Rebuild scaler if missing ─────────────────────────────────
def rebuild_scaler(X: np.ndarray, df: pd.DataFrame) -> StandardScaler:
    print("  Rebuilding scaler (70% split, seed=43)...")
    df = df.copy()
    df["_ym"] = df["published"].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    rng = np.random.default_rng(MONTHLY_SEED)
    shuffled = np.array(months, dtype=object)
    rng.shuffle(shuffled)
    n_tr = max(1, int(len(months) * 0.70))
    train_months = set(shuffled[:n_tr])
    train_idx = np.where(df["_ym"].isin(train_months))[0]
    if len(train_idx) == 0:
        train_idx = np.arange(len(X))
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    print(f"  Scaler fit on {len(train_idx):,} rows")
    return scaler


# ── 5. Groq cache coverage check ─────────────────────────────────
def check_groq_coverage(df: pd.DataFrame):
    cache = {}
    if GROQ_CACHE.exists():
        with open(GROQ_CACHE) as f:
            cache = json.load(f)
    titles  = df["title"].fillna("").tolist()
    cached  = sum(1 for t in titles if hashlib.md5(t.encode()).hexdigest() in cache)
    missing = len(titles) - cached
    pct     = cached / len(titles) * 100 if titles else 0
    print(f"  Groq cache coverage: {cached:,}/{len(titles):,} ({pct:.1f}%)")
    if missing > 0:
        print(f"  ⚠ {missing:,} titles not in cache → will show as 'neutral'")
        print(f"    To fill: python training/xgboost_v10_groq.py --groq-sample {len(titles)} --groq-limit {missing}")


# ── 6. Convert to cache items ────────────────────────────────────
def to_cache_items(df: pd.DataFrame, p15: np.ndarray, p1h: np.ndarray,
                   thr_15m: float, thr_1h: float) -> list:
    items = []
    for i in range(len(df)):
        row    = df.iloc[i]
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
        sig_type  = "BUY" if sentiment == "positive" else ("SELL" if sentiment == "negative" else "NEUTRAL")

        score = max(prob15, prob1h)
        impact = "High" if score >= 0.50 else ("Medium" if score >= 0.25 else "Low")

        items.append({
            "id":              f"hist_{pub_ts}_{hash(str(row.get('title',''))[:30]) % 100000}",
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
            "source":          "historical_xgb_v10",
        })
    return items


# ── 7. Preserve live items from existing cache (last 1 month only) ──
def load_live_cache() -> list:
    if not CACHE_FILE.exists():
        return []
    try:
        from datetime import timezone
        data  = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("news", [])
        HIST_SOURCES = {"historical_v6", "historical_v8", "historical_xgb_v9", "historical_xgb_v10"}
        cutoff_ts = (datetime.now(timezone.utc) - pd.DateOffset(months=MONTHS_WINDOW)).timestamp()
        live = [x for x in items
                if x.get("source") not in HIST_SOURCES
                and (x.get("published_ts") or 0) >= cutoff_ts]
        return live
    except Exception:
        return []


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Score but don't write news_cache.json")
    args = parser.parse_args()

    print("=" * 62)
    print("  SCORE HISTORICAL — XGBoost v10 (Groq/Llama-3.3-70B)  3M")
    print("=" * 62)

    print("\n[1/5] Loading data (last 3 months)...")
    df = load_data()

    print("\n[2/5] Building features...")
    check_groq_coverage(df)
    X_no_rag, type_dim = build_features(df)

    print("\n[3/5] Loading models...")
    clf_15m, clf_1h, scaler, n_expected, thr_15m, thr_1h = load_models()

    # Pad with zero RAG dims to match model
    rag_dim = n_expected - X_no_rag.shape[1]
    if rag_dim < 0:
        print(f"  ⚠ Feature mismatch: built {X_no_rag.shape[1]}, model expects {n_expected}")
        print(f"    Retrain v10 with --skip-rag to match, or check feature dims.")
        sys.exit(1)
    rag = np.zeros((len(df), rag_dim), dtype=np.float32)
    X   = np.hstack([X_no_rag, rag]).astype(np.float32)
    print(f"  RAG padding: {rag_dim} dims  →  total: {X.shape[1]}")

    if scaler is None:
        scaler = rebuild_scaler(X, df.copy())
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    X_scaled = scaler.transform(X).astype(np.float32)

    print(f"\n[4/5] Running inference on {len(df):,} rows...")
    p15 = clf_15m.predict_proba(X_scaled)[:, 1]
    p1h = clf_1h.predict_proba(X_scaled)[:, 1]
    print(f"  Mean prob 15m: {p15.mean():.3f}  1h: {p1h.mean():.3f}")
    print(f"  Pred impactful 15m: {(p15 >= thr_15m).mean()*100:.1f}%  "
          f"1h: {(p1h >= thr_1h).mean()*100:.1f}%")

    print("\n[5/5] Building news_cache.json...")
    hist_items = to_cache_items(df, p15, p1h, thr_15m, thr_1h)
    live_items = load_live_cache()
    print(f"  Historical items (3m): {len(hist_items):,}")
    print(f"  Live items preserved : {len(live_items):,}")

    merged = sorted(live_items + hist_items, key=lambda x: x.get("published_ts") or 0)

    payload = {
        "metadata": {
            "total_items":    len(merged),
            "live_items":     len(live_items),
            "hist_items":     len(hist_items),
            "months_window":  MONTHS_WINDOW,
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "model":          "xgboost_v10_groq",
            "sentiment_model": "groq/llama-3.3-70b-versatile",
        },
        "news": merged,
    }

    if args.dry_run:
        print(f"\n  [DRY RUN] Would write {len(merged):,} items to {CACHE_FILE.name}")
        return

    CACHE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    channels = {}
    for item in merged:
        ch = item.get("channel", "unknown")
        channels[ch] = channels.get(ch, 0) + 1

    print(f"\n{'='*62}")
    print(f"  ✅ Saved {len(merged):,} items → {CACHE_FILE.name}")
    print(f"  Channels:")
    for ch, cnt in sorted(channels.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ch:<35}: {cnt:,}")


if __name__ == "__main__":
    main()
