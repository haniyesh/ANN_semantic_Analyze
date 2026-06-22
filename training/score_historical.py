"""
score_historical.py
===================
Score news_cleaned_filtered_scored.csv using production_system_v8.pt
and merge the results into news_cache.json (live data is preserved).

Usage:
    .venv/bin/python score_historical.py
"""

import csv, json, sys, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

CSV_PATH   = HERE / "news_cleaned_filtered_scored.csv"
EMB_CACHE  = HERE / "cryptobert_v8_pipeline.npy"
MODEL_PATH = HERE / "production_system_v8.pt"
CACHE_FILE = HERE / "news_cache.json"

REG_SCALE  = 2.0
BATCH_SIZE = 512

from production_system_v8 import (
    CryptoImpactNetV5, EmbProjection, EMB_PROJ_DIMS,
    crypto_news_type_classify,
)

device = torch.device("cpu")  # Quadro M2000M (Maxwell sm_52) not supported by PyTorch cu128


# ── 1. Load CSV ───────────────────────────────────────────────────
def load_csv():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    for col in ["btc_price_at_news", "btc_price_15m", "btc_price_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["weight"] >= 5]
    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df.drop_duplicates(subset=["title", "published", "channel"])
    df = df.sort_values("published").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows  ({df['channel'].nunique()} channels)")
    return df


# ── 2. Build features (identical to production_system_v8.py) ─────
def build_features(df, emb):
    n = len(df)

    # Ensemble columns (9 probs from 3 models + 4 derived)
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
            extra.append("sentiment_reliable")
        sent_cols = ensemble_cols + extra
        print(f"  Using ensemble sentiment features ({len(sent_cols)} dims)")
    else:
        sent_cols = legacy_cols
        print(f"  Using legacy sentiment features ({len(sent_cols)} dims)")

    sent_arr = df[sent_cols].fillna(0).values.astype(np.float32)

    # News-type probs (11) via cosine similarity to prototypes
    print("  Computing news-type probs...")
    _, type_probs = crypto_news_type_classify(emb, return_probs=True)

    semantic = np.hstack([emb, sent_arr, type_probs]).astype(np.float32)
    print(f"  Semantic: {semantic.shape}")

    # Macro (5): weekend / low_liq / us / asia / fomc
    hour = df["published"].dt.hour.values
    dow  = df["published"].dt.dayofweek.values
    macro = np.column_stack([
        (dow >= 5).astype(np.float32),
        ((hour >= 2) & (hour <= 6)).astype(np.float32),
        ((hour >= 13) & (hour <= 21)).astype(np.float32),
        ((hour >= 0) & (hour <= 8)).astype(np.float32),
        df["fomc_week"].fillna(0).values.astype(np.float32),
    ]).astype(np.float32)

    # RAG = zeros (precomputed file doesn't cover all rows)
    rag = np.zeros((n, 10), dtype=np.float32)

    return semantic, macro, rag


# ── 3. Model inference ────────────────────────────────────────────
def load_model(ckpt):
    state        = ckpt["model_state"]
    sem_tower_in = state["sem_tower.net.0.weight"].shape[1]   # e.g. 41
    rag_dim      = state["rag_tower.net.0.weight"].shape[1]   # 10
    mac_dim      = state["mac_tower.net.0.weight"].shape[1]   # 5
    sem_dim = sem_tower_in + 768 - EMB_PROJ_DIMS

    emb_proj = EmbProjection().to(device)
    model    = CryptoImpactNetV5(
        sem_dim=sem_dim, rag_dim=rag_dim,
        macro_dim=mac_dim,
        emb_proj=emb_proj,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Dims — sem:{sem_dim} rag:{rag_dim} mac:{mac_dim}  device={device}")
    return model


def run_inference(model, scalers, semantic, macro, rag, thr15, thr1h):
    sem_s = scalers[0].transform(semantic)
    rag_s = scalers[1].transform(rag)
    mac_s = scalers[2].transform(macro)

    n = len(sem_s)
    p15_all, p1h_all, r15_all, r1h_all, dir_all = [], [], [], [], []

    with torch.no_grad():
        for s in range(0, n, BATCH_SIZE):
            e   = min(s + BATCH_SIZE, n)
            out = model(
                torch.FloatTensor(sem_s[s:e]).to(device),
                torch.FloatTensor(rag_s[s:e]).to(device),
                torch.FloatTensor(mac_s[s:e]).to(device),
            )
            p15_all.append(torch.sigmoid(out["cls_15m"]).cpu().numpy())
            p1h_all.append(torch.sigmoid(out["cls_1h"]).cpu().numpy())
            r15_all.append(out["reg_15m"].cpu().numpy())
            r1h_all.append(out["reg_1h"].cpu().numpy())
            dir_all.append(out["direction"].argmax(dim=1).cpu().numpy())
            if s % 2000 == 0:
                print(f"    {e}/{n} ({e*100//n}%)")

    p15 = np.concatenate(p15_all)
    p1h = np.concatenate(p1h_all)
    r15 = np.concatenate(r15_all)
    r1h = np.concatenate(r1h_all)

    mod15 = p15 * (1 + np.abs(r15) * REG_SCALE)
    mod1h = p1h * (1 + np.abs(r1h) * REG_SCALE)

    return {
        "prob_15m":  p15,
        "prob_1h":   p1h,
        "mod_15m":   mod15,
        "mod_1h":    mod1h,
        "pred_15m":  (mod15 >= thr15).astype(int),
        "pred_1h":   (mod1h >= thr1h).astype(int),
        "direction": np.concatenate(dir_all),
    }


# ── 4. Convert to cache format ────────────────────────────────────
def to_cache_items(df, preds):
    items = []
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)

        mod15 = float(preds["mod_15m"][idx])
        mod1h = float(preds["mod_1h"][idx])

        # Normalize raw scores (v8 model range 0.50–0.90+) → 0–1
        NORM_MIN, NORM_MAX = 0.50, 0.90
        score_15m = round(max(0.0, min(1.0, (mod15 - NORM_MIN) / (NORM_MAX - NORM_MIN))), 4)
        score_1h  = round(max(0.0, min(1.0, (mod1h - NORM_MIN) / (NORM_MAX - NORM_MIN))), 4)

        # Skip items below medium threshold (normalized < 0.35 = raw < 0.64)
        if max(score_15m, score_1h) < 0.35:
            continue

        pub_dt = row["published"]
        if pd.isnull(pub_dt):
            continue
        pub_ts = int(pub_dt.timestamp())

        btc_p  = float(row["btc_price_at_news"])
        btc_15 = float(row["btc_price_15m"])
        btc_1h = float(row["btc_price_1h"])
        c15m   = (btc_15 - btc_p) / btc_p * 100 if btc_p else 0
        c1h    = (btc_1h - btc_p) / btc_p * 100 if btc_p else 0

        sentiment = row.get("sentiment", "neutral")
        sig_type  = (
            "BUY"  if sentiment == "positive" else
            "SELL" if sentiment == "negative" else
            "NEUTRAL"
        )

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
            "prob_neutral":    float(row.get("prob_neutral") or 0),
            "type":            sig_type,
            "btc_change_15m":  round(c15m, 4),
            "btc_change_1h":   round(c1h,  4),
            "model_score":     round(score_15m, 4),
            "model_score_1h":  round(score_1h,  4),
            "score_normalized": True,
            "pred_15m":        int(preds["pred_15m"][idx]),
            "pred_1h":         int(preds["pred_1h"][idx]),
            "direction":       int(preds["direction"][idx]),
            "impact":          "High" if max(score_15m, score_1h) >= 0.50 else ("Medium" if max(score_15m, score_1h) >= 0.25 else "Low"),
            "news_type":       str(row.get("news_type", "")),
            "source":          "historical_v8",
        })
    return items


# ── 5. Merge with existing live cache ─────────────────────────────
def load_live_cache():
    if not CACHE_FILE.exists():
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("news", [])
        # Keep only live items (drop any previously scored historical items)
        return [x for x in items if x.get("source") not in ("historical_v6", "historical_v8")]
    except Exception:
        return []


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SCORE HISTORICAL — production_system_v8.pt")
    print("=" * 60)

    print("\n[1/5] Loading CSV...")
    df = load_csv()

    print("\n[2/5] Loading CryptoBERT embeddings...")
    emb = np.load(EMB_CACHE).astype(np.float32)
    if len(emb) != len(df):
        min_n = min(len(emb), len(df))
        print(f"  ⚠ Shape mismatch — truncating to {min_n}")
        emb = emb[:min_n]
        df  = df.iloc[:min_n].reset_index(drop=True)
    print(f"  Embeddings: {emb.shape}")

    print("\n[3/5] Building features...")
    semantic, macro, rag = build_features(df, emb)

    print("\n[4/5] Loading model...")
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    thr15 = ckpt["threshold_15m"]
    thr1h = ckpt["threshold_1h"]
    print(f"  threshold_15m={thr15}  threshold_1h={thr1h}")
    model   = load_model(ckpt)
    scalers = ckpt["scalers"]
    print(f"  Model ready")

    print(f"\n[5/5] Running inference on {len(df):,} rows...")
    preds = run_inference(model, scalers, semantic, macro, rag, thr15, thr1h)
    print(f"  Predicted impactful 15m: {preds['pred_15m'].mean()*100:.1f}%")
    print(f"  Predicted impactful 1h:  {preds['pred_1h'].mean()*100:.1f}%")

    print("\n  Converting to cache format...")
    hist_items = to_cache_items(df, preds)
    print(f"  Scored items (score >= 0.50): {len(hist_items):,}")

    print("  Loading live cache...")
    live_items = load_live_cache()
    print(f"  Live items: {len(live_items):,}")

    merged = sorted(
        live_items + hist_items,
        key=lambda x: x.get("published_ts") or x.get("id") or 0,
    )

    payload = {
        "metadata": {
            "total_items":  len(merged),
            "live_items":   len(live_items),
            "hist_items":   len(hist_items),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model":        "production_system_v8.pt",
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
