"""
create_sample_cache.py
======================
Loads real news from the last N months of the training CSV, re-scores
each item with the full XGBoost v9 pipeline (CryptoBERT + FinBERT + RAG),
and writes to news_cache.json.

Usage:
    .venv311/bin/python create_sample_cache.py               # last 3 months
    .venv311/bin/python create_sample_cache.py --months 6    # last 6 months
    .venv311/bin/python create_sample_cache.py --skip-rag    # skip Qdrant RAG
    .venv311/bin/python create_sample_cache.py --append      # add to existing cache
    .venv311/bin/python create_sample_cache.py --max 500     # limit to 500 items
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TRAINING_CSV = ROOT / "news_cleaned_filtered_scored.csv"
CACHE_PATH   = ROOT / "news_cache.json"


# ── Helpers ───────────────────────────────────────────────────────
def _hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]


def _load_bert_models():
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("  Loading CryptoBERT...")
    cb_tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cb_emb = AutoModel.from_pretrained("ElKulako/cryptobert").eval().to(device)
    cb_cls = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert").eval().to(device)

    print("  Loading FinBERT...")
    fb_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_mdl = AutoModel.from_pretrained("ProsusAI/finbert").eval().to(device)

    return {"cb_tok": cb_tok, "cb_emb": cb_emb, "cb_cls": cb_cls,
            "fb_tok": fb_tok, "fb_mdl": fb_mdl, "device": device}


def _encode(models: dict, title: str):
    device = models["device"]
    inp_cb = models["cb_tok"](title, padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
    inp_cb = {k: v.to(device) for k, v in inp_cb.items()}

    with torch.no_grad():
        cb_embedding = models["cb_emb"](**inp_cb).last_hidden_state[:, 0, :].cpu().numpy().flatten()
        cb_probs     = torch.softmax(models["cb_cls"](**inp_cb).logits, dim=1).cpu().numpy()[0]

    inp_fb = models["fb_tok"](title, padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
    inp_fb = {k: v.to(device) for k, v in inp_fb.items()}
    with torch.no_grad():
        fb_embedding = models["fb_mdl"](**inp_fb).last_hidden_state[:, 0, :].cpu().numpy().flatten()

    return cb_embedding.astype(np.float32), cb_probs, fb_embedding.astype(np.float32)


def _build_sentiment(cb_probs: np.ndarray) -> dict:
    cb_neg, cb_neu, cb_pos = float(cb_probs[0]), float(cb_probs[1]), float(cb_probs[2])
    net = cb_pos - cb_neg
    if cb_neu > max(cb_pos, cb_neg):
        sentiment, confidence, disc = "neutral", cb_neu, 0
    else:
        disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
               -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
        sentiment  = "positive" if disc > 0 else ("negative" if disc < 0 else "neutral")
        confidence = cb_pos if disc > 0 else (cb_neg if disc < 0 else cb_neu)

    return {
        "sentiment":       sentiment,
        "sentiment_score": round(float(disc), 4),
        "weight":          max(5, min(10, round(confidence * 10))),
        "confidence":      round(confidence * 100, 2),
        "prob_positive":   round(cb_pos, 4),
        "prob_negative":   round(cb_neg, 4),
        "prob_neutral":    round(cb_neu, 4),
        "cb_prob_pos":     round(cb_pos, 4),
        "cb_prob_neg":     round(cb_neg, 4),
        "cb_prob_neu":     round(cb_neu, 4),
    }


def _build_features(cb_emb, fb_emb, sent: dict, pub_dt: datetime) -> np.ndarray:
    from xgboost_v9 import crypto_news_type_classify
    type_probs = crypto_news_type_classify(cb_emb.reshape(1, -1))[0]

    sent_vec = np.array([
        sent["cb_prob_pos"], sent["cb_prob_neg"], sent["cb_prob_neu"],
        sent["cb_prob_pos"], sent["cb_prob_neg"], sent["cb_prob_neu"],
        sent["cb_prob_pos"], sent["cb_prob_neg"], sent["cb_prob_neu"],
        sent["cb_prob_pos"] - sent["cb_prob_neg"],
        sent["sentiment_score"],
        float(sent["weight"]),
        sent["confidence"] / 100.0,
    ], dtype=np.float32)

    hour = pub_dt.hour
    dow  = pub_dt.weekday()
    macro = np.array([
        float(dow >= 5),
        float(2 <= hour <= 6),
        float(13 <= hour <= 21),
        float(0 <= hour <= 8),
        0.0, 0.0, 0.0, 0.5,
    ], dtype=np.float32)

    return np.concatenate([cb_emb, fb_emb, sent_vec, type_probs, macro])


def _query_rag(title: str, published_ts: int) -> np.ndarray:
    from pipeline.rag_news import query_single
    ch_rates = {
        "the_block_crypto": 0.062,
        "coindesk":         0.058,
        "cointelegraph":    0.048,
        "WatcherGuru":      0.040,
    }
    hour = datetime.fromtimestamp(published_ts, tz=timezone.utc).hour
    macro_now = {
        "is_weekend":       0.0,
        "is_low_liquidity": 0.0,
        "is_us_hours":      float(13 <= hour <= 21),
        "is_asia_hours":    float(0 <= hour <= 8),
        "fomc_week":        0.0,
    }
    result = query_single(
        title=title,
        before_timestamp=published_ts,
        channel_impact_rates=ch_rates,
        macro_now=macro_now,
    )
    return result["features"].astype(np.float32)


def _load_xgb():
    import pickle, xgboost as xgb
    clf15 = xgb.XGBClassifier(); clf15.load_model(str(ROOT / "xgboost_v9_clf15m.json"))
    clf1h = xgb.XGBClassifier(); clf1h.load_model(str(ROOT / "xgboost_v9_clf1h.json"))
    with open(ROOT / "xgboost_v9_scaler.pkl", "rb") as f:
        import pickle
        scaler = pickle.load(f)

    thr15, thr1h = 0.295, 0.265
    res_path = ROOT / "xgboost_v9_results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        thr15 = res.get("threshold_15m", thr15)
        thr1h = res.get("threshold_1h",  thr1h)

    return clf15, clf1h, scaler, thr15, thr1h


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",   type=int,   default=5,     help="How many months back (default: 5)")
    parser.add_argument("--max",      type=int,   default=None,  help="Max items to score")
    parser.add_argument("--append",   action="store_true",       help="Add to existing cache")
    parser.add_argument("--skip-rag", action="store_true",       help="Skip Qdrant RAG query")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  CREATE SAMPLE CACHE  —  last {args.months} months  —  XGBoost v9")
    print("=" * 60)

    # Load training CSV, filter to last N months
    print(f"\n[1/5] Loading training CSV (last {args.months} months)...")
    full_df = pd.read_csv(TRAINING_CSV, low_memory=False)
    full_df["published"] = pd.to_datetime(full_df["published"], format="mixed", utc=True, errors="coerce")
    full_df = full_df.dropna(subset=["title", "published"])

    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=args.months)
    df = full_df[full_df["published"] >= cutoff].copy()

    # For channels with no data in the window, include their most recent 200 items
    # Only if their latest item is within 6 months (skip truly dead channels)
    FALLBACK_PER_CHANNEL = 200
    FALLBACK_MAX_AGE     = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)
    all_channels = full_df["channel"].unique()
    covered = set(df["channel"].unique())
    missing_channels = [ch for ch in all_channels if ch not in covered]
    fallback_frames = []
    for ch in missing_channels:
        ch_df = full_df[full_df["channel"] == ch].sort_values("published", ascending=False).head(FALLBACK_PER_CHANNEL)
        if len(ch_df) == 0:
            continue
        if ch_df["published"].max() < FALLBACK_MAX_AGE:
            print(f"  Skipping {ch} — last item too old ({ch_df['published'].max().date()})")
            continue
        fallback_frames.append(ch_df)
        print(f"  Fallback: {ch} — using last {len(ch_df)} items (latest: {ch_df['published'].max().date()})")

    if fallback_frames:
        df = pd.concat([df] + fallback_frames, ignore_index=True)

    df = df.sort_values("published", ascending=False).reset_index(drop=True)

    if args.max:
        df = df.head(args.max)

    print(f"  Items in range: {len(df):,}  ({cutoff.date()} → now)")
    if len(df) == 0:
        print("  No items found. Exiting.")
        return

    # Channel distribution
    print("  Channels:")
    for ch, cnt in df["channel"].value_counts().head(10).items():
        print(f"    {ch:<30}: {cnt:,}")

    print(f"\n[2/5] Loading models...")
    bert = _load_bert_models()
    clf15, clf1h, scaler, thr15, thr1h = _load_xgb()
    print(f"  XGBoost thresholds: 15m={thr15:.3f}  1h={thr1h:.3f}")

    print(f"\n[3/5] Scoring {len(df):,} items...")
    results = []
    skipped = 0

    for i, row in df.iterrows():
        title   = str(row["title"]).strip()
        channel = str(row.get("channel", ""))
        pub_dt  = row["published"].to_pydatetime()
        pub_ts  = int(pub_dt.timestamp())

        if i % 100 == 0:
            print(f"  {i}/{len(df)}  ({i*100//len(df)}%)...")

        try:
            cb_emb, cb_probs, fb_emb = _encode(bert, title)
            sent = _build_sentiment(cb_probs)

            if args.skip_rag:
                rag = np.zeros(10, dtype=np.float32)
            else:
                try:
                    rag = _query_rag(title, pub_ts)
                except Exception:
                    rag = np.zeros(10, dtype=np.float32)

            features = _build_features(cb_emb, fb_emb, sent, pub_dt)
            features = np.concatenate([features, rag]).astype(np.float32)

            X    = scaler.transform(features.reshape(1, -1)).astype(np.float32)
            p15  = float(clf15.predict_proba(X)[0, 1])
            p1h  = float(clf1h.predict_proba(X)[0, 1])
            pred15 = int(p15 >= thr15)
            pred1h = int(p1h >= thr1h)
            impact = "High" if max(p15, p1h) >= 0.67 else ("Medium" if max(p15, p1h) >= 0.40 else "Low")

            results.append({
                "id":               f"sample_{_hash(title)}",
                "time":             pub_dt.strftime("%H:%M:%S"),
                "title":            title,
                "link":             str(row.get("link", "")),
                "channel":          channel,
                "published":        pub_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "published_ts":     pub_ts,
                "sentiment":        sent["sentiment"],
                "sentiment_score":  sent["sentiment_score"],
                "confidence":       sent["confidence"],
                "weight":           sent["weight"],
                "prob_positive":    sent["prob_positive"],
                "prob_negative":    sent["prob_negative"],
                "prob_neutral":     sent["prob_neutral"],
                "type":             sent["sentiment"].upper(),
                "model_score":      round(p15, 4),
                "model_score_1h":   round(p1h, 4),
                "score_normalized": True,
                "pred_15m":         pred15,
                "pred_1h":          pred1h,
                "impact":           impact,
                "source":           "sample_cache",
            })
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  ⚠ Skipped: {title[:50]}... ({e})")

    print(f"  Scored : {len(results):,}")
    print(f"  Skipped: {skipped}")

    # Write cache
    print(f"\n[4/5] Writing {CACHE_PATH.name}...")
    if args.append and CACHE_PATH.exists():
        existing = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        existing_items = existing.get("news", existing) if isinstance(existing, dict) else existing
        existing_ids   = {x.get("id") for x in existing_items}
        new_items = [r for r in results if r["id"] not in existing_ids]
        all_items = existing_items + new_items
        print(f"  Appended {len(new_items)} new (skipped {len(results)-len(new_items)} duplicates)")
    else:
        all_items = results
        print(f"  Fresh cache: {len(all_items)} items")

    CACHE_PATH.write_text(
        json.dumps({"news": all_items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    print(f"\n[5/5] Impact distribution:")
    from collections import Counter
    impacts = Counter(r["impact"] for r in results)
    preds15 = sum(1 for r in results if r["pred_15m"])
    preds1h = sum(1 for r in results if r["pred_1h"])
    print(f"  High   : {impacts.get('High', 0):,}")
    print(f"  Medium : {impacts.get('Medium', 0):,}")
    print(f"  Low    : {impacts.get('Low', 0):,}")
    print(f"  pred_15m=1 : {preds15:,}  ({preds15*100//max(len(results),1)}%)")
    print(f"  pred_1h=1  : {preds1h:,}  ({preds1h*100//max(len(results),1)}%)")
    print(f"\n  Saved {len(all_items)} items → {CACHE_PATH}")


if __name__ == "__main__":
    main()
