"""
sentiment_score.py
==================
Type-routed sentiment scoring:
  1. CryptoBERT embeds the title → 768-dim vector
  2. Cosine similarity against 11 news-type prototypes → news type
  3. Route to the best sentiment model for that type:
       regulatory / macro_economic / etf / institutional  → FinBERT
       market_analysis                                    → RoBERTa
       hack / defi / mining / technical / exchange /
       partnership                                        → CryptoBERT

Speed  : 1 model inference per row (same as CryptoBERT-only)
Quality: better than CryptoBERT-only for financial/macro news

Usage:
  python services/sentiment_score.py news_cleaned_filtered.csv
  python services/sentiment_score.py news_cleaned_filtered.csv --output rescored.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

BATCH_SIZE = 32
SAVE_EVERY = 500

# ── News type routing ─────────────────────────────────────────────
FINBERT_TYPES    = {"regulatory", "macro_economic", "etf", "institutional"}
ROBERTA_TYPES    = {"market_analysis"}
CRYPTOBERT_TYPES = {"hack", "defi", "mining", "technical", "exchange", "partnership"}

NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis"
]

_NEWS_TYPE_PROTOTYPES = {
    "regulatory":     ["SEC charges crypto exchange securities violations",
                       "government bans cryptocurrency trading country"],
    "etf":            ["Bitcoin ETF approved by SEC trading",
                       "spot bitcoin fund launches stock exchange"],
    "hack":           ["crypto exchange hacked millions stolen",
                       "DeFi protocol exploited flash loan attack"],
    "macro_economic": ["Federal Reserve raises interest rates decision",
                       "inflation data CPI report released"],
    "exchange":       ["Binance lists new cryptocurrency token",
                       "Coinbase delists token regulatory concerns"],
    "defi":           ["DeFi protocol TVL record liquidity",
                       "Uniswap launches new version features"],
    "mining":         ["Bitcoin mining difficulty adjusts record",
                       "miner capitulation hashrate drops significantly"],
    "institutional":  ["MicroStrategy purchases Bitcoin treasury reserve",
                       "hedge fund allocates Bitcoin portfolio"],
    "technical":      ["Bitcoin network upgrade soft fork activates",
                       "Ethereum developers confirm upgrade date"],
    "partnership":    ["crypto company partnership bank deal",
                       "blockchain firm integrates payment processor"],
    "market_analysis":["Bitcoin price analysis bullish breakout target",
                       "technical analysis support level tested"],
}

_cache = {}


def load_models():
    global _cache
    if _cache:
        return _cache

    print("Loading models...")

    # CryptoBERT — used for type classification AND crypto sentiment
    print("  [1/3] CryptoBERT (embedding + classifier)...")
    cb_tok  = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cb_emb  = AutoModel.from_pretrained("ElKulako/cryptobert")              # for embeddings
    cb_cls  = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert")  # for sentiment
    cb_emb.eval()
    cb_cls.eval()

    # Build prototype matrix for news type classification (one-time)
    print("  Building type prototype matrix...")
    proto_embs = []
    for label in NEWS_TYPE_LABELS:
        sents  = _NEWS_TYPE_PROTOTYPES[label]
        inputs = cb_tok(sents, padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
        with torch.no_grad():
            out = cb_emb(**inputs).last_hidden_state[:, 0, :]
        proto_embs.append(out.mean(dim=0))
    proto_matrix = F.normalize(torch.stack(proto_embs), dim=1)  # (11, 768)

    # FinBERT — financial domain sentiment
    print("  [2/3] FinBERT...")
    fb_pipe = pipeline("text-classification", model="ProsusAI/finbert",
                       return_all_scores=True, device=-1)

    # RoBERTa — social/commentary sentiment
    print("  [3/3] RoBERTa...")
    rb_pipe = pipeline("text-classification",
                       model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                       return_all_scores=True, device=-1)

    _cache = {
        "cb_tok": cb_tok, "cb_emb": cb_emb, "cb_cls": cb_cls,
        "proto":  proto_matrix,
        "fb":     fb_pipe,
        "rb":     rb_pipe,
    }
    print("  ✅ All models ready\n")
    return _cache


def _classify_types(embeddings: torch.Tensor, proto: torch.Tensor) -> list[str]:
    """Cosine similarity → news type label for each embedding."""
    norm = F.normalize(embeddings, dim=1)
    sims = torch.mm(norm, proto.T)                    # (B, 11)
    idxs = sims.argmax(dim=1).tolist()
    return [NEWS_TYPE_LABELS[i] for i in idxs]


def _score_to_cols(prob_pos: float, prob_neg: float, prob_neu: float) -> dict:
    # If neutral dominates both positive and negative → neutral
    if prob_neu > max(prob_pos, prob_neg):
        disc       = 0
        sentiment  = "neutral"
        confidence = prob_neu
    else:
        net  = prob_pos - prob_neg
        # Check most-extreme first so all negative tiers are reachable
        disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
               -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
        sentiment  = "positive" if disc > 0 else ("negative" if disc < 0 else "neutral")
        # Confidence = probability of the *assigned* class
        confidence = prob_pos if disc > 0 else (prob_neg if disc < 0 else prob_neu)

    return {
        "sentiment":       sentiment,
        "sentiment_score": disc,
        "weight":          max(5, min(10, round(confidence * 10))),
        "confidence":      round(confidence, 4),
        "prob_positive":   round(prob_pos, 4),
        "prob_negative":   round(prob_neg, 4),
        "prob_neutral":    round(prob_neu, 4),
    }


def score_batch(titles: list[str], m: dict) -> list[dict]:
    """
    Score a batch. For each title:
      1. CryptoBERT embed + classify type
      2. Route to FinBERT / RoBERTa / CryptoBERT sentiment
    CryptoBERT sentiment is free (same forward pass as embedding).
    """
    titles = [str(t).strip() or "crypto news" for t in titles]

    # ── Step 1: CryptoBERT forward — get embeddings + sentiment logits ──
    inputs = m["cb_tok"](titles, padding=True, truncation=True,
                         max_length=128, return_tensors="pt")
    with torch.no_grad():
        emb_out = m["cb_emb"](**inputs).last_hidden_state[:, 0, :]  # (B, 768)
        cls_out = m["cb_cls"](**inputs).logits                       # (B, 3)

    cb_probs   = torch.softmax(cls_out, dim=1).numpy()  # (B, 3) — bear/neu/bull
    news_types = _classify_types(emb_out, m["proto"])   # list of type strings

    results = []
    for i, (title, ntype) in enumerate(zip(titles, news_types)):
        cb_neg, cb_neu, cb_pos = float(cb_probs[i][0]), float(cb_probs[i][1]), float(cb_probs[i][2])

        if ntype in FINBERT_TYPES:
            # ── FinBERT path ─────────────────────────────────────
            try:
                fb = {s["label"].lower(): s["score"]
                      for s in m["fb"](f"Bitcoin crypto market: {title}",
                                       truncation=True)[0]}
                pp = fb.get("positive", cb_pos)
                pn = fb.get("negative", cb_neg)
                pu = fb.get("neutral",  cb_neu)
            except Exception:
                pp, pn, pu = cb_pos, cb_neg, cb_neu

        elif ntype in ROBERTA_TYPES:
            # ── RoBERTa path ─────────────────────────────────────
            try:
                rb = {s["label"].lower(): s["score"]
                      for s in m["rb"](f"BREAKING: {title} #Bitcoin #Crypto",
                                       truncation=True)[0]}
                # RoBERTa labels: positive/negative/neutral
                pp = rb.get("positive", cb_pos)
                pn = rb.get("negative", cb_neg)
                pu = rb.get("neutral",  cb_neu)
            except Exception:
                pp, pn, pu = cb_pos, cb_neg, cb_neu

        else:
            # ── CryptoBERT path (already computed) ───────────────
            pp, pn, pu = cb_pos, cb_neg, cb_neu

        row = _score_to_cols(pp, pn, pu)
        row["news_type"] = ntype   # update news_type from classification
        results.append(row)

    return results


def score_dataframe(df: pd.DataFrame, model: dict, ckpt_path: Path) -> pd.DataFrame:
    sent_cols = ["sentiment", "sentiment_score", "weight",
                 "confidence", "prob_positive", "prob_negative", "prob_neutral"]
    for col in sent_cols:
        if col not in df.columns:
            df[col] = None

    w_col  = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    needs  = (df["prob_positive"].isna() | (w_col <= 5)).values
    indices = np.where(needs)[0].tolist()
    total   = len(indices)

    if total == 0:
        print("  ✅ All rows already scored")
        return df

    print(f"  Rows to score : {total:,}  (already done: {len(df) - total:,})")
    print(f"  Est. time     : ~{total / BATCH_SIZE * 0.35 / 60:.0f} min\n")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Scoring"):
        batch_idx = indices[start:start + BATCH_SIZE]
        titles    = df["title"].iloc[batch_idx].tolist()
        results   = score_batch(titles, model)

        for i, res in zip(batch_idx, results):
            for col, val in res.items():
                df.iloc[i, df.columns.get_loc(col)] = val

        if (start // BATCH_SIZE + 1) % (SAVE_EVERY // BATCH_SIZE) == 0:
            df.to_csv(ckpt_path, index=False)
            tqdm.write(f"  💾 checkpoint at {start + BATCH_SIZE}/{total}")

    df.to_csv(ckpt_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path  = Path(args.input_csv)
    output_path = Path(args.output) if args.output else \
                  input_path.parent / (input_path.stem + "_scored.csv")
    ckpt_path   = input_path.parent / (input_path.stem + "_score_ckpt.csv")

    if not input_path.exists():
        print(f"❌ Not found: {input_path}"); sys.exit(1)

    print("=" * 60)
    print("  TYPE-ROUTED SENTIMENT SCORER")
    print(f"  regulatory/macro/etf/inst → FinBERT")
    print(f"  market_analysis           → RoBERTa")
    print(f"  crypto-specific types     → CryptoBERT")
    print(f"  Input : {input_path.name}")
    print(f"  Output: {output_path.name}")
    print("=" * 60)

    if ckpt_path.exists():
        print(f"\n⚡ Resuming from checkpoint")
        df = pd.read_csv(ckpt_path, low_memory=False)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"  Loaded {len(df):,} rows")

    model = load_models()
    df    = score_dataframe(df, model, ckpt_path)

    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.sort_values("published").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    if ckpt_path.exists():
        ckpt_path.unlink()

    total = len(df)
    pos   = (df["sentiment"] == "positive").sum()
    neg   = (df["sentiment"] == "negative").sum()
    neu   = (df["sentiment"] == "neutral").sum()
    print(f"\n  Sentiment: positive={pos/total:.1%}  negative={neg/total:.1%}  neutral={neu/total:.1%}")

    if "news_type" in df.columns:
        print(f"\n  News type distribution:")
        for t, c in df["news_type"].value_counts().items():
            print(f"    {t:<20}: {c:,}")

    print(f"\n  ✅ Saved {total:,} rows → {output_path.name}")


if __name__ == "__main__":
    main()
