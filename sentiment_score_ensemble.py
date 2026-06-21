"""
sentiment_score_ensemble.py
===========================
Score every headline with ALL THREE models (CryptoBERT, FinBERT, RoBERTa)
and output 9 probability columns + derived sentiment/confidence.

Output columns added/updated:
  cb_prob_pos, cb_prob_neg, cb_prob_neu      — CryptoBERT
  fb_prob_pos, fb_prob_neg, fb_prob_neu      — FinBERT
  rb_prob_pos, rb_prob_neg, rb_prob_neu      — RoBERTa
  sentiment, sentiment_score, confidence     — derived from weighted avg of all 3 models
  weight, net_agreement, sentiment_reliable  — agreement signal + weight + reliability flag
  prob_positive, prob_negative, prob_neutral  — kept for backward compat (= weighted avg)

The downstream neural network gets all 9 raw probabilities as features,
so it can learn which model to trust per context.

Usage:
    python sentiment_score_ensemble.py news_cleaned_filtered.csv
    python sentiment_score_ensemble.py news_cleaned_filtered.csv --output rescored.csv
    python sentiment_score_ensemble.py --rescore   # re-score existing scored CSV in-place
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

def _detect_device():
    import torch
    if not torch.cuda.is_available():
        return -1
    try:
        # Verify kernels actually run on this GPU (fails on sm_50 with cu12x)
        torch.zeros(1).cuda()
        return 0
    except Exception:
        return -1

DEVICE     = _detect_device()
BATCH_SIZE = 64 if DEVICE == 0 else 32
SAVE_EVERY = 500

# ── News type routing (for deriving the human-readable sentiment) ─
NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis"
]

FINBERT_TYPES  = {"regulatory", "macro_economic", "etf", "institutional"}
ROBERTA_TYPES  = {"market_analysis"}
# Everything else → CryptoBERT

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

    dev = torch.device(f"cuda:{DEVICE}" if DEVICE >= 0 else "cpu")
    print(f"  Using device: {dev}")

    # CryptoBERT — embeddings + sentiment
    print("  [1/3] CryptoBERT...")
    cb_tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cb_emb = AutoModel.from_pretrained("ElKulako/cryptobert").to(dev)
    cb_cls = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert").to(dev)
    cb_emb.eval()
    cb_cls.eval()

    # Prototype matrix for news type classification
    print("  Building type prototype matrix...")
    proto_embs = []
    for label in NEWS_TYPE_LABELS:
        sents  = _NEWS_TYPE_PROTOTYPES[label]
        inputs = cb_tok(sents, padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = cb_emb(**inputs).last_hidden_state[:, 0, :]
        proto_embs.append(out.mean(dim=0))
    proto_matrix = F.normalize(torch.stack(proto_embs), dim=1)

    # FinBERT
    print("  [2/3] FinBERT...")
    fb_pipe = pipeline("text-classification", model="ProsusAI/finbert",
                       return_all_scores=True, device=DEVICE)

    # RoBERTa
    print("  [3/3] RoBERTa...")
    rb_pipe = pipeline("text-classification",
                       model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                       return_all_scores=True, device=DEVICE)

    _cache = {
        "cb_tok": cb_tok, "cb_emb": cb_emb, "cb_cls": cb_cls,
        "proto": proto_matrix, "fb": fb_pipe, "rb": rb_pipe,
    }
    print("  ✅ All models ready\n")
    return _cache


def _classify_types(embeddings: torch.Tensor, proto: torch.Tensor) -> list[str]:
    norm = F.normalize(embeddings, dim=1)
    sims = torch.mm(norm, proto.T)
    idxs = sims.argmax(dim=1).tolist()
    return [NEWS_TYPE_LABELS[i] for i in idxs]


def _fixed_score(prob_pos: float, prob_neg: float, prob_neu: float) -> tuple:
    """Fixed scoring logic. Returns (sentiment, score, confidence)."""
    if prob_neu > max(prob_pos, prob_neg):
        return "neutral", 0, prob_neu

    net = prob_pos - prob_neg
    score = (
         3 if net >  0.50 else
         2 if net >  0.25 else
         1 if net >  0.05 else
        -3 if net < -0.50 else
        -2 if net < -0.25 else
        -1 if net < -0.05 else 0
    )
    sentiment  = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    confidence = prob_pos if score > 0 else (prob_neg if score < 0 else prob_neu)
    return sentiment, score, confidence


def _net_agreement(cb, fb, rb):
    """
    Measure how much the 3 models agree.
    Returns float in [-1, 1]:
      +1 = all three agree strongly positive
      -1 = all three agree strongly negative
       0 = disagreement or mixed
    """
    nets = [cb[0] - cb[1], fb[0] - fb[1], rb[0] - rb[1]]  # (pos-neg) per model
    mean_net = np.mean(nets)
    # Penalize disagreement: if signs differ, reduce magnitude
    signs = [1 if n > 0 else (-1 if n < 0 else 0) for n in nets]
    agreement = 1.0 if len(set(signs)) == 1 else 0.5
    return round(float(mean_net * agreement), 4)


def score_batch(titles: list[str], m: dict) -> list[dict]:
    """
    Score a batch with ALL THREE models.
    Returns list of dicts with 9 probability columns + derived fields.
    """
    titles = [str(t).strip() or "crypto news" for t in titles]
    B = len(titles)

    # ── CryptoBERT: embeddings + sentiment ──
    dev    = next(m["cb_emb"].parameters()).device
    inputs = m["cb_tok"](titles, padding=True, truncation=True,
                         max_length=128, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        emb_out = m["cb_emb"](**inputs).last_hidden_state[:, 0, :]
        cls_out = m["cb_cls"](**inputs).logits

    cb_probs   = torch.softmax(cls_out, dim=1).cpu().numpy()  # (B, 3) bear/neu/bull
    news_types = _classify_types(emb_out.cpu(), m["proto"].cpu())

    # ── FinBERT: batch ──
    fb_results = []
    for title in titles:
        try:
            fb = {s["label"].lower(): s["score"]
                  for s in m["fb"](f"Bitcoin crypto market: {title}",
                                   truncation=True)[0]}
            fb_results.append((
                fb.get("positive", 0.0),
                fb.get("negative", 0.0),
                fb.get("neutral",  0.0),
            ))
        except Exception:
            fb_results.append((0.33, 0.33, 0.34))

    # ── RoBERTa: batch ──
    rb_results = []
    for title in titles:
        try:
            rb = {s["label"].lower(): s["score"]
                  for s in m["rb"](f"BREAKING: {title} #Bitcoin #Crypto",
                                   truncation=True)[0]}
            rb_results.append((
                rb.get("positive", 0.0),
                rb.get("negative", 0.0),
                rb.get("neutral",  0.0),
            ))
        except Exception:
            rb_results.append((0.33, 0.33, 0.34))

    # ── Combine results ──
    results = []
    for i in range(B):
        cb_neg, cb_neu, cb_pos = float(cb_probs[i][0]), float(cb_probs[i][1]), float(cb_probs[i][2])
        fb_pos, fb_neg, fb_neu = fb_results[i]
        rb_pos, rb_neg, rb_neu = rb_results[i]

        ntype = news_types[i]

        # Weighted average across all 3 models
        avg_pos = (cb_pos + fb_pos + rb_pos) / 3
        avg_neg = (cb_neg + fb_neg + rb_neg) / 3
        avg_neu = (cb_neu + fb_neu + rb_neu) / 3

        sentiment, score, confidence = _fixed_score(avg_pos, avg_neg, avg_neu)
        weight = max(5, min(10, round(confidence * 10)))

        agreement = _net_agreement(
            (cb_pos, cb_neg), (fb_pos, fb_neg), (rb_pos, rb_neg)
        )

        # Reliability: if any model's direction diverges too far from avg
        max_spread = max(
            abs(cb_pos - cb_neg) - abs(avg_pos - avg_neg),
            abs(fb_pos - fb_neg) - abs(avg_pos - avg_neg),
            abs(rb_pos - rb_neg) - abs(avg_pos - avg_neg),
        )
        reliable = max_spread < 0.3

        results.append({
            # All 9 raw probabilities — fed to neural network as features
            "cb_prob_pos":     round(cb_pos, 4),
            "cb_prob_neg":     round(cb_neg, 4),
            "cb_prob_neu":     round(cb_neu, 4),
            "fb_prob_pos":     round(fb_pos, 4),
            "fb_prob_neg":     round(fb_neg, 4),
            "fb_prob_neu":     round(fb_neu, 4),
            "rb_prob_pos":     round(rb_pos, 4),
            "rb_prob_neg":     round(rb_neg, 4),
            "rb_prob_neu":     round(rb_neu, 4),
            # Derived from weighted average of all 3 models
            "sentiment":           sentiment,
            "sentiment_score":     score,
            "confidence":          round(confidence, 4),
            "weight":              weight,
            "net_agreement":       agreement,
            "sentiment_reliable":  reliable,
            # Backward-compatible columns (= weighted avg)
            "prob_positive":   round(avg_pos, 4),
            "prob_negative":   round(avg_neg, 4),
            "prob_neutral":    round(avg_neu, 4),
            "news_type":       ntype,
        })

    return results


def score_dataframe(df: pd.DataFrame, model: dict, ckpt_path: Path,
                    force_all: bool = False) -> pd.DataFrame:
    all_cols = [
        "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
        "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
        "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
        "sentiment", "sentiment_score", "confidence", "weight",
        "net_agreement", "sentiment_reliable",
        "prob_positive", "prob_negative", "prob_neutral", "news_type",
    ]
    for col in all_cols:
        if col not in df.columns:
            df[col] = None

    if force_all:
        # Re-score everything
        indices = list(range(len(df)))
    else:
        # Only score rows missing the new ensemble columns
        needs = df["cb_prob_pos"].isna().values
        indices = np.where(needs)[0].tolist()

    total = len(indices)
    if total == 0:
        print("  ✅ All rows already scored")
        return df

    print(f"  Rows to score : {total:,}  (already done: {len(df) - total:,})")
    print(f"  Est. time     : ~{total / BATCH_SIZE * 0.5 / 60:.0f} min\n")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Ensemble scoring"):
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
    parser.add_argument("input_csv", nargs="?", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score the existing scored CSV in-place")
    args = parser.parse_args()

    HERE = Path(__file__).parent

    if args.rescore:
        input_path  = HERE / "news_cleaned_filtered_scored.csv"
        output_path = HERE / "news_cleaned_filtered_scored.csv"
        force_all   = True
    elif args.input_csv:
        input_path  = Path(args.input_csv)
        output_path = Path(args.output) if args.output else \
                      input_path.parent / (input_path.stem + "_scored.csv")
        force_all   = False
    else:
        print("Usage: python sentiment_score_ensemble.py <input.csv>")
        print("       python sentiment_score_ensemble.py --rescore")
        sys.exit(1)

    ckpt_path = input_path.parent / (input_path.stem + "_ensemble_ckpt.csv")

    if not input_path.exists():
        print(f"❌ Not found: {input_path}"); sys.exit(1)

    print("=" * 60)
    print("  ENSEMBLE SENTIMENT SCORER (CryptoBERT + FinBERT + RoBERTa)")
    print(f"  All 3 models run on every headline → 9 probability columns")
    print(f"  Input : {input_path.name}")
    print(f"  Output: {output_path.name}")
    print("=" * 60)

    # Resume from checkpoint if available
    if ckpt_path.exists():
        print(f"\n⚡ Resuming from checkpoint")
        df = pd.read_csv(ckpt_path, low_memory=False)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"  Loaded {len(df):,} rows")

    # Backup if overwriting
    if args.rescore:
        bak = HERE / "news_cleaned_filtered_scored_pre_ensemble.csv"
        if not bak.exists():
            print(f"  Saving backup → {bak.name}")
            pd.read_csv(input_path, low_memory=False).to_csv(bak, index=False)

    model = load_models()
    df    = score_dataframe(df, model, ckpt_path, force_all=force_all)

    # Sort by date
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.sort_values("published").reset_index(drop=True)

    df.to_csv(output_path, index=False)

    if ckpt_path.exists():
        ckpt_path.unlink()

    # Report
    total = len(df)
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    neu = (df["sentiment"] == "neutral").sum()
    print(f"\n  Sentiment distribution:")
    print(f"    positive: {pos:,}  ({pos/total:.1%})")
    print(f"    negative: {neg:,}  ({neg/total:.1%})")
    print(f"    neutral : {neu:,}  ({neu/total:.1%})")

    # Agreement stats
    if "net_agreement" in df.columns:
        ag = df["net_agreement"].dropna()
        print(f"\n  Model agreement (net_agreement):")
        print(f"    Mean:   {ag.mean():+.4f}")
        print(f"    Std:    {ag.std():.4f}")
        print(f"    Strong agree (|ag| > 0.3): {(ag.abs() > 0.3).sum():,} ({(ag.abs() > 0.3).mean():.1%})")
        print(f"    Disagree (|ag| < 0.05):    {(ag.abs() < 0.05).sum():,} ({(ag.abs() < 0.05).mean():.1%})")

    if "news_type" in df.columns:
        print(f"\n  News type distribution:")
        for t, c in df["news_type"].value_counts().items():
            print(f"    {t:<20}: {c:,}")

    print(f"\n  ✅ Saved {total:,} rows → {output_path.name}")
    print(f"  New columns: cb_prob_*, fb_prob_*, rb_prob_*, net_agreement")


if __name__ == "__main__":
    main()
