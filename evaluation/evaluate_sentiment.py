"""
evaluate_sentiment.py
=====================
Evaluate multiple sentiment models on a labeled crypto news test set.

Models tested:
  1. CryptoBERT   (ElKulako/cryptobert)
  2. FinBERT      (ProsusAI/finbert)
  3. RoBERTa      (cardiffnlp/twitter-roberta-base-sentiment-latest)
  4. FinBERT-Tone (yiyanghkust/finbert-tone)
  5. DistilRoBERTa-Finance (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
  6. Ensemble     (current production: CryptoBERT×0.5 + FinBERT×0.25 + RoBERTa×0.25, ratio-based)
  7. Claude Haiku (LLM zero-shot, via Anthropic API — optional, needs ANTHROPIC_API_KEY)

Usage:
    python evaluation/evaluate_sentiment.py
    python evaluation/evaluate_sentiment.py --skip-llm      # skip Claude API call
    python evaluation/evaluate_sentiment.py --verbose       # print per-headline results too
"""

import json, sys, os, argparse
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TEST_FILE = HERE / "test_headlines.json"

# ── helpers ────────────────────────────────────────────────────────
def normalize(label: str) -> str:
    """Map any model label string to positive / negative / neutral."""
    l = label.lower().strip()
    if any(x in l for x in ["positive", "bullish", "pos", "label_2", "buy"]):
        return "positive"
    if any(x in l for x in ["negative", "bearish", "neg", "label_0", "sell"]):
        return "negative"
    return "neutral"


def ratio_decision(pos: float, neg: float, threshold: float = 1.5) -> str:
    """Choose direction by ratio — same logic as production server."""
    if pos >= neg * threshold:
        return "positive"
    if neg >= pos * threshold:
        return "negative"
    return "neutral"


def print_table(results: dict, headlines: list, verbose: bool):
    model_names = list(results.keys())
    labels = [h["label"] for h in headlines]

    # Accuracy per model
    print("\n" + "═" * 62)
    print("  SENTIMENT MODEL EVALUATION — CRYPTO NEWS")
    print("═" * 62)
    print(f"  Test set: {len(headlines)} headlines  "
          f"(+{labels.count('positive')} / -{labels.count('negative')} / ={labels.count('neutral')})")
    print("─" * 62)

    summary = {}
    for name in model_names:
        preds = results[name]
        correct = sum(p == g for p, g in zip(preds, labels))
        acc = correct / len(labels) * 100

        # Per-class stats
        stats = {}
        for cls in ["positive", "negative", "neutral"]:
            tp = sum(p == g == cls for p, g in zip(preds, labels))
            fp = sum(p == cls and g != cls for p, g in zip(preds, labels))
            fn = sum(p != cls and g == cls for p, g in zip(preds, labels))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            stats[cls] = {"prec": prec, "rec": rec, "f1": f1}
        macro_f1 = sum(s["f1"] for s in stats.values()) / 3
        summary[name] = {"acc": acc, "macro_f1": macro_f1, "stats": stats, "preds": preds}

    # Sort by accuracy
    sorted_models = sorted(summary.items(), key=lambda x: -x[1]["acc"])

    print(f"\n  {'Model':<42} {'Acc':>6}  {'F1':>6}")
    print("  " + "─" * 56)
    for i, (name, s) in enumerate(sorted_models):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"  {medal} {name:<40} {s['acc']:>5.1f}%  {s['macro_f1']:>5.3f}")

    print("\n  Per-class F1 scores:")
    print(f"  {'Model':<42} {'Bullish':>8} {'Bearish':>8} {'Neutral':>8}")
    print("  " + "─" * 68)
    for name, s in sorted_models:
        ps = s["stats"]
        print(f"  {name:<42} {ps['positive']['f1']:>7.3f}  {ps['negative']['f1']:>7.3f}  {ps['neutral']['f1']:>7.3f}")

    if verbose:
        print("\n  Per-headline predictions:")
        print(f"  {'#':<3} {'Label':<9} {'Category':<16}", end="")
        for name in model_names:
            short = name[:10]
            print(f" {short:>10}", end="")
        print()
        print("  " + "─" * (28 + 11 * len(model_names)))
        for i, h in enumerate(headlines):
            truth = h["label"]
            sym = {"positive": "+", "negative": "-", "neutral": "="}
            print(f"  {h['id']:<3} {sym[truth]:<1}{truth:<8} {h['category']:<16}", end="")
            for name in model_names:
                pred = results[name][i]
                match = "✓" if pred == truth else "✗"
                pc = {"positive": "+", "negative": "-", "neutral": "="}.get(pred, "?")
                print(f" {match}{pc}{pred[:8]:>8}", end="")
            print()

    print("\n" + "═" * 62)
    winner = sorted_models[0][0]
    print(f"  Best model: {winner}  ({sorted_models[0][1]['acc']:.1f}% accuracy)")
    print("═" * 62 + "\n")


# ── model runners ──────────────────────────────────────────────────

def run_cryptobert(headlines: list) -> list:
    print("  [1/7] CryptoBERT …", flush=True)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert").eval()
    preds = []
    for h in headlines:
        inp = tok(h["title"], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(mdl(**inp).logits, dim=1)[0].tolist()
        neg, neu, pos = probs[0], probs[1], probs[2]
        preds.append(ratio_decision(pos, neg))
    return preds


def run_finbert(headlines: list) -> list:
    print("  [2/7] FinBERT …", flush=True)
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    preds = []
    for h in headlines:
        r = pipe(f"Bitcoin crypto market: {h['title']}")[0]
        preds.append(normalize(r["label"]))
    return preds


def run_roberta(headlines: list) -> list:
    print("  [3/7] RoBERTa (Twitter) …", flush=True)
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True)
    preds = []
    for h in headlines:
        r = pipe(f"Bitcoin price impact: {h['title']}")[0]
        preds.append(normalize(r["label"]))
    return preds


def run_finbert_tone(headlines: list) -> list:
    print("  [4/7] FinBERT-Tone …", flush=True)
    from transformers import pipeline
    try:
        pipe = pipeline("sentiment-analysis",
                        model="yiyanghkust/finbert-tone", truncation=True)
        preds = []
        for h in headlines:
            r = pipe(h["title"])[0]
            preds.append(normalize(r["label"]))
        return preds
    except Exception as e:
        print(f"       ⚠ Failed: {e}")
        return ["neutral"] * len(headlines)


def run_distilroberta(headlines: list) -> list:
    print("  [5/7] DistilRoBERTa-Finance …", flush=True)
    from transformers import pipeline
    try:
        pipe = pipeline("sentiment-analysis",
                        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                        truncation=True)
        preds = []
        for h in headlines:
            r = pipe(h["title"])[0]
            preds.append(normalize(r["label"]))
        return preds
    except Exception as e:
        print(f"       ⚠ Failed: {e}")
        return ["neutral"] * len(headlines)


def run_ensemble(cb_preds, fb_preds, rb_preds, headlines: list) -> list:
    """Production ensemble: CryptoBERT×0.5 + FinBERT×0.25 + RoBERTa×0.25, ratio-based."""
    print("  [6/7] Production Ensemble …", flush=True)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch

    cb_tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    cb_mdl = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert").eval()
    fb_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, top_k=None)
    rb_pipe = pipeline("sentiment-analysis",
                       model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                       truncation=True, top_k=None)
    preds = []
    for h in headlines:
        t = h["title"]
        inp = cb_tok(t, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(cb_mdl(**inp).logits, dim=1)[0].tolist()
        cb_neg, cb_neu, cb_pos = probs[0], probs[1], probs[2]

        fb_raw = {r["label"].lower(): r["score"] for r in fb_pipe(f"Bitcoin crypto market: {t}")[0]}
        fb_pos = fb_raw.get("positive", 0); fb_neg = fb_raw.get("negative", 0)

        rb_raw = {r["label"].lower(): r["score"] for r in rb_pipe(f"Bitcoin price impact: {t}")[0]}
        rb_pos = rb_raw.get("positive", 0); rb_neg = rb_raw.get("negative", 0)

        avg_pos = cb_pos * 0.5 + fb_pos * 0.25 + rb_pos * 0.25
        avg_neg = cb_neg * 0.5 + fb_neg * 0.25 + rb_neg * 0.25
        preds.append(ratio_decision(avg_pos, avg_neg))
    return preds


def run_claude(headlines: list) -> list:
    print("  [7/7] Claude Haiku (LLM zero-shot) …", flush=True)
    import anthropic
    client = anthropic.Anthropic()
    preds = []
    for h in headlines:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    f"Classify the market sentiment of this crypto news headline for Bitcoin/crypto price.\n"
                    f"Headline: \"{h['title']}\"\n"
                    f"Reply with exactly one word: positive, negative, or neutral."
                )
            }]
        )
        raw = msg.content[0].text.strip().lower()
        preds.append(normalize(raw))
    return preds


# ── main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip Claude API call")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-headline table")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one model: cryptobert|finbert|roberta|finbert_tone|distilroberta|ensemble|claude")
    args = parser.parse_args()

    headlines = json.loads(TEST_FILE.read_text())
    print(f"\nLoaded {len(headlines)} test headlines from {TEST_FILE.name}")
    print("Running models…\n")

    results = {}

    only = args.only

    if only in (None, "cryptobert"):
        results["CryptoBERT"] = run_cryptobert(headlines)

    if only in (None, "finbert"):
        results["FinBERT"] = run_finbert(headlines)

    if only in (None, "roberta"):
        results["RoBERTa-Twitter"] = run_roberta(headlines)

    if only in (None, "finbert_tone"):
        results["FinBERT-Tone"] = run_finbert_tone(headlines)

    if only in (None, "distilroberta"):
        results["DistilRoBERTa-Finance"] = run_distilroberta(headlines)

    if only in (None, "ensemble"):
        results["Ensemble (prod)"] = run_ensemble(
            results.get("CryptoBERT"), results.get("FinBERT"), results.get("RoBERTa-Twitter"),
            headlines
        )

    if (only in (None, "claude")) and not args.skip_llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            results["Claude Haiku"] = run_claude(headlines)
        else:
            print("  [7/7] Claude Haiku — skipped (ANTHROPIC_API_KEY not set)")

    # Save raw predictions
    out = HERE / "eval_results.json"
    out.write_text(json.dumps({
        "headlines": headlines,
        "predictions": results,
    }, indent=2))
    print(f"\n  Raw predictions saved → {out.name}")

    print_table(results, headlines, args.verbose)


if __name__ == "__main__":
    main()
