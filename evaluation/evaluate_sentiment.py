"""
evaluate_sentiment.py
=====================
Evaluate multiple sentiment models on a labeled crypto news test set.

BERT models (local, no API key needed):
  1. CryptoBERT          (ElKulako/cryptobert)
  2. FinBERT             (ProsusAI/finbert)
  3. RoBERTa-Twitter     (cardiffnlp/twitter-roberta-base-sentiment-latest)
  4. FinBERT-Tone        (yiyanghkust/finbert-tone)
  5. DistilRoBERTa-Fin   (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
  6. Ensemble (prod)     CryptoBERT×0.5 + FinBERT×0.25 + RoBERTa×0.25, ratio-based

Free LLM APIs (need free API key):
  7. DeepSeek-V3         https://platform.deepseek.com  (free tier)
  8. Groq / Llama-3.3    https://console.groq.com       (free tier, very fast)
  9. Groq / Mixtral-8x7B https://console.groq.com       (free tier)
 10. Groq / Gemma2-9B    https://console.groq.com       (free tier)
 11. Google Gemini Flash  https://aistudio.google.com   (free tier)
 12. Ollama (local)       any model installed locally   (completely free)

Paid LLM (optional):
 13. Claude Haiku         https://console.anthropic.com

Required packages (install what you need):
    pip install transformers torch
    pip install openai            # for DeepSeek + Groq (OpenAI-compatible)
    pip install google-genai      # for Gemini
    pip install ollama            # for local Ollama

Environment variables:
    DEEPSEEK_API_KEY=...
    GROQ_API_KEY=...
    GEMINI_API_KEY=...
    ANTHROPIC_API_KEY=...        # optional

Usage:
    python evaluation/evaluate_sentiment.py                  # BERT models only
    python evaluation/evaluate_sentiment.py --llm all        # + all LLMs
    python evaluation/evaluate_sentiment.py --llm deepseek groq_llama gemini
    python evaluation/evaluate_sentiment.py --only cryptobert
    python evaluation/evaluate_sentiment.py --verbose        # per-headline table
"""

import json, sys, os, argparse
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TEST_FILE = HERE / "test_headlines.json"

PROMPT_TEMPLATE = (
    'Classify the Bitcoin/crypto market sentiment of this news headline.\n'
    'Headline: "{title}"\n'
    'Reply with exactly one word: positive, negative, or neutral.'
)


# ── helpers ────────────────────────────────────────────────────────

def normalize(label: str) -> str:
    l = label.lower().strip().split()[0].rstrip(".,!")
    if any(x in l for x in ["positive", "bullish", "pos", "label_2", "buy", "optimistic"]):
        return "positive"
    if any(x in l for x in ["negative", "bearish", "neg", "label_0", "sell", "pessimistic"]):
        return "negative"
    return "neutral"


def ratio_decision(pos: float, neg: float, threshold: float = 1.5) -> str:
    if pos >= neg * threshold:
        return "positive"
    if neg >= pos * threshold:
        return "negative"
    return "neutral"


def _llm_openai_compat(headlines, base_url, api_key, model, name):
    """Generic runner for any OpenAI-compatible API (DeepSeek, Groq, etc.)."""
    try:
        from openai import OpenAI
    except ImportError:
        print(f"       ⚠ openai package not installed — pip install openai")
        return ["neutral"] * len(headlines)
    client = OpenAI(api_key=api_key, base_url=base_url)
    preds = []
    for h in headlines:
        try:
            r = client.chat.completions.create(
                model=model,
                max_tokens=5,
                temperature=0,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(title=h["title"])}],
            )
            raw = r.choices[0].message.content.strip()
            preds.append(normalize(raw))
        except Exception as e:
            print(f"       ⚠ {name} error on id={h['id']}: {e}")
            preds.append("neutral")
    return preds


# ── BERT runners ───────────────────────────────────────────────────

def run_cryptobert(headlines):
    print("  [1] CryptoBERT …", flush=True)
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


def run_finbert(headlines):
    print("  [2] FinBERT …", flush=True)
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    return [normalize(pipe(f"Bitcoin crypto market: {h['title']}")[0]["label"]) for h in headlines]


def run_roberta(headlines):
    print("  [3] RoBERTa-Twitter …", flush=True)
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True)
    return [normalize(pipe(f"Bitcoin price impact: {h['title']}")[0]["label"]) for h in headlines]


def run_finbert_tone(headlines):
    print("  [4] FinBERT-Tone …", flush=True)
    from transformers import pipeline
    try:
        pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", truncation=True)
        return [normalize(pipe(h["title"])[0]["label"]) for h in headlines]
    except Exception as e:
        print(f"       ⚠ Failed: {e}")
        return ["neutral"] * len(headlines)


def run_distilroberta(headlines):
    print("  [5] DistilRoBERTa-Finance …", flush=True)
    from transformers import pipeline
    try:
        pipe = pipeline("sentiment-analysis",
                        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                        truncation=True)
        return [normalize(pipe(h["title"])[0]["label"]) for h in headlines]
    except Exception as e:
        print(f"       ⚠ Failed: {e}")
        return ["neutral"] * len(headlines)


def run_ensemble(headlines):
    print("  [6] Ensemble (prod) …", flush=True)
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
        rb_raw = {r["label"].lower(): r["score"] for r in rb_pipe(f"Bitcoin price impact: {t}")[0]}
        avg_pos = cb_pos * 0.5 + fb_raw.get("positive", 0) * 0.25 + rb_raw.get("positive", 0) * 0.25
        avg_neg = cb_neg * 0.5 + fb_raw.get("negative", 0) * 0.25 + rb_raw.get("negative", 0) * 0.25
        preds.append(ratio_decision(avg_pos, avg_neg))
    return preds


# ── Free LLM runners ───────────────────────────────────────────────

def run_deepseek(headlines):
    print("  [7] DeepSeek-V3 (free API) …", flush=True)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("       ⚠ DEEPSEEK_API_KEY not set — get free key at platform.deepseek.com")
        return None
    return _llm_openai_compat(headlines,
        base_url="https://api.deepseek.com",
        api_key=key, model="deepseek-chat", name="DeepSeek-V3")


def run_groq_llama(headlines):
    print("  [8] Groq / Llama-3.3-70B (free API) …", flush=True)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("       ⚠ GROQ_API_KEY not set — get free key at console.groq.com")
        return None
    return _llm_openai_compat(headlines,
        base_url="https://api.groq.com/openai/v1",
        api_key=key, model="llama-3.3-70b-versatile", name="Groq/Llama3.3")


def run_groq_mixtral(headlines):
    print("  [9] Groq / Mixtral-8x7B (free API) …", flush=True)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("       ⚠ GROQ_API_KEY not set — get free key at console.groq.com")
        return None
    return _llm_openai_compat(headlines,
        base_url="https://api.groq.com/openai/v1",
        api_key=key, model="mixtral-8x7b-32768", name="Groq/Mixtral")


def run_groq_gemma(headlines):
    print("  [10] Groq / Gemma2-9B (free API) …", flush=True)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("       ⚠ GROQ_API_KEY not set — get free key at console.groq.com")
        return None
    return _llm_openai_compat(headlines,
        base_url="https://api.groq.com/openai/v1",
        api_key=key, model="gemma2-9b-it", name="Groq/Gemma2")


def run_gemini(headlines):
    print("  [11] Google Gemini Flash (free API) …", flush=True)
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("       ⚠ GEMINI_API_KEY not set — get free key at aistudio.google.com")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        preds = []
        for h in headlines:
            try:
                r = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=PROMPT_TEMPLATE.format(title=h["title"]),
                )
                preds.append(normalize(r.text.strip()))
            except Exception as e:
                print(f"       ⚠ Gemini error on id={h['id']}: {e}")
                preds.append("neutral")
        return preds
    except ImportError:
        print("       ⚠ google-genai not installed — pip install google-genai")
        return None


def run_ollama(headlines, model="deepseek-r1:7b"):
    print(f"  [12] Ollama / {model} (local, free) …", flush=True)
    try:
        import ollama
        preds = []
        for h in headlines:
            try:
                r = ollama.chat(model=model, messages=[
                    {"role": "user", "content": PROMPT_TEMPLATE.format(title=h["title"])}
                ])
                raw = r["message"]["content"].strip()
                # DeepSeek-R1 wraps answer in <think>...</think>
                if "</think>" in raw:
                    raw = raw.split("</think>")[-1].strip()
                preds.append(normalize(raw))
            except Exception as e:
                print(f"       ⚠ Ollama error on id={h['id']}: {e}")
                preds.append("neutral")
        return preds
    except ImportError:
        print("       ⚠ ollama not installed — pip install ollama (and install Ollama app)")
        return None


def run_claude(headlines):
    print("  [13] Claude Haiku (paid API) …", flush=True)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("       ⚠ ANTHROPIC_API_KEY not set")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        preds = []
        for h in headlines:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(title=h["title"])}],
            )
            preds.append(normalize(msg.content[0].text.strip()))
        return preds
    except ImportError:
        print("       ⚠ anthropic not installed — pip install anthropic")
        return None


# ── output ─────────────────────────────────────────────────────────

def print_table(results: dict, headlines: list, verbose: bool):
    labels = [h["label"] for h in headlines]
    print("\n" + "═" * 66)
    print("  SENTIMENT MODEL EVALUATION — CRYPTO NEWS")
    print("═" * 66)
    print(f"  Test set: {len(headlines)} headlines  "
          f"(+{labels.count('positive')} bullish / "
          f"-{labels.count('negative')} bearish / "
          f"={labels.count('neutral')} neutral)")
    print("─" * 66)

    summary = {}
    for name, preds in results.items():
        correct = sum(p == g for p, g in zip(preds, labels))
        acc = correct / len(labels) * 100
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

    sorted_models = sorted(summary.items(), key=lambda x: -x[1]["acc"])

    print(f"\n  {'Model':<38} {'Type':<6} {'Acc':>6}  {'MacroF1':>7}")
    print("  " + "─" * 60)
    bert_names = {"CryptoBERT","FinBERT","RoBERTa-Twitter","FinBERT-Tone","DistilRoBERTa-Finance","Ensemble (prod)"}
    for i, (name, s) in enumerate(sorted_models):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        kind = "BERT" if name in bert_names else "LLM "
        print(f"  {medal} {name:<37} {kind}  {s['acc']:>5.1f}%  {s['macro_f1']:>6.3f}")

    print(f"\n  {'Model':<38} {'Bullish':>8} {'Bearish':>8} {'Neutral':>8}")
    print("  " + "─" * 66)
    for name, s in sorted_models:
        ps = s["stats"]
        print(f"  {name:<38} {ps['positive']['f1']:>7.3f}  {ps['negative']['f1']:>7.3f}  {ps['neutral']['f1']:>7.3f}")

    if verbose:
        print(f"\n  Per-headline predictions  (✓=correct  ✗=wrong)")
        print(f"  {'#':<3} {'Truth':<9} {'Category':<18}", end="")
        for name in results:
            print(f" {name[:10]:>11}", end="")
        print()
        print("  " + "─" * (30 + 12 * len(results)))
        sym = {"positive": "▲", "negative": "▼", "neutral": "●"}
        for i, h in enumerate(headlines):
            truth = h["label"]
            print(f"  {h['id']:<3} {sym[truth]}{truth:<8} {h['category']:<18}", end="")
            for name, preds in results.items():
                pred = preds[i]
                mark = "✓" if pred == truth else "✗"
                print(f" {mark}{sym[pred]}{pred[:9]:>9}", end="")
            print()

    print("\n" + "═" * 66)
    winner = sorted_models[0]
    print(f"  Best: {winner[0]}  ({winner[1]['acc']:.1f}% acc,  F1={winner[1]['macro_f1']:.3f})")
    print("═" * 66 + "\n")


# ── main ───────────────────────────────────────────────────────────

LLM_RUNNERS = {
    "deepseek":    ("DeepSeek-V3",          run_deepseek),
    "groq_llama":  ("Groq/Llama-3.3-70B",   run_groq_llama),
    "groq_mixtral":("Groq/Mixtral-8x7B",    run_groq_mixtral),
    "groq_gemma":  ("Groq/Gemma2-9B",       run_groq_gemma),
    "gemini":      ("Gemini-2.0-Flash",      run_gemini),
    "ollama":      ("Ollama/DeepSeek-R1:7B", lambda h: run_ollama(h, "deepseek-r1:7b")),
    "claude":      ("Claude Haiku",          run_claude),
}

BERT_RUNNERS = [
    ("CryptoBERT",           run_cryptobert),
    ("FinBERT",              run_finbert),
    ("RoBERTa-Twitter",      run_roberta),
    ("FinBERT-Tone",         run_finbert_tone),
    ("DistilRoBERTa-Finance",run_distilroberta),
    ("Ensemble (prod)",      run_ensemble),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", nargs="*", default=[],
                        metavar="NAME",
                        help=f"LLM(s) to run: all | {' | '.join(LLM_RUNNERS)}")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this single model by key (e.g. cryptobert, deepseek)")
    parser.add_argument("--ollama-model", default="deepseek-r1:7b",
                        help="Ollama model name (default: deepseek-r1:7b)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Override ollama model
    LLM_RUNNERS["ollama"] = (f"Ollama/{args.ollama_model}", lambda h: run_ollama(h, args.ollama_model))

    headlines = json.loads(TEST_FILE.read_text())
    print(f"\nLoaded {len(headlines)} headlines from {TEST_FILE.name}")

    results = {}

    if args.only:
        key = args.only.lower()
        # BERT
        for name, fn in BERT_RUNNERS:
            if key == name.lower().replace(" ", "_").replace("-", "_") or key == name.lower():
                results[name] = fn(headlines)
                break
        # LLM
        if key in LLM_RUNNERS:
            label, fn = LLM_RUNNERS[key]
            r = fn(headlines)
            if r is not None:
                results[label] = r
    else:
        # Always run BERT models
        print("\nBERT models:")
        for name, fn in BERT_RUNNERS:
            results[name] = fn(headlines)

        # LLMs
        llm_keys = list(LLM_RUNNERS.keys()) if "all" in (args.llm or []) else (args.llm or [])
        if llm_keys:
            print("\nLLM models:")
            for key in llm_keys:
                if key not in LLM_RUNNERS:
                    print(f"  ⚠ Unknown LLM key '{key}'. Valid: {list(LLM_RUNNERS)}")
                    continue
                label, fn = LLM_RUNNERS[key]
                r = fn(headlines)
                if r is not None:
                    results[label] = r

    if not results:
        print("No results to show.")
        return

    out = HERE / "eval_results.json"
    out.write_text(json.dumps({"headlines": headlines, "predictions": results}, indent=2))
    print(f"\n  Saved → {out.name}")

    print_table(results, headlines, args.verbose)


if __name__ == "__main__":
    main()
