"""
build_groq_csv.py
=================
Score every row in news_cleaned_filtered_scored.csv with Groq/Llama-3.3-70B
and save a new CSV: news_cleaned_filtered_scored_groq.csv

Adds 4 new columns (does NOT overwrite BERT columns):
  groq_sentiment   — "positive" / "negative" / "neutral"
  groq_prob_pos    — 1.0 / 0.0
  groq_prob_neg    — 0.0 / 1.0 / 0.0
  groq_prob_neu    — 0.0 / 0.0 / 1.0

Uses the existing groq_sentiment_cache.json — only calls API for uncached rows.
Saves cache every 200 new entries so progress is never lost on interruption.

Usage:
    python training/build_groq_csv.py                # score everything
    python training/build_groq_csv.py --limit 5000   # only fetch 5000 new API calls this run
    python training/build_groq_csv.py --status        # show coverage stats without scoring
"""

import os, sys, json, time, hashlib, argparse, warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
CSV_IN     = ROOT / "news_cleaned_filtered_scored.csv"
CSV_OUT    = ROOT / "news_cleaned_filtered_scored_groq.csv"
CACHE_FILE = ROOT / "groq_sentiment_cache.json"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_RPM   = 30   # per key per minute

PROMPT = (
    'Classify the Bitcoin/crypto market sentiment of this news headline.\n'
    'Headline: "{title}"\n'
    'Reply with exactly one word: positive, negative, or neutral.'
)


def _key(title: str) -> str:
    return hashlib.md5(str(title).encode()).hexdigest()


def _normalize(text: str) -> str:
    t = text.lower().strip()
    if any(w in t for w in ["positive", "bullish", "buy"]):  return "positive"
    if any(w in t for w in ["negative", "bearish", "sell"]): return "negative"
    return "neutral"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def get_groq_clients():
    from openai import OpenAI
    keys = []
    for env in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]:
        k = os.environ.get(env)
        if k and k not in keys:
            keys.append(k)
    return [OpenAI(api_key=k, base_url="https://api.groq.com/openai/v1") for k in keys]


def fetch_groq_labels(titles: list[str], limit: int | None) -> dict:
    """Call Groq API for uncached titles. Returns updated cache."""
    clients = get_groq_clients()
    if not clients:
        print("  ✗ No GROQ_API_KEY found in environment")
        sys.exit(1)

    cache    = load_cache()
    to_fetch = [t for t in titles if _key(t) not in cache]
    if limit:
        to_fetch = to_fetch[:limit]

    if not to_fetch:
        print(f"  All {len(titles):,} titles already cached.")
        return cache

    n_keys   = len(clients)
    eff_rpm  = GROQ_RPM * n_keys
    interval = 60.0 / eff_rpm
    print(f"  {len(cache):,} cached · {len(to_fetch):,} to fetch · {n_keys} keys · {eff_rpm} req/min")
    eta_min  = len(to_fetch) / eff_rpm
    print(f"  ETA: {eta_min:.0f} min ({eta_min/60:.1f} h)")

    errors       = 0
    key_cooldown = [0.0] * n_keys

    for i, title in enumerate(to_fetch):
        if i % 500 == 0 and i > 0:
            pct = (len(cache) / len(titles)) * 100
            print(f"    … {i:,}/{len(to_fetch):,}  cached={len(cache):,} ({pct:.1f}%)  errors={errors}", flush=True)

        # pick client round-robin, skip cooled-down keys
        now = time.time()
        ki  = i % n_keys
        for attempt in range(n_keys):
            idx = (ki + attempt) % n_keys
            if now >= key_cooldown[idx]:
                ki = idx
                break

        label = "neutral"
        try:
            r = clients[ki].chat.completions.create(
                model=GROQ_MODEL, max_tokens=5, temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(title=title)}],
            )
            label = _normalize(r.choices[0].message.content.strip())
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                print(f"\n    Key[{ki}] rate-limited at {i}, cooling 60s…")
                key_cooldown[ki] = time.time() + 60
                alt = (ki + 1) % n_keys
                if alt != ki:
                    try:
                        r = clients[alt].chat.completions.create(
                            model=GROQ_MODEL, max_tokens=5, temperature=0,
                            messages=[{"role": "user", "content": PROMPT.format(title=title)}],
                        )
                        label = _normalize(r.choices[0].message.content.strip())
                    except Exception:
                        errors += 1
                else:
                    time.sleep(60)
                    errors += 1
            else:
                errors += 1

        cache[_key(title)] = label
        if (i + 1) % 200 == 0:
            save_cache(cache)
        time.sleep(interval)

    save_cache(cache)
    print(f"\n  Done: {len(to_fetch):,} fetched · {errors} errors · cache={len(cache):,}")
    return cache


def build_csv(cache: dict):
    """Apply cached labels to CSV and save new file."""
    print(f"\n  Loading {CSV_IN.name}…")
    df = pd.read_csv(CSV_IN, low_memory=False)

    titles = df["title"].fillna("").tolist()
    labels = [cache.get(_key(t), None) for t in titles]

    cached_count = sum(1 for l in labels if l is not None)
    print(f"  Coverage: {cached_count:,}/{len(df):,} ({cached_count/len(df)*100:.1f}%)")

    # IMPORTANT: rows without a Groq label stay EMPTY (NaN) — do NOT backfill
    # with BERT sentiment, that would contaminate the BERT-vs-Groq comparison.
    # Downstream (build_groq_features) counts and warns about empty labels.
    df["groq_sentiment"] = labels
    if cached_count < len(df):
        print(f"  ⚠️  {len(df) - cached_count:,} rows have no Groq label (left empty).")
        print(f"      Run again (optionally with --limit N) until coverage is 100%.")
    df["groq_prob_pos"]  = (df["groq_sentiment"] == "positive").astype(float)
    df["groq_prob_neg"]  = (df["groq_sentiment"] == "negative").astype(float)
    df["groq_prob_neu"]  = (df["groq_sentiment"] == "neutral").astype(float)

    dist = df["groq_sentiment"].value_counts()
    print(f"  Distribution: +{dist.get('positive',0):,}  -{dist.get('negative',0):,}  ={dist.get('neutral',0):,}")

    df.to_csv(CSV_OUT, index=False)
    print(f"  Saved → {CSV_OUT.name}  ({CSV_OUT.stat().st_size/1e6:.1f} MB)")


def show_status():
    cache  = load_cache()
    df     = pd.read_csv(CSV_IN, low_memory=False, usecols=["title"])
    titles = df["title"].fillna("").tolist()
    cached = sum(1 for t in titles if _key(t) in cache)
    pct    = cached / len(titles) * 100
    remaining = len(titles) - cached
    eta_h  = remaining / (GROQ_RPM * 2) / 60  # 2 keys
    print(f"\n  CSV rows     : {len(titles):,}")
    print(f"  Groq cache   : {len(cache):,}")
    print(f"  Coverage     : {cached:,} ({pct:.1f}%)")
    print(f"  Remaining    : {remaining:,}")
    print(f"  ETA (2 keys) : ~{eta_h:.1f} hours")
    if CSV_OUT.exists():
        print(f"  Output CSV   : exists ({CSV_OUT.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  Output CSV   : not built yet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max new API calls this run (default: unlimited)")
    parser.add_argument("--status", action="store_true",
                        help="Show coverage stats only, no API calls")
    args = parser.parse_args()

    print("=" * 55)
    print("  BUILD GROQ CSV — Groq/Llama-3.3-70B sentiment")
    print("=" * 55)

    if args.status:
        show_status()
        return

    print("\n[1/2] Fetching Groq labels…")
    df_titles = pd.read_csv(CSV_IN, low_memory=False, usecols=["title"])
    titles    = df_titles["title"].fillna("").tolist()
    cache     = fetch_groq_labels(titles, limit=args.limit)

    print("\n[2/2] Writing output CSV…")
    build_csv(cache)

    print("\n  Next steps:")
    print("  - Re-run with --limit N to fetch more labels incrementally")
    print("  - Once coverage > 50%, train: python training/xgboost_train_groq.py")
    print("    (update CSV_PATH in xgboost_train_groq.py to news_cleaned_filtered_scored_groq.csv)")


if __name__ == "__main__":
    main()
