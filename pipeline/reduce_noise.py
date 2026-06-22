"""
reduce_noise.py
===============
Identifies and removes noise from the scored CSV. Runs these filters:

1. WEIGHTED ENSEMBLE — derive sentiment from weighted average of all 3 models.
   A confident model pulls the average proportionally, naturally fixing
   CryptoBERT positive bias.

2. RELIABILITY FLAG — tag rows where one model's direction diverges
   too far from the ensemble average (spread > 0.3).

3. NEAR-DUPLICATE REMOVAL — same story from multiple channels within 2h window.
   Keep the earliest version only.

4. EXPANDED SPAM FILTER — catch patterns the original filter missed.

5. COMMENTARY FILTER — price prediction, weekly recap, "what to watch" articles
   that don't represent real events.

6. LOW INFORMATION — titles with < 6 words or < 25 characters.

Usage:
    python reduce_noise.py                    # dry run — shows stats only
    python reduce_noise.py --apply            # applies filters and saves
    python reduce_noise.py --apply --inplace  # overwrites original CSV
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

HERE = Path(__file__).parent
CSV  = HERE / "news_cleaned_filtered_scored.csv"


# ══════════════════════════════════════════════════════════════════
# LIVE FILTER — importable by main.py, score_historical_xgb.py,
#               api/server.py.  Single source of truth.
# ══════════════════════════════════════════════════════════════════

BLOCKED_CHANNELS: set[str] = {
    "CoinMarketCap",
    "whale_alert_io",
    "unusual_whales_TG1",
    "lookonchain",
    "BitcoinMagazineTelegram",
    "porter_news",
    "kaggle_btc_sent",
    "cryptoslatenews",
    "CoingraphNews",
}

# These channels must mention crypto to pass
CRYPTO_FILTERED_CHANNELS: set[str] = {"porter_news", "WatcherGuru"}

_PRICE     = r'\$[\d,]+\.?\d*[KkMmBb]?'          # $67,000  $67K  $1.2B
_COIN      = r'(?:bitcoin|btc|eth(?:ereum)?|crypto(?:currency)?)'
_MOVE_UP   = r'(?:reclaims?|surges?|jumps?|rises?|climbs?|bounces?|soars?|spikes?|hits?|reaches?|crosses?|breaks?\s+(?:above|past|out|through|above))'
_MOVE_DN   = r'(?:crashes?|drops?|plunges?|falls?|dips?|slides?|tumbles?|sinks?|falls?\s+(?:under|below|to))'
_MOVE_ANY  = rf'(?:{_MOVE_UP}|{_MOVE_DN}|at|above|below|near|around|over|under|past|to|back\s+to|trades?|hovers?)'

NOISE_TITLE_RE = re.compile(
    # ── editorial noise ───────────────────────────────────────────────
    r'price prediction|price forecast|price analysis|technical analysis|'
    r'convert bitcoin|convert btc|btc/[a-z]{2,5}|'
    r'\bhow to buy\b|best crypto(?:currencies)?|top \d+ crypto|'
    r'will .+ reach \$|can .+ hit \$\d|should you buy.*\?$|'
    r'weekly wrap|week in review|daily recap|morning update|'
    r'subscribe to our|join.*telegram|sponsored|press release|'
    # ── roundup / price-today articles ────────────────────────────────
    r'crypto\s+market\s+(?:today|roundup|recap|update)|'
    rf'{_COIN}\s+price\s+today|'
    rf'{_COIN}\s+price\s+(?:is\s+)?(?:now|currently)|'
    # ── JUST IN / BREAKING pure price alerts ─────────────────────────
    # Filter: "JUST IN: $67,000 Bitcoin"  (price before coin, no context)
    rf'(?:just\s*in[:\*\s]{{1,6}}|breaking[:\*\s]{{1,6}}){_PRICE}\s*{_COIN}\b|'
    # Filter: "JUST IN: Bitcoin at/hits $67K"  (coin + movement + price, nothing else)
    rf'(?:just\s*in[:\*\s]{{1,6}}|breaking[:\*\s]{{1,6}}){_COIN}\s+{_MOVE_ANY}\s*{_PRICE}|'
    # Filter: bare "JUST IN: $67K" with no real content after
    rf'just\s*in[:\*\s]{{1,6}}{_PRICE}(?:\s*[a-z]{{0,15}})?$|'
    # Filter: "NOW: Bitcoin hits $66,000"
    rf'now\s*:\s*{_COIN}\s+{_MOVE_ANY}\s*{_PRICE}|'
    # ── standalone coin + price-movement (no JUST IN needed) ─────────
    # Filter: "Bitcoin crashes under $60K" / "Bitcoin reclaims $65K"
    rf'{_COIN}\s+{_MOVE_DN}\s+(?:under|below|to|toward)?\s*{_PRICE}|'
    rf'{_COIN}\s+{_MOVE_UP}\s+(?:above|over|past|back\s+to|to)?\s*{_PRICE}|'
    # Filter: "$67,000 Bitcoin" / "$3,400 Ethereum" (price first, coin second)
    rf'{_PRICE}\s+{_COIN}\b(?!\s+(?:etf|fund|trust|mining|wallet|halving|regulation|protocol|network))|'
    # Filter: "Bitcoin at $67K" / "BTC at $67,000" (pure position, no context)
    rf'{_COIN}\s+(?:is\s+)?(?:now\s+)?at\s+{_PRICE}(?:\s*[,.]?\s*)?(?:up|down|after)?\s*$',
    re.IGNORECASE,
)

CRYPTO_KW_RE = re.compile(
    r'bitcoin|\bbtc\b|\beth\b|ethereum|crypto|blockchain|\bdefi\b|\bnft\b|'
    r'\bsec\b|\betf\b|binance|coinbase|stablecoin|\busdt\b|\busdc\b|solana|'
    r'\bsol\b|ripple|\bxrp\b|altcoin|mining|halving|\bwallet\b|exchange|'
    r'\btoken\b|\bweb3\b|grayscale|blackrock|microstrategy|strategy|'
    r'regulation|federal reserve|interest rate|\bfed\b',
    re.IGNORECASE,
)


def passes_news_filter(title: str, channel: str) -> bool:
    """
    Returns True if the news item should be processed / displayed.

    Checks (in order):
      1. Channel not in BLOCKED_CHANNELS
      2. Title length >= 30 characters
      3. Title does not match NOISE_TITLE_RE
      4. If channel in CRYPTO_FILTERED_CHANNELS, title must contain a crypto keyword
    """
    if channel in BLOCKED_CHANNELS:
        return False
    t = (title or "").strip()
    if len(t) < 25:
        return False
    if NOISE_TITLE_RE.search(t):
        return False
    if channel in CRYPTO_FILTERED_CHANNELS and not CRYPTO_KW_RE.search(t):
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# 1. WEIGHTED ENSEMBLE — replace single-model sentiment with avg of 3
# ══════════════════════════════════════════════════════════════════
def fix_cryptobert_bias(df):
    """
    Replace single-model sentiment with weighted average of all 3 models.
    A confident model (e.g. FinBERT at 0.92 negative) pulls the average
    proportionally, naturally fixing CryptoBERT positive bias.
    """
    has_ensemble = all(c in df.columns for c in [
        "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
        "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
        "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
    ])
    if not has_ensemble:
        print("  ⚠ No ensemble columns — skipping weighted avg")
        return df, 0

    old_sentiment = df["sentiment"].copy()

    # Weighted average across all 3 models
    avg_pos = (df["cb_prob_pos"] + df["fb_prob_pos"] + df["rb_prob_pos"]) / 3
    avg_neg = (df["cb_prob_neg"] + df["fb_prob_neg"] + df["rb_prob_neg"]) / 3
    avg_neu = (df["cb_prob_neu"] + df["fb_prob_neu"] + df["rb_prob_neu"]) / 3

    neu_mask = avg_neu > np.maximum(avg_pos, avg_neg)
    net = avg_pos - avg_neg

    score = np.where(neu_mask, 0,
            np.where(net >  0.50,  3,
            np.where(net >  0.25,  2,
            np.where(net >  0.05,  1,
            np.where(net < -0.50, -3,
            np.where(net < -0.25, -2,
            np.where(net < -0.05, -1, 0)))))))

    sentiment = np.where(score > 0, "positive",
                np.where(score < 0, "negative", "neutral"))

    confidence = np.where(score > 0, avg_pos,
                 np.where(score < 0, avg_neg, avg_neu))

    df["sentiment"] = sentiment
    df["sentiment_score"] = score
    df["confidence"] = np.round(confidence, 4)
    df["prob_positive"] = np.round(avg_pos, 4)
    df["prob_negative"] = np.round(avg_neg, 4)
    df["prob_neutral"] = np.round(avg_neu, 4)
    df["weight"] = np.clip(np.round(confidence * 10), 5, 10).astype(int)

    changed = (df["sentiment"] != old_sentiment).sum()
    return df, changed


# ══════════════════════════════════════════════════════════════════
# 2. RELIABILITY FLAG — spread-based disagreement detection
# ══════════════════════════════════════════════════════════════════
def flag_disagreement(df):
    """
    Add 'sentiment_reliable' column based on model spread:
      True  = no model's direction diverges > 0.3 from ensemble average
      False = at least one model strongly disagrees with the average
    """
    has_ensemble = all(c in df.columns for c in [
        "cb_prob_pos", "cb_prob_neg",
        "fb_prob_pos", "fb_prob_neg",
        "rb_prob_pos", "rb_prob_neg",
    ])
    if not has_ensemble:
        df["sentiment_reliable"] = True
        return df, 0

    avg_pos = (df["cb_prob_pos"] + df["fb_prob_pos"] + df["rb_prob_pos"]) / 3
    avg_neg = (df["cb_prob_neg"] + df["fb_prob_neg"] + df["rb_prob_neg"]) / 3
    avg_dir = abs(avg_pos - avg_neg)

    cb_dir = abs(df["cb_prob_pos"] - df["cb_prob_neg"])
    fb_dir = abs(df["fb_prob_pos"] - df["fb_prob_neg"])
    rb_dir = abs(df["rb_prob_pos"] - df["rb_prob_neg"])

    # Max divergence of any single model from the ensemble direction
    max_spread = np.maximum(
        np.maximum(cb_dir - avg_dir, fb_dir - avg_dir),
        rb_dir - avg_dir
    )

    df["sentiment_reliable"] = max_spread < 0.3
    unreliable = (max_spread >= 0.3).sum()
    return df, unreliable


# ══════════════════════════════════════════════════════════════════
# 3. NEAR-DUPLICATE REMOVAL
# ══════════════════════════════════════════════════════════════════
def remove_near_duplicates(df, time_window_hours=2, similarity_threshold=0.70):
    """
    Remove near-duplicate headlines within a time window.
    Keep the earliest version. Uses title similarity ratio.
    """
    df = df.sort_values("published").reset_index(drop=True)
    drop_indices = set()

    # Normalize titles for comparison
    titles_lower = df["title"].str.lower().str.strip().values
    timestamps = pd.to_datetime(df["published"], utc=True, errors="coerce")

    # Group by approximate time windows for efficiency
    # Only compare within 2-hour windows
    window_ns = time_window_hours * 3600 * 1e9

    i = 0
    while i < len(df):
        if i in drop_indices:
            i += 1
            continue

        j = i + 1
        while j < len(df):
            if j in drop_indices:
                j += 1
                continue

            # Check time window
            ts_i = timestamps.iloc[i]
            ts_j = timestamps.iloc[j]
            if pd.isna(ts_i) or pd.isna(ts_j):
                j += 1
                continue
            if (ts_j - ts_i).total_seconds() > time_window_hours * 3600:
                break

            # Check similarity
            ratio = SequenceMatcher(None, titles_lower[i], titles_lower[j]).ratio()
            if ratio >= similarity_threshold:
                drop_indices.add(j)

            j += 1
        i += 1

        if i % 5000 == 0 and i > 0:
            print(f"    Dedup progress: {i:,}/{len(df):,} ({len(drop_indices):,} dupes found)")

    return drop_indices


# ══════════════════════════════════════════════════════════════════
# 4. EXPANDED SPAM FILTER
# ══════════════════════════════════════════════════════════════════
EXTRA_SPAM_PATTERNS = [
    # YouTube / channel promotion
    r"subscribe to our",
    r"youtube\.com/channel",
    r"t\.me/\w+\?start",
    r"join.*telegram",

    # Price prediction clickbait
    r"price prediction.*time to buy",
    r"should you buy.*\?$",
    r"will .+ reach \$",
    r"can .+ hit \$\d",

    # Weekly/daily recap (not real-time events)
    r"weekend watch",
    r"weekly wrap",
    r"week in review",
    r"daily recap",
    r"morning update",

    # Ads / sponsored
    r"sponsored",
    r"paid partnership",
    r"press release",

    # Generic filler
    r"^crypto news:",
    r"^breaking:",
    r"top \d+ crypto",
    r"\+ more news$",
]
_SPAM_RE = [re.compile(p, re.IGNORECASE) for p in EXTRA_SPAM_PATTERNS]


def is_extra_spam(title: str) -> bool:
    return any(rx.search(title) for rx in _SPAM_RE)


# ══════════════════════════════════════════════════════════════════
# 5. COMMENTARY FILTER
# ══════════════════════════════════════════════════════════════════
COMMENTARY_PATTERNS = [
    r"price prediction",
    r"price forecast",
    r"price analysis",
    r"technical analysis",
    r"current price of",
    r"how (high|low) (can|will)",
    r"where .+ headed",
    r"bull(ish)? or bear(ish)?",
    r"^\d+ (reasons|things|ways)",
]
_COMMENTARY_RE = [re.compile(p, re.IGNORECASE) for p in COMMENTARY_PATTERNS]


def is_commentary(title: str) -> bool:
    return any(rx.search(title) for rx in _COMMENTARY_RE)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply filters and save")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  NOISE REDUCTION")
    print("=" * 60)

    df = pd.read_csv(CSV, low_memory=False)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    orig_len = len(df)
    print(f"\n  Loaded: {orig_len:,} rows")

    # ── 1. Fix CryptoBERT positive bias ──
    print("\n[1/6] Fixing CryptoBERT positive bias...")
    df, n_overridden = fix_cryptobert_bias(df)
    print(f"  Overridden: {n_overridden:,} rows (positive → negative)")

    # ── 2. Flag model disagreement ──
    print("\n[2/6] Flagging model disagreement...")
    df, n_unreliable = flag_disagreement(df)
    print(f"  Unreliable sentiment: {n_unreliable:,} rows")

    # ── 3. Mark extra spam ──
    print("\n[3/6] Expanded spam filter...")
    spam_mask = df["title"].apply(is_extra_spam)
    n_spam = spam_mask.sum()
    print(f"  Extra spam found: {n_spam:,} rows")
    if n_spam > 0:
        print("  Examples:")
        for t in df.loc[spam_mask, "title"].head(5).values:
            print(f"    → {t[:80]}")

    # ── 4. Mark commentary ──
    print("\n[4/6] Commentary filter...")
    commentary_mask = df["title"].apply(is_commentary)
    n_commentary = commentary_mask.sum()
    print(f"  Commentary found: {n_commentary:,} rows")

    # ── 5. Low information ──
    print("\n[5/6] Low information filter...")
    word_counts = df["title"].str.split().str.len().fillna(0)
    char_counts = df["title"].str.len().fillna(0)
    low_info_mask = (word_counts < 6) | (char_counts < 25)
    n_low_info = low_info_mask.sum()
    print(f"  Low info: {n_low_info:,} rows (< 6 words or < 25 chars)")

    # ── 6. Near-duplicates ──
    print("\n[6/6] Near-duplicate detection (this may take a few minutes)...")
    dup_indices = remove_near_duplicates(df)
    dup_mask = pd.Series(False, index=df.index)
    dup_mask.iloc[list(dup_indices)] = True
    n_dupes = len(dup_indices)
    print(f"  Near-duplicates: {n_dupes:,} rows")

    # ── Combined removal mask ──
    remove_mask = spam_mask | commentary_mask | low_info_mask | dup_mask
    n_remove = remove_mask.sum()

    # Add flags to DataFrame (keep rows but flag them)
    df["is_spam_v2"] = spam_mask
    df["is_commentary"] = commentary_mask
    df["is_low_info"] = low_info_mask
    df["is_duplicate"] = dup_mask
    df["noise_flag"] = remove_mask

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Original rows:           {orig_len:>8,}")
    print(f"  Sentiment overridden:    {n_overridden:>8,}  (label fixed, not removed)")
    print(f"  Unreliable sentiment:    {n_unreliable:>8,}  (flagged, not removed)")
    print(f"  Extra spam:              {n_spam:>8,}  (removed)")
    print(f"  Commentary:              {n_commentary:>8,}  (removed)")
    print(f"  Low information:         {n_low_info:>8,}  (removed)")
    print(f"  Near-duplicates:         {n_dupes:>8,}  (removed)")
    print(f"  Total to remove:         {n_remove:>8,}  ({n_remove/orig_len:.1%})")
    print(f"  Clean rows remaining:    {orig_len - n_remove:>8,}")

    # New sentiment distribution
    clean = df[~remove_mask]
    print(f"\n  Sentiment distribution (clean subset):")
    for label in ["positive", "negative", "neutral"]:
        c = (clean["sentiment"] == label).sum()
        print(f"    {label:<12}: {c:>8,} ({c/len(clean):.1%})")

    reliable = clean["sentiment_reliable"].sum() if "sentiment_reliable" in clean.columns else len(clean)
    print(f"    reliable:     {reliable:>8,} ({reliable/len(clean):.1%})")

    if args.apply:
        if args.inplace:
            out_path = CSV
        else:
            out_path = HERE / "news_cleaned_filtered_scored_denoised.csv"

        # Option A: Save full CSV with flags (so you can experiment)
        df.to_csv(HERE / "news_scored_with_flags.csv", index=False)
        print(f"\n  📋 Full CSV with flags → news_scored_with_flags.csv")

        # Option B: Save clean subset only
        clean_df = df[~remove_mask].drop(
            columns=["is_spam_v2", "is_commentary", "is_low_info",
                     "is_duplicate", "noise_flag"],
            errors="ignore",
        )
        clean_df.to_csv(out_path, index=False)
        print(f"  ✅ Clean CSV ({len(clean_df):,} rows) → {out_path.name}")
    else:
        print(f"\n  ℹ Dry run — no files changed. Run with --apply to save.")
        print(f"    --apply           → saves to news_cleaned_filtered_scored_denoised.csv")
        print(f"    --apply --inplace → overwrites original CSV")


if __name__ == "__main__":
    main()
