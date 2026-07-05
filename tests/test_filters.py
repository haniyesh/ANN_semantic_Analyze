"""Unit tests for pipeline/spam_filter.py, pipeline/reduce_noise.py,
and config.news_importance.

These run without ML models, heavy I/O, or network calls.
"""
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.spam_filter import (
    is_spam, is_extra_spam, is_commentary, is_low_info, clean_title
)
from pipeline.reduce_noise import passes_news_filter
from config import news_importance


# ── spam_filter: is_spam (substring) ─────────────────────────────

def test_is_spam_weekly_recap():
    assert is_spam("Weekly Recap: Top crypto news")

def test_is_spam_market_snapshot():
    assert is_spam("Market snapshot: BTC consolidates")

def test_is_spam_false_positive_guard():
    # Real breaking news must NOT be flagged as spam
    assert not is_spam("SEC approves Bitcoin spot ETF application")
    assert not is_spam("Michael Saylor announces $1B Bitcoin purchase")


# ── spam_filter: is_extra_spam (regex) ───────────────────────────

def test_extra_spam_subscribe():
    assert is_extra_spam("Subscribe to our YouTube channel for more crypto!")

def test_extra_spam_price_prediction_clickbait():
    assert is_extra_spam("Will BTC reach $200,000? Should you buy now?")

def test_extra_spam_top_n_crypto():
    assert is_extra_spam("Top 5 crypto tokens to watch this week")

def test_extra_spam_breaking_NOT_flagged():
    # Critical: blanket ^breaking: was removed — real news must pass
    assert not is_extra_spam("Breaking: Fed holds rates steady amid inflation concerns")
    assert not is_extra_spam("BREAKING: SEC approves Bitcoin ETF")

def test_extra_spam_crypto_news_prefix():
    assert is_extra_spam("Crypto news: nothing interesting today")


# ── spam_filter: is_commentary ───────────────────────────────────

def test_commentary_price_prediction():
    assert is_commentary("Bitcoin price prediction for Q3 2025")

def test_commentary_technical_analysis():
    assert is_commentary("Technical analysis: BTC support at $60K")

def test_commentary_real_news_not_flagged():
    assert not is_commentary("Blackrock files for Ethereum ETF with SEC")


# ── spam_filter: is_low_info ─────────────────────────────────────

def test_low_info_too_short():
    assert is_low_info("BTC up")

def test_low_info_sufficient():
    assert not is_low_info("Blackrock Bitcoin ETF sees record $500M inflow")


# ── reduce_noise: passes_news_filter ─────────────────────────────

def test_passes_filter_real_event():
    assert passes_news_filter("SEC approves Bitcoin spot ETF in landmark ruling", "cointelegraph")

def test_passes_filter_breaking_real_event():
    # BREAKING + real event should not be noise-filtered
    assert passes_news_filter("BREAKING: Fed cuts rates by 50 basis points", "coindesk")

def test_blocked_channel():
    assert not passes_news_filter("Bitcoin price analysis", "CryptoNews")  # blocked

def test_price_only_alert_filtered():
    # "BREAKING: Bitcoin hits $67,000" — pure price alert, no context
    assert not passes_news_filter("BREAKING: Bitcoin hits $67,000", "WatcherGuru")

def test_too_short_filtered():
    assert not passes_news_filter("BTC up", "cointelegraph")

def test_roundup_filtered():
    assert not passes_news_filter("Bitcoin price today: what you need to know", "coindesk")


# ── config: news_importance ──────────────────────────────────────

def test_importance_breaking_boost():
    item = {
        "title": "BREAKING: Bitcoin ETF approved",
        "confidence": 80, "prob_positive": 0.8, "prob_negative": 0.1,
        "prob_neutral": 0.1, "channel": "cointelegraph",
    }
    result = news_importance(item)
    assert result["tier"] in ("Key", "Notable")
    assert result["score"] >= 50

def test_importance_low_confidence():
    item = {
        "title": "Some unknown token does something",
        "confidence": 20, "prob_positive": 0.3, "prob_negative": 0.3,
        "prob_neutral": 0.4, "channel": "google_news",
    }
    result = news_importance(item)
    assert result["tier"] == "Regular"

def test_importance_returns_expected_keys():
    item = {
        "title": "Ethereum upgrade goes live",
        "confidence": 60, "prob_positive": 0.6, "prob_negative": 0.2,
        "prob_neutral": 0.2, "channel": "coindesk",
    }
    result = news_importance(item)
    assert "tier" in result
    assert "score" in result
    assert result["tier"] in ("Key", "Notable", "Regular")
    assert 0 <= result["score"] <= 100

def test_importance_high_authority_channel():
    base = {"title": "ETF sees record inflows", "confidence": 65,
            "prob_positive": 0.7, "prob_negative": 0.1, "prob_neutral": 0.2}
    coindesk_score = news_importance({**base, "channel": "coindesk"})["score"]
    unknown_score  = news_importance({**base, "channel": "unknown_channel"})["score"]
    assert coindesk_score >= unknown_score


# ── config: FOMC_WEEK_DATES ──────────────────────────────────────

def test_fomc_week_dates_populated():
    from config import FOMC_WEEK_DATES
    from datetime import date
    assert len(FOMC_WEEK_DATES) > 0
    # The 2025-01-29 meeting — window should cover ±3 days
    assert date(2025, 1, 26) in FOMC_WEEK_DATES   # -3
    assert date(2025, 1, 29) in FOMC_WEEK_DATES   # meeting day
    assert date(2025, 2,  1) in FOMC_WEEK_DATES   # +3

def test_fomc_week_dates_non_fomc_day():
    from config import FOMC_WEEK_DATES
    from datetime import date
    # A random non-FOMC day should not be in the set
    assert date(2025, 3, 1) not in FOMC_WEEK_DATES
