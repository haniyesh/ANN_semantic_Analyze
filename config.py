import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_ID     = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH   = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_SESSION    = os.getenv("TELEGRAM_SESSION")
ADMIN_ID            = int(os.getenv("ADMIN_ID", "0"))

# Channels to monitor (comma-separated in .env)
_raw_channels       = os.getenv("TELEGRAM_CHANNELS", "")
TELEGRAM_CHANNELS   = [c.strip() for c in _raw_channels.split(",") if c.strip()]
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID") or os.getenv("CHANNEL_ID")
CHANNEL_ID          = TELEGRAM_CHANNEL_ID or (TELEGRAM_CHANNELS[0] if TELEGRAM_CHANNELS else None)
BOT_TOKEN           = TELEGRAM_BOT_TOKEN

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

# ── AI / ML ───────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_API_KEYS       = [k.strip() for k in os.getenv("GROQ_API_KEYS", ",".join(filter(None, [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]))).split(",") if k.strip()]
GROQ_CLASSIFICATION_MODEL = os.getenv("GROQ_CLASSIFICATION_MODEL", "llama-3.1-8b-instant")
HF_API_KEY          = os.getenv("HF_API_KEY")

# ── News scoring thresholds — importance-tiered, calibrated to model output ───
# Recalibrated 2026-07 against storage/news_cache.json (3,501 items). The gate is
# on the production 15-minute score AND a confidence floor. One-hour fields are
# retained only for historical research compatibility.
#
# Why the old values were wrong: `confidence` is the max of a 3-class softmax, so
# its natural floor is ~0.33 and ~95% of items already exceed 0.50 — the old 0.50
# confidence gate filtered almost nothing. And a 0.50 score gate tagged ~19% of
# all news "Hot" (the alert tier), vs the ~0.4% originally intended. The values
# below come from the measured score/confidence percentiles so each tier maps to
# the share of news its importance warrants.
#
# Tier    | 15m score           | Confidence | ~% of cache | Action
# Hot     | ≥ 0.80             | ≥ 0.78     | ~1–2%       | Telegram alert (high conviction)
# Medium  | ≥ 0.55             | ≥ 0.70     | ~12–15%     | Highlighted badge
# Show    | ≥ 0.30             | ≥ 0.50     | ~20–25%     | Shown in dashboard feed
# Hidden  | below Show gate                              | rest        | not displayed
DASHBOARD_API        = os.getenv("DASHBOARD_API", "http://localhost:8000")
SCORE_15M_MIN        = 0.0
SCORE_15M_MAX        = 1.0
SCORE_1H_MIN         = 0.0
SCORE_1H_MAX         = 1.0

# Impact badge / gate thresholds — applied to the production 15-minute score.
SCORE_THRESHOLD_HOT    = 0.80   # Hot badge / alert
SCORE_THRESHOLD_MEDIUM = 0.55   # Medium badge
SCORE_THRESHOLD_SHOW   = 0.30   # minimum score to display in feed
SCORE_THRESHOLD_HIGH   = SCORE_THRESHOLD_HOT   # alias for legacy code

# Confidence floors — one per tier, scaled by news importance
CONF_SHOW   = 0.50   # display floor (matches server CONF_MIN=50 and dashboard CONF_MIN=50)
CONF_MEDIUM = 0.70   # medium tier confidence (Telegram bot only)
CONF_HOT    = 0.78   # hot tier / alert (Telegram bot only)
CONF_MIN    = CONF_SHOW   # legacy alias = display floor

# "Show" tier — minimum to display in dashboard feed (score AND confidence gate)
IMPORTANT_MIN_SCORE      = SCORE_THRESHOLD_SHOW
IMPORTANT_MIN_CONFIDENCE = CONF_SHOW
IMPORTANT_MIN_SCORE_1H   = SCORE_THRESHOLD_SHOW

# "Hot" tier — triggers Telegram alert (uses max of both scores)
HOT_MIN_MODEL_SCORE      = SCORE_THRESHOLD_HOT
HOT_MIN_CONFIDENCE       = CONF_HOT
HOT_MIN_MODEL_SCORE_1H   = SCORE_THRESHOLD_HOT   # 1h also checked via max()
HOT_MIN_SCORE_1H         = SCORE_THRESHOLD_HOT
HOT_MAX_AGE_MIN          = 30
BATCH_SIZE           = 3


def impact_tier(score_15m, score_1h=None) -> str:
    """Canonical live impact badge from the production 15-minute score.

    ``score_1h`` remains accepted for compatibility with historical callers but
    intentionally does not affect live routing.
    Returns one of: 'Hot' | 'Medium' | 'Show' | 'Low'."""
    s = abs(float(score_15m or 0))
    if s >= SCORE_THRESHOLD_HOT:
        return "Hot"
    if s >= SCORE_THRESHOLD_MEDIUM:
        return "Medium"
    return "Show" if s >= SCORE_THRESHOLD_SHOW else "Low"

# ── News Importance (editorial importance — independent of price impact) ──
import re as _re
_IMPORTANCE_KW = _re.compile(r'\b(JUST IN|BREAKING|MASSIVE|BIG|ALERT|NOW|UPDATE|URGENT)\b', _re.IGNORECASE)
_CHANNEL_AUTH  = {"cointelegraph": 1.0, "coindesk": 1.0, "the_block_crypto": 0.95, "WatcherGuru": 0.85, "google_news": 0.7}

def news_importance(item: dict) -> dict:
    """Return { tier: 'Key'|'Notable'|'Regular', score: 0-100 }"""
    conf = float(item.get("confidence", 0)) / 100.0
    probs = [float(item.get("prob_positive", 0)), float(item.get("prob_negative", 0)), float(item.get("prob_neutral", 0))]
    sent_strength = max(probs) if probs else 0.0
    ch_auth = _CHANNEL_AUTH.get(item.get("channel", ""), 0.5)
    kw_boost = 0.15 if _IMPORTANCE_KW.search(item.get("title", "")) else 0.0
    raw = (conf * 0.35) + (sent_strength * 0.25) + (ch_auth * 0.25) + kw_boost
    pct = min(100, round(raw * 100))
    tier = "Key" if pct >= 70 else ("Notable" if pct >= 55 else "Regular")
    return {"tier": tier, "score": pct}

# ── External APIs ─────────────────────────────────────────────────────────────
BINANCE_API   = os.getenv("BINANCE_API",   "https://api.binance.com/api/v3")
COINGECKO_API = os.getenv("COINGECKO_API", "https://api.coingecko.com/api/v3")

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# sentiment model
SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")

# ── Validation ────────────────────────────────────────────────────────────────
def validate_bot():
    """Call at main.py startup to catch missing bot env vars early."""
    required = {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_API_ID":    TELEGRAM_API_ID,
        "TELEGRAM_API_HASH":  TELEGRAM_API_HASH,
        "TELEGRAM_SESSION":   TELEGRAM_SESSION,
        "GROQ_API_KEY":       GROQ_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required env vars for bot: {', '.join(missing)}")
    if not DATABASE_URL:
        print("⚠️  DB_URL not set — running in cache-only mode (news_cache.json)")
    print("✅ Bot config OK")


def validate_api():
    """Call at api/server.py startup to catch missing API env vars early."""
    ingest_key = os.getenv("INGEST_API_KEY", "")
    if not ingest_key:
        print("⚠️  INGEST_API_KEY not set — POST /news endpoint will be disabled")
    print("✅ API config OK")


def validate():
    """Legacy: validates both bot and API vars. Use validate_bot()/validate_api() directly."""
    validate_bot()
    validate_api()


# ── FOMC calendar ─────────────────────────────────────────────────────────────
# Fed meeting dates (announcement day).  ±3-day window is treated as fomc_week=1.
# Update this list when the Fed publishes a new calendar.
from datetime import date as _date, timedelta as _td

_FOMC_DATES_RAW = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
_s: set = set()
for _raw in _FOMC_DATES_RAW:
    _center = _date.fromisoformat(_raw)
    for _off in range(-3, 4):
        _s.add(_center + _td(days=_off))
FOMC_WEEK_DATES: frozenset = frozenset(_s)
del _s, _raw, _center, _off


if __name__ == "__main__":
    # Run this file directly to test your .env is set up correctly:
    # python config.py
    validate()
    print(f"  Channels : {TELEGRAM_CHANNELS}")
    print(f"  API Port : {API_PORT}")
    print("  Model    : xgb_impact_clf_15m_bert_rag.json (15-minute RAG)")
