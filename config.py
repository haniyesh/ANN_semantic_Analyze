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
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
CHANNEL_ID          = TELEGRAM_CHANNEL_ID or (TELEGRAM_CHANNELS[0] if TELEGRAM_CHANNELS else None)
BOT_TOKEN           = TELEGRAM_BOT_TOKEN

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DB_URL")

# ── AI / ML ───────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_API_KEYS       = [k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()]
GROQ_CLASSIFICATION_MODEL = os.getenv("GROQ_CLASSIFICATION_MODEL", "llama-3.1-8b-instant")
HF_API_KEY          = os.getenv("HF_API_KEY")

# Path to the trained PyTorch model file
MODEL_PATH          = os.getenv("MODEL_PATH", "production_system_v8.pt")

# ── News scoring thresholds — 4-tier system ───────────────────────────────────
# Tier    | Score  | Confidence | ~% of data
# Hot     | ≥0.55  | ≥60%       | ~0.4%
# Medium  | ≥0.30  | ≥55%       | ~1.5%
# Show    | ≥0.20  | ≥50%       | ~5%
# Hidden  | <0.20  | —          | ~95%  (never sent to dashboard)
DASHBOARD_API        = os.getenv("DASHBOARD_API", "http://localhost:8000")
MODEL_PATH           = "production_system_v8.pt"
SCORE_15M_MIN        = 0.0
SCORE_15M_MAX        = 1.0
SCORE_1H_MIN         = 0.0
SCORE_1H_MAX         = 1.0

# Impact badge thresholds — uses max(score_15m, score_1h), NOT for display filtering
SCORE_THRESHOLD_HOT    = 0.50   # Hot badge
SCORE_THRESHOLD_MEDIUM = 0.25   # Medium badge
SCORE_THRESHOLD_SHOW   = 0.0    # No score gate for display (confidence-only)
SCORE_THRESHOLD_HIGH   = SCORE_THRESHOLD_HOT   # alias for legacy code

# Display filter uses confidence only — no score gate
CONF_MIN    = 0.50   # 50% — minimum confidence to display
CONF_HOT    = CONF_MIN   # kept for backward compat, no separate hot confidence
CONF_MEDIUM = CONF_MIN
CONF_SHOW   = CONF_MIN

# "Show" tier — minimum to display in dashboard feed (confidence only, no score gate)
IMPORTANT_MIN_SCORE      = 0.0    # no score gate
IMPORTANT_MIN_CONFIDENCE = CONF_MIN
IMPORTANT_MIN_SCORE_1H   = 0.0

# "Hot" tier — triggers Telegram alert (uses max of both scores)
HOT_MIN_MODEL_SCORE      = SCORE_THRESHOLD_HOT
HOT_MIN_CONFIDENCE       = CONF_MIN
HOT_MIN_MODEL_SCORE_1H   = SCORE_THRESHOLD_HOT   # 1h also checked via max()
HOT_MIN_SCORE_1H         = SCORE_THRESHOLD_HOT
HOT_MAX_AGE_MIN          = 30
BATCH_SIZE           = 3

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

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_FILE      = "storage/news_cache.json"
MAX_CACHE_ITEMS = 10_000   # keep only the latest N items

# ── External APIs ─────────────────────────────────────────────────────────────
BINANCE_API   = os.getenv("BINANCE_API",   "https://api.binance.com/api/v3")
COINGECKO_API = os.getenv("COINGECKO_API", "https://api.coingecko.com/api/v3")

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# sentiment model
SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")

BINANCE_API   = os.getenv("BINANCE_API",   "https://api.binance.com/api/v3")
COINGECKO_API = os.getenv("COINGECKO_API", "https://api.coingecko.com/api/v3")
# ── Validation ────────────────────────────────────────────────────────────────
def validate():
    """
    Call this at startup to catch missing required settings early.
    Raises ValueError if a required key is missing.
    """
    required = {
        "TELEGRAM_BOT_TOKEN":  TELEGRAM_BOT_TOKEN,
        "TELEGRAM_API_ID":     TELEGRAM_API_ID,
        "TELEGRAM_API_HASH":   TELEGRAM_API_HASH,
        "TELEGRAM_SESSION":    TELEGRAM_SESSION,
        "DATABASE_URL":        DATABASE_URL,
        "GROQ_API_KEY":        GROQ_API_KEY,
        "SENTIMENT_MODEL_NAME": SENTIMENT_MODEL_NAME,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    print("✅ Config loaded successfully")


if __name__ == "__main__":
    # Run this file directly to test your .env is set up correctly:
    # python config.py
    validate()
    print(f"  Channels : {TELEGRAM_CHANNELS}")
    print(f"  API Port : {API_PORT}")
    print(f"  Model    : {MODEL_PATH}")