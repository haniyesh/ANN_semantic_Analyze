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

# ── News scoring thresholds ────────────────────────────────────────────────────
DASHBOARD_API        = os.getenv("DASHBOARD_API", "http://localhost:8000")
MODEL_PATH           = "production_system_v8.pt"
SCORE_15M_MIN        = 0.0
SCORE_15M_MAX        = 1.0
SCORE_1H_MIN         = 0.0
SCORE_1H_MAX         = 1.0
SCORE_THRESHOLD_HIGH   = 0.67
SCORE_THRESHOLD_MEDIUM = 0.40
IMPORTANT_MIN_CONFIDENCE = 0.65
IMPORTANT_MIN_SCORE      = 0.40
IMPORTANT_MIN_SCORE_1H   = 0.40
HOT_MIN_MODEL_SCORE      = 0.67
HOT_MIN_MODEL_SCORE_1H   = 0.60
HOT_MIN_CONFIDENCE       = 0.60
HOT_MIN_SCORE_1H     = 0.60
HOT_MAX_AGE_MIN      = 30
BATCH_SIZE           = 3
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