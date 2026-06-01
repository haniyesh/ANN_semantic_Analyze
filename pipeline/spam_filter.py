import re

# ==============================
# 🧹 EMOJI CLEANER
# ==============================
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00010000-\U0010FFFF"
    "☀-⛿"
    "✀-➿"
    "]+",
    flags=re.UNICODE,
)

# ==============================
# 🚫 SPAM PHRASES
# ==============================
SPAM_PHRASES = [
    # Promotions
    "vip community", "vip group", "vip channel", "vip signal",
    "join our", "join now", "join us",
    "sign up", "subscribe",
    "trading signals", "signal group", "our signals", "our channel",

    # Scams
    "free crypto", "airdrop", "giveaway", "claim now",
    "make money", "earn money", "get rich",
    "100x", "guaranteed profit",
    "dm me", "dm for",

    # Clickbait
    "click here", "click link", "click the link",
    "limited offer", "exclusive offer",
    "don't miss", "dont miss",

    # Digest / recap posts (not real news)
    "historical snapshots", "liquidity analysis", "market snapshot",
    "top crypto news:", "(24h)", "(weekly)",
    "weekly roundup", "weekly recap", "daily recap",
    "daily roundup", "morning brief",
    "weekly digest", "daily digest",

    # Generic analysis spam
    "market update:", "price analysis:",
    "technical analysis:", "| analysis:", "| snapshot",
]

MIN_WORDS = 5


# ==============================
# 🧹 CLEAN TITLE
# ==============================
def clean_title(text: str) -> str:
    """Strip emojis and extra whitespace."""
    return _EMOJI_RE.sub("", text).strip()


# ==============================
# 🔍 IS SPAM
# ==============================
def is_spam(title: str) -> bool:
    """Returns True if the title contains any spam phrase."""
    t = title.lower()
    return any(phrase in t for phrase in SPAM_PHRASES)


# ==============================
# 📏 IS TOO SHORT
# ==============================
def is_too_short(title: str, min_length: int = 20) -> bool:
    """Returns True if the title is too short to be real news."""
    return len(title.strip()) < min_length


# ==============================
# 📰 IS REAL HEADLINE
# ==============================
def is_real_headline(text: str) -> bool:
    """Returns False for URLs, pure numbers, timestamps, single words."""
    t = text.strip()
    if not t:                           return False
    if t.lower().startswith("time:"):   return False
    if re.match(r"^\d+$", t):          return False
    if re.match(r"^https?://\S+$", t): return False
    if len(t.split()) < 2:             return False
    return True


# ==============================
# ✅ PASSES ALL PRE-FILTERS
# ==============================
def passes_pre_filters(title: str) -> tuple:
    """
    Main function — run this on every incoming message.
    Returns (cleaned_title, skip_reason)
    skip_reason is None if the title passed all filters.

    Usage:
        clean, reason = passes_pre_filters(title)
        if reason:
            return   # skip this message
    """
    clean = clean_title(title)

    if not is_real_headline(clean):
        return clean, "not a headline"

    if len(clean.split()) < MIN_WORDS:
        return clean, f"too short ({len(clean.split())} words)"

    if is_spam(clean):
        return clean, "spam phrase detected"

    if is_too_short(clean):
        return clean, "title under 20 characters"

    return clean, None  # ✅ passed