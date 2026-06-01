import asyncio
import re
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from config import BOT_TOKEN, CHANNEL_ID


# ==============================
# 🧹 HELPERS
# ==============================
def clean_url(url: str) -> str:
    """Remove UTM parameters from URL."""
    if not url:
        return None
    return url.split("?")[0]


def clean_text(text: str) -> str:
    """Remove URLs from text."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text.strip()


def format_time(published=None) -> str:
    """Convert published time to Istanbul timezone string."""
    try:
        if published:
            if isinstance(published, str):
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%a, %d %b %Y %H:%M:%S GMT",
                ]:
                    try:
                        dt = datetime.strptime(published, fmt)
                        return dt.astimezone(ZoneInfo("Europe/Istanbul")).strftime("🕐 %d %b %Y • %H:%M (TR)")
                    except Exception:
                        continue
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                return dt.astimezone(ZoneInfo("Europe/Istanbul")).strftime("🕐 %d %b %Y • %H:%M (TR)")
    except Exception:
        pass
    return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("🕐 %d %b %Y • %H:%M (TR)")


# ==============================
# 📤 SEND ALERT
# ==============================
def send_telegram_alert(news: dict, signal: str, impact_score: float, related_news: list = None):
    """Send a news signal alert to the Telegram channel."""
    if not BOT_TOKEN or not CHANNEL_ID:
        print("[ALERT] BOT_TOKEN or CHANNEL_ID not set")
        return

    title    = clean_text(news.get("title", news.get("text", "")))
    time_str = format_time(news.get("published"))
    link     = clean_url(news.get("link"))
    link_text = f"🔗 {link}" if link else "🔗 Telegram"

    # Deduplicate related news — max 3, no duplicates
    main_key = title[:50]
    seen     = {main_key}
    unique_related = []
    for r in (related_news or []):
        r_title = r.get("title", r.get("text", ""))
        r_key   = r_title[:50]
        if r_key in seen:
            continue
        seen.add(r_key)
        unique_related.append(r)
        if len(unique_related) >= 3:
            break

    related_text = "\n".join([
        f"• {clean_text(r.get('title', r.get('text', '')))[:100]}"
        for r in unique_related
    ]) or "No similar past news found."

    text = (
        f"🔔 {title}\n\n"
        f"{time_str}\n\n"
        f"📊 Category: {news.get('type', 'unknown')}\n"
        f"🧠 Impact Score: {impact_score:.2f}\n"
        f"📢 Signal: {signal}\n\n"
        f"{link_text}\n\n"
        f"🔍 Related past news:\n{related_text}"
    )

    if len(text) > 4096:
        text = text[:4090] + "..."

    resp = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={
            "chat_id":                  CHANNEL_ID,
            "text":                     text,
            "disable_web_page_preview": True,
        },
    )

    if resp.status_code == 200:
        print("[ALERT] ✅ Sent to Telegram")
    else:
        print(f"[ALERT] ❌ Error: {resp.text}")


# ==============================
# 🟢 TELEGRAM LISTENER
# ==============================
async def start(news_queue):
    """Placeholder Telegram listener entrypoint.

    The actual telegram feed implementation is not present in this file,
    but this coroutine keeps the service alive while the processor loop runs.
    """
    print("[TELEGRAM] Listener placeholder active — no Telegram feed implemented.")
    await asyncio.Event().wait()
