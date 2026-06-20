import os
import httpx
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import re
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID") or os.getenv("TELEGRAM_CHANNEL_ID")


def clean_url(url):
    if not url:
        return None
    return url.split('?')[0]


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text.strip()


async def send_telegram_alert_async(title: str, signal: str, impact_score: float, similar_news: list = None):
    if not BOT_TOKEN or not CHANNEL_ID:
        print("[ALERT] BOT_TOKEN or CHANNEL_ID not set")
        return

    now = datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%d %b %Y %H:%M (TR)")

    related_text = ""
    if similar_news:
        lines = []
        for s in similar_news[:3]:
            t = clean_text(s.get("title", ""))
            c = s.get("change", s.get("btc_change_15m", 0))
            lines.append(f"  {t[:80]} -> BTC {c:+.2f}%")
        if lines:
            related_text = "\n\nRelated:\n" + "\n".join(lines)

    text = (
        f"{signal} {title}\n\n"
        f"{now}\n"
        f"Impact: {impact_score:.2f}\n"
        f"Signal: {signal}{related_text}"
    )

    if len(text) > 4096:
        text = text[:4090] + "..."

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data={
            "chat_id": CHANNEL_ID,
            "text": text,
            "disable_web_page_preview": True,
        }, timeout=10)

    if resp.status_code == 200:
        print("[ALERT] Sent to Telegram")
    else:
        print(f"[ALERT] Telegram error: {resp.text}")
