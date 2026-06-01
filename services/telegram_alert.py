import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import re
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")

def clean_url(url):
    if not url:
        return None
    return url.split('?')[0]  # ✅ remove UTM parameters

def format_time(published=None):
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
                        tr_time = dt.astimezone(ZoneInfo("Europe/Istanbul"))
                        return tr_time.strftime("🕐 %d %b %Y • %H:%M (TR)")
                    except:
                        continue
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                tr_time = dt.astimezone(ZoneInfo("Europe/Istanbul"))
                return tr_time.strftime("🕐 %d %b %Y • %H:%M (TR)")
        return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("🕐 %d %b %Y • %H:%M (TR)")
    except:
        return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("🕐 %d %b %Y • %H:%M (TR)")
def clean_text(text):
    # ✅ remove URLs from text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text.strip()
def send_telegram_alert(news, signal, impact_score, related_news=None):
    if not BOT_TOKEN or not CHANNEL_ID:
        print("❌ BOT_TOKEN or CHANNEL_ID not set")
        return

    time_str = format_time(news.get("published"))
    link = clean_url(news.get('link'))
    link_text = f"🔗 {link}" if link else "🔗 Telegram"
    title = clean_text(news.get('title', news.get('text', '')))
    # ✅ max 3 related, no duplicates, exclude main news
    main_title = title[:50]
    seen = set()
    seen.add(main_title)
    unique_related = []
    for r in (related_news or []):
        r_title = r.get('title', r.get('text', ''))
        r_key = r_title[:50]
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

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(url, data={
        "chat_id": CHANNEL_ID,
        "text": text,
        "disable_web_page_preview": True
    })

    if response.status_code == 200:
        print(f"✅ Alert sent to Telegram")
    else:
        print(f"❌ Telegram error: {response.text}")
