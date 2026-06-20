import asyncio
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from config import BOT_TOKEN, CHANNEL_ID


def clean_url(url: str) -> str:
    if not url:
        return None
    return url.split("?")[0]


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text.strip()


def format_time(published=None) -> str:
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
                        return dt.astimezone(ZoneInfo("Europe/Istanbul")).strftime("%d %b %Y %H:%M (TR)")
                    except Exception:
                        continue
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                return dt.astimezone(ZoneInfo("Europe/Istanbul")).strftime("%d %b %Y %H:%M (TR)")
    except Exception:
        pass
    return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%d %b %Y %H:%M (TR)")


async def start(news_queue):
    from telethon import TelegramClient, events
    from config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION, TELEGRAM_CHANNELS

    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        print("[TELEGRAM] TELEGRAM_API_ID / TELEGRAM_API_HASH not set -- listener disabled.")
        await asyncio.Event().wait()
        return

    if not TELEGRAM_CHANNELS:
        print("[TELEGRAM] TELEGRAM_CHANNELS not set in .env -- listener disabled.")
        await asyncio.Event().wait()
        return

    from pathlib import Path
    session_name = str(Path(__file__).resolve().parent.parent / "telegram_session")

    client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)

    @client.on(events.NewMessage(chats=TELEGRAM_CHANNELS))
    async def handler(event):
        try:
            msg  = event.message
            text = (msg.text or "").strip()
            if not text:
                return

            title = text.splitlines()[0][:300]

            channel = ""
            if event.chat:
                channel = getattr(event.chat, "username", "") or \
                          getattr(event.chat, "title",    "") or "telegram"

            pub_dt = msg.date

            news_queue.append({
                "title":   title,
                "text":    text,
                "source":  channel,
                "link":    f"https://t.me/{channel}/{msg.id}" if channel else "",
                "pub_dt":  pub_dt,
            })
            print(f"[TELEGRAM] {channel} | {title[:70]}")

        except Exception as e:
            print(f"[TELEGRAM] Handler error: {e}")

    print(f"[TELEGRAM] Connecting -- monitoring {len(TELEGRAM_CHANNELS)} channels...")
    await client.start()
    print(f"[TELEGRAM] Listening: {TELEGRAM_CHANNELS}")
    await client.run_until_disconnected()
