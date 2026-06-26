"""
telegram_alert.py
=================
Thin helper for posting HOT signal alerts to a Telegram channel.

Used by the live pipeline when a news item crosses the HOT threshold. Kept
intentionally small and dependency-light: it fails soft (logs and returns
False) so an alerting outage never crashes the scoring pipeline.

Env:
  BOT_TOKEN          — Telegram bot token
  SIGNAL_CHANNEL_ID  — chat/channel id to post to
"""

import os


def _format_alert(payload: dict) -> str:
    emoji = {"BUY": "🟢", "SELL": "🔴"}.get(payload.get("type", ""), "🟡")
    lines = [
        f"{emoji} *{payload.get('type', 'SIGNAL')} SIGNAL*",
        "",
        f"📰 {payload.get('title', '')}",
        "",
        f"📡 `{payload.get('channel', '')}`  |  ⏰ `{payload.get('age_minutes', 0)}m ago`",
        f"🎯 Conf: `{payload.get('confidence', 0)}%`  |  "
        f"🤖 Score: `{payload.get('model_score', 0)}`",
    ]
    link = payload.get("link", "")
    if link:
        lines += ["", f"🔗 {link}"]
    return "\n".join(lines)


async def send_alert(payload: dict) -> bool:
    """Post a formatted HOT alert. Returns True on success, False otherwise.

    Never raises — alerting failures must not break the pipeline.
    """
    token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("SIGNAL_CHANNEL_ID")
    if not token or not chat_id:
        print("  ⚠️  telegram_alert: BOT_TOKEN / SIGNAL_CHANNEL_ID not set — skipping alert")
        return False
    try:
        from telegram import Bot
        bot = Bot(token=token)
        await bot.send_message(
            chat_id=chat_id,
            text=_format_alert(payload),
            parse_mode="Markdown",
            disable_web_page_preview=False,
        )
        return True
    except Exception as e:
        print(f"  ⚠️  telegram_alert send failed: {type(e).__name__}: {e}")
        return False
