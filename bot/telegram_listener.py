import asyncio
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_SEEN_MAX       = 20_000  # bound the dedup set so memory stays flat
_FALLBACK_DAYS  = 30      # used only for channels with no stored history

# Process-lifetime dedup of message links. Shared by the live handler and the
# backfill so a message is never queued twice (e.g. a live message arriving
# during backfill, or a restart re-reading the same history).
_seen_links: "OrderedDict[str, None]" = OrderedDict()


def _mark_seen(link: str) -> bool:
    """Return True if this link is new (and record it); False if already seen."""
    if not link:
        return True  # no stable id — let it through, can't dedup
    if link in _seen_links:
        return False
    _seen_links[link] = None
    if len(_seen_links) > _SEEN_MAX:
        _seen_links.popitem(last=False)  # evict oldest
    return True


def _load_cursors(cache_path: Path) -> dict:
    """Return {channel_username: max_telegram_msg_id} from the on-disk cache.

    Only entries whose link field contains a parseable Telegram message URL
    (https://t.me/<channel>/<id>) are counted. Channels with no stored
    messages get no entry — the caller falls back to a date window.
    """
    cursors: dict[str, int] = {}
    try:
        with open(cache_path) as f:
            items = json.load(f)
    except Exception:
        return cursors
    for it in items:
        link = it.get("link") or ""
        m = re.search(r"t\.me/([^/]+)/(\d+)$", link)
        if not m:
            continue
        ch_name = m.group(1)
        msg_id  = int(m.group(2))
        if cursors.get(ch_name, 0) < msg_id:
            cursors[ch_name] = msg_id
    return cursors


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

    ROOT         = Path(__file__).resolve().parent.parent
    session_name = str(ROOT / "telegram_session")
    cache_path   = ROOT / "storage" / "news_cache.json"

    if not sys.stdin.isatty():
        raise RuntimeError(
            "[TELEGRAM] Not running in a TTY — client.start() would hang waiting "
            "for an interactive login code. Run once interactively to create the "
            "session file, then restart under systemd/Docker."
        )

    client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)

    def _make_link(channel: str, msg_id: int) -> str:
        return f"https://t.me/{channel}/{msg_id}" if channel else ""

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

            link = _make_link(channel, msg.id)
            pub_dt = msg.date

            if not _mark_seen(link):
                return  # already queued/processed this message

            news_queue.append({
                "title":   title,
                "text":    text,
                "source":  channel,
                "link":    link,
                "pub_dt":  pub_dt,
            })
            print(f"[TELEGRAM] {channel} | {title[:70]}")

        except Exception as e:
            print(f"[TELEGRAM] Handler error: {e}")

    print(f"[TELEGRAM] Connecting -- monitoring {len(TELEGRAM_CHANNELS)} channels...")
    await client.start()

    # ── Cursor-based backfill ─────────────────────────────────────
    # For each channel, resume from the last Telegram message ID stored in
    # cache (zero duplicates, no fixed time window). Channels with no stored
    # history fall back to _FALLBACK_DAYS.
    cursors = _load_cursors(cache_path)
    fallback_cutoff = datetime.now(timezone.utc) - timedelta(days=_FALLBACK_DAYS)

    total_backfill = 0
    skipped = 0
    for ch in TELEGRAM_CHANNELS:
        last_id = cursors.get(ch, 0)
        count = 0
        try:
            if last_id:
                print(f"[TELEGRAM] {ch}: resuming from msg_id {last_id}")
                async for msg in client.iter_messages(ch, min_id=last_id, limit=None):
                    if not msg or not msg.date:
                        continue
                    text = (msg.text or "").strip()
                    if not text:
                        continue
                    link = _make_link(ch, msg.id)
                    if not _mark_seen(link):
                        skipped += 1
                        continue
                    msg_dt = msg.date if msg.date.tzinfo else msg.date.replace(tzinfo=timezone.utc)
                    title = text.splitlines()[0][:300]
                    news_queue.append({
                        "title":  title,
                        "text":   text,
                        "source": ch,
                        "link":   link,
                        "pub_dt": msg_dt,
                    })
                    count += 1
            else:
                print(f"[TELEGRAM] {ch}: no history found — fetching last {_FALLBACK_DAYS} days")
                async for msg in client.iter_messages(ch, reverse=False, limit=None):
                    if not msg or not msg.date:
                        continue
                    msg_dt = msg.date if msg.date.tzinfo else msg.date.replace(tzinfo=timezone.utc)
                    if msg_dt < fallback_cutoff:
                        break
                    text = (msg.text or "").strip()
                    if not text:
                        continue
                    link = _make_link(ch, msg.id)
                    if not _mark_seen(link):
                        skipped += 1
                        continue
                    title = text.splitlines()[0][:300]
                    news_queue.append({
                        "title":  title,
                        "text":   text,
                        "source": ch,
                        "link":   link,
                        "pub_dt": msg_dt,
                    })
                    count += 1
        except Exception as e:
            print(f"[TELEGRAM] Backfill error for {ch}: {e}")
        print(f"[TELEGRAM]   {ch}: {count} new messages queued")
        total_backfill += count

    print(f"[TELEGRAM] Backfill complete — {total_backfill} queued, {skipped} in-process duplicates skipped")
    print(f"[TELEGRAM] Listening: {TELEGRAM_CHANNELS}")
    await client.run_until_disconnected()
