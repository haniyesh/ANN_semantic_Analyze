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


def _load_cursor_state(cache_path: Path, cursor_path: Path) -> dict[str, int]:
    """Load cursors, preferring explicit durable state over cache inference.

    The state file is intentionally authoritative: operators can rewind a
    channel for gap recovery even when the display cache contains a newer,
    sparse message that would otherwise make intermediate IDs look processed.
    Cache-derived values are used only for channels absent from durable state.
    """
    cursors = _load_cursors(cache_path)
    try:
        stored = json.loads(cursor_path.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            for channel, msg_id in stored.items():
                msg_id = int(msg_id)
                cursors[channel] = msg_id
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return cursors


def _save_cursor_state(cursor_path: Path, cursors: dict[str, int]) -> None:
    """Atomically persist acknowledged processing progress."""
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cursor_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cursors, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(cursor_path)


def acknowledge_message(channel: str, msg_id: int) -> None:
    """Advance a channel only after its queued message finished processing."""
    if not channel or not msg_id:
        return
    root = Path(__file__).resolve().parent.parent
    cursor_path = root / "storage" / "telegram_cursors.json"
    try:
        cursors = json.loads(cursor_path.read_text(encoding="utf-8"))
        if not isinstance(cursors, dict):
            cursors = {}
    except (FileNotFoundError, json.JSONDecodeError):
        cursors = {}
    if int(msg_id) > int(cursors.get(channel, 0) or 0):
        cursors[channel] = int(msg_id)
        _save_cursor_state(cursor_path, cursors)


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
    cursor_path  = ROOT / "storage" / "telegram_cursors.json"

    # Do not download and queue channels that the production noise policy will
    # reject later. This previously replayed hundreds of useless messages on
    # every restart because rejected items never entered news_cache.json.
    from pipeline.reduce_noise import BLOCKED_CHANNELS
    active_channels = [ch for ch in TELEGRAM_CHANNELS if ch not in BLOCKED_CHANNELS]
    blocked_channels = [ch for ch in TELEGRAM_CHANNELS if ch in BLOCKED_CHANNELS]
    for ch in blocked_channels:
        print(f"[TELEGRAM] {ch}: skipped (blocked by news filter)")

    cursors = _load_cursor_state(cache_path, cursor_path)

    if not sys.stdin.isatty():
        raise RuntimeError(
            "[TELEGRAM] Not running in a TTY — client.start() would hang waiting "
            "for an interactive login code. Run once interactively to create the "
            "session file, then restart under systemd/Docker."
        )

    client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    backfill_in_progress = True
    live_buffer = []

    def _make_link(channel: str, msg_id: int) -> str:
        return f"https://t.me/{channel}/{msg_id}" if channel else ""

    @client.on(events.NewMessage(chats=active_channels))
    async def handler(event):
        try:
            msg  = event.message
            channel = ""
            if event.chat:
                channel = getattr(event.chat, "username", "") or \
                          getattr(event.chat, "title",    "") or "telegram"

            text = (msg.text or "").strip()
            if not text:
                return

            title = text.splitlines()[0][:300]

            link = _make_link(channel, msg.id)
            pub_dt = msg.date

            if not _mark_seen(link):
                return  # already queued/processed this message

            payload = {
                "title":   title,
                "text":    text,
                "source":  channel,
                "link":    link,
                "pub_dt":  pub_dt,
                "telegram_channel": channel,
                "telegram_msg_id": msg.id,
            }
            (live_buffer if backfill_in_progress else news_queue).append(payload)
            print(f"[TELEGRAM] {channel} | {title[:70]}")

        except Exception as e:
            print(f"[TELEGRAM] Handler error: {e}")

    print(f"[TELEGRAM] Connecting -- monitoring {len(active_channels)} channels...")
    await client.start()

    # ── Cursor-based backfill ─────────────────────────────────────
    # For each channel, resume from the last Telegram message ID stored in
    # cache (zero duplicates, no fixed time window). Channels with no stored
    # history fall back to _FALLBACK_DAYS.
    fallback_cutoff = datetime.now(timezone.utc) - timedelta(days=_FALLBACK_DAYS)

    total_backfill = 0
    skipped = 0
    pending_backfill = []
    for ch in active_channels:
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
                    pending_backfill.append({
                        "title":  title,
                        "text":   text,
                        "source": ch,
                        "link":   link,
                        "pub_dt": msg_dt,
                        "telegram_channel": ch,
                        "telegram_msg_id": msg.id,
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
                    pending_backfill.append({
                        "title":  title,
                        "text":   text,
                        "source": ch,
                        "link":   link,
                        "pub_dt": msg_dt,
                        "telegram_channel": ch,
                        "telegram_msg_id": msg.id,
                    })
                    count += 1
        except Exception as e:
            print(f"[TELEGRAM] Backfill error for {ch}: {e}")
        print(f"[TELEGRAM]   {ch}: {count} new messages queued")
        total_backfill += count

    # Telethon returns newest first by default. Process oldest first so every
    # acknowledged cursor represents a contiguous completed prefix. Live
    # messages received during the scan are buffered behind that backlog.
    pending_backfill.sort(key=lambda item: item["pub_dt"])
    live_buffer.sort(key=lambda item: item["pub_dt"])
    news_queue.extend(pending_backfill)
    news_queue.extend(live_buffer)
    backfill_in_progress = False

    print(f"[TELEGRAM] Backfill complete — {total_backfill} queued, {skipped} in-process duplicates skipped")
    print(f"[TELEGRAM] Listening: {active_channels}")
    await client.run_until_disconnected()
