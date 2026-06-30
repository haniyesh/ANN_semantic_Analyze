"""Guard for Telegram listener message deduplication.

A restart re-reads BACKFILL_DAYS of history, and a live message can arrive
during backfill — without dedup the same message is queued (and scored) twice.
"""
from bot.telegram_listener import _mark_seen, _seen_links, _SEEN_MAX


def setup_function(_):
    _seen_links.clear()


def test_same_link_seen_once():
    link = "https://t.me/chan/100"
    assert _mark_seen(link) is True
    assert _mark_seen(link) is False
    assert _mark_seen(link) is False


def test_empty_link_always_passes():
    assert _mark_seen("") is True
    assert _mark_seen("") is True  # no stable id, cannot dedup


def test_eviction_bounds_memory():
    for i in range(_SEEN_MAX + 100):
        _mark_seen(f"https://t.me/c/{i}")
    assert len(_seen_links) <= _SEEN_MAX
