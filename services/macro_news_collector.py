"""
Macro & Geopolitical News Collector
====================================
Real-time news collector for BTC/ETH-impacting events:
  - CryptoPanic API  → crypto-specific news (BTC/ETH filtered)
  - RSS feeds        → Reuters, BBC, Al Jazeera (war/macro/sanctions)
  - Tree of Alpha    → aggregated crypto + macro news

All news is pushed into the shared news_queue used by the main pipeline.

Usage (called from main.py):
    from services.macro_news_collector import start as start_macro_collector
    await asyncio.gather(..., start_macro_collector(news_queue))
"""

import asyncio
import hashlib
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import deque

import httpx

# ─── Keywords that indicate BTC/ETH-moving events ───────────────────────────


# ─── RSS sources ─────────────────────────────────────────────────────────────
RSS_FEEDS = [
    {"url": "https://feeds.reuters.com/reuters/topNews",       "source": "reuters"},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml",     "source": "bbc_world"},
    {"url": "https://www.aljazeera.com/xml/rss/all.xml",        "source": "aljazeera"},
    {"url": "https://rss.dw.com/rdf/rss-en-world",             "source": "dw_world"},
]

RSS_POLL_INTERVAL   = 60   # seconds between RSS polls
CRYPTOPANIC_INTERVAL = 30  # seconds between CryptoPanic polls
TREEOFALPHA_INTERVAL = 20  # seconds between Tree of Alpha polls

# ─── Dedup cache (keep last 2000 hashes) ─────────────────────────────────────
_seen: deque = deque(maxlen=2000)


def _is_new(title: str) -> bool:
    h = hashlib.md5(title.strip().lower().encode()).hexdigest()
    if h in _seen:
        return False
    _seen.append(h)
    return True


def _has_impact(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in IMPACT_KEYWORDS)


def _make_news(title: str, source: str, url: str = "", pub_dt: datetime = None) -> dict:
    return {
        "title":   title,
        "source":  source,
        "link":    url or title,
        "pub_dt":  pub_dt or datetime.now(timezone.utc),
        "text":    title,
    }


# ─── CryptoPanic ─────────────────────────────────────────────────────────────
async def _poll_cryptopanic(client: httpx.AsyncClient, queue: deque, token: str):
    """
    Free tier: https://cryptopanic.com/api/free/v1/posts/
    Get a free API key at https://cryptopanic.com/developers/api/
    """
    try:
        resp = await client.get(
            "https://cryptopanic.com/api/free/v1/posts/",
            params={
                "auth_token": token,
                "currencies": "BTC,ETH",
                "filter":     "important",
                "public":     "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for post in data.get("results", []):
            title = post.get("title", "")
            url   = post.get("url", "")
            if title and _is_new(title):
                pub_dt = None
                if post.get("published_at"):
                    try:
                        pub_dt = datetime.fromisoformat(
                            post["published_at"].replace("Z", "+00:00")
                        )
                    except Exception:
                        pass
                queue.append(_make_news(title, "cryptopanic", url, pub_dt))
                print(f"[CRYPTOPANIC] {title[:80]}")
    except Exception as e:
        print(f"[CRYPTOPANIC] Error: {e}")


async def cryptopanic_loop(queue: deque, token: str):
    async with httpx.AsyncClient() as client:
        while True:
            await _poll_cryptopanic(client, queue, token)
            await asyncio.sleep(CRYPTOPANIC_INTERVAL)


# ─── Tree of Alpha ────────────────────────────────────────────────────────────
async def _poll_treeofalpha(client: httpx.AsyncClient, queue: deque):
    try:
        resp = await client.get(
            "https://news.treeofalpha.com/api/news",
            params={"limit": 20},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json()
        for item in items:
            title = item.get("title", "")
            url   = item.get("url", "")
            src   = f"treeofalpha_{item.get('source', 'unknown').lower()}"

            # Only keep items that mention BTC/ETH coins or match impact keywords
            suggestions = item.get("suggestions", [])
            coins = [s.get("coin", "") for s in suggestions]
            is_btc_eth = any(c in ("BTC", "ETH") for c in coins)

            if title and _is_new(title):
                pub_dt = None
                ts = item.get("time")
                if ts:
                    pub_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                queue.append(_make_news(title, src, url, pub_dt))
                print(f"[TREEOFALPHA] {title[:80]}")
    except Exception as e:
        print(f"[TREEOFALPHA] Error: {e}")


async def treeofalpha_loop(queue: deque):
    async with httpx.AsyncClient() as client:
        while True:
            await _poll_treeofalpha(client, queue)
            await asyncio.sleep(TREEOFALPHA_INTERVAL)


# ─── RSS feeds ────────────────────────────────────────────────────────────────
async def _poll_rss(client: httpx.AsyncClient, queue: deque, feed: dict):
    try:
        resp = await client.get(feed["url"], timeout=10,
                                headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        # Handle both RSS and Atom formats
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        for item in items:
            title_el = item.find("title") or item.find("atom:title", ns)
            link_el  = item.find("link")  or item.find("atom:link",  ns)

            if title_el is None:
                continue

            title = (title_el.text or "").strip()
            url   = (link_el.text or link_el.get("href", "") if link_el is not None else "")

            if title and _is_new(title):
                queue.append(_make_news(title, feed["source"], url))
                print(f"[RSS:{feed['source'].upper()}] {title[:80]}")

    except Exception as e:
        print(f"[RSS:{feed['source']}] Error: {e}")


async def rss_loop(queue: deque):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        while True:
            tasks = [_poll_rss(client, queue, feed) for feed in RSS_FEEDS]
            await asyncio.gather(*tasks)
            await asyncio.sleep(RSS_POLL_INTERVAL)


# ─── Entry point ─────────────────────────────────────────────────────────────
async def start(queue: deque, cryptopanic_token: str = ""):
    """
    Start all collectors. Call this from main.py:

        from services.macro_news_collector import start as start_macro_collector
        await asyncio.gather(
            start_telegram_listener(news_queue),
            processor_loop(pool),
            start_macro_collector(news_queue, cryptopanic_token="YOUR_TOKEN"),
        )

    CryptoPanic token is optional — RSS + Tree of Alpha work without one.
    Get a free token at: https://cryptopanic.com/developers/api/
    """
    print("[MACRO COLLECTOR] Starting...")
    collectors = [rss_loop(queue), treeofalpha_loop(queue)]

    if cryptopanic_token:
        collectors.append(cryptopanic_loop(queue, cryptopanic_token))
        print("[MACRO COLLECTOR] CryptoPanic ✅")
    else:
        print("[MACRO COLLECTOR] CryptoPanic skipped (no token) — add CRYPTOPANIC_TOKEN to .env")

    print(f"[MACRO COLLECTOR] RSS feeds: {[f['source'] for f in RSS_FEEDS]} ✅")
    print("[MACRO COLLECTOR] Tree of Alpha ✅")

    await asyncio.gather(*collectors)
