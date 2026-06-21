"""
fetch_news_gap.py
=================
Fills the news gap in news_cleaned_filtered_scored.csv using
Google News RSS with date-windowed queries.

Covers: Oct 2024 → Jun 2026 (or any custom range)
Source: Google News RSS (free, no API key, supports date filters)
Output: appended directly to news_cleaned_filtered_scored.csv

Strategy:
  - Weekly windows: after:YYYY-MM-DD before:YYYY-MM-DD
  - Query: bitcoin OR BTC OR Ethereum OR ETH (market/price/regulation focus)
  - Fetch BTC price from Binance for each article timestamp
  - Skip duplicates by title hash
  - Checkpoint every 200 rows

Usage:
  python services/fetch_news_gap.py
  python services/fetch_news_gap.py --from 2025-01-01 --to 2026-06-12
  python services/fetch_news_gap.py --dry-run
"""

import argparse
import csv
import hashlib
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import requests

HERE     = Path(__file__).parent.parent
OUT_CSV  = HERE / "news_cleaned_filtered_scored.csv"
CKPT_CSV = HERE / "services" / "gap_fetch_checkpoint.csv"
BINANCE  = "https://api.binance.com/api/v3/klines"

QUERIES = [
    'bitcoin OR BTC price market',
    'bitcoin OR BTC SEC regulation ETF',
    'bitcoin OR BTC halving mining institutional',
    'Ethereum ETH price market upgrade',
]

FIELDNAMES = [
    "title", "link", "channel", "published",
    "btc_price_at_news", "btc_price_15m", "btc_price_1h",
    "eth_price_at_news", "eth_price_15m", "eth_price_1h",
    "sentiment", "sentiment_score", "weight", "confidence",
    "prob_positive", "prob_negative", "prob_neutral", "news_type",
    "fomc_week", "is_weekend", "is_low_liquidity", "is_us_hours",
    "is_asia_hours", "hour_utc", "day_of_week",
    "btc_pct_change_15m", "btc_pct_change_1h",
    "eth_pct_change_15m", "eth_pct_change_1h",
    "hour_of_day", "word_count", "sentiment_binary",
    "is_spam", "is_relevant", "_hash",
]


def _hash(title: str) -> str:
    return hashlib.md5(title.strip().lower().encode()).hexdigest()[:12]


def load_existing_hashes() -> set:
    seen = set()
    if not OUT_CSV.exists():
        return seen
    with open(OUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row.get("_hash") or _hash(row.get("title", ""))
            if h:
                seen.add(h)
    return seen


def fetch_google_news(query: str, after: date, before: date) -> list[dict]:
    url = (
        f"https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query + f' after:{after} before:{before}')}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.text)
        items = []
        for item in root.findall(".//item"):
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            if title and pubdate:
                items.append({"title": title, "link": link, "pubdate": pubdate})
        return items
    except Exception:
        return []


def parse_pubdate(pubdate: str):
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(pubdate)
    except Exception:
        return None


def fetch_btc_window(ts_ms: int):
    try:
        resp = requests.get(BINANCE, params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": ts_ms, "limit": 62,
        }, timeout=10)
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return None, None, None
        def close(idx):
            return float(data[idx][4]) if idx < len(data) else None
        return close(0), close(15), close(min(61, len(data) - 1))
    except Exception:
        return None, None, None


def week_windows(start: date, end: date):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        yield cur, nxt
        cur = nxt


def main(start_date: date, end_date: date, dry_run: bool = False):
    print("=" * 60)
    print("  NEWS GAP FETCHER — Google News RSS")
    print(f"  Range : {start_date} → {end_date}")
    print(f"  Output: {OUT_CSV.name}")
    print("=" * 60)

    existing = load_existing_hashes()
    print(f"\n  Existing rows   : {len(existing):,}")

    windows = list(week_windows(start_date, end_date))
    print(f"  Weekly windows  : {len(windows)}")
    print(f"  Queries/window  : {len(QUERIES)}")
    print(f"  Est. articles   : ~{len(windows) * len(QUERIES) * 40:,}")

    if dry_run:
        print("\n  [dry-run] Fetching sample window only...")
        sample = fetch_google_news(QUERIES[0], windows[0][0], windows[0][1])
        print(f"  Sample ({windows[0][0]} → {windows[0][1]}): {len(sample)} articles")
        for a in sample[:5]:
            print(f"    {a['pubdate'][:16]}  {a['title'][:65]}")
        return

    # Load checkpoint if exists
    ckpt_rows = []
    if CKPT_CSV.exists():
        with open(CKPT_CSV) as f:
            ckpt_rows = list(csv.DictReader(f))
        existing.update(r.get("_hash", "") for r in ckpt_rows)
        print(f"\n⚡ Resuming — checkpoint has {len(ckpt_rows)} rows")

    new_rows = list(ckpt_rows)
    total_new = 0
    skipped = 0

    for wi, (w_start, w_end) in enumerate(windows):
        print(f"\n  [{wi+1:>3}/{len(windows)}] {w_start} → {w_end}", end="", flush=True)
        window_new = 0

        for query in QUERIES:
            articles = fetch_google_news(query, w_start, w_end)
            time.sleep(1.5)

            for art in articles:
                h = _hash(art["title"])
                if h in existing:
                    skipped += 1
                    continue

                dt = parse_pubdate(art["pubdate"])
                if dt is None:
                    continue

                ts_ms = int(dt.timestamp() * 1000)
                p0, p15, p1h = fetch_btc_window(ts_ms)
                time.sleep(0.25)

                if p0 is None:
                    continue

                pct15 = round((p15 - p0) / p0 * 100, 6) if p15 else None
                pct1h = round((p1h - p0) / p0 * 100, 6) if p1h else None
                pub_str = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

                row = {k: "" for k in FIELDNAMES}
                row.update({
                    "title":              art["title"],
                    "link":               art["link"],
                    "channel":            "google_news",
                    "published":          pub_str,
                    "btc_price_at_news":  round(p0, 2),
                    "btc_price_15m":      round(p15, 2) if p15 else "",
                    "btc_price_1h":       round(p1h, 2) if p1h else "",
                    "weight":             "5",
                    "is_weekend":         "1" if dt.weekday() >= 5 else "0",
                    "hour_utc":           str(dt.hour),
                    "day_of_week":        dt.strftime("%A"),
                    "hour_of_day":        str(dt.hour),
                    "word_count":         str(len(art["title"].split())),
                    "btc_pct_change_15m": pct15 if pct15 is not None else "",
                    "btc_pct_change_1h":  pct1h if pct1h is not None else "",
                    "_hash":              h,
                })

                new_rows.append(row)
                existing.add(h)
                total_new += 1
                window_new += 1

        print(f"  +{window_new}", end="", flush=True)

        # Checkpoint every 200 new rows
        if total_new > 0 and total_new % 200 < len(QUERIES) * 10:
            with open(CKPT_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(new_rows)

    # Append to main CSV
    print(f"\n\n  Appending {total_new:,} new rows to {OUT_CSV.name}...")
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(r for r in new_rows if r not in ckpt_rows)

    if CKPT_CSV.exists():
        CKPT_CSV.unlink()

    print(f"\n{'='*60}")
    print(f"  ✅ Done")
    print(f"  New rows added  : {total_new:,}")
    print(f"  Skipped (dupe)  : {skipped:,}")
    print(f"  Output          : {OUT_CSV.name}")
    print(f"{'='*60}")
    print(f"\n  Next: run sentiment scoring on the new rows:")
    print(f"  python services/sentiment_score.py news_cleaned_filtered_scored.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date",
                        default="2024-10-01",
                        help="Start date YYYY-MM-DD (default: 2024-10-01)")
    parser.add_argument("--to", dest="to_date",
                        default=str(date.today()),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(
        start_date = date.fromisoformat(args.from_date),
        end_date   = date.fromisoformat(args.to_date),
        dry_run    = args.dry_run,
    )
