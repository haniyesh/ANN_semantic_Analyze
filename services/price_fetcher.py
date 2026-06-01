import asyncio
import requests
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from config import BINANCE_API, COINGECKO_API

BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT",
    
}

COIN_IDS = {
    "BTC": "bitcoin",   "ETH": "ethereum",
   
}


# ==============================
# 🪙 COIN DETECTION
# ==============================
def extract_coin_from_text(text: str) -> Optional[str]:
    """Detect which coin the news is about."""
    t = text.lower()

    BTC_KEYWORDS = ["bitcoin", " btc ", "btc's", "bitcoins", "#bitcoin", "$btc", "btcusd", "btc/usd"]
    ETH_KEYWORDS = ["ethereum", " eth ", "eth's", "#ethereum", "$eth", "ethusd", "eth/usd", "ether"]

    btc_score = sum(1 for kw in BTC_KEYWORDS if kw in t)
    eth_score = sum(1 for kw in ETH_KEYWORDS if kw in t)

    if btc_score > 0 and btc_score >= eth_score:
        return "BTC"
    elif eth_score > 0:
        return "ETH"
    elif any(w in t for w in ["crypto", "cryptocurrency"]):
        return "BTC"
    return None


# ==============================
# 💰 CURRENT PRICE
# ==============================
def get_current_price_sync(symbol: str) -> Optional[float]:
    """Get live price from Binance — no rate limits."""
    binance_symbol = BINANCE_SYMBOLS.get(symbol.upper())
    if not binance_symbol:
        return None
    try:
        resp = requests.get(
            f"{BINANCE_API}/ticker/price",
            params={"symbol": binance_symbol},
            timeout=10,
        )
        if resp.status_code == 200:
            return float(resp.json()["price"])
    except Exception as e:
        print(f"[PRICE ERROR] {symbol}: {e}")
    return None


async def get_current_price(symbol: str) -> Optional[float]:
    return await asyncio.to_thread(get_current_price_sync, symbol)


# ==============================
# 📅 HISTORICAL PRICE
# ==============================
def get_historical_price_sync(symbol: str, timestamp: datetime) -> Optional[float]:
    """Get price at a specific past time from Binance 1m candles."""
    binance_symbol = BINANCE_SYMBOLS.get(symbol.upper())
    if not binance_symbol:
        return None
    ts_ms = int(timestamp.replace(tzinfo=timezone.utc).timestamp() * 1000)
    try:
        resp = requests.get(
            f"{BINANCE_API}/klines",
            params={
                "symbol":    binance_symbol,
                "interval":  "1m",
                "startTime": ts_ms - 60000,
                "endTime":   ts_ms + 60000,
                "limit":     5,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return float(data[0][4])  # close price
    except Exception as e:
        print(f"[HISTORICAL PRICE ERROR] {symbol}: {e}")
    return None


async def get_historical_price(symbol: str, timestamp: datetime) -> Optional[float]:
    return await asyncio.to_thread(get_historical_price_sync, symbol, timestamp)


# ==============================
# 📊 OHLC DATA
# ==============================
def get_ohlc_sync(symbol: str, interval: str = "15m", limit: int = 100) -> List[Dict]:
    """Get candlestick data from Binance."""
    binance_symbol = BINANCE_SYMBOLS.get(symbol.upper())
    if not binance_symbol:
        return []
    try:
        resp = requests.get(
            f"{BINANCE_API}/klines",
            params={"symbol": binance_symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if resp.status_code == 200:
            return [
                {
                    "timestamp": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                    "open":      float(c[1]),
                    "high":      float(c[2]),
                    "low":       float(c[3]),
                    "close":     float(c[4]),
                    "volume":    float(c[5]),
                }
                for c in resp.json()
            ]
    except Exception as e:
        print(f"[OHLC ERROR] {symbol}: {e}")
    return []


async def get_ohlc(symbol: str, interval: str = "15m", limit: int = 100) -> List[Dict]:
    return await asyncio.to_thread(get_ohlc_sync, symbol, interval, limit)


# ==============================
# 📈 PRICE AT MULTIPLE TIMES
# ==============================
async def get_price_at_times(symbol: str, news_time: datetime,
                              intervals: List[int] = [15, 60, 240]) -> Dict[int, Optional[float]]:
    """Get price at news time and at multiple intervals after (in minutes)."""
    results = {}
    results[0] = await get_historical_price(symbol, news_time)
    for minutes in intervals:
        target = news_time + timedelta(minutes=minutes)
        results[minutes] = await get_historical_price(symbol, target)
    return results


# ==============================
# 📉 CALCULATE MOVEMENT
# ==============================
def calculate_movement(price_at_news: float, price_after: float) -> Dict:
    """
    Calculate % change and direction.
    Thresholds match what the model was trained on:
      >= 1.0% → BULLISH
      <= -1.0% → BEARISH
      >= 0.3% → BULLISH
      <= -0.3% → BEARISH
      else → NEUTRAL
    """
    if not price_at_news or not price_after:
        return {}
    change = ((price_after - price_at_news) / price_at_news) * 100
    if change >= 1.0:
        direction = "BULLISH"
    elif change <= -1.0:
        direction = "BEARISH"
    elif change >= 0.3:
        direction = "BULLISH"
    elif change <= -0.3:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    return {
        "change_percent": round(change, 2),
        "direction":      direction,
    }


# ==============================
# 🔄 PRICE TRACKER
# ==============================
class PriceTracker:
    """
    Tracks pending news items and fetches their price movements
    after 15 minutes.
    """
    def __init__(self):
        self.pending = []

    async def add_pending(self, news_id: int, symbol: str, news_time: datetime):
        self.pending.append({
            "news_id":   news_id,
            "symbol":    symbol,
            "news_time": news_time,
            "added_at":  datetime.now(timezone.utc),
        })

    async def process_pending(self, callback=None):
        """Process items that have been waiting >= 15 minutes."""
        now        = datetime.now(timezone.utc)
        to_process = [
            t for t in self.pending
            if (now - t["added_at"]).total_seconds() >= 15 * 60
        ]
        for track in to_process:
            self.pending.remove(track)
            try:
                prices = await get_price_at_times(
                    track["symbol"],
                    track["news_time"],
                    intervals=[15, 60, 240],
                )
                result = {
                    "news_id":       track["news_id"],
                    "symbol":        track["symbol"],
                    "price_at_news": prices.get(0),
                    "price_15m":     prices.get(15),
                    "price_1h":      prices.get(60),
                    "price_4h":      prices.get(240),
                    "movement_15m":  calculate_movement(prices.get(0), prices.get(15)),
                    "movement_1h":   calculate_movement(prices.get(0), prices.get(60)),
                }
                if callback:
                    await callback(result)
            except Exception as e:
                print(f"[PRICE TRACKER ERROR] news_id={track['news_id']}: {e}")