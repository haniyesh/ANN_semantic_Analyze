import ssl
import asyncpg
from datetime import datetime, timezone
from config import DATABASE_URL


# ==============================
# 🔌 DATABASE CONNECTION
# ==============================
async def create_pool():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        ssl=ssl_context,
        min_size=1,
        max_size=10,
        timeout=8.0,               # fail fast if DB unreachable
        command_timeout=30.0,
        max_inactive_connection_lifetime=300.0,
    )
    return pool


# ==============================
# 🏗️ CREATE ALL TABLES
# ==============================
async def create_tables(pool):
    async with pool.acquire() as conn:

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS news_full (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                link TEXT UNIQUE,
                source VARCHAR(50),
                coin VARCHAR(10),
                category VARCHAR(50),
                signal VARCHAR(20),
                impact_score REAL,
                published_at TIMESTAMPTZ,
                processed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_news (
                id SERIAL PRIMARY KEY,
                link TEXT UNIQUE NOT NULL,
                processed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_movements (
                id SERIAL PRIMARY KEY,
                news_id INTEGER REFERENCES news_full(id),
                symbol VARCHAR(10),
                price_at_news REAL,
                price_15m REAL,
                price_1h REAL,
                price_4h REAL,
                price_24h REAL,
                movement_15m REAL,
                movement_1h REAL,
                movement_4h REAL,
                movement_24h REAL,
                direction_15m VARCHAR(10),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)


# ==============================
# 💾 SAVE NEWS
# ==============================
async def save_news(pool, title: str, link: str, source: str,
                    coin: str, category: str, signal: str,
                    impact_score: float, published_at: datetime) -> int:

    if published_at and published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    async with pool.acquire() as conn:
        row_id = await conn.fetchval("""
            INSERT INTO news_full
            (title, link, source, coin, category, signal, impact_score, published_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (link) DO UPDATE SET
                signal = EXCLUDED.signal,
                impact_score = EXCLUDED.impact_score
            RETURNING id
        """, title, link, source, coin, category, signal, impact_score, published_at)
    return row_id


# ==============================
# ✅ CHECK IF NEWS IS PROCESSED
# ==============================
async def is_processed(pool, link: str) -> bool:
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT 1 FROM processed_news WHERE link=$1", link
        )
    return result is not None


# ==============================
# 💾 MARK NEWS AS PROCESSED
# ==============================
async def mark_processed(pool, link: str):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO processed_news (link)
            VALUES ($1)
            ON CONFLICT (link) DO NOTHING
        """, link)


# ==============================
# 💰 SAVE PRICE MOVEMENT
# ==============================
async def save_price_movement(pool, news_id: int, symbol: str,
                               price_at_news: float = None,
                               price_15m: float = None,
                               price_1h: float = None,
                               price_4h: float = None,
                               price_24h: float = None,
                               movement_15m: float = None,
                               movement_1h: float = None,
                               movement_4h: float = None,
                               movement_24h: float = None,
                               direction: str = None):

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO price_movements
            (news_id, symbol, price_at_news, price_15m, price_1h, price_4h, price_24h,
             movement_15m, movement_1h, movement_4h, movement_24h, direction_15m)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        """, news_id, symbol, price_at_news, price_15m, price_1h,
            price_4h, price_24h, movement_15m, movement_1h,
            movement_4h, movement_24h, direction)


# ==============================
# 📰 GET NEWS
# ==============================
async def get_news(pool, date: str = None, limit: int = 100):
    async with pool.acquire() as conn:
        if date:
            return await conn.fetch("""
                SELECT n.*, pm.movement_15m, pm.direction_15m, pm.price_at_news
                FROM news_full n
                LEFT JOIN price_movements pm ON n.id = pm.news_id
                WHERE DATE(n.published_at) = $1
                ORDER BY n.published_at DESC
                LIMIT $2
            """, date, limit)
        else:
            return await conn.fetch("""
                SELECT n.*, pm.movement_15m, pm.direction_15m, pm.price_at_news
                FROM news_full n
                LEFT JOIN price_movements pm ON n.id = pm.news_id
                ORDER BY n.published_at DESC
                LIMIT $1
            """, limit)


# ==============================
# 📊 GET STATS
# ==============================
async def get_stats(pool):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*) AS total_news,
                COUNT(*) FILTER (WHERE n.coin = 'BTC') AS btc_count,
                COUNT(*) FILTER (WHERE n.coin = 'ETH') AS eth_count,
                COUNT(*) FILTER (WHERE pm.direction_15m = 'BULLISH') AS bullish_count,
                COUNT(*) FILTER (WHERE pm.direction_15m = 'BEARISH') AS bearish_count,
                AVG(ABS(pm.movement_15m)) AS avg_movement
            FROM news_full n
            LEFT JOIN price_movements pm ON n.id = pm.news_id
            WHERE n.coin IN ('BTC', 'ETH')
        """)
    return dict(row)