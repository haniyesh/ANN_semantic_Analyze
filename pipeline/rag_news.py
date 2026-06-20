"""
rag_news.py
===========
Qdrant Cloud RAG with macro-conditioned re-weighting.

Performance: uses ThreadPoolExecutor (16 threads) for parallel Qdrant queries.
  Sequential: ~4-8 hours for 14k rows
  Parallel  : ~20-30 minutes for 14k rows

Resilience:
  - Retry logic: each query retries 3x with exponential backoff
  - Checkpoint every 500 completed rows (resume from crash)
  - Timeout 120s per client

Macro features used in re-weighting (updated from 2 → 5):
  is_weekend       : weekend liquidity effect
  is_low_liquidity : thin market hours (02-06 UTC)
  is_us_hours      : US session active (13-21 UTC)
  is_asia_hours    : Asia session (00-08 UTC)
  fomc_week        : Fed meeting week (kept for backward compat, defaults 0)

.env: QDRANT_URL=... QDRANT_API_KEY=...
pip install qdrant-client fastembed
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Range, PayloadSchemaType,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
COLLECTION_NAME      = "crypto_news"
VECTOR_SIZE          = 384
EMBED_MODEL          = "BAAI/bge-small-en-v1.5"
TOP_K                = 10
BATCH_SIZE           = 50
IMPACT_THRESHOLD_15M = 0.5
IMPACT_THRESHOLD_1H  = 0.5
SIMILARITY_THRESHOLD = 0.72
NUM_THREADS          = 16
CHECKPOINT_EVERY     = 500

# Macro re-weighting boost factors
# How much each matching context boosts a retrieved result
WEEKEND_BOOST    = 0.2   # weekend liquidity effect
LOW_LIQ_BOOST    = 0.3   # thin market hours amplify moves most
US_HOURS_BOOST   = 0.2   # US session — fastest price reaction
ASIA_HOURS_BOOST = 0.1   # Asia session — different market behavior
FOMC_BOOST       = 0.4   # Fed meeting week (kept, defaults to 0 if missing)

HERE             = Path(__file__).parent
RAG_FEAT_CACHE   = HERE / "rag_features_qdrant.pkl"
RAG_DETAIL_CACHE = HERE / "rag_details_qdrant.pkl"
CHECKPOINT_PATH  = HERE / "rag_checkpoint.pkl"

_embedder: TextEmbedding | None = None
_embedder_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════
# CLIENT + EMBEDDER
# ══════════════════════════════════════════════════════════════════
def get_embedder() -> TextEmbedding:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            print(f"  Loading embedding model: {EMBED_MODEL}...")
            _embedder = TextEmbedding(EMBED_MODEL)
        return _embedder


def get_client() -> QdrantClient:
    """Fresh Qdrant client — each thread gets its own (thread-safe)."""
    url     = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    return QdrantClient(url=url, api_key=api_key, timeout=120)


# ══════════════════════════════════════════════════════════════════
# RETRY WRAPPER
# ══════════════════════════════════════════════════════════════════
def _query_with_retry(
    client: QdrantClient,
    query: list,
    query_filter: Filter,
    limit: int,
    max_retries: int = 3,
) -> list:
    """Query Qdrant with exponential backoff. Returns [] on total failure."""
    for attempt in range(max_retries):
        try:
            return client.query_points(
                collection_name=COLLECTION_NAME,
                query=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ).points
        except Exception:
            if attempt == max_retries - 1:
                return []
            time.sleep(2 ** attempt)   # 1s → 2s → 4s
    return []


# ══════════════════════════════════════════════════════════════════
# MACRO-CONDITIONED RE-WEIGHTING
# ══════════════════════════════════════════════════════════════════
def macro_reweight_rag(
    results: list,
    macro_now: dict,
    channel_impact_rates: dict,
) -> tuple[np.ndarray, list[float]]:
    """
    Re-weight retrieved results by macro context + channel impact rate.

    Args:
        results              : Qdrant search results (list of ScoredPoint)
        macro_now            : dict of current macro context:
                               {
                                 "is_weekend":       float,
                                 "is_low_liquidity": float,
                                 "is_us_hours":      float,
                                 "is_asia_hours":    float,
                                 "fomc_week":        float,  (optional, defaults 0)
                               }
        channel_impact_rates : dict of channel → historical impact rate

    Weight formula:
      base_weight  = cosine_similarity (from Qdrant)
      macro_boost  = 1 + sum(boost × match for each macro feature)
      channel_mult = channel_rate / mean_rate  (normalizes around 1.0)
      final_weight = base × macro_boost × channel_mult  (then sum-normalized)

    Returns:
        features : (10,) float32 — macro-reweighted RAG feature vector
        weights  : list of final weight per result (for explanation)
    """
    if not results:
        return np.zeros(10, dtype=np.float32), []

    mean_rate = (
        np.mean(list(channel_impact_rates.values()))
        if channel_impact_rates else 0.163
    )

    btc_15m = np.array([r.payload.get("btc_change_15m",   0.0) for r in results])
    btc_1h  = np.array([r.payload.get("btc_change_1h",    0.0) for r in results])
    imp_15m = np.array([r.payload.get("is_impactful_15m",  0)   for r in results])
    imp_1h  = np.array([r.payload.get("is_impactful_1h",   0)   for r in results])
    sims    = np.array([r.score                                   for r in results])

    # Extract current macro context
    weekend_now    = float(macro_now.get("is_weekend",       0.0))
    low_liq_now    = float(macro_now.get("is_low_liquidity", 0.0))
    us_now         = float(macro_now.get("is_us_hours",      0.0))
    asia_now       = float(macro_now.get("is_asia_hours",    0.0))
    fomc_now       = float(macro_now.get("fomc_week",        0.0))

    weights = np.ones(len(results), dtype=np.float32)

    for i, r in enumerate(results):
        base = float(r.score)

        # Match score: 1.0 = contexts match, 0.0 = contexts differ
        weekend_match = 1.0 - abs(weekend_now - float(r.payload.get("is_weekend",       0.0)))
        low_liq_match = 1.0 - abs(low_liq_now - float(r.payload.get("is_low_liquidity", 0.0)))
        us_match      = 1.0 - abs(us_now       - float(r.payload.get("is_us_hours",      0.0)))
        asia_match    = 1.0 - abs(asia_now     - float(r.payload.get("is_asia_hours",    0.0)))
        fomc_match    = 1.0 - abs(fomc_now     - float(r.payload.get("fomc_week",        0.0)))

        macro_boost = (
            1.0
            + WEEKEND_BOOST    * weekend_match   # 0.2
            + LOW_LIQ_BOOST    * low_liq_match   # 0.3 — highest: thin market = big moves
            + US_HOURS_BOOST   * us_match         # 0.2
            + ASIA_HOURS_BOOST * asia_match       # 0.1
            + FOMC_BOOST       * fomc_match       # 0.4 — highest single: Fed meeting
        )

        channel      = r.payload.get("channel", "")
        ch_rate      = channel_impact_rates.get(channel, mean_rate)
        channel_mult = ch_rate / (mean_rate + 1e-8)

        weights[i] = base * macro_boost * channel_mult

    weights = weights / (weights.sum() + 1e-8)

    features = np.array([
        float(np.sum(weights * btc_15m)),   # weighted avg BTC 15m change
        float(np.max(btc_15m)),             # max BTC 15m change
        float(np.std(btc_15m)),             # std of BTC 15m changes
        float(np.sum(weights * imp_15m)),   # weighted hit rate 15m
        float(np.sum(weights * btc_1h)),    # weighted avg BTC 1h change
        float(np.max(btc_1h)),              # max BTC 1h change
        float(np.std(btc_1h)),              # std of BTC 1h changes
        float(np.sum(weights * imp_1h)),    # weighted hit rate 1h
        float(np.mean(sims)),               # avg cosine similarity
        float(np.sum(imp_15m)),             # raw count of impactful retrieved
    ], dtype=np.float32)

    return features, weights.tolist()


def _build_macro_now(row: pd.Series) -> dict:
    """Build macro_now dict from a DataFrame row."""
    hour = pd.Timestamp(row["published"]).hour
    dow  = pd.Timestamp(row["published"]).dayofweek
    return {
        "is_weekend":       float(dow >= 5),
        "is_low_liquidity": float(2 <= hour <= 6),
        "is_us_hours":      float(13 <= hour <= 21),
        "is_asia_hours":    float(0 <= hour <= 8),
        "fomc_week":        float(row.get("fomc_week", 0.0)),
    }


def _build_macro_now_from_ts(timestamp: int) -> dict:
    """Build macro_now dict from a Unix timestamp."""
    dt   = pd.Timestamp(timestamp, unit="s", tz="UTC")
    hour = dt.hour
    dow  = dt.dayofweek
    return {
        "is_weekend":       float(dow >= 5),
        "is_low_liquidity": float(2 <= hour <= 6),
        "is_us_hours":      float(13 <= hour <= 21),
        "is_asia_hours":    float(0 <= hour <= 8),
        "fomc_week":        0.0,   # not known at inference time without compute_macro
    }


# ══════════════════════════════════════════════════════════════════
# COLLECTION SETUP
# ══════════════════════════════════════════════════════════════════
def setup_collection(client: QdrantClient) -> bool:
    """Creates collection if missing. Returns True if already existed."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        count = client.get_collection(COLLECTION_NAME).points_count
        print(f"  Qdrant '{COLLECTION_NAME}' exists — {count:,} vectors")
        return True

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    for field, schema in [
        ("timestamp",        PayloadSchemaType.INTEGER),
        ("channel",          PayloadSchemaType.KEYWORD),
        ("is_weekend",       PayloadSchemaType.FLOAT),
        ("is_low_liquidity", PayloadSchemaType.FLOAT),
        ("is_us_hours",      PayloadSchemaType.FLOAT),
        ("is_asia_hours",    PayloadSchemaType.FLOAT),
        ("fomc_week",        PayloadSchemaType.FLOAT),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field, field_schema=schema,
        )
    print(f"  Created Qdrant collection '{COLLECTION_NAME}' (384 dims, cosine)")
    return False


def fix_missing_indexes(client: QdrantClient):
    """Add missing payload indexes to existing collection. Run once if needed."""
    for field, schema in [
        ("timestamp",        PayloadSchemaType.INTEGER),
        ("channel",          PayloadSchemaType.KEYWORD),
        ("is_weekend",       PayloadSchemaType.FLOAT),
        ("is_low_liquidity", PayloadSchemaType.FLOAT),
        ("is_us_hours",      PayloadSchemaType.FLOAT),
        ("is_asia_hours",    PayloadSchemaType.FLOAT),
        ("fomc_week",        PayloadSchemaType.FLOAT),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field, field_schema=schema,
            )
            print(f"  ✅ Created index: {field}")
        except Exception as e:
            print(f"  ⚠️  {field}: {e}")


def get_existing_ids(client: QdrantClient) -> set:
    existing, offset = set(), None
    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000, offset=offset,
            with_payload=False, with_vectors=False,
        )
        for point in result:
            existing.add(point.id)
        if next_offset is None:
            break
        offset = next_offset
    return existing


# ══════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════
def upload_vectors(df: pd.DataFrame, client: QdrantClient, existing_ids: set):
    """
    Embed and upload only rows not already in Qdrant. Idempotent.
    df must have been through compute_macro.py (has macro feature columns).
    """
    new_df = df[~df.index.isin(existing_ids)].copy()
    if len(new_df) == 0:
        print(f"  All {len(df):,} rows already in Qdrant.")
        return

    print(f"  Uploading {len(new_df):,} new vectors...")
    embedder   = get_embedder()
    titles     = new_df["title"].fillna("").tolist()
    embeddings = list(embedder.embed(titles))

    def sf(val):
        try:
            f = float(val)
            return f if f == f else 0.0
        except Exception:
            return 0.0

    for start in range(0, len(titles), BATCH_SIZE):
        end   = min(start + BATCH_SIZE, len(titles))
        batch = new_df.iloc[start:end]
        points = []

        for j, (idx, row) in enumerate(batch.iterrows()):
            _pub = pd.Timestamp(row["published"])
            ts   = 0 if pd.isna(_pub) else int(_pub.timestamp())
            b15m = sf(row.get("btc_change_15m", 0))
            b1h  = sf(row.get("btc_change_1h",  0))
            hour = 0 if pd.isna(_pub) else _pub.hour
            dow  = 0 if pd.isna(_pub) else _pub.dayofweek

            points.append(PointStruct(
                id     = int(idx),
                vector = embeddings[start + j].tolist(),
                payload = {
                    "timestamp":        ts,
                    "title":            titles[start + j][:200],
                    "channel":          str(row.get("channel", "")),
                    "published":        str(row.get("published", "")),
                    "link":             str(row.get("link", "")),
                    "btc_change_15m":   b15m,
                    "btc_change_1h":    b1h,
                    "is_impactful_15m": int(abs(b15m) >= IMPACT_THRESHOLD_15M),
                    "is_impactful_1h":  int(abs(b1h)  >= IMPACT_THRESHOLD_1H),
                    "is_impactful":     int(
                        abs(b15m) >= IMPACT_THRESHOLD_15M or
                        abs(b1h)  >= IMPACT_THRESHOLD_1H
                    ),
                    # Macro context — computed from timestamp, no external data needed
                    "is_weekend":       float(dow >= 5),
                    "is_low_liquidity": float(2 <= hour <= 6),
                    "is_us_hours":      float(13 <= hour <= 21),
                    "is_asia_hours":    float(0 <= hour <= 8),
                    "fomc_week":        sf(row.get("fomc_week", 0.0)),
                    "btc_price_at":     sf(row.get("btc_price_at_news", 0)),
                },
            ))

        for attempt in range(5):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 10 * (attempt + 1)
                print(f"    ⚠ upsert failed ({e.__class__.__name__}), retry {attempt+1}/5 in {wait}s...")
                import time; time.sleep(wait)
        print(f"    {end}/{len(titles)} uploaded...")

    print(f"  Done. Total in Qdrant: {len(existing_ids) + len(new_df):,}")


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════
def _save_checkpoint(rag_features: np.ndarray, rag_details: list, completed: set):
    pd.to_pickle({
        "rag_features": rag_features.copy(),
        "rag_details":  rag_details,
        "completed":    completed,
    }, CHECKPOINT_PATH)


def _load_checkpoint(n: int) -> tuple[np.ndarray, list, set]:
    if not CHECKPOINT_PATH.exists():
        return np.zeros((n, 10), dtype=np.float32), [None] * n, set()

    ckpt        = pd.read_pickle(CHECKPOINT_PATH)
    saved_feats = ckpt.get("rag_features", np.zeros((n, 10), dtype=np.float32))
    saved_dets  = ckpt.get("rag_details",  [None] * n)
    completed   = ckpt.get("completed", set())

    if saved_feats.shape[0] != n:
        print(f"  ⚠️  Checkpoint size mismatch — starting fresh")
        return np.zeros((n, 10), dtype=np.float32), [None] * n, set()

    print(f"  Resuming from checkpoint: {len(completed):,}/{n:,} rows done")
    return saved_feats, saved_dets, completed


# ══════════════════════════════════════════════════════════════════
# PARALLEL WORKER
# ══════════════════════════════════════════════════════════════════
def _process_one_row(
    i: int,
    embedding: list,
    timestamp: int,
    macro_now: dict,
    channel_impact_rates: dict,
    top_k: int,
) -> tuple[int, np.ndarray, dict]:
    """
    Query Qdrant for one row, apply macro re-weighting.
    Each call gets its own client — fully thread-safe.
    """
    client  = get_client()
    results = _query_with_retry(
        client,
        query=embedding,
        query_filter=Filter(must=[FieldCondition(
            key="timestamp",
            range=Range(lt=timestamp),
        )]),
        limit=top_k,
    )

    if not results:
        return i, np.zeros(10, dtype=np.float32), {"similar_news": [], "macro_weights": []}

    features, weights = macro_reweight_rag(results, macro_now, channel_impact_rates)

    detail = {
        "similar_news": [
            {
                "title":            r.payload.get("title", ""),
                "channel":          r.payload.get("channel", ""),
                "published":        r.payload.get("published", ""),
                "link":             r.payload.get("link", ""),
                "btc_change_15m":   r.payload.get("btc_change_15m", 0.0),
                "btc_change_1h":    r.payload.get("btc_change_1h",  0.0),
                "is_impactful_15m": r.payload.get("is_impactful_15m", 0),
                "similarity_score": float(r.score),
                "macro_weight":     float(weights[k]) if k < len(weights) else 0.0,
                "weekend_match":    float(r.payload.get("is_weekend",       0.0)) == macro_now.get("is_weekend", 0.0),
                "low_liq_match":    float(r.payload.get("is_low_liquidity", 0.0)) == macro_now.get("is_low_liquidity", 0.0),
                "us_match":         float(r.payload.get("is_us_hours",      0.0)) == macro_now.get("is_us_hours", 0.0),
            }
            for k, r in enumerate(results)
        ],
        "macro_weights": weights,
    }

    return i, features, detail


# ══════════════════════════════════════════════════════════════════
# BUILD RAG FEATURES FOR TRAINING
# ══════════════════════════════════════════════════════════════════
def build_rag_features_qdrant(
    df: pd.DataFrame,
    channel_impact_rates: dict,
    top_k: int = TOP_K,
    rebuild: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Builds 10-dim macro-reweighted RAG feature array for every row in df.
    Uses 5 macro features: is_weekend, is_low_liquidity, is_us_hours,
    is_asia_hours, fomc_week (from compute_macro.py or computed on-the-fly).

    Time-safe: row i only retrieves news with timestamp < row i's timestamp.
    Parallel: 16 threads, ~20-30 min for 14k rows.
    Checkpoint: saves every 500 rows, auto-resumes on crash.
    """
    if not rebuild and RAG_FEAT_CACHE.exists() and RAG_DETAIL_CACHE.exists():
        feats   = pd.read_pickle(RAG_FEAT_CACHE).values.astype(np.float32)
        details = pd.read_pickle(RAG_DETAIL_CACHE)
        if len(details) == len(df):
            print(f"  Qdrant RAG cache hit: {RAG_FEAT_CACHE.name}")
            return feats, details

    client = get_client()

    already_exists = setup_collection(client)
    existing_ids   = get_existing_ids(client) if already_exists else set()
    upload_vectors(df, client, existing_ids)

    n          = len(df)
    def _safe_ts(val):
        try:
            t = pd.Timestamp(val)
            return 0 if pd.isna(t) else int(t.timestamp())
        except Exception:
            return 0
    timestamps = [_safe_ts(df.iloc[i]["published"]) for i in range(n)]

    # Build macro_now for each row
    macro_contexts = []
    for i in range(n):
        row  = df.iloc[i]
        hour = pd.Timestamp(row["published"]).hour
        dow  = pd.Timestamp(row["published"]).dayofweek
        macro_contexts.append({
            "is_weekend":       float(dow >= 5),
            "is_low_liquidity": float(2 <= hour <= 6),
            "is_us_hours":      float(13 <= hour <= 21),
            "is_asia_hours":    float(0 <= hour <= 8),
            "fomc_week":        float(row.get("fomc_week", 0.0))
                                if "fomc_week" in df.columns else 0.0,
        })

    print(f"  Embedding {n:,} titles for RAG queries...")
    embedder       = get_embedder()
    titles         = df["title"].fillna("").tolist()
    all_embeddings = list(embedder.embed(titles))

    # Load checkpoint
    if rebuild:
        rag_features  = np.zeros((n, 10), dtype=np.float32)
        rag_details   = [{"similar_news": [], "macro_weights": []}] * n
        completed_set = set()
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
    else:
        rag_features, rag_details, completed_set = _load_checkpoint(n)
        rag_details = [
            d if d is not None else {"similar_news": [], "macro_weights": []}
            for d in rag_details
        ]

    todo = [i for i in range(n) if i >= top_k and i not in completed_set]
    print(f"  Computing RAG features: {len(todo):,} rows ({NUM_THREADS} threads)...")

    lock       = threading.Lock()
    done_count = [0]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(
                _process_one_row,
                i,
                all_embeddings[i].tolist(),
                timestamps[i],
                macro_contexts[i],
                channel_impact_rates,
                top_k,
            ): i
            for i in todo
        }

        for future in as_completed(futures):
            row_i, features, detail = future.result()
            with lock:
                rag_features[row_i] = features
                rag_details[row_i]  = detail
                completed_set.add(row_i)
                done_count[0] += 1

                if done_count[0] % 500 == 0 or done_count[0] == len(todo):
                    pct = done_count[0] * 100 // len(todo)
                    print(f"    {done_count[0]:,}/{len(todo):,} ({pct}%) completed")

                if done_count[0] % CHECKPOINT_EVERY == 0:
                    _save_checkpoint(rag_features, rag_details, completed_set)

    pd.DataFrame(rag_features).to_pickle(RAG_FEAT_CACHE)
    pd.to_pickle(rag_details, RAG_DETAIL_CACHE)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"  RAG done: {rag_features.shape}  →  {RAG_FEAT_CACHE.name}")
    return rag_features, rag_details


# ══════════════════════════════════════════════════════════════════
# LIVE BOT INFERENCE
# ══════════════════════════════════════════════════════════════════
def query_single(
    title: str,
    before_timestamp: int,
    channel_impact_rates: dict,
    macro_now: dict | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    Real-time macro-reweighted RAG query for the live Telegram bot.

    Args:
        title                : headline text
        before_timestamp     : Unix timestamp — only retrieve news before this
        channel_impact_rates : dict of channel → impact rate
        macro_now            : macro context dict. If None, computed from timestamp.
        top_k                : number of similar past news to retrieve
    """
    if macro_now is None:
        macro_now = _build_macro_now_from_ts(before_timestamp)

    client   = get_client()
    embedder = get_embedder()
    emb      = list(embedder.embed([title]))[0].tolist()

    results = _query_with_retry(
        client,
        query=emb,
        query_filter=Filter(must=[FieldCondition(
            key="timestamp", range=Range(lt=before_timestamp),
        )]),
        limit=top_k,
    )
    results = [r for r in results if r.score >= SIMILARITY_THRESHOLD]

    if not results:
        return {
            "features":      np.zeros(10, dtype=np.float32),
            "similar_news":  [], "hit_rate": 0.0,
            "direction":     "N/A", "macro_weights": [],
        }

    features, weights = macro_reweight_rag(results, macro_now, channel_impact_rates)

    imp_15m  = np.array([r.payload.get("is_impactful_15m", 0)  for r in results])
    btc_1h   = np.array([r.payload.get("btc_change_1h",  0.0)  for r in results])
    hit_rate = float(np.mean(imp_15m))
    imp_vals = btc_1h[imp_15m == 1]
    avg_1h   = float(np.mean(imp_vals)) if len(imp_vals) > 0 else 0.0
    direction = "PUMP 📈" if avg_1h > 0 else ("DUMP 📉" if avg_1h < 0 else "N/A")

    return {
        "features": features,
        "similar_news": [
            {
                "title":            r.payload.get("title", ""),
                "channel":          r.payload.get("channel", ""),
                "link":             r.payload.get("link", ""),
                "btc_change_15m":   r.payload.get("btc_change_15m", 0.0),
                "btc_change_1h":    r.payload.get("btc_change_1h",  0.0),
                "is_impactful_15m": r.payload.get("is_impactful_15m", 0),
                "similarity_score": float(r.score),
                "macro_weight":     float(weights[k]) if k < len(weights) else 0.0,
                "weekend_match":    float(r.payload.get("is_weekend",       0.0)) == macro_now.get("is_weekend", 0.0),
                "low_liq_match":    float(r.payload.get("is_low_liquidity", 0.0)) == macro_now.get("is_low_liquidity", 0.0),
                "us_match":         float(r.payload.get("is_us_hours",      0.0)) == macro_now.get("is_us_hours", 0.0),
            }
            for k, r in enumerate(results)
        ],
        "hit_rate":      hit_rate,
        "direction":     direction,
        "macro_weights": weights,
    }


def add_new_point(
    point_id: int,
    title: str,
    channel: str,
    published_ts: int,
    btc_change_15m: float,
    btc_change_1h: float,
    link: str = "",
    fomc_week: float = 0.0,
):
    """
    Add one news item after its real outcome is known.
    Macro features computed automatically from published_ts.
    """
    client   = get_client()
    embedder = get_embedder()
    emb      = list(embedder.embed([title]))[0].tolist()

    dt   = pd.Timestamp(published_ts, unit="s", tz="UTC")
    hour = dt.hour
    dow  = dt.dayofweek

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=point_id, vector=emb,
            payload={
                "timestamp":        published_ts,
                "title":            title[:200],
                "channel":          channel,
                "published":        str(dt),
                "link":             link,
                "btc_change_15m":   btc_change_15m,
                "btc_change_1h":    btc_change_1h,
                "is_impactful_15m": int(abs(btc_change_15m) >= IMPACT_THRESHOLD_15M),
                "is_impactful_1h":  int(abs(btc_change_1h)  >= IMPACT_THRESHOLD_1H),
                "is_impactful":     int(
                    abs(btc_change_15m) >= IMPACT_THRESHOLD_15M or
                    abs(btc_change_1h)  >= IMPACT_THRESHOLD_1H
                ),
                "is_weekend":       float(dow >= 5),
                "is_low_liquidity": float(2 <= hour <= 6),
                "is_us_hours":      float(13 <= hour <= 21),
                "is_asia_hours":    float(0 <= hour <= 8),
                "fomc_week":        fomc_week,
            },
        )],
    )


# ══════════════════════════════════════════════════════════════════
# check_news() — standalone similarity search
# ══════════════════════════════════════════════════════════════════
def check_news(title: str) -> bool:
    """Standalone similarity search. Does NOT apply macro re-weighting."""
    client   = get_client()
    embedder = get_embedder()
    emb      = list(embedder.embed([title]))[0].tolist()

    results = _query_with_retry(
        client, query=emb,
        query_filter=Filter(must=[FieldCondition(
            key="timestamp", range=Range(lt=int(time.time())),
        )]),
        limit=TOP_K,
    )
    results = [r for r in results if r.score >= SIMILARITY_THRESHOLD]

    print(f"\n{'='*65}\n  NEWS: {title}\n{'='*65}")

    if not results:
        print(f"  No similar news above similarity {SIMILARITY_THRESHOLD}")
        return False

    impactful = [r for r in results if r.payload.get("is_impactful") == 1]
    hit_rate  = len(impactful) / len(results)
    avg_1h    = sum(r.payload.get("btc_change_1h",  0) for r in impactful) / max(len(impactful), 1)
    avg_15m   = sum(r.payload.get("btc_change_15m", 0) for r in impactful) / max(len(impactful), 1)
    direction = "PUMP 📈" if avg_1h > 0 else "DUMP 📉"
    confidence = ("🚨 HIGH" if hit_rate >= 0.6 else
                  "⚠️  MODERATE" if hit_rate >= 0.35 else "ℹ️  LOW")

    print(f"  Similar: {len(results)}  |  Impactful: {len(impactful)}  ({hit_rate*100:.0f}%)")
    print(f"  Confidence: {confidence}  |  Direction: {direction if impactful else 'N/A'}")
    if impactful:
        print(f"  BTC avg 1h: {avg_1h:+.2f}%  |  BTC avg 15m: {avg_15m:+.2f}%")

    print(f"\n  Top 3:")
    for i, r in enumerate(results[:3]):
        p   = r.payload
        tag = "✅" if p.get("is_impactful") == 1 else "➖"
        print(f"  [{i+1}] {r.score:.3f} {tag} "
              f"BTC 1h:{p.get('btc_change_1h',0):+.2f}% | "
              f"{p.get('title','')[:80]}")
        print(f"       {p.get('link','')}")

    return hit_rate >= 0.35