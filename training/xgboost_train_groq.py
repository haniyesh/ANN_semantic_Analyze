"""
XGBoost Trainer — Groq/Llama-3.3-70B Sentiment
================================================
Same as xgboost_train_bert.py but replaces the 3-BERT ensemble sentiment features
with Groq/Llama-3.3-70B predictions, to compare LLM sentiment vs BERT.

Sentiment features replaced:
  BERT : cb_prob_{pos/neg/neu}, fb_prob_{pos/neg/neu}, rb_prob_{pos/neg/neu}, net_agreement (10 dims)
  Groq : groq_pos, groq_neg, groq_neu  (3 dims, one-hot from LLM label)

Everything else is identical: dual BERT embeddings, price context, RAG, thresholds.

Usage:
    python xgboost_train_groq.py                        # train + evaluate (compute Groq if needed)
    python xgboost_train_groq.py --skip-rag             # faster debug
    python xgboost_train_groq.py --compare              # show BERT vs Groq table
    python xgboost_train_groq.py --groq-sample 5000     # only call Groq for first N rows
    python xgboost_train_groq.py --groq-limit 1000      # limit new Groq API calls per run
"""

import os, sys, json, time, hashlib, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))              # pipeline.*
sys.path.insert(0, str(ROOT / "training"))   # sibling training scripts

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, mean_absolute_error, r2_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
HERE             = Path(__file__).parent.parent
SENTIMENT_CSV    = HERE / "news_cleaned_filtered_scored.csv"
CRYPTOBERT_CACHE = HERE / "cryptobert_embeddings_cache.npy"
FINBERT_CACHE    = HERE / "finbert_embeddings_cache.npy"
FEAR_GREED_CACHE = HERE / "fear_greed_cache.json"
GROQ_CACHE       = HERE / "groq_sentiment_cache.json"

FINBERT_MODEL    = "ProsusAI/finbert"
DUAL_EMB_DIM     = 768 + 768

XGB_MODEL_BASE   = HERE / "xgb_groq"      # prefix for Groq-variant model artifacts
XGB_RESULTS_PATH = HERE / "xgb_groq_results.json"
V9_RESULTS_PATH  = HERE / "xgb_bert_results.json"

THRESHOLD_15M = 0.3
THRESHOLD_1H  = 0.5
MONTHLY_SEED  = 43
MIN_PRECISION = 0.20

GROQ_MODEL    = "llama-3.3-70b-versatile"
GROQ_RPM      = 30      # requests per minute (free tier safe limit)

NEWS_TYPE_LABELS = [
    "regulatory", "etf", "hack", "macro_economic", "exchange",
    "defi", "mining", "institutional", "technical", "partnership", "market_analysis",
]

_NEWS_TYPE_PROTOTYPES_TEXT = {
    "regulatory":     ["SEC charges crypto exchange securities violations", "government bans cryptocurrency trading country"],
    "etf":            ["Bitcoin ETF approved by SEC trading", "spot bitcoin fund launches stock exchange"],
    "hack":           ["crypto exchange hacked millions stolen", "DeFi protocol exploited flash loan attack"],
    "macro_economic": ["Federal Reserve raises interest rates decision", "inflation data CPI report released"],
    "exchange":       ["Binance lists new cryptocurrency token", "Coinbase delists token regulatory concerns"],
    "defi":           ["DeFi protocol TVL record liquidity", "Uniswap launches new version features"],
    "mining":         ["Bitcoin mining difficulty adjusts record", "miner capitulation hashrate drops significantly"],
    "institutional":  ["MicroStrategy purchases Bitcoin treasury reserve", "hedge fund allocates Bitcoin portfolio"],
    "technical":      ["Bitcoin network upgrade soft fork activates", "Ethereum developers confirm upgrade date"],
    "partnership":    ["crypto company partnership bank deal", "blockchain firm integrates payment processor"],
    "market_analysis":["Bitcoin price analysis bullish breakout target", "technical analysis support level tested"],
}

_proto_matrix = None

PROMPT_TEMPLATE = (
    'Classify the Bitcoin/crypto market sentiment of this news headline.\n'
    'Headline: "{title}"\n'
    'Reply with exactly one word: positive, negative, or neutral.'
)


# ══════════════════════════════════════════════════════════════════
# GROQ SENTIMENT  — cached LLM predictions
# ══════════════════════════════════════════════════════════════════
def _title_key(title: str) -> str:
    return hashlib.md5(title.encode()).hexdigest()


def _normalize(text: str) -> str:
    t = text.lower().strip()
    if any(w in t for w in ["positive", "bullish", "buy"]):  return "positive"
    if any(w in t for w in ["negative", "bearish", "sell"]): return "negative"
    return "neutral"


def load_groq_cache() -> dict:
    if GROQ_CACHE.exists():
        with open(GROQ_CACHE) as f:
            return json.load(f)
    return {}


def save_groq_cache(cache: dict):
    with open(GROQ_CACHE, "w") as f:
        json.dump(cache, f)


def _get_groq_clients():
    """Return list of OpenAI clients, one per available Groq API key."""
    from openai import OpenAI
    keys = []
    for env in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]:
        k = os.environ.get(env)
        if k and k not in keys:
            keys.append(k)
    if not keys:
        return []
    return [OpenAI(api_key=k, base_url="https://api.groq.com/openai/v1") for k in keys]


def compute_groq_sentiment(titles: list[str], groq_limit: int = None) -> dict:
    """
    Call Groq API for titles not yet in cache. Rotates between all available
    API keys to multiply effective rate limit. Saves results incrementally.
    Returns full cache dict {md5_key: label}.
    """
    clients = _get_groq_clients()
    if not clients:
        print("  ⚠ No GROQ_API_KEY found — using 'neutral' fallback")
        return {}

    cache    = load_groq_cache()
    to_fetch = [t for t in titles if _title_key(t) not in cache]
    if groq_limit:
        to_fetch = to_fetch[:groq_limit]

    if not to_fetch:
        print(f"  Groq cache: {len(cache):,} entries, all titles cached")
        return cache

    n_keys   = len(clients)
    eff_rpm  = GROQ_RPM * n_keys
    interval = 60.0 / eff_rpm
    print(f"  Groq cache: {len(cache):,} cached, {len(to_fetch):,} new to fetch")
    print(f"  Model: {GROQ_MODEL}  Keys: {n_keys}  Effective rate: {eff_rpm} req/min")

    errors = 0
    # Track per-key cooldowns after rate-limit hits
    key_cooldown = [0.0] * n_keys

    for i, title in enumerate(to_fetch):
        if i % 100 == 0 and i > 0:
            print(f"    … {i}/{len(to_fetch)}  errors={errors}", flush=True)

        # Pick client: round-robin, skip keys in cooldown
        ki     = i % n_keys
        now    = time.time()
        for attempt in range(n_keys):
            idx = (ki + attempt) % n_keys
            if now >= key_cooldown[idx]:
                ki = idx
                break

        label = "neutral"
        try:
            r = clients[ki].chat.completions.create(
                model=GROQ_MODEL, max_tokens=5, temperature=0,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(title=title)}],
            )
            label = _normalize(r.choices[0].message.content.strip())
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                print(f"\n    Key[{ki}] rate limit at {i}, cooling 60s...")
                key_cooldown[ki] = time.time() + 60
                # Retry immediately with another key
                alt = (ki + 1) % n_keys
                if alt != ki:
                    try:
                        r = clients[alt].chat.completions.create(
                            model=GROQ_MODEL, max_tokens=5, temperature=0,
                            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(title=title)}],
                        )
                        label = _normalize(r.choices[0].message.content.strip())
                    except Exception:
                        errors += 1
                else:
                    time.sleep(60)
                    errors += 1
            else:
                errors += 1

        cache[_title_key(title)] = label
        if (i + 1) % 200 == 0:
            save_groq_cache(cache)
        time.sleep(interval)

    save_groq_cache(cache)
    print(f"  Groq done: {len(to_fetch)} fetched, {errors} errors, total cache={len(cache):,}")
    return cache


GROQ_CSV = HERE / "news_cleaned_filtered_scored_groq.csv"


def build_groq_features(df: pd.DataFrame, groq_limit: int = None) -> np.ndarray:
    """
    Returns (N, 3) array: [groq_pos, groq_neg, groq_neu] one-hot features.

    Label sources, in priority order:
      1. news_cleaned_filtered_scored_groq.csv (groq_sentiment column,
         matched by title) — the authoritative scored file
      2. groq_sentiment_cache.json (md5-keyed cache; may trigger API calls)
      3. 'neutral' fallback — counted and WARNED, because build_groq_csv.py
         backfills uncached rows with BERT sentiment, which would contaminate
         the BERT-vs-Groq comparison if unnoticed.
    """
    titles = df["title"].fillna("").tolist()

    csv_map = {}
    if GROQ_CSV.exists():
        gdf = pd.read_csv(GROQ_CSV, low_memory=False,
                          usecols=["title", "groq_sentiment"])
        csv_map = dict(zip(gdf["title"].fillna(""), gdf["groq_sentiment"]))
        print(f"  Groq labels from {GROQ_CSV.name}: {len(csv_map):,} titles")

    missing_from_csv = [t for t in titles if t not in csv_map]
    cache = {}
    if missing_from_csv:
        cache = compute_groq_sentiment(missing_from_csv, groq_limit=groq_limit)

    n_fallback = 0
    feats = []
    for title in titles:
        label = csv_map.get(title)
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = cache.get(_title_key(title))
        if label not in ("positive", "negative", "neutral"):
            label = "neutral"
            n_fallback += 1
        feats.append([
            1.0 if label == "positive" else 0.0,
            1.0 if label == "negative" else 0.0,
            1.0 if label == "neutral"  else 0.0,
        ])

    if n_fallback:
        print(f"  ⚠️  {n_fallback:,}/{len(titles):,} titles have NO Groq label "
              f"(defaulted to neutral). For a clean BERT-vs-Groq comparison, "
              f"run training/build_groq_csv.py until coverage is 100%.")

    arr = np.array(feats, dtype=np.float32)
    pos_pct = arr[:, 0].mean() * 100
    neg_pct = arr[:, 1].mean() * 100
    neu_pct = arr[:, 2].mean() * 100
    print(f"  Groq sentiment dist: +{pos_pct:.1f}%  -{neg_pct:.1f}%  ={neu_pct:.1f}%")
    return arr


# ══════════════════════════════════════════════════════════════════
# NEWS TYPE PROTOTYPES
# ══════════════════════════════════════════════════════════════════
def _build_proto_matrix():
    global _proto_matrix
    if _proto_matrix is not None:
        return _proto_matrix
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    print("  Building news type prototype embeddings (one-time)...")
    tok = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    mdl = AutoModel.from_pretrained("ElKulako/cryptobert").eval()
    proto_embs = []
    for label in NEWS_TYPE_LABELS:
        sentences = _NEWS_TYPE_PROTOTYPES_TEXT[label]
        inputs    = tok(sentences, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            emb = mdl(**inputs).last_hidden_state[:, 0, :]
        proto_embs.append(emb.mean(dim=0))
    mat = torch.stack(proto_embs)
    _proto_matrix = F.normalize(mat, dim=1)
    return _proto_matrix


def crypto_news_type_classify(embeddings: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    proto = _build_proto_matrix()
    emb_t = F.normalize(torch.FloatTensor(embeddings), dim=1)
    sims  = torch.mm(emb_t, proto.T)
    return F.softmax(sims * 5.0, dim=1).numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("[1/7] LOAD DATA")
    df = pd.read_csv(SENTIMENT_CSV, low_memory=False)
    for col in df.columns:
        orig = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            df[col] = orig

    orig = len(df)
    df = df.dropna(subset=["btc_price_at_news", "btc_price_15m", "btc_price_1h"])
    df = df.drop_duplicates(subset=["title", "published", "channel"])

    df["btc_change_15m"]   = (df["btc_price_15m"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_15m"]   = df["btc_change_15m"].abs()
    df["is_impactful_15m"] = (df["abs_change_15m"] > THRESHOLD_15M).astype(int)
    df["direction_15m"]    = (df["btc_change_15m"] > 0).astype(int)

    df["btc_change_1h"]   = (df["btc_price_1h"] - df["btc_price_at_news"]) / df["btc_price_at_news"] * 100
    df["abs_change_1h"]   = df["btc_change_1h"].abs()
    df["is_impactful_1h"] = (df["abs_change_1h"] > THRESHOLD_1H).astype(int)
    df["direction_1h"]    = (df["btc_change_1h"] > 0).astype(int)

    df["published"] = pd.to_datetime(df["published"], format="mixed", utc=True)
    df = df.sort_values("published").reset_index(drop=True)

    print(f"  Rows: {len(df):,} (from {orig:,})")
    print(f"  Impactful 15m: {df['is_impactful_15m'].mean()*100:.1f}%  "
          f"1h: {df['is_impactful_1h'].mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════
def compute_cryptobert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if CRYPTOBERT_CACHE.exists():
        emb = np.load(CRYPTOBERT_CACHE).astype(np.float32)
        if len(emb) == len(df):
            print("  CryptoBERT cache hit")
            return emb
    print(f"  Computing CryptoBERT embeddings for {len(df):,} rows...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    device    = torch.device("cpu")   # CPU-only avoids WSL2 CUDA allocator crash
    tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model     = AutoModel.from_pretrained("ElKulako/cryptobert").eval().to(device)
    titles, embs = df["title"].fillna("").tolist(), []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)}...")
            inputs = tokenizer(titles[i:i+32], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            embs.append(model(**inputs).last_hidden_state[:, 0, :].numpy())
    emb = np.vstack(embs).astype(np.float32)
    np.save(CRYPTOBERT_CACHE, emb)
    return emb


def compute_finbert_embeddings(df: pd.DataFrame) -> np.ndarray:
    if FINBERT_CACHE.exists():
        emb = np.load(FINBERT_CACHE).astype(np.float32)
        if len(emb) == len(df):
            print("  FinBERT cache hit")
            return emb
    print(f"  Computing FinBERT embeddings for {len(df):,} rows...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModel.from_pretrained(FINBERT_MODEL).eval().to(device)
    titles, embs = df["title"].fillna("").tolist(), []
    with torch.no_grad():
        for i in range(0, len(titles), 32):
            if i % 2000 == 0:
                print(f"    {i}/{len(titles)}...")
            inputs = tokenizer(titles[i:i+32], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            embs.append(model(**inputs).last_hidden_state[:, 0, :].numpy())
    emb = np.vstack(embs).astype(np.float32)
    np.save(FINBERT_CACHE, emb)
    return emb


# ══════════════════════════════════════════════════════════════════
# PRICE CONTEXT FEATURES
# ══════════════════════════════════════════════════════════════════
def fetch_fear_greed_index(published_series: pd.Series) -> np.ndarray:
    fg_map = {}
    if FEAR_GREED_CACHE.exists():
        try:
            with open(FEAR_GREED_CACHE) as f:
                items = json.load(f)
            for item in items:
                date = pd.Timestamp(int(item["timestamp"]), unit="s").date()
                fg_map[date] = float(item["value"]) / 100.0
        except Exception:
            pass
    if not fg_map:
        try:
            import urllib.request
            with urllib.request.urlopen("https://api.alternative.me/fng/?limit=0&format=json", timeout=15) as r:
                items = json.loads(r.read())["data"]
            with open(FEAR_GREED_CACHE, "w") as f:
                json.dump(items, f)
            for item in items:
                date = pd.Timestamp(int(item["timestamp"]), unit="s").date()
                fg_map[date] = float(item["value"]) / 100.0
        except Exception as e:
            print(f"  Fear/Greed unavailable ({e}), using 0.5")
    return np.array([fg_map.get(pd.Timestamp(ts).date(), 0.5) for ts in published_series], dtype=np.float32)


def compute_price_context(df: pd.DataFrame) -> np.ndarray:
    changes = pd.Series(df["btc_change_15m"].values.astype(np.float32))
    shifted = changes.shift(1)
    btc_vol = shifted.rolling(20, min_periods=2).std().fillna(0).values.astype(np.float32)
    btc_mom = shifted.rolling(5,  min_periods=1).mean().fillna(0).values.astype(np.float32)
    fg      = fetch_fear_greed_index(df["published"])
    return np.column_stack([btc_vol, btc_mom, fg]).astype(np.float32)


def build_macro_features(df: pd.DataFrame) -> np.ndarray:
    hour = df["published"].dt.hour.values
    dow  = df["published"].dt.dayofweek.values
    is_weekend       = (dow >= 5).astype(np.float32)
    is_low_liquidity = ((hour >= 2) & (hour <= 6)).astype(np.float32)
    is_us_hours      = ((hour >= 13) & (hour <= 21)).astype(np.float32)
    is_asia_hours    = ((hour >= 0) & (hour <= 8)).astype(np.float32)
    fomc_week        = df["fomc_week"].fillna(0).values.astype(np.float32) if "fomc_week" in df.columns else np.zeros(len(df), dtype=np.float32)
    return np.column_stack([is_weekend, is_low_liquidity, is_us_hours, is_asia_hours, fomc_week]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  — Groq replaces BERT ensemble sentiment
# ══════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, train_idx: np.ndarray,
                   skip_rag: bool = False, groq_limit: int = None):
    print("[2/7] FEATURE ENGINEERING")

    cb_emb = compute_cryptobert_embeddings(df)
    fb_emb = compute_finbert_embeddings(df)

    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        type_onehot = pd.get_dummies(df["news_type"].fillna("market_analysis"))
        for label in NEWS_TYPE_LABELS:
            if label not in type_onehot.columns:
                type_onehot[label] = 0
        type_probs = type_onehot[NEWS_TYPE_LABELS].values.astype(np.float32)
    else:
        type_probs = crypto_news_type_classify(cb_emb)

    # Groq sentiment (3 dims) replaces 10-dim BERT ensemble
    print("  Computing Groq/Llama-3.3-70B sentiment features...")
    groq_feats = build_groq_features(df, groq_limit=groq_limit)

    # Keep legacy scalar features from CSV
    scalar_cols = ["sentiment_score", "weight", "confidence"]
    scalar_df   = df[scalar_cols].fillna(0).values.astype(np.float32)

    sent_feats = np.hstack([groq_feats, scalar_df])  # (N, 6)

    timing_mac = build_macro_features(df)
    price_ctx  = compute_price_context(df)
    macro      = np.hstack([timing_mac, price_ctx]).astype(np.float32)

    print(f"  Semantic : {DUAL_EMB_DIM} emb + 6 sent(groq+scalar) + 11 type = {DUAL_EMB_DIM + 6 + 11} dims")
    print(f"  Macro    : {macro.shape[1]} dims")

    if skip_rag:
        print("  RAG      : SKIPPED")
        rag = np.zeros((len(df), 1), dtype=np.float32)
    else:
        from pipeline.rag_news import build_rag_features_qdrant
        ch_rates = df.iloc[train_idx].groupby("channel")["is_impactful_15m"].mean().to_dict()
        rag, _   = build_rag_features_qdrant(
            df, channel_impact_rates=ch_rates,
            train_idx=train_idx, rebuild=True,
        )
        print(f"  RAG      : {rag.shape[1]} dims")

    X = np.hstack([cb_emb, fb_emb, sent_feats, type_probs, macro, rag]).astype(np.float32)
    print(f"  Total features: {X.shape[1]}")

    feat_names = (
        [f"cb_{i}"  for i in range(768)] +
        [f"fb_{i}"  for i in range(768)] +
        ["groq_pos", "groq_neg", "groq_neu", "sentiment_score", "weight", "confidence"] +
        [f"type_{l}" for l in NEWS_TYPE_LABELS] +
        ["is_weekend", "is_low_liq", "is_us_hours", "is_asia_hours", "fomc_week",
         "btc_vol", "btc_mom", "fear_greed"] +
        ([f"rag_{i}" for i in range(rag.shape[1])] if not skip_rag else ["rag_dummy"])
    )

    return (
        X, feat_names,
        df["btc_change_15m"].values.astype(np.float32),
        df["is_impactful_15m"].values.astype(np.float32),
        df["btc_change_1h"].values.astype(np.float32),
        df["is_impactful_1h"].values.astype(np.float32),
        df["direction_15m"].values.astype(np.int64),
    )


# ══════════════════════════════════════════════════════════════════
# XGBOOST TRAINING
# ══════════════════════════════════════════════════════════════════
def train_xgboost_models(X_tr, X_vl,
                          y_c15_tr, y_c15_vl,
                          y_c1h_tr, y_c1h_vl,
                          y_r15_tr, y_r1h_tr):
    try:
        import xgboost as xgb
    except ImportError:
        os.system("pip install xgboost -q")
        import xgboost as xgb

    print(f"\n[4/7] TRAINING XGBOOST MODELS")
    neg_15 = int((y_c15_tr == 0).sum()); pos_15 = int((y_c15_tr == 1).sum())
    neg_1h = int((y_c1h_tr == 0).sum()); pos_1h = int((y_c1h_tr == 1).sum())
    print(f"  15m scale_pos_weight: {neg_15/pos_15:.2f}")
    print(f"  1h  scale_pos_weight: {neg_1h/pos_1h:.2f}")

    base_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=MONTHLY_SEED,
        device="cuda", tree_method="hist",
        early_stopping_rounds=20, eval_metric="logloss",
    )

    print("  Training clf_15m...")
    clf_15m = xgb.XGBClassifier(**base_params, objective="binary:logistic",
                                  scale_pos_weight=neg_15/pos_15)
    clf_15m.fit(X_tr, y_c15_tr, eval_set=[(X_vl, y_c15_vl)], verbose=False)
    print(f"    Best iteration: {clf_15m.best_iteration}")

    print("  Training clf_1h...")
    clf_1h = xgb.XGBClassifier(**base_params, objective="binary:logistic",
                                 scale_pos_weight=neg_1h/pos_1h)
    clf_1h.fit(X_tr, y_c1h_tr, eval_set=[(X_vl, y_c1h_vl)], verbose=False)
    print(f"    Best iteration: {clf_1h.best_iteration}")

    reg_params = {k: v for k, v in base_params.items()
                  if k not in ["scale_pos_weight", "eval_metric", "n_jobs"]}
    reg_params["eval_metric"] = "rmse"
    print("  Training reg_15m...")
    reg_15m = xgb.XGBRegressor(**reg_params, objective="reg:squarederror")
    reg_15m.fit(X_tr, y_r15_tr, eval_set=[(X_vl, y_r15_tr[:len(X_vl)])], verbose=False)

    return clf_15m, clf_1h, reg_15m


# ══════════════════════════════════════════════════════════════════
# THRESHOLD + EVALUATION
# ══════════════════════════════════════════════════════════════════
def find_threshold(probs, y_cls, label="15m"):
    best_f1, best_t, found = 0.0, 0.50, False
    for t in np.linspace(0.02, 0.90, 177):
        preds = (probs >= t).astype(int)
        prec  = precision_score(y_cls, preds, zero_division=0)
        if prec < MIN_PRECISION:
            continue
        f1 = f1_score(y_cls, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, found = f1, t, True
    if not found:
        print(f"    {label}: WARNING — fallback threshold=0.50")
    else:
        print(f"    {label}: threshold={best_t:.2f}  val F1={best_f1:.3f}")
    return best_t


def eval_horizon(label, probs, threshold, y_cls, reg_pred, y_reg):
    preds   = (probs >= threshold).astype(int)
    f1      = f1_score(y_cls, preds, zero_division=0)
    prec    = precision_score(y_cls, preds, zero_division=0)
    rec     = recall_score(y_cls, preds, zero_division=0)
    acc     = accuracy_score(y_cls, preds)
    auc     = roc_auc_score(y_cls, probs) if len(np.unique(y_cls)) > 1 else 0.0
    mae     = mean_absolute_error(y_reg, reg_pred)
    r2      = r2_score(y_reg, reg_pred)
    dir_acc = (np.sign(y_reg) == np.sign(reg_pred)).mean()
    cm      = confusion_matrix(y_cls, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ── {label} ──")
    print(f"    Threshold : {threshold:.2f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}  AUC: {auc:.3f}")
    print(f"    Acc: {acc:.3f}  DirAcc: {dir_acc:.1%}  MAE: {mae:.4f}%  R²: {r2:.3f}")
    print(f"    Confusion: NO[TN={tn:>5} FP={fp:>5}]  YES[FN={fn:>5} TP={tp:>5}]")

    return {
        "F1": float(f1), "Precision": float(prec), "Recall": float(rec),
        "Accuracy": float(acc), "ROC_AUC": float(auc),
        "MAE": float(mae), "R2": float(r2), "DirAcc": float(dir_acc),
        "Threshold": float(threshold), "CM": cm.tolist(),
    }


# ══════════════════════════════════════════════════════════════════
# COMPARISON vs v9 (BERT sentiment)
# ══════════════════════════════════════════════════════════════════
def print_comparison(v10_results: dict):
    if not V9_RESULTS_PATH.exists():
        print(f"\n  ⚠ BERT results not found at {V9_RESULTS_PATH} — run xgboost_train_bert.py first")
        return

    with open(V9_RESULTS_PATH) as f:
        v9 = json.load(f)

    metrics = ["F1", "Precision", "Recall", "Accuracy", "ROC_AUC"]
    print(f"\n{'='*72}")
    print(f"  XGBoost v9 (BERT sentiment)  vs  v10 (Groq/Llama-3.3-70B sentiment)")
    print(f"{'='*72}")

    for horizon, key in [("15-minute", "15_minute"), ("1-hour", "1_hour")]:
        print(f"\n  ── {horizon} ──")
        print(f"  {'Metric':<14} {'v9 (BERT)':>12} {'v10 (Groq)':>12} {'Winner':>12}  {'Δ':>8}")
        print(f"  {'─'*60}")
        v9_h  = v9.get(key, {})
        v10_h = v10_results.get(key, {})
        for m in metrics:
            a = v9_h.get(m, 0.0)
            b = v10_h.get(m, 0.0)
            diff   = b - a
            winner = "Groq ✅" if b > a else ("BERT ✅" if a > b else "TIE")
            bar    = "▲" * min(int(abs(diff) * 50), 10) if diff > 0 else "▼" * min(int(abs(diff) * 50), 10)
            print(f"  {m:<14} {a:>12.3f} {b:>12.3f} {winner:>12}  {diff:>+7.3f} {bar}")

    v9_avg  = (v9.get("15_minute",  {}).get("F1", 0) + v9.get("1_hour",  {}).get("F1", 0)) / 2
    v10_avg = (v10_results.get("15_minute", {}).get("F1", 0) + v10_results.get("1_hour", {}).get("F1", 0)) / 2
    print(f"\n  v9  (BERT sentiment) avg F1 : {v9_avg:.3f}")
    print(f"  v10 (Groq sentiment) avg F1 : {v10_avg:.3f}")
    if v10_avg > v9_avg:
        print(f"  Groq/LLM sentiment wins by {v10_avg - v9_avg:+.3f} F1")
    elif v9_avg > v10_avg:
        print(f"  BERT sentiment wins by {v9_avg - v10_avg:+.3f} F1")
    else:
        print(f"  TIE")


def print_feature_importance(clf_15m, feat_names, top_n=20):
    importances = clf_15m.feature_importances_
    pairs       = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    meaningful  = [(n, v) for n, v in pairs if not n.startswith("cb_") and not n.startswith("fb_")]
    print(f"\n{'─'*50}")
    print(f"  TOP {top_n} FEATURES (non-embedding) — clf_15m")
    print(f"{'─'*50}")
    for name, val in meaningful[:top_n]:
        bar = "█" * int(val * 500)
        print(f"  {name:<30} {val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare",      action="store_true", help="Print v9(BERT) vs v10(Groq) table")
    parser.add_argument("--skip-rag",     action="store_true", help="Skip RAG features")
    parser.add_argument("--load-only",    action="store_true", help="Load saved models, skip training")
    parser.add_argument("--groq-limit",   type=int, default=None,
                        help="Max new Groq API calls per run (default: unlimited)")
    parser.add_argument("--groq-sample",  type=int, default=None,
                        help="Only use first N rows of the dataset (quick test)")
    args = parser.parse_args()

    print("=" * 65)
    print("  XGBOOST v10 — Groq/Llama-3.3-70B Sentiment")
    print(f"  seed={MONTHLY_SEED}  threshold_15m={THRESHOLD_15M}  min_precision={MIN_PRECISION}")
    print("=" * 65)

    df = load_data()
    if args.groq_sample:
        df = df.head(args.groq_sample).reset_index(drop=True)
        print(f"  Using first {len(df):,} rows (--groq-sample)")

    print(f"\n[3/7] CHRONOLOGICAL SPLIT (70/15/15)")
    # True chronological: earliest 70% = train, next 15% = val, latest 15% = test.
    # Data is already sorted by published in load_data().
    n = len(df)
    n_tr  = int(n * 0.70)
    n_val = int(n * 0.15)
    tri    = np.arange(0, n_tr)
    vi     = np.arange(n_tr, n_tr + n_val)
    te_idx = np.arange(n_tr + n_val, n)

    # Sanity check: no temporal leakage
    train_max = df.iloc[tri]["published"].max()
    val_min   = df.iloc[vi]["published"].min()
    assert train_max <= val_min, f"Train/val overlap: train_max={train_max} > val_min={val_min}"
    print(f"  Train: {len(tri):,} [{df.iloc[0]['published'].date()} → {train_max.date()}]")
    print(f"  Val:   {len(vi):,}  [{val_min.date()} → {df.iloc[vi[-1]]['published'].date()}]")
    print(f"  Test:  {len(te_idx):,}  [{df.iloc[te_idx[0]]['published'].date()} → {df.iloc[-1]['published'].date()}]")

    (X, feat_names,
     y_r15, y_c15, y_r1h, y_c1h, y_dir) = build_features(
        df, tri, skip_rag=args.skip_rag, groq_limit=args.groq_limit)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[tri]).astype(np.float32)
    X_vl   = scaler.transform(X[vi]).astype(np.float32)
    X_te   = scaler.transform(X[te_idx]).astype(np.float32)

    clf15_path  = str(HERE / "xgb_impact_clf_15m_groq.json")
    clf1h_path  = str(HERE / "xgb_impact_clf_1h_groq.json")
    reg15_path  = str(HERE / "xgb_price_reg_15m_groq.json")
    scaler_path = str(HERE / "xgb_feature_scaler_groq.pkl")

    if args.load_only and Path(clf15_path).exists():
        import xgboost as xgb
        print(f"\n[4/7] LOADING saved models")
        clf_15m = xgb.XGBClassifier(); clf_15m.load_model(clf15_path)
        clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(clf1h_path)
        reg_15m = xgb.XGBRegressor();  reg_15m.load_model(reg15_path)
    else:
        clf_15m, clf_1h, reg_15m = train_xgboost_models(
            X_tr, X_vl,
            y_c15[tri], y_c15[vi],
            y_c1h[tri], y_c1h[vi],
            y_r15[tri], y_r1h[tri],
        )
        clf_15m.save_model(clf15_path)
        clf_1h.save_model(clf1h_path)
        reg_15m.save_model(reg15_path)
        import pickle
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Models saved → {XGB_MODEL_BASE}*")

    print(f"\n[5/7] THRESHOLD SEARCH (min_precision={MIN_PRECISION})")
    p15_vl  = clf_15m.predict_proba(X_vl)[:, 1]
    p1h_vl  = clf_1h.predict_proba(X_vl)[:, 1]
    thr_15m = find_threshold(p15_vl, y_c15[vi], "15m")
    thr_1h  = find_threshold(p1h_vl, y_c1h[vi], "1h")

    print(f"\n[6/7] TEST SET EVALUATION")
    print(f"{'='*65}\n  XGBoost v10 TEST RESULTS\n{'='*65}")

    p15_te   = clf_15m.predict_proba(X_te)[:, 1]
    p1h_te   = clf_1h.predict_proba(X_te)[:, 1]
    r15_te   = reg_15m.predict(X_te)
    dir_pred = (p15_te >= 0.5).astype(int)

    # Save per-row test predictions for bootstrap CIs / paired significance
    # tests (used by training/compare_matrix.py).
    preds_path = str(XGB_MODEL_BASE) + "_test_preds.npz"
    np.savez(
        preds_path,
        p15=p15_te, p1h=p1h_te, r15=r15_te,
        y_c15=y_c15[te_idx], y_c1h=y_c1h[te_idx],
        y_r15=y_r15[te_idx], y_r1h=y_r1h[te_idx],
        thr_15m=thr_15m, thr_1h=thr_1h,
        published=df["published"].iloc[te_idx].astype("int64").values,
    )
    print(f"  Test predictions saved → {preds_path}")

    r15 = eval_horizon("15-minute", p15_te, thr_15m, y_c15[te_idx], r15_te, y_r15[te_idx])
    r1h = eval_horizon("1-hour",    p1h_te, thr_1h,  y_c1h[te_idx], r15_te, y_r1h[te_idx])

    dir_acc = accuracy_score(y_dir[te_idx], dir_pred)
    dir_f1  = f1_score(y_dir[te_idx], dir_pred, zero_division=0)
    print(f"\n  Direction: Acc={dir_acc:.1%}  F1={dir_f1:.3f}")

    xgb_results = {
        "15_minute": r15, "1_hour": r1h,
        "direction": {"Acc": float(dir_acc), "F1": float(dir_f1)},
        "threshold_15m": float(thr_15m), "threshold_1h": float(thr_1h),
        "groq_model": GROQ_MODEL,
        "groq_cache_size": len(load_groq_cache()),
    }
    with open(XGB_RESULTS_PATH, "w") as f:
        json.dump(xgb_results, f, indent=2)
    print(f"\n  Saved → {XGB_RESULTS_PATH}")

    print_feature_importance(clf_15m, feat_names)

    if args.compare or V9_RESULTS_PATH.exists():
        print_comparison(xgb_results)

    print(f"\n{'='*65}\n  SUMMARY — XGBoost v10 (Groq/Llama-3.3-70B)\n{'='*65}")
    print(f"  Features      : {X.shape[1]} total")
    print(f"  Train/Val/Test: {len(tri):,} / {len(vi):,} / {len(te_idx):,}")
    print(f"  Groq cache    : {xgb_results['groq_cache_size']:,} entries")
    print(f"  clf_15m F1    : {r15['F1']:.3f}")
    print(f"  clf_1h  F1    : {r1h['F1']:.3f}")


if __name__ == "__main__":
    main()
