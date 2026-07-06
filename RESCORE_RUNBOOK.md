# Re-score recent news so the dashboard is consistent across June 19

## Why the dashboard goes "strict" after June 19

`storage/news_cache.json` is fed by two scoring pipelines on **different score scales**:

- `historical_xgb_groq` — the backfill scorer. `model_score` runs high (0.30–0.80). Passes the display gate.
- `telegram_2025_2026` — the live fetcher. `model_score` is compressed low (0.03–0.15). Hidden by the gate.

The backfill scorer (`training/score_historical_xgb_groq.py`) is what produces the high, consistent
scores (`source: historical_xgb_groq`). **It simply has not been re-run since ~June 19.** Since then,
recent news kept arriving in the dashboard through the live pipeline (`telegram_2025_2026`), which
puts `model_score` on a much lower scale, so those items fall below the display gate.

The good news: the priced recent news is **already in `news_cleaned_filtered_scored.csv`** (rows through
July 4 have their 15m/1h forward prices filled). So re-scoring needs **no dataset rebuild and no Kaggle
files** — just re-run the scorer.

It is **not** a threshold problem — the two periods are on different score scales, and re-running the
backfill puts them back on one scale.

## Ignore the build_dataset preflight error

`python scripts/build_dataset.py --check` fails asking for `bitcoin_sentiments_21_24.csv`, `BTC.csv`,
`ETH.csv`. Those are the Kaggle **training** raws used to rebuild the model from scratch. You are not
retraining — you are re-scoring existing, already-priced news. **Skip build_dataset entirely.**

## The fix: just re-run the backfill scorer

Run from the project root, using your project's Python (scripts reference `.venv311/bin/python`).

### 1. Confirm prerequisites
- `GROQ_API_KEY` set in `.env` (backfill uses Groq/Llama sentiment).
- Model files present: `xgb_impact_clf_15m_groq.json`, `xgb_impact_clf_1h_groq.json`,
  `xgb_feature_scaler_groq.pkl`.

### 2. Dry-run the scorer (no write) to sanity-check
```bash
python training/score_historical_xgb_groq.py --dry-run
```
Check the printed "last 3m" date range now ends at ~today (July 4–5), and "Mean prob 15m/1h" looks
reasonable.

### 3. Re-score for real (rewrites storage/news_cache.json)
```bash
python training/score_historical_xgb_groq.py
```

### 4. Restart the API server
So it reloads the cache and the new config thresholds.

## Verify it worked

- Open a July item in `storage/news_cache.json` — its `source` should now be `historical_xgb_groq`
  (not `telegram_2025_2026`) and `model_score` should be on the high scale (many ≥ 0.30).
- The dashboard feed should no longer thin out after June 19.

## Important caveat — the newest ~1 hour can never be backfill-scored

Forward prices need 1 hour to elapse, so the most recent hour of news will always rely on the live
pipeline (low scores) until the next backfill run. To keep the dashboard consistent, **re-run the
scorer (steps 2–3) on a schedule** (e.g. daily). Everything older than ~1 hour will then always
carry consistent, high-quality scores.
