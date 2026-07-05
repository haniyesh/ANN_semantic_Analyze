#!/usr/bin/env python3
"""
build_dataset.py
================
One command to regenerate the training dataset from raw sources, so results
are reproducible from a clean clone (given the raw Kaggle inputs + API keys).

Pipeline:
  raw Kaggle files
    -> services/merge_all_sources.py   (prices from Binance, label build)
       => news_cleaned_filtered.csv
    -> services/sentiment_score.py     (3-model ensemble sentiment columns)
       => news_cleaned_filtered_scored.csv
    -> training/xgboost_train_bert.py   (embeddings, RAG, train, evaluate)
       => xgb_impact_clf_15m_bert.json, xgb_impact_clf_1h_bert.json, xgb_feature_scaler_bert.pkl, xgb_bert_results.json

Required raw inputs (NOT committed — download from Kaggle into repo root):
  - bitcoin_sentiments_21_24.csv
  - BTC.csv
  - ETH.csv

Required env (.env):
  - QDRANT_URL, QDRANT_API_KEY   (or pass --skip-rag)

Usage:
  python scripts/build_dataset.py            # full pipeline
  python scripts/build_dataset.py --skip-rag # train without RAG features
  python scripts/build_dataset.py --check    # only verify inputs/env, do nothing
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_INPUTS = ["bitcoin_sentiments_21_24.csv", "BTC.csv", "ETH.csv"]
MERGED = ROOT / "news_cleaned_filtered.csv"
SCORED = ROOT / "news_cleaned_filtered_scored.csv"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n" + "-" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n✗ step failed (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)


def preflight(skip_rag: bool) -> list[str]:
    """Return a list of problems; empty means good to go."""
    problems = []

    # An existing merged CSV can stand in for the raw inputs.
    have_raw = all((ROOT / f).exists() for f in RAW_INPUTS)
    if not have_raw and not MERGED.exists():
        missing = [f for f in RAW_INPUTS if not (ROOT / f).exists()]
        problems.append(
            "Missing raw inputs and no news_cleaned_filtered.csv to fall back on. "
            f"Download from Kaggle into the repo root: {', '.join(missing)}"
        )

    if not skip_rag and (not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY")):
        problems.append(
            "QDRANT_URL / QDRANT_API_KEY not set. Set them in .env, "
            "or run with --skip-rag to train without RAG features."
        )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rag", action="store_true", help="train without RAG features")
    ap.add_argument("--check", action="store_true", help="verify inputs/env only")
    ap.add_argument("--keep-unreliable-ts", action="store_true",
                    help="keep noon-UTC placeholder-timestamp rows (NOT recommended)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass

    print("=" * 60)
    print("  DATASET BUILD — reproducible pipeline")
    print("=" * 60)

    problems = preflight(args.skip_rag)
    if problems:
        print("\nPreflight problems:")
        for p in problems:
            print(f"  ✗ {p}")
        sys.exit(2)
    print("  ✓ preflight passed")

    if args.check:
        print("  (--check) inputs and env look OK — exiting without building.")
        return

    py = sys.executable

    # Step 1: merge raw sources -> news_cleaned_filtered.csv (skip if present)
    if MERGED.exists():
        print(f"\n[1/3] {MERGED.name} exists — skipping merge "
              f"(delete it to force a rebuild from raw inputs)")
    else:
        _run([py, "services/merge_all_sources.py"])

    # Step 2: ensemble sentiment scoring -> *_scored.csv
    if SCORED.exists():
        print(f"\n[2/3] {SCORED.name} exists — skipping scoring "
              f"(delete it to re-score)")
    else:
        _run([py, "services/sentiment_score.py", str(MERGED),
              "--output", str(SCORED)])

    # Step 3: train + evaluate
    env = dict(os.environ)
    if args.keep_unreliable_ts:
        env["KEEP_UNRELIABLE_TS"] = "1"
    train_cmd = [py, "training/xgboost_train_bert.py"]
    if args.skip_rag:
        train_cmd.append("--skip-rag")
    print(f"\n[3/3] training")
    print(f"\n$ {' '.join(train_cmd)}\n" + "-" * 60)
    if subprocess.run(train_cmd, cwd=ROOT, env=env).returncode != 0:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ dataset + model build complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
