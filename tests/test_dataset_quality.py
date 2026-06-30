"""Guards for dataset-quality fixes.

1. Rows with date-level (noon-UTC placeholder) timestamps must be flagged
   `timestamp_reliable=False` so training can quarantine them.
2. The dataset build preflight must fail clearly when raw inputs and env are
   missing, and pass when fallbacks exist.
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_quarantines_unreliable_timestamps():
    """Mirror the filter in xgboost_v9.load_data()."""
    df = pd.DataFrame({
        "timestamp_reliable": [True, False, True, False],
        "v": [1, 2, 3, 4],
    })
    rel = df["timestamp_reliable"].astype(str).str.lower().isin(["true", "1", "1.0"])
    kept = df[rel]
    assert list(kept["v"]) == [1, 3]
    assert int((~rel).sum()) == 2


def test_build_row_flag_propagates():
    m = _load_module("services/merge_all_sources.py", "merge_all_sources")
    reliable = m._build_row("a real headline here", "2023-01-02T12:00:00Z", "c",
                            100, 101, 102, 50, 51, 52, m._default_sentiment(),
                            timestamp_reliable=False)
    assert reliable["timestamp_reliable"] is False
    ok = m._build_row("another real headline", "2023-01-02T09:13:00Z", "c",
                      100, 101, 102, 50, 51, 52, m._default_sentiment())
    assert ok["timestamp_reliable"] is True


def test_dataset_preflight_detects_missing_inputs(tmp_path, monkeypatch):
    bd = _load_module("scripts/build_dataset.py", "build_dataset")
    # Point the module at an empty dir with no raw inputs and no merged csv.
    monkeypatch.setattr(bd, "ROOT", tmp_path)
    monkeypatch.setattr(bd, "MERGED", tmp_path / "news_cleaned_filtered.csv")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    problems = bd.preflight(skip_rag=False)
    assert any("raw inputs" in p for p in problems)
    assert any("QDRANT" in p for p in problems)
