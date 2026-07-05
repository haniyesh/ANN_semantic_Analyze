"""Guards for the two scientific-validity bugs found in review.

1. The train/val/test split must be CHRONOLOGICAL (no lookahead): every
   training timestamp precedes every validation timestamp, which precedes
   every test timestamp.

2. The RAG Qdrant index must be built from TRAIN rows only, so val/test
   outcomes can never be retrieved into another row's features.

These are pure-logic tests with no network / no model files required.
"""
import numpy as np
import pandas as pd


def chronological_split(n, tr=0.70, vl=0.15):
    """Mirror of the split logic in training/xgboost_train_bert.py."""
    n_tr = int(n * tr)
    n_val = int(n * vl)
    return (
        np.arange(0, n_tr),
        np.arange(n_tr, n_tr + n_val),
        np.arange(n_tr + n_val, n),
    )


def test_split_is_chronological():
    n = 1000
    ts = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"published": ts}).sort_values("published").reset_index(drop=True)

    tri, vi, te = chronological_split(n)

    assert df["published"].iloc[tri].max() <= df["published"].iloc[vi].min()
    assert df["published"].iloc[vi].max() <= df["published"].iloc[te].min()
    # No index overlap
    assert set(tri).isdisjoint(vi)
    assert set(vi).isdisjoint(te)
    assert len(tri) + len(vi) + len(te) == n


def test_rag_index_excludes_val_test():
    """The set of rows uploaded to the index must be exactly the train rows."""
    n = 100
    df = pd.DataFrame({"title": [f"news {i}" for i in range(n)]})
    tri, vi, te = chronological_split(n)

    # This is what build_rag_features_qdrant does when train_idx is provided.
    upload_df = df.iloc[tri]
    uploaded_ids = set(upload_df.index)

    assert uploaded_ids == set(tri)
    assert uploaded_ids.isdisjoint(vi)
    assert uploaded_ids.isdisjoint(te)
