"""Canonical 15-minute DualBERT + RAG feature contract.

Keep this module dependency-light so training, live inference, API analysis,
and contract tests all use the exact same ordering and validation.
"""
from __future__ import annotations

import numpy as np


EMBEDDING_DIM = 768
SENTIMENT_KEYS = (
    "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
    "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
    "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
    "net_agreement", "sentiment_score", "weight", "confidence",
)
NEWS_TYPE_DIM = 11
MACRO_DIM = 8
RAG_DIM = 10
TOTAL_DIM = EMBEDDING_DIM * 2 + len(SENTIMENT_KEYS) + NEWS_TYPE_DIM + MACRO_DIM + RAG_DIM


def _vector(name: str, value, expected: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected:
        raise ValueError(f"{name} has {arr.shape[0]} values; expected {expected}")
    return arr


def build_bert_rag_features(
    sent: dict,
    cb_embedding,
    fb_embedding,
    type_probs,
    macro,
    rag_features,
) -> np.ndarray:
    """Return the canonical 1,578-value inference vector."""
    sent_vec = np.asarray([
        sent.get(key, 5 if key == "weight" else 0.0)
        for key in SENTIMENT_KEYS
    ], dtype=np.float32)

    features = np.concatenate([
        _vector("CryptoBERT embedding", cb_embedding, EMBEDDING_DIM),
        _vector("FinBERT embedding", fb_embedding, EMBEDDING_DIM),
        sent_vec,
        _vector("news-type probabilities", type_probs, NEWS_TYPE_DIM),
        _vector("macro features", macro, MACRO_DIM),
        _vector("RAG features", rag_features, RAG_DIM),
    ]).astype(np.float32)
    if features.shape[0] != TOTAL_DIM:
        raise AssertionError(f"Feature contract produced {features.shape[0]}, expected {TOTAL_DIM}")
    return features
