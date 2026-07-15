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


def build_bert_rag_feature_matrix(
    sentiment_rows,
    cb_embeddings,
    fb_embeddings,
    type_probabilities,
    macro_rows,
    rag_rows,
) -> np.ndarray:
    """Batch equivalent of :func:`build_bert_rag_features` for training.

    The function intentionally delegates every row to the inference builder.
    Training therefore cannot silently introduce a different feature order.
    ``sentiment_rows`` may be a DataFrame or an iterable of dictionaries.
    """
    if hasattr(sentiment_rows, "to_dict"):
        sentiments = sentiment_rows.to_dict(orient="records")
    else:
        sentiments = list(sentiment_rows)

    arrays = [
        np.asarray(cb_embeddings), np.asarray(fb_embeddings),
        np.asarray(type_probabilities), np.asarray(macro_rows),
        np.asarray(rag_rows),
    ]
    row_count = len(sentiments)
    for name, values in zip(
        ("CryptoBERT embeddings", "FinBERT embeddings", "news types", "macro", "RAG"),
        arrays,
    ):
        if values.ndim != 2 or values.shape[0] != row_count:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {row_count} rows"
            )

    return np.stack([
        build_bert_rag_features(
            sentiments[i], arrays[0][i], arrays[1][i], arrays[2][i],
            arrays[3][i], arrays[4][i],
        )
        for i in range(row_count)
    ]).astype(np.float32)
