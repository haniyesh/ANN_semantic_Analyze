"""
ANN Trainer — Groq/Llama-3.3-70B Sentiment
============================================
Multi-Stream ANN with Groq/Llama-3.3-70B sentiment features instead of the
3-BERT ensemble — the Groq counterpart of xgboost_train_groq.py.

Sentiment features:
  BERT : cb/fb/rb per-model probs + net_agreement  (13 dims)
  Groq : groq_pos, groq_neg, groq_neu + scalars    (6 dims)

Everything else is identical to ann_train.py: dual BERT embeddings (1536d),
news-type probs (11d), macro timing + price context (8d), RAG stream.

Outputs:
  ann_groq_weights.pt          — trained model weights
  ann_groq_results.json        — metrics (F1, AUC, thresholds)
  ann_groq_test_preds.npz      — per-row test predictions for compare_matrix.py

Usage:
    python training/ann_train_groq.py
    python training/ann_train_groq.py --skip-rag
    python training/ann_train_groq.py --seed 44
"""

import sys
from pathlib import Path

# ── Inject --sentiment groq before ann_train.main() parses sys.argv ───────────
# Remove any existing --sentiment flag so we can enforce groq.
_args = [a for i, a in enumerate(sys.argv)
         if not (a == "--sentiment" or (i > 0 and sys.argv[i - 1] == "--sentiment"))]
sys.argv = _args + ["--sentiment", "groq"]

# ── Delegate entirely to ann_train — no logic duplication ─────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ann_train import main

if __name__ == "__main__":
    main()
