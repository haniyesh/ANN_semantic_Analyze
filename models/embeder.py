import numpy as np
from fastembed import TextEmbedding

_model = None


def _load():
    global _model
    if _model is None:
        try:
            print("[EMBEDDER] Loading FastEmbed model...")
            _model = TextEmbedding("BAAI/bge-small-en-v1.5")
            print("[EMBEDDER] Loaded ✅")
        except Exception as e:
            print(f"[EMBEDDER ERROR] {e}")


def embed(text: str) -> np.ndarray:
    """Returns a 384-dimension embedding vector for the text."""
    _load()
    if _model is None:
        return np.zeros(384)
    try:
        vectors = list(_model.embed([text[:512]]))
        return np.array(vectors[0])
    except Exception as e:
        print(f"[EMBEDDER ERROR] {e}")
        return np.zeros(384)