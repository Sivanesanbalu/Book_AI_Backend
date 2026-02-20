from sentence_transformers import SentenceTransformer
import numpy as np

# --------------------------------------------------
# LOAD MODEL ONCE (GLOBAL SINGLETON)
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# NORMALIZE VECTOR
# --------------------------------------------------
def normalize(vec):
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return vec / norm


# --------------------------------------------------
# GET EMBEDDING
# --------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    """
    Converts book title text into semantic vector
    Used by search_engine.py
    """

    if not text:
        return np.zeros((1, 384), dtype="float32")

    emb = model.encode([text])[0].astype("float32")
    emb = normalize(np.array([emb]))

    return emb