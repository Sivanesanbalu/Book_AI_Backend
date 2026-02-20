import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = Lock()


# ---------------- SAFE LAZY LOAD ----------------
def get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                print("Loading embedding model...")
                _model = SentenceTransformer(MODEL_NAME)
                print("Embedding model loaded")

    return _model


# ---------------- NORMALIZE ----------------
def normalize(vec):
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return vec / norm


# ---------------- GET EMBEDDING ----------------
def get_embedding(text: str) -> np.ndarray:

    if not text:
        return np.zeros((1, 384), dtype="float32")

    model = get_model()

    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    return emb