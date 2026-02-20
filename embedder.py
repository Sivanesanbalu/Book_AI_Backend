import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock
import torch

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = Lock()


# ---------------- SAFE LAZY LOAD ----------------
def get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                print("🔄 Loading embedding model...")

                _model = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu"
                )

                # IMPORTANT: reduce RAM usage (Render safe)
                _model.max_seq_length = 64

                print("✅ Embedding model loaded")

    return _model


# ---------------- SAFE ZERO VECTOR ----------------
def zero_vector():
    vec = np.zeros((1, 384), dtype="float32")
    vec[0][0] = 1e-6   # prevents cosine divide-by-zero
    return vec


# ---------------- GET EMBEDDING ----------------
def get_embedding(text: str) -> np.ndarray:

    if not text or len(text.strip()) < 2:
        return zero_vector()

    model = get_model()

    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False
    ).astype("float32")

    return emb