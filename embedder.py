import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock
import re

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = Lock()


# ---------------------------------------------------
# TEXT NORMALIZATION (VERY IMPORTANT FOR OCR)
# ---------------------------------------------------
def clean_text(text: str) -> str:

    text = text.lower()

    # common OCR mistakes
    text = text.replace("systern", "system")
    text = text.replace("cornputer", "computer")
    text = text.replace("operatng", "operating")

    # remove junk characters
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # remove useless words
    stop_words = {
        "edition","third","fourth","fifth","sixth",
        "international","student","version","volume",
        "vol","part","series","publication","press"
    }

    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)


# ---------------------------------------------------
# SAFE MODEL LOAD (Render friendly)
# ---------------------------------------------------
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

                # very important for memory + OCR stability
                _model.max_seq_length = 32

                print("✅ Embedding model loaded")

    return _model


# ---------------------------------------------------
# SAFE ZERO VECTOR
# ---------------------------------------------------
def zero_vector():
    vec = np.zeros((1, 384), dtype="float32")
    vec[0][0] = 1e-6
    return vec


# ---------------------------------------------------
# MAIN EMBEDDING FUNCTION
# ---------------------------------------------------
def get_embedding(text: str) -> np.ndarray:

    if not text or len(text.strip()) < 2:
        return zero_vector()

    text = clean_text(text)

    if len(text) < 2:
        return zero_vector()

    model = get_model()

    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False
    ).astype("float32")

    # stability fix (prevents tiny drift between scans)
    emb = np.round(emb, 6)

    return emb