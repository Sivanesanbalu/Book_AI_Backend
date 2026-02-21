import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock
import torch
import re

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = Lock()
_encode_lock = Lock()


# ---------------------------------------------------
# TEXT NORMALIZATION (BALANCED — NOT TOO HARD)
# ---------------------------------------------------
def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.lower()

    # OCR corrections
    fixes = {
        "systern":"system",
        "cornputer":"computer",
        "operatng":"operating",
        "lernng":"learning",
        "machne":"machine",
        "artifcial":"artificial",
        "inteligence":"intelligence",
        "pyth0n":"python",
        "alg0rithm":"algorithm",
        "databse":"database",
        "netw0rk":"network"
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    # remove symbols
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    words = text.split()

    # remove publishing noise (but KEEP title meaning words)
    stop_words = {
        "edition","third","fourth","fifth","sixth","seventh",
        "international","student","version","volume",
        "vol","part","series","publication","press","publisher",
        "education","asia","india"
    }

    words = [w for w in words if w not in stop_words and len(w) > 2]

    # keep max 8 words (important: don't over trim)
    words = words[:8]

    return " ".join(words)


# ---------------------------------------------------
# LOAD MODEL (ONLY ONCE — THREAD SAFE)
# ---------------------------------------------------
def get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                print("🔄 Loading embedding model...")

                torch.set_num_threads(1)

                _model = SentenceTransformer(MODEL_NAME, device="cpu")
                _model.max_seq_length = 64
                _model.eval()

                print("✅ Embedding model loaded")

    return _model


# ---------------------------------------------------
# VALID TITLE CHECK (prevents garbage embedding)
# ---------------------------------------------------
def is_valid_title(text: str):

    if not text:
        return False

    words = text.split()

    if len(words) < 2:
        return False

    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.40:
        return False

    return True


# ---------------------------------------------------
# EMBEDDING FUNCTION (FINAL STABLE)
# ---------------------------------------------------
def get_embedding(text: str) -> np.ndarray:

    text = clean_text(text)

    if not is_valid_title(text):
        return None   # IMPORTANT: never send fake vector

    model = get_model()

    with _encode_lock:
        emb = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False
        ).astype("float32")

    return emb