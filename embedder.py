import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock
import torch
import re

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = Lock()
_encode_lock = Lock()   # prevents concurrent encode crash


# ---------------------------------------------------
# TEXT NORMALIZATION (OCR HARDENED)
# ---------------------------------------------------
def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.lower()

    # common OCR mistakes (expandable dictionary)
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

    # remove punctuation garbage
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # remove meaningless tokens
    stop_words = {
        "edition","third","fourth","fifth","sixth","seventh",
        "international","student","version","volume",
        "vol","part","series","publication","press","publisher",
        "pearson","mcgraw","wiley","oxford","university"
    }

    words = [w for w in text.split() if w not in stop_words and len(w) > 2]

    return " ".join(words)


# ---------------------------------------------------
# SAFE MODEL LOAD (only once)
# ---------------------------------------------------
def get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                print("🔄 Loading embedding model...")

                torch.set_num_threads(1)  # CRITICAL for CPU stability

                _model = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu"
                )

                _model.max_seq_length = 32
                _model.eval()

                print("✅ Embedding model loaded")

    return _model


# ---------------------------------------------------
# SAFE ZERO VECTOR (prevents FAISS crash)
# ---------------------------------------------------
def zero_vector():
    vec = np.zeros((1, 384), dtype="float32")
    vec[0][0] = 1e-6
    return vec


# ---------------------------------------------------
# VALIDATE TEXT QUALITY
# ---------------------------------------------------
def is_low_quality(text: str):

    if not text:
        return True

    words = text.split()

    if len(words) < 2:
        return True

    # mostly numbers
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.35:
        return True

    # repeated characters like "aaaaa"
    if len(set(text)) < len(text) * 0.3:
        return True

    return False


# ---------------------------------------------------
# MAIN EMBEDDING FUNCTION
# ---------------------------------------------------
def get_embedding(text: str) -> np.ndarray:

    if not text:
        return zero_vector()

    text = clean_text(text)

    if is_low_quality(text):
        return zero_vector()

    model = get_model()

    # SentenceTransformer is NOT thread safe → lock required
    with _encode_lock:
        emb = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False
        ).astype("float32")

    # stabilize similarity across scans
    emb = np.round(emb, 6)

    return emb