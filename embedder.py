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
# TEXT NORMALIZATION (OCR HARDENED)
# ---------------------------------------------------
def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.lower()

    # common OCR mistakes
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
    text = re.sub(r'\s+', ' ', text).strip()

    # remove useless publishing words
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

                torch.set_num_threads(1)

                _model = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu"
                )

                _model.max_seq_length = 64
                _model.eval()

                print("✅ Embedding model loaded")

    return _model


# ---------------------------------------------------
# SAFE ZERO VECTOR
# ---------------------------------------------------
def zero_vector():
    return np.zeros((1, 384), dtype="float32")


# ---------------------------------------------------
# LIGHT QUALITY CHECK (DO NOT DESTROY TEXT)
# ---------------------------------------------------
def soft_filter(text: str):

    if not text:
        return ""

    words = text.split()

    # keep at least 2 words
    if len(words) < 2:
        return text

    # trim too long OCR garbage
    if len(text) > 120:
        text = " ".join(words[:12])

    return text


# ---------------------------------------------------
# MAIN EMBEDDING FUNCTION
# ---------------------------------------------------
def get_embedding(text: str) -> np.ndarray:

    if not text:
        return zero_vector()

    text = clean_text(text)
    text = soft_filter(text)

    if not text:
        return zero_vector()

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