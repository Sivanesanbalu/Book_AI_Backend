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
# TEXT NORMALIZATION (BALANCED)
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
        "netw0rk":"network",
        "macnine":"machine",
        "powere":"power",
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    # remove symbols
    text = re.sub(r'[^a-z0-9 ]', ' ', text)

    words = text.split()

    # publishing noise only (SAFE)
    stop_words = {
        "edition","international","student","version",
        "volume","vol","publication","press","publisher",
        "education","asia","india"
    }

    words = [w for w in words if w not in stop_words]

    # limit words but don't destroy meaning
    words = words[:10]

    text = re.sub(r'\s+', ' ', " ".join(words)).strip()

    return text


# ---------------------------------------------------
# LOAD MODEL (THREAD SAFE)
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
# VALID TITLE CHECK (RELAXED)
# ---------------------------------------------------
def is_valid_title(text: str):

    if not text:
        return False

    words = text.split()

    # allow single word titles
    if len(words) == 0:
        return False

    digit_ratio = sum(c.isdigit() for c in text) / max(len(text),1)

    # relaxed rule (important)
    if digit_ratio > 0.65:
        return False

    return True


# ---------------------------------------------------
# EMBEDDING FUNCTION (SAFE)
# ---------------------------------------------------
def get_embedding(text: str) -> np.ndarray:

    text = clean_text(text)

    if not is_valid_title(text):
        # return neutral vector (prevents faiss crash)
        return np.zeros((1,384), dtype="float32")

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