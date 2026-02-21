import os
import json
import faiss
import numpy as np
from image_embedder import get_image_embedding
from threading import Lock

DATA_FILE = "data/books.json"
INDEX_FILE = "data/books.faiss"

DIM = 512
MATCH_THRESHOLD = 0.88      # real world match
DUPLICATE_THRESHOLD = 0.94  # exact same cover

os.makedirs("data", exist_ok=True)

_books = []
_index = None
_lock = Lock()


# ---------------- SAFE LOAD ----------------
def rebuild_index():
    global _index

    _index = faiss.IndexFlatIP(DIM)

    if len(_books) == 0:
        return

    vectors = []

    for b in _books:
        if "embedding" in b:
            vectors.append(np.array(b["embedding"], dtype="float32"))

    if len(vectors):
        _index.add(np.stack(vectors))


def load_db():
    global _books
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            _books = json.load(f)
    else:
        _books = []

    rebuild_index()


def save_db():
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_books, f)
    os.replace(tmp, DATA_FILE)


load_db()


# ---------------- SEARCH ----------------
def search_book(image_path):

    if _index.ntotal == 0:
        return None, 0

    emb = get_image_embedding(image_path)
    if emb is None:
        return None, 0

    D, I = _index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if idx >= len(_books):
        return None, 0

    if score < MATCH_THRESHOLD:
        return None, score

    return _books[idx], score


# ---------------- ADD ----------------
def add_book(image_path, title):

    global _index

    emb = get_image_embedding(image_path)
    if emb is None:
        return False

    with _lock:

        # duplicate detection
        if _index.ntotal > 0:
            D, _ = _index.search(emb, 1)
            if float(D[0][0]) > DUPLICATE_THRESHOLD:
                print("⚠️ Duplicate book ignored")
                return False

        # store embedding INSIDE JSON (very important)
        _books.append({
            "title": title,
            "embedding": emb.flatten().tolist()
        })

        save_db()

        # rebuild index (guaranteed sync)
        rebuild_index()

    print("➕ Added:", title)
    return True