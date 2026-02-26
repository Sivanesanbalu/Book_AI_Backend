import os
import json
import faiss
import numpy as np
from image_embedder import get_image_embedding
from threading import Lock

DATA_FILE = "data/books.json"
DIM = 768

# tuned real world thresholds
MATCH_THRESHOLD = 0.72
DUPLICATE_THRESHOLD = 0.87

os.makedirs("data", exist_ok=True)

_books = []
_index = None
_lock = Lock()
_loaded = False   # 🔥 LAZY FLAG


# ---------------- NORMALIZE ----------------
def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------- REBUILD INDEX ----------------
def rebuild_index():
    global _index

    _index = faiss.IndexFlatIP(DIM)

    if len(_books) == 0:
        return

    vectors = []

    for b in _books:
        if "embedding" in b:
            vec = np.array(b["embedding"], dtype="float32").reshape(1, -1)
            vec = normalize(vec)
            vectors.append(vec[0])

    if len(vectors):
        _index.add(np.stack(vectors))


# ---------------- LOAD DB ----------------
def load_db():
    global _books

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            _books = json.load(f)
    else:
        _books = []

    rebuild_index()


# ---------------- ENSURE LOADED (LAZY INIT) ----------------
def ensure_loaded():
    global _loaded

    if _loaded:
        return

    print("📚 Loading book database (first request only)...")
    load_db()
    _loaded = True
    print("✅ Book DB Ready")


# ---------------- SAVE DB ----------------
def save_db():
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_books, f)
    os.replace(tmp, DATA_FILE)


# ---------------- SEARCH BOOK ----------------
def search_book(image_path):

    ensure_loaded()  # 🔥 LAZY LOAD HERE

    if _index is None or _index.ntotal == 0:
        return None, 0

    emb = get_image_embedding(image_path)
    if emb is None:
        return None, 0

    emb = normalize(emb)

    D, I = _index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if idx >= len(_books):
        return None, 0

    if score < MATCH_THRESHOLD:
        return None, score

    return _books[idx], score


# ---------------- ADD BOOK ----------------
def add_book(image_path, title):

    global _index

    ensure_loaded()  # 🔥 LAZY LOAD HERE

    emb = get_image_embedding(image_path)
    if emb is None:
        return False

    emb = normalize(emb)

    with _lock:

        if _index is not None and _index.ntotal > 0:
            D, _ = _index.search(emb, 1)
            if float(D[0][0]) > DUPLICATE_THRESHOLD:
                print("⚠️ Already exists → not adding again")
                return True

        _books.append({
            "title": title,
            "embedding": emb.flatten().tolist()
        })

        save_db()
        rebuild_index()

    print("➕ Added:", title)
    return True