import os
import json
import numpy as np
import time
from threading import Lock
from image_embedder import get_image_embedding, normalize, DIM

# Check if faiss is available
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

DATA_FILE = "data/books.json"
MATCH_THRESHOLD = 0.68
DUPLICATE_THRESHOLD = 0.88

os.makedirs("data", exist_ok=True)

_books = []
_index = None
_matrix = None  # fallback numpy matrix
_lock = Lock()
_loaded = False


# ---------------- REBUILD INDEX ----------------
def rebuild_index():
    global _index, _matrix

    if HAS_FAISS:
        _index = faiss.IndexFlatIP(DIM)
    else:
        _index = None

    if len(_books) == 0:
        _matrix = None
        return

    vectors = []
    for b in _books:
        if "embedding" in b:
            vec = np.array(b["embedding"], dtype="float32").reshape(1, -1)
            vec = normalize(vec)
            vectors.append(vec[0])

    if len(vectors):
        stacked = np.stack(vectors).astype("float32")
        if HAS_FAISS and _index is not None:
            _index.add(stacked)
        _matrix = stacked
    else:
        _matrix = None


# ---------------- LOAD DB ----------------
def load_db():
    global _books

    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                _books = json.load(f)
        except Exception as e:
            print("⚠️ Error reading books.json:", e)
            _books = []
    else:
        _books = []

    rebuild_index()


# ---------------- ENSURE LOADED ----------------
def ensure_loaded():
    global _loaded
    if _loaded:
        return

    print("📚 Loading book vector database...")
    load_db()
    _loaded = True
    print(f"✅ Book DB Ready ({len(_books)} indexed items, Engine: {'FAISS' if HAS_FAISS else 'NumPy Vector Space'})")


# ---------------- FORCE RELOAD ----------------
def force_reload():
    global _loaded
    _loaded = False
    ensure_loaded()


# ---------------- WAIT UNTIL INDEX READY ----------------
def wait_until_index_ready(timeout=5):
    start = time.time()
    while True:
        total = getattr(_index, "ntotal", 0) if HAS_FAISS and _index else (len(_matrix) if _matrix is not None else 0)
        if total == len(_books):
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(0.02)


# ---------------- SAVE DB ----------------
def save_db():
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_books, f, indent=2)
    os.replace(tmp, DATA_FILE)


# ---------------- SEARCH BOOK ----------------
def search_book(image_path):
    ensure_loaded()

    if len(_books) == 0:
        return None, 0

    emb = get_image_embedding(image_path)
    if emb is None:
        return None, 0

    emb = normalize(emb)

    if HAS_FAISS and _index is not None and getattr(_index, "ntotal", 0) > 0:
        D, I = _index.search(emb, 1)
        score = float(D[0][0])
        idx = int(I[0][0])
    elif _matrix is not None and len(_matrix) > 0:
        # Cosine similarity via dot product (both normalized)
        sims = np.dot(_matrix, emb.T).flatten()
        idx = int(np.argmax(sims))
        score = float(sims[idx])
    else:
        return None, 0

    if idx < 0 or idx >= len(_books):
        return None, 0

    if score < MATCH_THRESHOLD:
        return None, score

    return _books[idx], score


# ---------------- ADD BOOK ----------------
def add_book(image_path, title, author="", cover_url="", overview="", genre="General"):
    ensure_loaded()

    emb = get_image_embedding(image_path)
    if emb is None:
        return False

    emb = normalize(emb)

    with _lock:
        # duplicate check
        if len(_books) > 0:
            if HAS_FAISS and _index is not None and getattr(_index, "ntotal", 0) > 0:
                D, _ = _index.search(emb, 1)
                best_score = float(D[0][0])
            elif _matrix is not None and len(_matrix) > 0:
                sims = np.dot(_matrix, emb.T).flatten()
                best_score = float(np.max(sims))
            else:
                best_score = 0.0

            if best_score > DUPLICATE_THRESHOLD:
                print(f"⚠️ Book '{title}' visually exists (score: {best_score:.3f}) → updating metadata")
                # update metadata
                for b in _books:
                    if b.get("title") == title:
                        if author: b["author"] = author
                        if cover_url: b["cover_url"] = cover_url
                        if overview: b["overview"] = overview
                        if genre: b["genre"] = genre
                save_db()
                return True

        # add new book
        _books.append({
            "title": title,
            "author": author,
            "genre": genre,
            "cover_url": cover_url,
            "overview": overview,
            "embedding": emb.flatten().tolist()
        })

        save_db()
        force_reload()
        wait_until_index_ready()

    print("➕ Added to vector index:", title)
    return True