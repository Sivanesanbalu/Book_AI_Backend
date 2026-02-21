import faiss
import json
import os
import numpy as np
from embedder import get_embedding
from threading import Lock
import re
from rapidfuzz import fuzz

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DB_PATH = os.path.join(DATA_DIR, "books_db.json")

DIM = 384

# tuned for OCR noise
SEMANTIC_ACCEPT = 0.65
FALLBACK_ACCEPT = 120
STRONG_DUPLICATE = 0.70
STRING_MIN = 70

lock = Lock()
os.makedirs(DATA_DIR, exist_ok=True)

_index = None
_books_cache = None


# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- LOAD DATABASE ----------------
def load_db():
    global _books_cache

    if _books_cache is not None:
        return _books_cache

    if not os.path.exists(DB_PATH):
        _books_cache = []
        return _books_cache

    with open(DB_PATH, "r", encoding="utf-8") as f:
        _books_cache = json.load(f)

    return _books_cache


def save_db(data):
    global _books_cache
    _books_cache = data
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------- BUILD INDEX ----------------
def build_index(books):

    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 80
    index.hnsw.efConstruction = 40

    if len(books) == 0:
        return index

    embeddings = []
    for b in books:
        embeddings.append(get_embedding(b["title"]))

    embeddings = np.vstack(embeddings).astype("float32")
    index.add(embeddings)

    return index


# ---------------- LOAD INDEX SAFELY ----------------
def load_index():
    global _index

    if _index is not None:
        return _index

    books = load_db()
    rebuild = True

    if os.path.exists(INDEX_PATH):
        try:
            idx = faiss.read_index(INDEX_PATH)
            if idx.ntotal == len(books):
                _index = idx
                rebuild = False
                print("FAISS loaded")
        except:
            pass

    if rebuild:
        print("Rebuilding FAISS index")
        _index = build_index(books)
        faiss.write_index(_index, INDEX_PATH)

    return _index


# ---------------- SEARCH BOOK (FIXED) ----------------
def search_book(text: str):

    text = normalize_text(text)

    if len(text) < 3:
        return None, 0.0

    books = load_db()
    if len(books) == 0:
        return None, 0.0

    index = load_index()

    query = get_embedding(text)

    # 🔥 search top 3 candidates
    k = min(3, len(books))
    D, I = index.search(query, k)

    best_match = None
    best_score = 0

    for rank in range(k):

        idx = int(I[0][rank])
        if idx < 0 or idx >= len(books):
            continue

        similarity = float(D[0][rank])
        stored_title = books[idx]["title"]

        string_score = fuzz.token_set_ratio(text, stored_title)

        # -------- Primary Accept --------
        if similarity >= SEMANTIC_ACCEPT and string_score >= STRING_MIN:
            return books[idx], similarity

        # -------- Fallback Score --------
        combined = similarity * 100 + string_score
        if combined > best_score:
            best_score = combined
            best_match = (books[idx], similarity)

    # -------- Final fallback --------
    if best_match and best_score > FALLBACK_ACCEPT:
        return best_match

    return None, 0.0


# ---------------- ADD BOOK ----------------
def add_book(title: str):

    title = normalize_text(title)
    if len(title) < 3:
        return None

    with lock:

        books = load_db()

        # check duplicate
        existing, sim = search_book(title)
        if existing and sim > STRONG_DUPLICATE:
            return existing

        # strong string duplicate
        for b in books:
            if fuzz.token_set_ratio(title, b["title"]) > 92:
                return b

        # add new
        books.append({"title": title})
        save_db(books)

        emb = get_embedding(title)
        index = load_index()
        index.add(emb)

        faiss.write_index(index, INDEX_PATH)

        return {"title": title}