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

# tuned thresholds for book titles
SEMANTIC_THRESHOLD = 0.72
STRONG_DUPLICATE = 0.80
STRING_THRESHOLD = 85

lock = Lock()
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- GLOBAL CACHE ----------------
_index = None
_books_cache = None


# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(text: str) -> str:
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


# ---------------- BUILD INDEX (COSINE SIMILARITY) ----------------
def build_index(books):

    # cosine similarity index
    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 80
    index.hnsw.efConstruction = 40

    if len(books) == 0:
        return index

    embeddings = []

    for b in books:
        emb = get_embedding(b["title"])
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")
    index.add(embeddings)

    return index


# ---------------- SAFE INDEX LOAD ----------------
def load_index():
    global _index

    if _index is not None:
        return _index

    books = load_db()
    rebuild = True

    if os.path.exists(INDEX_PATH):
        try:
            idx = faiss.read_index(INDEX_PATH)

            # important safety check
            if idx.ntotal == len(books):
                _index = idx
                rebuild = False
                print("FAISS index loaded from disk")

        except:
            pass

    if rebuild:
        print("Rebuilding FAISS index safely...")
        _index = build_index(books)
        faiss.write_index(_index, INDEX_PATH)

    return _index


# ---------------- SEARCH BOOK ----------------
def search_book(text: str):

    text = normalize_text(text)

    if len(text) < 3:
        return None, 0.0

    books = load_db()
    if len(books) == 0:
        return None, 0.0

    index = load_index()

    query = get_embedding(text)
    D, I = index.search(query, 1)

    similarity = float(D[0][0])
    idx = int(I[0][0])

    if idx < 0 or idx >= len(books):
        return None, 0.0

    # semantic filter
    if similarity < SEMANTIC_THRESHOLD:
        return None, similarity

    stored_title = books[idx]["title"]

    # string verification (prevents color / design confusion)
    string_score = fuzz.token_sort_ratio(text, stored_title)

    if string_score < STRING_THRESHOLD:
        return None, similarity

    return books[idx], similarity


# ---------------- ADD BOOK ----------------
def add_book(title: str):

    global _index

    title = normalize_text(title)
    if len(title) < 3:
        return None

    with lock:

        books = load_db()

        # semantic duplicate protection
        existing, sim = search_book(title)
        if existing and sim > STRONG_DUPLICATE:
            return existing

        # strong string duplicate
        for b in books:
            if fuzz.token_sort_ratio(title, b["title"]) > 92:
                return b

        # add new book
        books.append({"title": title})
        save_db(books)

        emb = get_embedding(title)
        index = load_index()
        index.add(emb)

        faiss.write_index(index, INDEX_PATH)

        return {"title": title}