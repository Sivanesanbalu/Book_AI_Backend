import os
import json
import faiss
import numpy as np
from threading import Lock
from rapidfuzz import fuzz

from embedding import get_embedding, clean_text

DATA_FILE = "data/books.json"
INDEX_FILE = "data/books.faiss"

os.makedirs("data", exist_ok=True)

_books_cache = []
_index = None
_lock = Lock()


# ---------------- LOAD DB ----------------
def load_books():
    global _books_cache

    if not os.path.exists(DATA_FILE):
        _books_cache = []
        return []

    with open(DATA_FILE, "r", encoding="utf8") as f:
        _books_cache = json.load(f)

    return _books_cache


def save_books(books):
    with open(DATA_FILE, "w", encoding="utf8") as f:
        json.dump(books, f, indent=2)


# ---------------- BUILD INDEX ----------------
def build_index():

    global _index

    books = load_books()

    if not books:
        _index = None
        return None

    vectors = []
    for b in books:
        vec = get_embedding(b["title"])
        vectors.append(vec[0])

    emb = np.array(vectors).astype("float32")

    _index = faiss.IndexFlatIP(384)
    _index.add(emb)

    faiss.write_index(_index, INDEX_FILE)

    print("📚 FAISS index rebuilt:", len(vectors), "books")

    return _index


def load_index():
    global _index

    if os.path.exists(INDEX_FILE):
        _index = faiss.read_index(INDEX_FILE)
    else:
        build_index()

    return _index


load_books()
load_index()


# ---------------- HYBRID SEARCH ----------------
def search_book(text):

    global _index, _books_cache

    if _index is None or not _books_cache:
        return None, 0

    query = clean_text(text)
    emb = get_embedding(query)

    # ----- semantic search -----
    D, I = _index.search(emb, min(5, len(_books_cache)))

    best_book = None
    best_score = 0

    for sim, idx in zip(D[0], I[0]):

        book = _books_cache[int(idx)]
        title = clean_text(book["title"])

        semantic_score = float(sim)              # 0 → 1
        fuzzy_score = fuzz.token_set_ratio(query, title) / 100  # 0 → 1

        # combined confidence
        final_score = (semantic_score * 0.65) + (fuzzy_score * 0.35)

        if final_score > best_score:
            best_score = final_score
            best_book = book

    if best_score < 0.55:
        return None, best_score

    return best_book, best_score


# ---------------- ADD BOOK ----------------
def add_book(title):

    global _books_cache

    title = clean_text(title)

    for b in _books_cache:
        if clean_text(b["title"]) == title:
            return

    _books_cache.append({"title": title})
    save_books(_books_cache)

    build_index()

    print("➕ Added book:", title)