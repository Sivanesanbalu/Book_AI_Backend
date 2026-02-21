
import os
import json
import faiss
import numpy as np
from threading import Lock

from embedding import get_embedding, clean_text   # 🔥 IMPORTANT

DATA_FILE = "data/books.json"
INDEX_FILE = "data/books.faiss"

os.makedirs("data", exist_ok=True)

# ---------------- GLOBAL CACHE ----------------
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


# load once
load_books()
load_index()


# ---------------- SEARCH ----------------
def search_book(text):

    global _index, _books_cache

    if _index is None or not _books_cache:
        return None, 0

    text = clean_text(text)
    emb = get_embedding(text)

    D, I = _index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    # tuned threshold (very important)
    if score < 0.45:
        return None, score

    return _books_cache[idx], score


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
