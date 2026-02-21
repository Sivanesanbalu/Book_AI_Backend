import os
import json
import faiss
import numpy as np
from rapidfuzz import fuzz

from embedding import get_embedding, clean_text

DATA_FILE = "data/books.json"
INDEX_FILE = "data/books.faiss"

os.makedirs("data", exist_ok=True)

_books_cache = []
_index = None


# ---------------- LOAD DATABASE ----------------
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


# ---------------- BUILD FAISS INDEX ----------------
def build_index():

    global _index
    books = load_books()

    vectors = []
    valid_books = []

    for b in books:
        emb = get_embedding(b["title"])

        # skip garbage titles
        if emb is None:
            continue

        vectors.append(emb[0])
        valid_books.append(b)

    _books_cache[:] = valid_books

    if not vectors:
        _index = None
        return None

    emb = np.array(vectors).astype("float32")

    _index = faiss.IndexFlatIP(384)
    _index.add(emb)

    faiss.write_index(_index, INDEX_FILE)

    print("📚 Index built:", len(valid_books), "books")
    return _index


def load_index():
    global _index

    if os.path.exists(INDEX_FILE):
        _index = faiss.read_index(INDEX_FILE)
    else:
        build_index()

    return _index


# load on startup
load_books()
load_index()


# ---------------- HYBRID SEARCH ----------------
def search_book(text):

    global _index, _books_cache

    if _index is None or not _books_cache:
        return None, 0

    query = clean_text(text)
    emb = get_embedding(query)

    if emb is None:
        return None, 0

    D, I = _index.search(emb, min(5, len(_books_cache)))

    best_book = None
    best_score = 0

    for sim, idx in zip(D[0], I[0]):

        book = _books_cache[int(idx)]
        title = clean_text(book["title"])

        semantic = float(sim)
        fuzzy = fuzz.token_set_ratio(query, title) / 100

        score = (semantic * 0.75) + (fuzzy * 0.25)

        if score > best_score:
            best_score = score
            best_book = book

    # STRICT threshold (prevents false positive)
    if best_score < 0.70:
        return None, best_score

    return best_book, best_score


# ---------------- DUPLICATE CHECK ----------------
def is_duplicate(title):

    book, score = search_book(title)

    if book and score > 0.82:
        print("📕 Duplicate:", book["title"], "score:", score)
        return True

    return False


# ---------------- ADD BOOK ----------------
def add_book(title):

    global _books_cache

    title = clean_text(title)

    # reject weak titles
    if len(title.split()) < 2:
        print("⚠️ Ignored weak OCR:", title)
        return

    if is_duplicate(title):
        return

    _books_cache.append({"title": title})
    save_books(_books_cache)
    build_index()

    print("➕ Added:", title)