import os
import json
import faiss
import numpy as np
from rapidfuzz import fuzz

from embedder import get_embedding, clean_text

# Prevent CPU overload on Render
faiss.omp_set_num_threads(1)

DATA_FILE = "data/books.json"
INDEX_FILE = "data/books.faiss"

os.makedirs("data", exist_ok=True)

_books_cache = []
_index = None
DIM = 384


# ---------------------------------------------------------
# VALID TITLE CHECK  ⭐ IMPORTANT
# ---------------------------------------------------------
def is_valid_title(title: str):
    words = title.split()

    if len(words) < 2:
        return False

    # reject garbage OCR
    bad = ["and", "the", "for", "with", "from", "into", "book"]
    if sum(w in bad for w in words) >= len(words) - 1:
        return False

    if len(title) < 6:
        return False

    return True


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


# ---------------- BUILD INDEX ----------------
def build_index():

    global _index, _books_cache

    books = load_books()

    vectors = []
    valid_books = []

    for b in books:
        title = clean_text(b["title"])

        if not is_valid_title(title):
            continue

        emb = get_embedding(title)
        if emb is None:
            continue

        vectors.append(emb[0])
        valid_books.append({"title": title})

    _books_cache = valid_books

    if not vectors:
        _index = None
        return None

    emb = np.array(vectors).astype("float32")

    _index = faiss.IndexFlatIP(DIM)
    _index.add(emb)

    faiss.write_index(_index, INDEX_FILE)

    print("📚 Index built:", len(valid_books), "books")
    return _index


def load_index():
    global _index

    if os.path.exists(INDEX_FILE):
        _index = faiss.read_index(INDEX_FILE)
        print("⚡ FAISS index loaded instantly")
    else:
        build_index()

    return _index


# Load at startup
load_books()
load_index()


# ---------------- HYBRID SEARCH ----------------
def search_book(text):

    global _index, _books_cache

    if _index is None or not _books_cache:
        return None, 0

    query = clean_text(text)

    if not is_valid_title(query):
        return None, 0

    emb = get_embedding(query)
    if emb is None:
        return None, 0

    k = min(5, len(_books_cache))
    D, I = _index.search(emb, k)

    best_book = None
    best_score = 0

    for sim, idx in zip(D[0], I[0]):

        book = _books_cache[int(idx)]
        title = book["title"]

        semantic = float(sim)
        fuzzy = fuzz.token_set_ratio(query, title) / 100

        # ⭐ balanced hybrid score
        score = (semantic * 0.72) + (fuzzy * 0.28)

        if score > best_score:
            best_score = score
            best_book = book

    if best_score < 0.68:
        return None, best_score

    return best_book, best_score


# ---------------- DUPLICATE CHECK ----------------
def is_duplicate(title):

    book, score = search_book(title)

    if book and score > 0.80:
        print("📕 Duplicate:", book["title"], "score:", score)
        return True

    return False


# ---------------- INSTANT ADD ----------------
def add_book(title):

    global _books_cache, _index

    title = clean_text(title)

    if not is_valid_title(title):
        print("⚠️ Ignored weak OCR:", title)
        return

    if is_duplicate(title):
        return

    emb = get_embedding(title)
    if emb is None:
        return

    # save JSON
    _books_cache.append({"title": title})
    save_books(_books_cache)

    # FAISS incremental add
    if _index is None:
        _index = faiss.IndexFlatIP(DIM)

    _index.add(np.array([emb[0]], dtype="float32"))
    faiss.write_index(_index, INDEX_FILE)

    print("➕ Added instantly:", title)