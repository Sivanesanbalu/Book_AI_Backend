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
SEMANTIC_THRESHOLD = 0.60
STRING_THRESHOLD = 82

lock = Lock()
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- GLOBAL CACHE ----------------
_index = None
_books_cache = None


# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
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


# ---------------- INDEX ----------------
def build_index(books):
    index = faiss.IndexHNSWFlat(DIM, 32)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 40

    if len(books) == 0:
        return index

    embeddings = []
    for b in books:
        emb = get_embedding(b["title"])
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    index.add(embeddings)

    return index


def load_index():
    global _index

    if _index is not None:
        return _index

    books = load_db()

    if os.path.exists(INDEX_PATH):
        try:
            _index = faiss.read_index(INDEX_PATH)
            if _index.ntotal == len(books):
                print("FAISS index loaded from disk")
                return _index
        except:
            pass

    print("Rebuilding FAISS index...")
    _index = build_index(books)
    faiss.write_index(_index, INDEX_PATH)
    return _index


# ---------------- SEARCH ----------------
def search_book(text: str):

    text = normalize_text(text)
    if len(text) < 4:
        return None, 0.0

    books = load_db()
    if len(books) == 0:
        return None, 0.0

    index = load_index()

    query = get_embedding(text)
    D, I = index.search(query, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if idx >= len(books) or score < SEMANTIC_THRESHOLD:
        return None, score

    stored_title = books[idx]["title"]

    string_score = fuzz.token_set_ratio(text, stored_title)
    if string_score < STRING_THRESHOLD:
        return None, score

    return books[idx], score


# ---------------- ADD BOOK ----------------
def add_book(title: str):

    global _index

    title = normalize_text(title)
    if len(title) < 4:
        return None

    with lock:

        books = load_db()

        for b in books:
            if fuzz.token_set_ratio(title, b["title"]) > 90:
                return b

        books.append({"title": title})
        save_db(books)

        emb = get_embedding(title)
        index = load_index()
        index.add(emb)

        faiss.write_index(index, INDEX_PATH)

        return {"title": title}