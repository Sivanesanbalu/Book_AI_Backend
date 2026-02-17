import faiss
import json
import os
import numpy as np
from embedder import get_embedding
from threading import Lock

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DB_PATH = os.path.join(DATA_DIR, "books_db.json")

DIM = 384
SIMILARITY_THRESHOLD = 0.78   # tuned for book titles

lock = Lock()

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------- DB ---------------- #

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------- INDEX ---------------- #

def get_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)

    # Cosine similarity index
    index = faiss.IndexFlatIP(DIM)
    return index


def save_index(index):
    faiss.write_index(index, INDEX_PATH)


def normalize(vec):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


# ---------------- SEARCH ---------------- #

def search_book(text: str):

    books = load_db()
    if len(books) == 0:
        return None, 0.0

    index = get_index()

    query = get_embedding(text).reshape(1, -1)
    query = normalize(query)

    if index.ntotal == 0:
        return None, 0.0

    D, I = index.search(query, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if score >= SIMILARITY_THRESHOLD:
        return books[idx], score

    return None, score


# ---------------- ADD BOOK ---------------- #

def add_book(title: str):

    with lock:

        books = load_db()
        index = get_index()

        emb = get_embedding(title).reshape(1, -1)
        emb = normalize(emb)

        index.add(emb)

        books.append({
            "title": title
        })

        save_db(books)
        save_index(index)
