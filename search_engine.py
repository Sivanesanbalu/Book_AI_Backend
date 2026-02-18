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
SIMILARITY_THRESHOLD = 0.65   # LOWER for OCR tolerance

lock = Lock()
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------- UTIL ---------------- #

def normalize(vec):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


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

def rebuild_index(books):
    """Always rebuild FAISS from DB (guaranteed sync)"""
    index = faiss.IndexFlatIP(DIM)

    if len(books) == 0:
        return index

    embeddings = []
    for b in books:
        emb = get_embedding(b["title"]).reshape(1, -1)
        emb = normalize(emb)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    return index


def get_index():
    books = load_db()

    # if index missing OR count mismatch → rebuild
    if (not os.path.exists(INDEX_PATH)):
        return rebuild_index(books)

    index = faiss.read_index(INDEX_PATH)

    if index.ntotal != len(books):
        return rebuild_index(books)

    return index


# ---------------- SEARCH ---------------- #

def search_book(text: str):

    books = load_db()
    if len(books) == 0:
        return None, 0.0

    index = get_index()

    query = get_embedding(text).reshape(1, -1)
    query = normalize(query)

    D, I = index.search(query, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if idx < len(books) and score >= SIMILARITY_THRESHOLD:
        return books[idx], score

    return None, score


# ---------------- ADD BOOK ---------------- #

def add_book(title: str):

    with lock:

        books = load_db()

        # Prevent duplicates
        for b in books:
            if title.lower() in b["title"].lower() or b["title"].lower() in title.lower():
                return b

        books.append({"title": title})
        save_db(books)

        # Always rebuild index (safe)
        rebuild_index(books)

        return {"title": title}
