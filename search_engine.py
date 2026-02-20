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
SEMANTIC_THRESHOLD = 0.60     # FAISS meaning similarity
STRING_THRESHOLD = 82         # title verification

lock = Lock()
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------- TEXT NORMALIZATION ---------------- #

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- VECTOR NORMALIZATION ---------------- #

def normalize(vec):
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return vec / norm


# ---------------- DB ---------------- #

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------- INDEX (FAST SEARCH) ---------------- #

def build_hnsw():
    index = faiss.IndexHNSWFlat(DIM, 32)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 40
    return index


def rebuild_index(books):
    index = build_hnsw()

    if len(books) == 0:
        faiss.write_index(index, INDEX_PATH)
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

    if not os.path.exists(INDEX_PATH):
        return rebuild_index(books)

    index = faiss.read_index(INDEX_PATH)

    if index.ntotal != len(books):
        return rebuild_index(books)

    return index


# ---------------- SEARCH (AI + VERIFY) ---------------- #

def search_book(text: str):

    text = normalize_text(text)

    if len(text) < 4:
        return None, 0.0

    books = load_db()
    if len(books) == 0:
        return None, 0.0

    index = get_index()

    query = get_embedding(text).reshape(1, -1)
    query = normalize(query)

    D, I = index.search(query, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if idx >= len(books) or score < SEMANTIC_THRESHOLD:
        return None, score

    # -------- SECOND VERIFICATION (IMPORTANT) --------
    stored_title = books[idx]["title"]

    string_score = fuzz.token_set_ratio(text, stored_title)

    if string_score < STRING_THRESHOLD:
        return None, score

    return books[idx], score


# ---------------- ADD BOOK ---------------- #

def add_book(title: str):

    title = normalize_text(title)

    if len(title) < 4:
        return None

    with lock:

        books = load_db()

        # prevent duplicates strongly
        for b in books:
            if fuzz.token_set_ratio(title, b["title"]) > 90:
                return b

        books.append({"title": title})
        save_db(books)

        rebuild_index(books)

        return {"title": title}