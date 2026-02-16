import faiss
import json
import os
from embedder import get_embedding
import numpy as np

INDEX_PATH = "data/index.faiss"
DB_PATH = "data/books_db.json"
DIM = 384


def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        return json.load(f)


def save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(DIM)


def save_index(index):
    faiss.write_index(index, INDEX_PATH)


# 🔍 Check book already exists
def search_book(text: str):
    books = load_db()
    if len(books) == 0:
        return None, 999

    index = get_index()
    query = get_embedding(text).reshape(1, -1)

    D, I = index.search(query, 1)
    distance = float(D[0][0])
    idx = int(I[0][0])

    if distance < 0.8:
        return books[idx], distance

    return None, distance


# ➕ Add new book
def add_book(title: str):
    books = load_db()
    books.append({"title": title})
    save_db(books)

    emb = get_embedding(title).reshape(1, -1)
    index = get_index()
    index.add(emb)
    save_index(index)
