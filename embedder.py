from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

INDEX_FILE = "data/book_index.faiss"
META_FILE = "data/book_meta.pkl"

DIM = 384
THRESHOLD = 0.78   # cosine similarity threshold


# ---------------- NORMALIZE ----------------
def normalize(vec):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


# ---------------- LOAD INDEX ----------------
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatIP(DIM)  # cosine similarity
    metadata = []


# ---------------- EMBEDDING ----------------
def get_embedding(text: str) -> np.ndarray:
    emb = model.encode([text])[0].astype("float32")
    emb = normalize(np.array([emb]))
    return emb


# ---------------- ADD BOOK ----------------
def add_book(title: str):
    emb = get_embedding(title)
    index.add(emb)
    metadata.append({"title": title})
    save_index()


# ---------------- SEARCH BOOK ----------------
def search_book(title: str):
    if len(metadata) == 0 or index.ntotal == 0:
        return None, 0.0

    emb = get_embedding(title)
    D, I = index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])

    if score >= THRESHOLD:
        return metadata[idx], score

    return None, score


# ---------------- SAVE ----------------
def save_index():
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
