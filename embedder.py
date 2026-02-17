from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

INDEX_FILE = "data/book_index.faiss"
META_FILE = "data/book_meta.pkl"

dimension = 384  # MiniLM output size

# Load or create index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    metadata = []

def get_embedding(text: str) -> np.ndarray:
    emb = model.encode([text])[0]
    return emb.astype("float32")

def add_book(title: str):
    emb = get_embedding(title)
    index.add(np.array([emb]))
    metadata.append(title)
    save_index()

def search_book(title: str, threshold=1.2):
    if len(metadata) == 0:
        return None

    emb = get_embedding(title)
    D, I = index.search(np.array([emb]), 1)

    distance = D[0][0]
    idx = I[0][0]

    if distance < threshold:
        return metadata[idx]

    return None

def save_index():
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
