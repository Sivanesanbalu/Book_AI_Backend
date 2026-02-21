from embedder import get_model
from search_engine import load_index

print("🚀 Warming up AI models...")

get_model()      # loads sentence transformer
load_index()     # loads FAISS index

print("✅ Server ready")