from image_embedder import get_image_embedding
from PIL import Image
import numpy as np
import os
import torch

def start_ai():
    print("🚀 Warming AI model...")

    dummy = "warmup.jpg"

    if not os.path.exists(dummy):
        img = Image.new("RGB", (224, 224), color="white")
        img.save(dummy)

    try:
        # ---- MULTIPLE PASSES (stabilize embedding) ----
        emb1 = get_image_embedding(dummy)
        emb2 = get_image_embedding(dummy)
        emb3 = get_image_embedding(dummy)

        # Force torch kernel compile + cpu optimization
        _ = np.dot(emb1, emb2.T)
        _ = np.dot(emb2, emb3.T)

        # disable torch randomness
        torch.set_num_threads(1)

        print("✅ AI Ready & Stable")

    except Exception as e:
        print("❌ Warmup failed:", e)