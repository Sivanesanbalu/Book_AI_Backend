from image_embedder import get_image_embedding
from PIL import Image
import os

def start_ai():
    print("🚀 Warming AI model...")

    dummy = "warmup.jpg"

    if not os.path.exists(dummy):
        img = Image.new("RGB", (224, 224), color="white")
        img.save(dummy)

    try:
        get_image_embedding(dummy)
        print("✅ AI Ready")
    except Exception as e:
        print("❌ Warmup failed:", e)