from PIL import Image
import torch
import open_clip
import gc
from threading import Lock

# ---------------- SETTINGS ----------------
torch.set_num_threads(1)
torch.set_grad_enabled(False)

device = "cpu"

_model = None
_preprocess = None
_model_lock = Lock()


# ---------------- LOAD MODEL ----------------
def load_model():
    global _model, _preprocess

    print("🔄 Loading CLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device
    )

    model.eval()

    _model = model
    _preprocess = preprocess

    print("✅ CLIP ready")


# called once at server start
def warmup():
    if _model is None:
        with _model_lock:
            if _model is None:
                load_model()

                # dummy run prevents first user lag
                img = Image.new("RGB", (224, 224), "white")
                image = _preprocess(img).unsqueeze(0)

                with torch.no_grad():
                    _model.encode_image(image)


# ---------------- IMAGE EMBEDDING ----------------
def get_image_embedding(image_path: str):

    global _model

    if _model is None:
        warmup()

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = _preprocess(img).unsqueeze(0)

        with torch.no_grad():
            emb = _model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        result = emb.cpu().numpy().astype("float32")

        # IMPORTANT: free RAM
        del image
        del emb
        gc.collect()

        return result

    except Exception as e:
        print("❌ embedding error:", e)
        return None