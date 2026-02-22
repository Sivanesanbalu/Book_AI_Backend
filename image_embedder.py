import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ---------------- MEMORY SAFE SETTINGS ----------------
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

device = "cpu"

# -------- LIGHTWEIGHT BUT ACCURATE MODEL --------
model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
model.eval()
model.to(device)


# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- NORMALIZE VECTOR --------
def normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    return v / norm


# -------- LIGHTING NORMALIZATION --------
def remove_lighting(img):
    arr = np.array(img).astype("float32")
    mean = arr.mean()
    arr = (arr - mean) * 1.15 + 128
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype("uint8"))


# -------- EMBEDDING --------
def extract(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy().astype("float32")
    return emb


def get_image_embedding(path):
    try:
        # safe open (prevents memory leak)
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
        img = img.copy()

        # remove lighting differences
        img = remove_lighting(img)

        embeddings = []

        # original
        embeddings.append(extract(img))

        # slight zoom crop (real world camera variation)
        w, h = img.size
        crop = img.crop((w*0.12, h*0.12, w*0.88, h*0.88))
        embeddings.append(extract(crop))

        emb = np.mean(np.vstack(embeddings), axis=0, keepdims=True)

        del img
        return normalize(emb)

    except Exception as e:
        print("Embedding error:", e)
        return None