import torch
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
import timm

# -------- MODEL LOAD --------
device = "cpu"

model = timm.create_model("tf_efficientnetv2_b0", pretrained=True, num_classes=0)
model.eval()
model.to(device)


# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- NORMALIZE VECTOR --------
def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# -------- PREPROCESS --------
def enhance(img: Image.Image):
    img = img.convert("RGB")

    # contrast stabilize (very important)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.2)

    return img


# -------- MULTI VIEW EMBEDDING --------
def extract(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img).cpu().numpy().astype("float32")
    return emb


def get_image_embedding(path):

    try:
        base = Image.open(path)
        base = enhance(base)

        embeddings = []

        # 4 angle views (REAL WORLD FIX)
        for angle in [0, 90, 180, 270]:
            img = base.rotate(angle, expand=True)
            embeddings.append(extract(img))

        # slight zoom crop
        w, h = base.size
        crop = base.crop((w*0.1, h*0.1, w*0.9, h*0.9))
        embeddings.append(extract(crop))

        emb = np.mean(np.vstack(embeddings), axis=0, keepdims=True)

        return normalize(emb)

    except Exception as e:
        print("Embedding error:", e)
        return None