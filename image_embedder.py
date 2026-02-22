import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm

device = "cpu"

# Better for similarity tasks than efficientnetv2
model = timm.create_model("convnext_base", pretrained=True, num_classes=0)
model.eval()
model.to(device)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def get_image_embedding(path):
    try:
        img = Image.open(path).convert("RGB")

        # VERY IMPORTANT — remove lighting differences
        img = Image.fromarray(
            np.uint8(
                (np.array(img) - np.array(img).mean()) * 1.2 + 128
            ).clip(0,255)
        )

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy().astype("float32")

        return normalize(emb)

    except Exception as e:
        print("Embedding error:", e)
        return None