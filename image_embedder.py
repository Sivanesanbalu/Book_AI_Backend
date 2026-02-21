from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T
from threading import Lock
import numpy as np

device = "cpu"

_model = None
_preprocess = None
_model_lock = Lock()

def load_model():
    global _model, _preprocess

    print("🔄 Loading lightweight vision model...")

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = torch.nn.Identity()
    model.eval()

    _model = model.to(device)

    _preprocess = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    print("✅ Vision model ready")

def warmup():
    if _model is None:
        with _model_lock:
            if _model is None:
                load_model()

                img = Image.new("RGB",(224,224),"white")
                t = _preprocess(img).unsqueeze(0)

                with torch.no_grad():
                    _model(t)

def get_image_embedding(path:str):

    global _model

    if _model is None:
        warmup()

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            t = _preprocess(img).unsqueeze(0)

        with torch.no_grad():
            emb = _model(t)

        emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().astype("float32")

    except Exception as e:
        print("embedding error:",e)
        return None