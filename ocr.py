import os
os.environ["OMP_NUM_THREADS"] = "1"

from paddleocr import PaddleOCR
import cv2
import re
from threading import Lock

# ---------------------------------------------------
# SAFE LAZY LOAD MODEL (IMPORTANT FOR RENDER)
# ---------------------------------------------------
_ocr = None
_lock = Lock()

def get_ocr():
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                print("Loading OCR model...")
                _ocr = PaddleOCR(
                    use_angle_cls=False,   # faster
                    lang="en",
                    show_log=False,
                    use_gpu=False
                )
                print("OCR ready")
    return _ocr


# ---------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# IMAGE PREPROCESS (VERY IMPORTANT)
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)

    # resize big images (massive speed boost)
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # focus center (title usually center)
    h, w = img.shape[:2]
    crop = img[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    return gray


# ---------------------------------------------------
# FILTER BAD LINES
# ---------------------------------------------------
def valid(text, conf):

    if conf < 0.60:
        return False

    if len(text) < 3:
        return False

    if re.search(r'\d{10,13}', text):
        return False

    if re.search(r'\b(rs|inr|\$|edition|ed\.)\b', text.lower()):
        return False

    return True


# ---------------------------------------------------
# SMART TITLE PICKER
# ---------------------------------------------------
def pick_title(detections):

    candidates = []

    for det in detections:
        box = det[0]
        text = det[1][0].strip()
        conf = float(det[1][1])

        if not valid(text, conf):
            continue

        # height importance (title usually biggest)
        height = abs(box[0][1] - box[2][1])
        score = len(text) * conf * (height + 1)

        candidates.append((text, score))

    if not candidates:
        return ""

    best = max(candidates, key=lambda x: x[1])[0]
    return normalize(best)


# ---------------------------------------------------
# MAIN OCR FUNCTION
# ---------------------------------------------------
def extract_text(path: str) -> str:
    try:
        img = preprocess(path)

        ocr = get_ocr()
        result = ocr.ocr(img)

        if not result or not result[0]:
            return ""

        title = pick_title(result[0])

        print("DETECTED TITLE:", title)
        return title

    except Exception as e:
        print("OCR FAILED:", e)
        return ""