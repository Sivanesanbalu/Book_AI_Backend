import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import pytesseract
import re
from threading import Lock

# ---------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# IMAGE PREPROCESS (CENTER FOCUS)
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)
    if img is None:
        return None

    # resize big images (speed boost)
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # crop center (book title mostly center)
    h, w = img.shape[:2]
    crop = img[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.adaptiveThreshold(gray,255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,31,2)

    return gray


# ---------------------------------------------------
# FILTER BAD LINES
# ---------------------------------------------------
def valid(text):

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
def pick_title(lines):

    candidates = []

    for line in lines:

        if not valid(line):
            continue

        score = len(line)  # longest meaningful line wins
        candidates.append((line, score))

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
        if img is None:
            return ""

        # Tesseract config tuned for book titles
        config = r'--oem 3 --psm 6'

        data = pytesseract.image_to_string(img, config=config)

        lines = [l.strip() for l in data.split("\n") if l.strip()]

        title = pick_title(lines)

        print("DETECTED TITLE:", title)
        return title

    except Exception as e:
        print("OCR FAILED:", e)
        return ""