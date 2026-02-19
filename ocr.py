
import pytesseract
import cv2
import numpy as np
import os
import shutil
import re

# ---------------- AUTO DETECT TESSERACT ----------------
def configure_tesseract():
    # Windows
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        return

    # Linux (Render / Docker)
    possible_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

    raise RuntimeError("Tesseract not installed in system!")

configure_tesseract()


############################################################
# AUTO ROTATE
############################################################
def correct_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))

    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


############################################################
# PREPROCESS
############################################################
def preprocess(image):
    image = correct_rotation(image)
    image = cv2.resize(image, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


############################################################
# CLEAN TEXT
############################################################
def clean_text(text: str):

    lines = text.split("\n")
    candidates = []

    for line in lines:
        line = line.strip()

        if len(line) < 4:
            continue

        line = re.sub(r'[^A-Za-z0-9 ]', '', line)
        line = re.sub(r'\s+', ' ', line)

        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line),1)
        if alpha_ratio < 0.55:
            continue

        candidates.append(line)

    if not candidates:
        return ""

    candidates.sort(key=len, reverse=True)
    return candidates[0].lower().strip()


############################################################
# MAIN OCR
############################################################
def extract_text(path: str) -> str:

    image = cv2.imread(path)

    if image is None:
        return ""

    processed = preprocess(image)

    raw_text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6 -l eng"
    )

    final_text = clean_text(raw_text)

    print("OCR DETECTED:", final_text)

    return final_text
