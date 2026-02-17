import pytesseract
import cv2
import numpy as np
from PIL import Image

# Windows path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


############################################################
# IMAGE PREPROCESSING — REAL WORLD CAMERA FIX
############################################################
def preprocess(image):

    # resize (important for OCR accuracy)
    scale_percent = 150
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # convert gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # adaptive threshold (better than binary)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    return thresh


############################################################
# TEXT CLEANING — ONLY TITLE IMPORTANT
############################################################
def clean_text(text: str):

    lines = text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()

        # ignore empty
        if len(line) < 3:
            continue

        # ignore numbers / isbn heavy lines
        if sum(c.isalpha() for c in line) < 3:
            continue

        cleaned.append(line)

    # keep first 3 lines (usually title + author)
    return " ".join(cleaned[:3]).lower()


############################################################
# MAIN OCR FUNCTION
############################################################
def extract_text(path: str) -> str:

    image = cv2.imread(path)

    processed = preprocess(image)

    raw_text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6 -l eng"
    )

    final_text = clean_text(raw_text)

    print("\nOCR DETECTED:", final_text)

    return final_text
