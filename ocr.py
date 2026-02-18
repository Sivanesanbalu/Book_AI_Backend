import pytesseract
import cv2
import numpy as np
import os

# IMPORTANT: Render / Linux auto detect
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


############################################################
# AUTO ROTATE (fix tilted book)
############################################################
def correct_rotation(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))

    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


############################################################
# IMAGE PREPROCESSING
############################################################
def preprocess(image):

    # rotate fix
    image = correct_rotation(image)

    # upscale for better OCR
    image = cv2.resize(image, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove shadow
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # OTSU threshold (better for books)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


############################################################
# TEXT CLEANING (SMART TITLE PICK)
############################################################
def clean_text(text: str):

    lines = text.split("\n")
    candidates = []

    for line in lines:
        line = line.strip()

        if len(line) < 4:
            continue

        # ignore ISBN / numbers
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line),1)
        if alpha_ratio < 0.5:
            continue

        candidates.append(line)

    if not candidates:
        return ""

    # biggest line usually title
    candidates.sort(key=len, reverse=True)

    return candidates[0].lower()


############################################################
# MAIN OCR
############################################################
def extract_text(path: str) -> str:

    image = cv2.imread(path)

    if image is None:
        print("OCR ERROR: image not loaded")
        return ""

    processed = preprocess(image)

    raw_text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6 -l eng"
    )

    final_text = clean_text(raw_text)

    print("\nOCR DETECTED:", final_text)

    return final_text
