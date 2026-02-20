import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import pytesseract
import re


# ---------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# IMAGE PREPROCESS (TITLE REGION FOCUS)
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)
    if img is None:
        return None

    # Resize large images (huge speed boost)
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # Focus middle 70% (title area)
    h, w = img.shape[:2]
    crop = img[int(h*0.15):int(h*0.80), int(w*0.08):int(w*0.92)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=15)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    # adaptive threshold handles dark/light covers
    th = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,2
    )

    return th


# ---------------------------------------------------
# REMOVE JUNK TEXT
# ---------------------------------------------------
def valid(text, conf):

    if conf < 60:
        return False

    if len(text) < 3:
        return False

    # remove numbers (isbn/price)
    if re.search(r'\d{3,}', text):
        return False

    # remove common noise words
    bad_words = ["edition","press","publisher","volume","vol","rs","inr"]
    if any(w in text.lower() for w in bad_words):
        return False

    return True


# ---------------------------------------------------
# PICK TITLE USING BIGGEST TEXT AREA
# ---------------------------------------------------
def extract_text(path: str) -> str:
    try:
        img = preprocess(path)
        if img is None:
            return ""

        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT
        )

        candidates = []

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if not valid(text, conf):
                continue

            w = data["width"][i]
            h = data["height"][i]

            # KEY IDEA → title = biggest visible text
            score = w * h * conf
            candidates.append((text, score, data["top"][i]))

        if not candidates:
            return ""

        # sort top-to-bottom, then group lines
        candidates.sort(key=lambda x: x[2])

        # combine nearby lines (multi-line titles)
        final_lines = []
        current_line = [candidates[0]]

        for i in range(1, len(candidates)):
            if abs(candidates[i][2] - current_line[-1][2]) < 40:
                current_line.append(candidates[i])
            else:
                final_lines.append(current_line)
                current_line = [candidates[i]]

        final_lines.append(current_line)

        # choose group with largest total score
        best_group = max(final_lines, key=lambda g: sum(x[1] for x in g))

        title = " ".join(x[0] for x in best_group)

        title = normalize(title)

        print("DETECTED TITLE:", title)
        return title

    except Exception as e:
        print("OCR FAILED:", e)
        return ""