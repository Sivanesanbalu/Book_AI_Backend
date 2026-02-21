import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import pytesseract
import numpy as np
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
# IMAGE PREPROCESS (BOOK COVER OPTIMIZED)
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)
    if img is None:
        return None

    # resize (consistent OCR scale)
    h, w = img.shape[:2]
    scale = 1200 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # crop center region (titles usually center-top)
    h, w = img.shape[:2]
    crop = img[int(h*0.05):int(h*0.75), int(w*0.05):int(w*0.95)]

    # grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE improves printed fonts
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # light sharpening (keeps stylish fonts)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp


# ---------------------------------------------------
# FILTER NON-TITLE WORDS
# ---------------------------------------------------
def is_bad_text(text):

    text_low = text.lower()

    # too many numbers
    if sum(c.isdigit() for c in text) > len(text) * 0.4:
        return True

    # long numeric sequences (ISBN etc)
    if re.search(r'\d{4,}', text):
        return True

    # publisher words
    bad = ["edition","press","publisher","volume","vol","rs","inr","isbn","copyright"]
    if any(w in text_low for w in bad):
        return True

    if len(text) < 2:
        return True

    return False


# ---------------------------------------------------
# GROUP WORDS INTO LINES (DYNAMIC THRESHOLD)
# ---------------------------------------------------
def group_lines(data, img_height):

    words = []

    for i in range(len(data["text"])):

        txt = data["text"][i].strip()
        conf = int(data["conf"][i])

        if conf < 60 or is_bad_text(txt):
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        words.append((txt, x, y, w, h))

    if not words:
        return []

    words.sort(key=lambda x: x[2])

    lines = []
    current = [words[0]]

    for w in words[1:]:

        if abs(w[2] - current[-1][2]) < img_height * 0.04:
            current.append(w)
        else:
            lines.append(current)
            current = [w]

    lines.append(current)
    return lines


# ---------------------------------------------------
# PICK BEST TITLE CANDIDATES
# ---------------------------------------------------
def pick_titles(lines, img_height):

    scored = []

    for line in lines:

        text = " ".join(w[0] for w in line)
        text = normalize(text)

        word_count = len(text.split())
        if word_count < 1:
            continue

        # bigger text = title
        area = sum(w[3]*w[4] for w in line)

        # higher position = title
        avg_y = sum(w[2] for w in line)/len(line)
        position_score = 1 - (avg_y/img_height)

        # longer title bonus
        length_bonus = min(word_count / 5, 2)

        score = area * (1.2 + position_score + length_bonus)

        scored.append((text, score))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    # remove duplicates
    results = []
    for text, _ in scored:
        if all(text not in r and r not in text for r in results):
            results.append(text)
        if len(results) == 5:
            break

    return results


# ---------------------------------------------------
# MAIN OCR FUNCTION
# ---------------------------------------------------
def extract_text(path: str) -> list[str]:

    try:
        img = preprocess(path)
        if img is None:
            return []

        h = img.shape[0]

        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 11 -l eng",
            output_type=pytesseract.Output.DICT
        )

        lines = group_lines(data, h)
        titles = pick_titles(lines, h)

        print("OCR CANDIDATES:", titles)

        return titles

    except Exception as e:
        print("OCR FAILED:", e)
        return []