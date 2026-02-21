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
# IMAGE PREPROCESS (REAL CAMERA OPTIMIZED)
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)
    if img is None:
        return None

    # resize for OCR stability
    h, w = img.shape[:2]
    scale = 1400 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # focus on title area (top-middle)
    h, w = img.shape[:2]
    crop = img[int(h*0.05):int(h*0.65), int(w*0.05):int(w*0.95)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp


# ---------------------------------------------------
# FILTER BAD TEXT (SMART FILTER)
# ---------------------------------------------------
def is_bad_text(text):

    text_low = text.lower()

    if len(text_low) == 1 and text_low not in ["c","r"]:
        return True

    # ignore ISBN / price heavy strings
    if sum(c.isdigit() for c in text_low) > len(text_low)*0.6:
        return True

    garbage = ["edition","press","publisher","volume","vol","isbn","copyright"]
    if text_low in garbage:
        return True

    return False


# ---------------------------------------------------
# GROUP WORDS INTO LINES
# ---------------------------------------------------
def group_lines(data, img_height):

    words = []

    for i in range(len(data["text"])):

        txt = data["text"][i].strip()

        if txt == "":
            continue

        try:
            conf = int(data["conf"][i])
        except:
            continue

        # 🔥 MOBILE CAMERA THRESHOLD
        if conf < 35:
            continue

        if is_bad_text(txt):
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
# PICK TITLE CANDIDATES (REALISTIC SCORING)
# ---------------------------------------------------
def pick_titles(lines, img_height):

    scored = []

    for line in lines:

        text = " ".join(w[0] for w in line)
        text = normalize(text)

        words = text.split()
        word_count = len(words)

        # titles usually 2+ words
        if word_count < 2:
            continue

        area = sum(w[3]*w[4] for w in line)

        avg_y = sum(w[2] for w in line)/len(line)
        position_score = 1 - (avg_y/img_height)

        phrase_bonus = min(word_count * 0.8, 4)

        score = (area * 0.6) + (position_score * 8000) + (phrase_bonus * 5000)

        scored.append((text, score))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

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
            config="--oem 3 --psm 6 -l eng",
            output_type=pytesseract.Output.DICT
        )

        lines = group_lines(data, h)
        titles = pick_titles(lines, h)

        print("OCR CANDIDATES:", titles)

        return titles

    except Exception as e:
        print("OCR FAILED:", e)
        return []