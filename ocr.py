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
# IMAGE PREPROCESS
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)
    if img is None:
        return None

    # resize large images
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # focus title zone (top-middle)
    h, w = img.shape[:2]
    crop = img[int(h*0.10):int(h*0.70), int(w*0.10):int(w*0.90)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=12)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    th = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,2
    )

    return th


# ---------------------------------------------------
# TEXT FILTER
# ---------------------------------------------------
def is_bad_text(text):

    text_low = text.lower()

    if sum(c.isdigit() for c in text) > len(text) * 0.4:
        return True

    if re.search(r'\d{4,}', text):
        return True

    bad = ["edition","press","publisher","volume","vol","rs","inr","isbn"]
    if any(w in text_low for w in bad):
        return True

    if len(text) < 3:
        return True

    return False


# ---------------------------------------------------
# GROUP WORDS INTO LINES
# ---------------------------------------------------
def group_lines(data):

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
        if abs(w[2] - current[-1][2]) < 35:
            current.append(w)
        else:
            lines.append(current)
            current = [w]

    lines.append(current)
    return lines


# ---------------------------------------------------
# PICK MULTIPLE TITLES (MAIN FIX)
# ---------------------------------------------------
def pick_titles(lines, img_height):

    scored = []

    for line in lines:

        text = " ".join(w[0] for w in line)
        text = normalize(text)

        word_count = len(text.split())
        if word_count < 2:
            continue

        area = sum(w[3]*w[4] for w in line)

        avg_y = sum(w[2] for w in line)/len(line)
        position_score = 1 - (avg_y/img_height)

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
# MAIN OCR
# ---------------------------------------------------
def extract_text(path: str) -> list[str]:
    try:
        img = preprocess(path)
        if img is None:
            return []

        h = img.shape[0]

        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT
        )

        lines = group_lines(data)

        titles = pick_titles(lines, h)

        print("OCR CANDIDATES:", titles)

        return titles

    except Exception as e:
        print("OCR FAILED:", e)
        return []