import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import pytesseract
import re
from collections import Counter

# ---------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# AUTO ROTATE (critical for book spines)
# ---------------------------------------------------
def auto_rotate(img):

    try:
        osd = pytesseract.image_to_osd(img)
        angle = int(re.search('Rotate: (\d+)', osd).group(1))

        if angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    except:
        pass

    return img


# ---------------------------------------------------
# MULTI PREPROCESS
# ---------------------------------------------------
def preprocess_versions(path):

    img = cv2.imread(path)
    if img is None:
        return []

    img = auto_rotate(img)

    h, w = img.shape[:2]
    crop = img[int(h*0.10):int(h*0.75), int(w*0.10):int(w*0.90)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    versions = []

    # normal
    versions.append(gray)

    # sharpen
    versions.append(cv2.convertScaleAbs(gray, alpha=1.8, beta=20))

    # threshold
    versions.append(cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,2
    ))

    # inverted (very important for dark covers)
    versions.append(cv2.bitwise_not(gray))

    return versions


# ---------------------------------------------------
# VALID TEXT FILTER
# ---------------------------------------------------
def is_valid_candidate(text):

    if len(text) < 4:
        return False

    if sum(c.isdigit() for c in text) > len(text)*0.4:
        return False

    bad = ["edition","press","publisher","isbn","volume","vol","rs","inr"]
    if any(w in text for w in bad):
        return False

    if len(text.split()) < 2:
        return False

    return True


# ---------------------------------------------------
# EXTRACT TEXT LINES
# ---------------------------------------------------
def extract_candidates(img):

    data = pytesseract.image_to_data(
        img,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    lines = {}

    for i, word in enumerate(data["text"]):

        word = word.strip()
        if not word:
            continue

        conf = int(data["conf"][i])
        if conf < 55:
            continue

        line_num = data["line_num"][i]
        lines.setdefault(line_num, []).append(word)

    results = []

    for line in lines.values():
        text = normalize(" ".join(line))
        if is_valid_candidate(text):
            results.append(text)

    return results


# ---------------------------------------------------
# MERGE SIMILAR TITLES
# ---------------------------------------------------
def merge_similar(texts):

    merged = []

    for t in texts:
        added = False
        for i, m in enumerate(merged):
            if t in m or m in t:
                merged[i] = max(t, m, key=len)
                added = True
                break
        if not added:
            merged.append(t)

    return merged


# ---------------------------------------------------
# MAIN OCR
# ---------------------------------------------------
def extract_text(path: str):

    try:
        versions = preprocess_versions(path)

        all_candidates = []

        for img in versions:
            all_candidates.extend(extract_candidates(img))

        if not all_candidates:
            print("OCR: nothing detected")
            return []

        # vote
        vote = Counter(all_candidates)

        # sort by frequency + length
        ranked = sorted(
            vote.items(),
            key=lambda x: (x[1], len(x[0])),
            reverse=True
        )

        candidates = [t[0] for t in ranked[:8]]

        candidates = merge_similar(candidates)

        print("OCR FINAL CANDIDATES:", candidates)

        return candidates

    except Exception as e:
        print("OCR FAILED:", e)
        return []