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
# MULTI PREPROCESS (3 VERSIONS)
# ---------------------------------------------------
def preprocess_versions(path):

    img = cv2.imread(path)
    if img is None:
        return []

    h, w = img.shape[:2]

    # focus book center
    crop = img[int(h*0.10):int(h*0.75), int(w*0.10):int(w*0.90)]

    versions = []

    # 1 NORMAL
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    versions.append(gray)

    # 2 SHARP
    sharp = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
    versions.append(sharp)

    # 3 THRESHOLD
    th = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,2
    )
    versions.append(th)

    return versions


# ---------------------------------------------------
# FILTER BAD TEXT
# ---------------------------------------------------
def is_valid_candidate(text):

    if len(text) < 4:
        return False

    if sum(c.isdigit() for c in text) > len(text)*0.4:
        return False

    bad = ["edition","press","publisher","isbn","volume","vol","rs","inr"]
    t = text.lower()

    if any(w in t for w in bad):
        return False

    words = text.split()
    if len(words) < 2:
        return False

    return True


# ---------------------------------------------------
# EXTRACT CANDIDATES FROM IMAGE
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
        text = " ".join(line)
        text = normalize(text)

        if is_valid_candidate(text):
            results.append(text)

    return results


# ---------------------------------------------------
# MAIN OCR WITH VOTING
# ---------------------------------------------------
def extract_text(path: str) -> str:
    try:

        versions = preprocess_versions(path)

        all_candidates = []

        for img in versions:
            candidates = extract_candidates(img)
            all_candidates.extend(candidates)

        if not all_candidates:
            print("OCR: nothing detected")
            return ""

        # voting (MOST FREQUENT TITLE)
        vote = Counter(all_candidates)
        best, count = vote.most_common(1)[0]

        print("OCR candidates:", vote)
        print("FINAL TITLE:", best)

        return best

    except Exception as e:
        print("OCR FAILED:", e)
        return ""