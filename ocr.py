import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import pytesseract
import numpy as np
import re


# ---------------------------------------------------
# ORDER POINTS (for perspective correction)
# ---------------------------------------------------
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ---------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# SMART BOOK DETECTION + PERSPECTIVE FIX
# ---------------------------------------------------
def preprocess(path):

    img = cv2.imread(path)
    if img is None:
        return None

    original = img.copy()

    # resize for contour detection
    ratio = img.shape[0] / 800.0
    img = cv2.resize(img, (int(img.shape[1]/ratio), 800))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edged = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    book_contour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            book_contour = approx
            break

    # if book detected → flatten
    if book_contour is not None:

        pts = book_contour.reshape(4,2) * ratio
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0,0],
            [maxWidth-1,0],
            [maxWidth-1,maxHeight-1],
            [0,maxHeight-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    else:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # OCR optimized cleanup
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,2
    )

    return gray


# ---------------------------------------------------
# FILTER BAD TEXT
# ---------------------------------------------------
def is_bad_text(text):

    text_low = text.lower()

    if sum(c.isdigit() for c in text) > len(text)*0.4:
        return True

    if re.search(r'\d{4,}', text):
        return True

    bad_words = [
        "edition","press","publisher","volume",
        "vol","isbn","copyright","rs","inr"
    ]

    if any(w in text_low for w in bad_words):
        return True

    if len(text) < 2:
        return True

    return False


# ---------------------------------------------------
# GROUP WORDS INTO LINES
# ---------------------------------------------------
def group_lines(data, img_height):

    words = []

    for i in range(len(data["text"])):

        txt = data["text"][i].strip()
        conf = int(data["conf"][i])

        if conf < 55 or is_bad_text(txt):
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
        if abs(w[2] - current[-1][2]) < img_height * 0.045:
            current.append(w)
        else:
            lines.append(current)
            current = [w]

    lines.append(current)
    return lines


# ---------------------------------------------------
# PICK BEST TITLE
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

        length_bonus = min(word_count/4, 2)

        score = area * (1.2 + position_score + length_bonus)

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

        print("📖 OCR TITLES:", titles)

        return titles

    except Exception as e:
        print("OCR FAILED:", e)
        return []