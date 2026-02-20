from paddleocr import PaddleOCR
import re
import numpy as np

# load once
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)


# ---------------- CLEAN TEXT ----------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- GROUP TITLE LINES ----------------
def merge_title_lines(detections):

    lines = []

    for det in detections:
        box = det[0]
        text = det[1][0]
        conf = float(det[1][1])

        if conf < 0.55:
            continue

        if len(text) < 3:
            continue

        # center Y position
        y_center = (box[0][1] + box[2][1]) / 2
        height = abs(box[0][1] - box[2][1])

        lines.append({
            "text": text.strip(),
            "y": y_center,
            "h": height
        })

    if not lines:
        return ""

    # sort vertically
    lines.sort(key=lambda x: x["y"])

    # group close lines (same title block)
    groups = []
    current = [lines[0]]

    for i in range(1, len(lines)):
        prev = current[-1]
        curr = lines[i]

        # if vertically close → same block
        if abs(curr["y"] - prev["y"]) < max(prev["h"], curr["h"]) * 1.8:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]

    groups.append(current)

    # choose biggest group
    best_group = max(groups, key=lambda g: len(g))

    title = " ".join([l["text"] for l in best_group])

    return normalize(title)


# ---------------- MAIN OCR ----------------
def extract_text(path: str) -> str:
    try:
        result = ocr.ocr(path)

        if not result or not result[0]:
            print("OCR: nothing detected")
            return ""

        title = merge_title_lines(result[0])

        print("DETECTED TITLE:", title)
        return title

    except Exception as e:
        print("OCR FAILED:", e)
        return ""