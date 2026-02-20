from paddleocr import PaddleOCR
import re

# ---------------------------------------------------
# LOAD MODEL ONCE (VERY IMPORTANT)
# ---------------------------------------------------
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)


# ---------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# REMOVE BAD LINES
# ---------------------------------------------------
def is_valid_line(text: str, conf: float) -> bool:

    if conf < 0.55:
        return False

    if len(text) < 3:
        return False

    # remove isbn
    if re.search(r'\d{10,13}', text):
        return False

    # remove price / edition
    if re.search(r'\b(rs|inr|\$|edition|ed\.)\b', text.lower()):
        return False

    return True


# ---------------------------------------------------
# SMART TITLE GROUPING
# ---------------------------------------------------
def extract_title(detections):

    lines = []

    for det in detections:
        box = det[0]
        text = det[1][0].strip()
        conf = float(det[1][1])

        if not is_valid_line(text, conf):
            continue

        y_center = (box[0][1] + box[2][1]) / 2
        height = abs(box[0][1] - box[2][1])

        lines.append({
            "text": text,
            "y": y_center,
            "h": height
        })

    if not lines:
        return ""

    # sort vertically
    lines.sort(key=lambda x: x["y"])

    # group nearby lines → same title block
    groups = []
    current = [lines[0]]

    for i in range(1, len(lines)):
        prev = current[-1]
        curr = lines[i]

        if abs(curr["y"] - prev["y"]) < max(prev["h"], curr["h"]) * 1.8:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]

    groups.append(current)

    # choose biggest visual block (book title usually largest)
    best_group = max(groups, key=lambda g: len(g))

    title = " ".join([l["text"] for l in best_group])

    return normalize(title)


# ---------------------------------------------------
# MAIN OCR FUNCTION
# ---------------------------------------------------
def extract_text(path: str) -> str:
    try:
        result = ocr.ocr(path)

        if not result or not result[0]:
            return ""

        title = extract_title(result[0])

        print("DETECTED TITLE:", title)
        return title

    except Exception as e:
        print("OCR FAILED:", e)
        return ""