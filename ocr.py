from paddleocr import PaddleOCR
import re

# ---------------------------------------------------
# LOAD MODEL ONLY ONCE (IMPORTANT FOR PERFORMANCE)
# ---------------------------------------------------
ocr = PaddleOCR(
    use_angle_cls=True,     # handles rotated books
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
# TITLE EXTRACTION LOGIC
# chooses strongest visible title text
# ---------------------------------------------------
def extract_title(detections):

    candidates = []

    for line in detections:
        text = line[1][0]
        confidence = float(line[1][1])

        text = text.strip()

        # ignore weak detections
        if confidence < 0.60:
            continue

        # ignore very short words
        if len(text) < 4:
            continue

        # ignore isbn / numeric lines
        if re.search(r'\d{10,13}', text):
            continue

        # weighted scoring
        weight = len(text) * confidence
        candidates.append((text, weight))

    if not candidates:
        return ""

    # best candidate = title
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0][0]

    return normalize(best)


# ---------------------------------------------------
# MAIN OCR FUNCTION
# ---------------------------------------------------
def extract_text(path: str) -> str:
    try:
        result = ocr.ocr(path)

        if not result or not result[0]:
            print("OCR: nothing detected")
            return ""

        title = extract_title(result[0])

        print("\nDETECTED TITLE:", title)

        return title

    except Exception as e:
        print("OCR FAILED:", e)
        return ""