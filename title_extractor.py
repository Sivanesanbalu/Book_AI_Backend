import re

def clean_line(line: str):
    line = line.strip()

    if len(line) < 3:
        return None

    # remove garbage symbols
    line = re.sub(r'[^a-zA-Z0-9 :\-]', '', line)

    if len(line.split()) <= 1:
        return None

    return line


def extract_title(ocr_text: str):

    lines = ocr_text.split("\n")

    candidates = []

    for line in lines:
        line = clean_line(line)
        if not line:
            continue

        words = line.split()

        # title usually short phrase
        if 2 <= len(words) <= 8:
            candidates.append(line)

    if not candidates:
        return None

    # choose most meaningful line
    candidates.sort(key=len, reverse=True)
    return candidates[0]