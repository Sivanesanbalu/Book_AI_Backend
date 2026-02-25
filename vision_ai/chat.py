import requests
import base64
import os
import re

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def extract_title(text: str):
    """Clean AI response and extract probable book title"""

    if not text:
        return None

    text = text.replace("\n", " ").strip()

    # remove common phrases
    text = re.sub(r"(?i)the book (title )?(is|appears to be|looks like)", "", text)
    text = re.sub(r"(?i)written by.*", "", text)
    text = re.sub(r"(?i)by [A-Z][a-z]+.*", "", text)

    # keep only main part
    text = text.strip(" :.-")

    # too short -> invalid
    if len(text) < 4:
        return None

    return text


def detect_book_title(image_path):

    if not GROQ_API_KEY:
        return None

    img_base64 = encode_image(image_path)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # 🔥 MUCH STRONGER PROMPT
    prompt = """
You are a vision OCR system.

Task:
Read the book cover text and return ONLY the book title.

Rules:
- Ignore subtitle
- Ignore author
- Ignore publisher
- Do NOT explain
- Do NOT add sentences
- Output only title text

If unreadable return: UNKNOWN
"""

    data = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
            ]}
        ],
        "temperature": 0
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)

        if r.status_code != 200:
            return None

        raw = r.json()["choices"][0]["message"]["content"]

        if "unknown" in raw.lower():
            return None

        return extract_title(raw)

    except:
        return None