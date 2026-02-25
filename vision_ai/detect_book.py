import requests, base64, os, re
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def extract_title(text):
    if not text:
        return None

    text = text.split("\n")[0]

    if " - " in text:
        text = text.split(" - ")[0]

    text = re.sub(r"(?i)by .*", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def detect_book_title(image_path):

    img_base64 = encode_image(image_path)

    prompt = """
Read the book cover and return only the book title.
Ignore subtitles and stickers.
Format: Title - Author
"""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama-3.2-11b-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }],
            "temperature": 0
        },
        timeout=60
    )

    if r.status_code != 200:
        return None

    raw = r.json()["choices"][0]["message"]["content"]
    return extract_title(raw)