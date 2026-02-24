import requests
import base64
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def detect_book_title(image_path):

    if not GROQ_API_KEY:
        return None

    img_base64 = encode_image(image_path)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = """
Look at this image of a book cover.
Tell ONLY the book title and author.
If unclear say UNKNOWN.
Return short text only.
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

        text = r.json()["choices"][0]["message"]["content"]

        if "unknown" in text.lower():
            return None

        return text.strip()

    except:
        return None