import requests
import base64
import os
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


def clean_title(text: str):
    """
    Cleans AI output so Google Books API can match correctly
    """

    if not text:
        return None

    # normalize separators
    text = text.replace(" by ", " - ")
    text = text.replace("—", "-")
    text = text.replace("|", "-")

    # keep first line only
    text = text.split("\n")[0]

    # remove sentences AI may add
    text = re.sub(r"The book.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"This is.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Here is.*?:", "", text, flags=re.IGNORECASE)

    # remove quotes
    text = text.replace('"', '').replace("'", "")

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_book(image_path):
    try:
        # convert image → base64
        with open(image_path, "rb") as img:
            b64 = base64.b64encode(img.read()).decode()

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify the book title and author from this cover. Only output: TITLE - AUTHOR"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }

        # request
        res = requests.post(url, headers=headers, json=payload, timeout=60)

        print("GROQ RAW:", res.text)

        data = res.json()

        # 🛑 if model error / quota / blocked
        if "choices" not in data:
            print("VISION ERROR:", data)
            return None

        raw_text = data["choices"][0]["message"]["content"].strip()

        if not raw_text:
            return None

        cleaned = clean_title(raw_text)

        print("VISION CLEAN:", cleaned)

        return cleaned

    except Exception as e:
        print("VISION EXCEPTION:", e)
        return None