import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


def detect_book(image_path):
    try:
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

        res = requests.post(url, headers=headers, json=payload, timeout=60)

        print("GROQ RESPONSE:", res.text)

        data = res.json()

        # ❗ IMPORTANT FIX
        if "choices" not in data:
            print("VISION ERROR:", data)
            return None

        text = data["choices"][0]["message"]["content"].strip()

        if not text:
            return None

        return text

    except Exception as e:
        print("VISION EXCEPTION:", e)
        return None