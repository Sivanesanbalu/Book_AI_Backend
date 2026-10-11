import requests
import base64
import os
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


def clean_title(text: str):
    """
    Cleans AI output so Google Books / OpenLibrary can match correctly
    """
    if not text:
        return None

    # normalize separators
    text = text.replace(" by ", " - ")
    text = text.replace("—", "-")
    text = text.replace("|", "-")

    # keep first line only
    text = text.split("\n")[0]

    # remove sentences AI may prepend
    text = re.sub(r"The book.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"This is.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Here is.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Title:\s*", "", text, flags=re.IGNORECASE)

    # remove quotes
    text = text.replace('"', '').replace("'", "")

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_book_with_groq(image_path: str):
    """
    Call Groq Vision API with latest LLaMA 3.2 Vision models
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        with open(image_path, "rb") as img:
            b64 = base64.b64encode(img.read()).decode()

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Try llama-3.2-11b-vision-preview
        models_to_try = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview"
        ]

        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify the book title and author from this cover. Only output: TITLE - AUTHOR. If unknown, output UNKNOWN."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 80,
                "temperature": 0.1
            }

            res = requests.post(url, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                if "choices" in data and len(data["choices"]) > 0:
                    raw_text = data["choices"][0]["message"]["content"].strip()
                    if raw_text and "UNKNOWN" not in raw_text.upper():
                        cleaned = clean_title(raw_text)
                        if cleaned:
                            print(f"✅ Groq Vision Identified: {cleaned}")
                            return cleaned
            else:
                print(f"⚠️ Groq Vision {model} error: {res.status_code} - {res.text}")

    except Exception as e:
        print("⚠️ Groq Vision exception:", e)

    return None


def detect_book(image_path: str):
    """
    Master book detection pipeline
    """
    # 1. Try Groq Vision API if API key is provided
    detected = detect_book_with_groq(image_path)
    if detected:
        return detected

    # 2. Check local vector index for best visual match
    try:
        from image_search import search_book
        book, score = search_book(image_path)
        if book and score >= 0.70:
            title = book.get("title", "")
            author = book.get("author", "")
            print(f"✅ Visual Vector Match: {title} (score: {score:.3f})")
            return f"{title} - {author}" if author else title
    except Exception as e:
        print("⚠️ Local vector detection error:", e)

    return None