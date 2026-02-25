import requests
import os
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

def summarize_book(book):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    user_prompt = f"""
    Book Title: {book['title']}
    Author: {book['authors']}

    Description:
    {book['description']}

    Explain this book in a simple short overview for students.
    """

    payload = {
        "model": "llama-3.3-70b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 300
    }

    res = requests.post(url, headers=headers, json=payload)
    return res.json()["choices"][0]["message"]["content"]