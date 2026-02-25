import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def explain_book(title, book_data, question, mode="book"):

    if not GROQ_API_KEY:
        return "AI not ready"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # ================= CHAT MODE =================
    if mode == "chat":

        prompt = f"""
You are a friendly teaching assistant.

Explain clearly.
First intuition then definition.
Give simple real-life example.

Question:
{question}
"""

        body = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 350
        }

        r = requests.post(url, headers=headers, json=body, timeout=60)

        if r.status_code != 200:
            return "AI busy, retry"

        return r.json()["choices"][0]["message"]["content"]


    # ================= BOOK MODE =================
    description = book_data.get("description", "").strip()

    if len(description) < 40:
        return f"I found the book '{title}', but description not available."

    prompt = f"""
Explain this book simply for a student.

BOOK: {title}
AUTHOR: {book_data.get("authors")}
CATEGORY: {book_data.get("categories")}

DESCRIPTION:
{description}

QUESTION:
{question}
"""

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 260
    }

    r = requests.post(url, headers=headers, json=body, timeout=60)

    if r.status_code != 200:
        return "AI explanation failed"

    return r.json()["choices"][0]["message"]["content"]