import requests, os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def analyze_book(title, book_data, question):

    description = book_data.get("description", "")

    prompt = f"""
You are a senior student advising a junior.

Explain the usefulness of the book.

Give:
1) What this book teaches
2) Who should read it
3) Is it worth reading
4) Difficulty level

Speak naturally like a human, not marketing.

BOOK:
Title: {title}
Author: {book_data.get("authors")}
Category: {book_data.get("categories")}
Description: {description}

User question: {question}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 350
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        return r.json()["choices"][0]["message"]["content"]

    except:
        return "Couldn't analyze the book."