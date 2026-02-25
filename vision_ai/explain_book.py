import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def explain_book(title, book_data, question):

    if not GROQ_API_KEY:
        return "AI not ready"

    description = book_data.get("description", "").strip()

    # 🚨 No description protection
    if len(description) < 40:
        return f"I found the book '{book_data['title']}', but I could not find enough information to explain it."

    prompt = f"""
You are a friendly book tutor helping a student understand a book.

STRICT RULES:
- Use ONLY the provided description
- Do NOT invent story or content
- If information missing → clearly say "not mentioned in description"
- Explain clearly in simple English
- Be helpful and structured
- Maximum 10 lines

BOOK DETAILS:
Title: {book_data['title']}
Author: {book_data['authors']}
Category: {book_data['categories']}

DESCRIPTION:
{description}

USER QUESTION:
{question}

RESPONSE STYLE:
If summary requested → give short summary
If learning requested → explain what user will learn
If difficulty asked → infer from description only
If general → explain purpose of the book
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # little naturalness but still safe
        "max_tokens": 260
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)

        if r.status_code != 200:
            return "AI explanation failed"

        answer = r.json()["choices"][0]["message"]["content"].strip()

        # cleanup overly tiny responses
        if len(answer) < 20:
            return "I could not confidently understand this book from available information."

        return answer

    except:
        return "AI error while explaining"