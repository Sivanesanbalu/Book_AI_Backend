import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def explain_book(title, book_data, question):

    if not GROQ_API_KEY:
        return "AI not ready"

    description = book_data.get("description", "").strip()

    # 🚨 CRITICAL: If no description -> don't hallucinate
    if len(description) < 40:
        return f"I found the book '{book_data['title']}', but I could not find enough information to explain it."

    prompt = f"""
You are a careful teacher.

RULES:
- Use ONLY the given description
- DO NOT add your own story
- DO NOT guess the plot
- If info missing, say information not available
- Answer in simple English
- Maximum 8 lines

BOOK INFO:
Title: {book_data['title']}
Author: {book_data['authors']}
Category: {book_data['categories']}

DESCRIPTION:
{description}

QUESTION:
{question}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,   # 🔥 reduce imagination
        "max_tokens": 220
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)

        if r.status_code != 200:
            return "AI explanation failed"

        answer = r.json()["choices"][0]["message"]["content"].strip()

        # safety cleanup
        if len(answer) < 15:
            return "I could not confidently understand this book."

        return answer

    except:
        return "AI error while explaining"