import requests
import os
import time

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_book_reply(book_name, topic, user_question):

    if not user_question:
        user_question = "Explain this book"

    # ---------- PROMPT ----------
    prompt = f"""
You are a friendly teacher helping a student understand a book.

Book Name: {book_name}
Subject: {topic}

Student Question:
{user_question}

Instructions:
- Always answer the student's question
- But mainly explain the book
- Give around 8 to 12 simple lines
- Use very easy English
- If question unrelated, gently connect it back to the book
- Teach like a school teacher

Structure:
1) Small direct answer
2) What this book teaches
3) Who should read it
4) What student will learn
5) Real life usage
"""

    # ---------- IF NO API KEY ----------
    if not GROQ_API_KEY:
        return fallback_summary(book_name, topic)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful educational book assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 450
    }

    # ---------- REQUEST WITH RETRY ----------
    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)

        # retry once if rate limit
        if r.status_code == 429:
            time.sleep(2)
            r = requests.post(url, headers=headers, json=data, timeout=60)

        if r.status_code != 200:
            print("Groq API failed:", r.text)
            return fallback_summary(book_name, topic)

        response = r.json()

        if "choices" not in response:
            return fallback_summary(book_name, topic)

        content = response["choices"][0]["message"].get("content", "")

        return content.strip() if content else fallback_summary(book_name, topic)

    except Exception as e:
        print("LLM ERROR:", e)
        return fallback_summary(book_name, topic)


# ---------- FALLBACK (WHEN SERVER WAKING) ----------
def fallback_summary(book_name, topic):
    return f"""
I detected the book: {book_name}

AI is starting... please ask again in few seconds.

Basic idea:
This book belongs to {topic} subject and explains concepts step by step for students.
"""