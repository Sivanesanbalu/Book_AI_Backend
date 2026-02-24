import requests
import os
import time

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_book_reply(book_name, topic, user_question, book_content):

    if not user_question:
        user_question = "Explain this book"

    # ---------- PROMPT (REAL RAG PROMPT) ----------
    prompt = f"""
    You are a strict school teacher.

    You MUST answer ONLY using the BOOK CONTENT below.
    If answer not present, say: "This part is not in the book content."

    BOOK NAME:
    {book_name}

    SUBJECT:
    {topic}

    BOOK CONTENT:
    {book_content}

    STUDENT QUESTION:
    {user_question}

    Rules:
    - Do NOT use outside knowledge
    - Do NOT guess
    - Do NOT add extra syllabus
    - Explain in very simple student language
    - 6 to 10 short lines
    """

    # ---------- NO API KEY ----------
    if not GROQ_API_KEY:
        return fallback_summary(book_name, topic)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        # FAST + FREE + GOOD FOR RAG
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You explain books using provided knowledge only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,   # VERY IMPORTANT (stop hallucination)
        "max_tokens": 400
    }

    # ---------- REQUEST ----------
    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)

        # retry once
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

        if not content or len(content) < 20:
            return fallback_summary(book_name, topic)

        return content.strip()

    except Exception as e:
        print("LLM ERROR:", e)
        return fallback_summary(book_name, topic)


# ---------- FALLBACK ----------
def fallback_summary(book_name, topic):
    return f"""
I detected the book: {book_name}

The AI server is starting or book data not available.

This book belongs to {topic}.
Please scan again in a moment for full explanation.
"""