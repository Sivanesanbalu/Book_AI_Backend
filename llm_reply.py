import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_book_reply(book_name, topic, user_question):

    if not user_question:
        user_question = "Explain this book"

    # 🧠 SMART PROMPT
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
1) Small direct answer to the question
2) What this book teaches
3) Who should read it
4) What student will learn
5) Why it is useful in real life
"""

    if not GROQ_API_KEY:
        return fallback_summary(book_name, topic)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",   # ✅ NEW MODEL
        "messages": [
            {"role": "system", "content": "You are a helpful educational book assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 450
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)

        if r.status_code != 200:
            print("Groq API failed:", r.text)
            return fallback_summary(book_name, topic)

        response = r.json()

        if "choices" not in response:
            return fallback_summary(book_name, topic)

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        print("LLM ERROR:", e)
        return fallback_summary(book_name, topic)


def fallback_summary(book_name, topic):
    return f"""
This book '{book_name}' belongs to {topic} subject.

It explains the core ideas in a simple and step by step way.
Students can understand fundamentals clearly.

This book is good for beginners and exam preparation.
You will learn important concepts and practical understanding.

Reading this helps improve knowledge and confidence in the subject.
"""