import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_book_reply(book_name, topic, user_question):

    if not user_question:
        user_question = "Explain this book simply"

    prompt = f"""
A student uploaded a book photo.

Book name: {book_name}
Subject: {topic}

Student question: {user_question}

Give 5 to 8 simple educational lines:
1) What the book is about
2) Who should read it
3) What they will learn
Answer like a teacher explaining to a student.
"""

    # if API key missing → never crash
    if not GROQ_API_KEY:
        return fallback_summary(book_name, topic)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 300
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=40)

        # API failed
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


# ALWAYS WORKING (no internet / no credits / server sleep)
def fallback_summary(book_name, topic):
    return f"""
This book '{book_name}' is related to {topic}.

It introduces important concepts in a clear and structured manner.
Students can build strong understanding step by step.

Suitable for beginners, learners and exam preparation.
Reading this book improves knowledge and fundamentals.
"""