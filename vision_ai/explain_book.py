import os, requests
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def explain_book(title,data,question):

    desc=data.get("description","")

    prompt=f"""
Explain this book for a student deciding to read it.

Tell:
• what it teaches
• who should read
• difficulty level
• real life usefulness

Book: {title}
Author: {data.get("authors")}
Category: {data.get("categories")}
Description: {desc}
"""

    r=requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization":f"Bearer {GROQ_API_KEY}"},
    json={
        "model":"llama-3.1-8b-instant",
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.4
    })

    return r.json()["choices"][0]["message"]["content"]