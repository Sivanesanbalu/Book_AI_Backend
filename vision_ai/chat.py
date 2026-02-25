from fastapi import APIRouter
import os, requests

router = APIRouter(prefix="/ai", tags=["AI Chat"])
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@router.post("/chat")
async def chat_only(data: dict):

    question = data.get("question", "").strip()

    if question == "":
        return {"answer": "Ask me anything — ML, coding, concepts 🙂"}

    prompt = f"""
You are a friendly teaching assistant.

Explain step by step.
First intuition → then definition → example.
Simple English. Avoid theory overload.

Student Question:
{question}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 350
    }

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json=body,
        timeout=60
    )

    if r.status_code != 200:
        return {"answer": "AI busy, retry"}

    return {"answer": r.json()["choices"][0]["message"]["content"]}