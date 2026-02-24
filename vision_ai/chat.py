from fastapi import APIRouter
from .explain_book import explain_book

router = APIRouter(prefix="/ai", tags=["AI Chat"])


@router.post("/chat")
async def chat_only(data: dict):
    question = data.get("question", "")

    if question.strip() == "":
        return {"answer": "Ask me something about books 😊"}

    # general assistant mode
    fake_book = {
        "title": "General Knowledge",
        "authors": "AI Assistant",
        "categories": "Education",
        "description": "Answer student questions in simple explanation"
    }

    answer = explain_book("General", fake_book, question)

    return {"answer": answer}