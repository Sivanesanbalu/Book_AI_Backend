from fastapi import APIRouter
from pydantic import BaseModel
from image_search import search_book
from llm_reply import generate_book_reply
import asyncio
import os

router = APIRouter()

class AskRequest(BaseModel):
    uid: str
    image_path: str | None = None
    question: str | None = None

@router.post("/ask")
async def ask_ai(data: AskRequest):

    book_name = "Unknown Book"
    topic = "General Education"

    # detect book if image provided
    if data.image_path and os.path.exists(data.image_path):
        book, score = await asyncio.to_thread(search_book, data.image_path)

        if book:
            book_name = book["title"]
            topic = book.get("topic", "Education")

    # generate AI answer
    reply = await asyncio.to_thread(
        generate_book_reply,
        book_name,
        topic,
        data.question
    )

    return {
        "book": book_name,
        "topic": topic,
        "answer": reply
    }