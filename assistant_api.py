from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from image_search import search_book
from llm_reply import generate_book_reply
from book_data import get_book_summary
import asyncio, os, uuid, shutil

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= TEXT REQUEST =================
class AskRequest(BaseModel):
    question: str | None = None

@router.post("/ask")
async def ask_ai(data: AskRequest):

    question = data.question or "Explain this clearly in simple words"

    reply = await asyncio.to_thread(
        generate_book_reply,
        "General Book",
        "Education",
        question,
        "No book content available"
    )

    return {"answer": reply}


# ================= IMAGE REQUEST =================
@router.post("/ask-image")
async def ask_ai_image(
    question: str = Form(None),
    file: UploadFile = File(...)
):

    question = question or "Explain this book clearly in simple words"
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")

    try:
        # SAVE IMAGE
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await file.close()

        # -------- COMPUTER VISION --------
        book, score = await asyncio.to_thread(search_book, temp_path)

        if not book:
            return {
                "answer": "I cannot detect the book clearly. Please scan the front cover properly."
            }

        book_name = book.get("title", "Unknown Book")

        # -------- RAG RETRIEVAL --------
        summary = get_book_summary(book_name)

        if not summary:
            return {
                "book": book_name,
                "answer": f"I found the book '{book_name}', but its learning content is not in my knowledge yet."
            }

        topic = summary.get("category", "Education")
        book_content = summary.get("description", "")

        if not book_content.strip():
            return {
                "book": book_name,
                "answer": f"I detected '{book_name}', but I don't have its explanation data."
            }

        # -------- LLM --------
        reply = await asyncio.to_thread(
            generate_book_reply,
            book_name,
            topic,
            question,
            book_content
        )

        return {
            "book": book_name,
            "topic": topic,
            "answer": reply
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)