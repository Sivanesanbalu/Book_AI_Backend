from fastapi import APIRouter, UploadFile, File, Form
import shutil, os

from image_search import search_book   # your existing function
from llm_reply import generate_book_reply

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/assistant")
async def assistant(file: UploadFile = File(...), question: str = Form("")):

    path = f"{UPLOAD_DIR}/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔎 YOUR OLD AI
    result = search_book(path)

    if result is None:
        return {"reply": "I couldn't recognize this book clearly."}

    book_name = result.get("book", "Unknown Book")
    topic = result.get("category", "General Knowledge")

    # 🧠 NEW AI BRAIN
    answer = generate_book_reply(book_name, topic, question)

    return {
        "book": book_name,
        "topic": topic,
        "reply": answer
    }