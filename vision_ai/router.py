from fastapi import APIRouter, UploadFile, File, Form
import os, uuid, shutil

from .detect_book import detect_book_title
from .fetch_book import fetch_book_details
from .explain_book import explain_book

router = APIRouter(prefix="/ai", tags=["AI Vision"])

UPLOAD_DIR = "uploads_ai"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/understand-book")
async def understand_book(
    question: str = Form("Explain this book"),
    file: UploadFile = File(...)
):

    temp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")

    try:
        # save image
        with open(temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # STEP 1 — identify book
        title = detect_book_title(temp)

        # 🔍 DEBUG RETURN (VERY IMPORTANT)
        if not title:
            return {
                "debug": "vision_failed",
                "answer": "I cannot read the book cover clearly"
            }

        # STEP 2 — search internet
        book_data = fetch_book_details(title)

        if not book_data:
            return {
                "debug": title,
                "book": title,
                "answer": "Book detected but details unavailable"
            }

        # STEP 3 — explain
        explanation = explain_book(title, book_data, question)

        return {
            "debug": title,
            "book": title,
            "answer": explanation
        }

    finally:
        if os.path.exists(temp):
            os.remove(temp)