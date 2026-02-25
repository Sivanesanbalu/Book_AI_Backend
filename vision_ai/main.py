import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from vision_ai.vision import detect_book
from vision_ai.book_fetcher import get_book_info
from vision_ai.ai_summary import summarize_book
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/ask-book-ai")
async def ask_book_ai(file: UploadFile = File(...)):

    # save image
    path = f"{UPLOAD_DIR}/{uuid.uuid4()}.jpg"
    with open(path, "wb") as f:
        f.write(await file.read())

    # 1 detect
    book_name = detect_book(path)

    # 2 fetch
    book = get_book_info(book_name)

    if not book:
        return JSONResponse({"error": "Book not found"}, status_code=404)

    # 3 summarize
    overview = summarize_book(book)

    return {
        "detected": book_name,
        "title": book["title"],
        "overview": overview
    }