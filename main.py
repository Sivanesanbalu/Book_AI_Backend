import uvicorn
import os

from fastapi import FastAPI, UploadFile, File
import shutil, os
import uvicorn

from ocr import extract_text
from search_engine import search_book, add_book
from firebase_service import save_book_to_cloud

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 🔎 SCAN BOOK (Check duplicate)
@app.post("/scan")
async def scan_book(file: UploadFile = File(...)):

    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(path)

    book, score = search_book(text)

    if book:
        return {
            "status": "duplicate",
            "message": "You already own this book",
            "title": book["title"],
            "confidence": score
        }

    return {
        "status": "new",
        "title_detected": text
    }


# 📸 CAPTURE BOOK (Save to collection)
@app.post("/capture")
async def capture_book(file: UploadFile = File(...)):

    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(path)

    add_book(text)
    save_book_to_cloud(text)

    return {
        "status": "saved",
        "title": text
    }


# ⭐⭐⭐ THIS PART FIXES RENDER DEPLOY ⭐⭐⭐
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
