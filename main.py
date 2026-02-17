import os
import shutil
from fastapi import FastAPI, UploadFile, File

from ocr import extract_text
from search_engine import search_book
from firebase_service import save_book_for_user, user_has_book

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


############################################################
# 🔎 SCAN — ONLY CHECK (Lens Mode)
############################################################
@app.post("/scan")
async def scan_book(uid: str, file: UploadFile = File(...)):

    path = f"{UPLOAD_DIR}/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(path)

    # 1️⃣ Check AI global database
    book, score = search_book(text)

    # 2️⃣ Check user's personal library
    already_owned = user_has_book(uid, text)

    if already_owned:
        return {
            "status": "owned",
            "message": "You already have this book",
            "title": text
        }

    if book:
        return {
            "status": "known_book",
            "title": book["title"],
            "confidence": float(score)
        }

    return {
        "status": "unknown",
        "detected_text": text
    }


############################################################
# 📸 CAPTURE — SAVE TO USER LIBRARY ONLY
############################################################
@app.post("/capture")
async def capture_book(uid: str, file: UploadFile = File(...)):

    path = f"{UPLOAD_DIR}/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(path)

    # save only to that user
    save_book_for_user(uid, text)

    return {
        "status": "saved",
        "title": text
    }
