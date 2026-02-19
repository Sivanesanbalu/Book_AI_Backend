import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File

from ocr import extract_text
from search_engine import search_book, add_book   # 🔥 IMPORTANT CHANGE
from firebase_service import save_book_for_user, user_has_book

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- helper ----------
def save_temp_file(upload_file: UploadFile) -> str:
    unique_name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, unique_name)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return path


############################################################
# 🔎 SCAN — ONLY CHECK
############################################################
@app.post("/scan")
async def scan_book(uid: str, file: UploadFile = File(...)):

    path = save_temp_file(file)

    try:
        text = extract_text(path)

        if not text or len(text) < 3:
            return {"status": "no_text"}

        book, score = search_book(text)

        if book:
            clean_title = book["title"]

            if user_has_book(uid, clean_title):
                return {
                    "status": "owned",
                    "title": clean_title
                }

            return {
                "status": "known_book",
                "title": clean_title,
                "confidence": float(score)
            }

        return {
            "status": "unknown",
            "detected_text": text
        }

    finally:
        if os.path.exists(path):
            os.remove(path)


############################################################
# 📸 CAPTURE — SAVE BOOK + TRAIN AI
############################################################
@app.post("/capture")
async def capture_book(uid: str, file: UploadFile = File(...)):

    path = save_temp_file(file)

    try:
        text = extract_text(path)

        if not text or len(text) < 3:
            return {"status": "failed"}

        book, score = search_book(text)

        # -------------------------------
        # BOOK ALREADY EXISTS IN AI DB
        # -------------------------------
        if book:
            final_title = book["title"]

        # -------------------------------
        # NEW BOOK -> TRAIN AI DATABASE
        # -------------------------------
        else:
            final_title = text
            add_book(final_title)     # 🔥 THIS FIXES YOUR PROBLEM

        # SAVE TO USER LIBRARY
        save_book_for_user(uid, final_title)

        return {
            "status": "saved",
            "title": final_title
        }

    finally:
        if os.path.exists(path):
            os.remove(path)
