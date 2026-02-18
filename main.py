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

    # save image
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OCR read
    text = extract_text(path)

    # search in AI database
    book, score = search_book(text)

    # if book matched -> check ownership using CLEAN TITLE
    if book:
        clean_title = book["title"]

        already_owned = user_has_book(uid, clean_title)

        if already_owned:
            return {
                "status": "owned",
                "title": clean_title
            }

        return {
            "status": "known_book",
            "title": clean_title,
            "confidence": float(score)
        }

    # no match
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

    # save image
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OCR read
    text = extract_text(path)

    # match with AI DB to get clean title
    book, score = search_book(text)

    if book:
        final_title = book["title"]
    else:
        final_title = text

    # save CLEAN title to user library
    save_book_for_user(uid, final_title)

    return {
        "status": "saved",
        "title": final_title
    }
