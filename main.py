import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from ocr import extract_text
from search_engine import search_book, add_book
from firebase_service import save_book_for_user, user_has_book

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------
# SAVE TEMP FILE
# ------------------------------------------------
def save_temp_file(upload_file: UploadFile) -> str:
    unique_name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, unique_name)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return path


# ------------------------------------------------
# NON BLOCKING OCR
# ------------------------------------------------
async def run_ocr(path: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_text, path)


############################################################
# 🔎 SCAN — ONLY CHECK
############################################################
@app.post("/scan")
async def scan_book(uid: str = Query(...), file: UploadFile = File(...)):

    path = save_temp_file(file)

    try:
        title = await run_ocr(path)

        if not title or len(title) < 4:
            return {"status": "no_text"}

        book, score = search_book(title)

        # GLOBAL BOOK FOUND
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
                "confidence": round(float(score), 3)
            }

        # USER FALLBACK CHECK
        if user_has_book(uid, title):
            return {
                "status": "owned",
                "title": title
            }

        return {
            "status": "unknown",
            "detected_title": title
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)


############################################################
# 📸 CAPTURE — SAVE BOOK
############################################################
@app.post("/capture")
async def capture_book(uid: str = Query(...), file: UploadFile = File(...)):

    path = save_temp_file(file)

    try:
        title = await run_ocr(path)

        if not title or len(title) < 4:
            return {"status": "failed"}

        book, score = search_book(title)

        # BOOK ALREADY KNOWN
        if book:
            final_title = book["title"]
        else:
            final_title = title
            add_book(final_title)

        if user_has_book(uid, final_title):
            return {
                "status": "already_saved",
                "title": final_title
            }

        save_book_for_user(uid, final_title)

        return {
            "status": "saved",
            "title": final_title
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)