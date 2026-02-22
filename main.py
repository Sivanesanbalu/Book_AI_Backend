import os
import uuid
import shutil
import asyncio
from threading import Lock
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

import startup
from image_search import search_book, add_book
from firebase_service import save_book_for_user, user_has_book

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 4 * 1024 * 1024
index_lock = Lock()


# ---------------- STARTUP ----------------
@app.on_event("startup")
def boot():
    startup.start_ai()


# ---------------- IMAGE VALIDATION ----------------
def validate_image(path: str):
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except:
        return False


# ---------------- SAVE TEMP FILE ----------------
def save_temp(file: UploadFile):
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_FILE_SIZE:
        raise ValueError("file_too_large")

    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file.file.close()
    return path


# ---------------- CAPTURE API ----------------
@app.post("/capture")
async def capture(uid: str = Query(...), file: UploadFile = File(...)):
    path = None

    # ---------- SAVE FILE ----------
    try:
        path = save_temp(file)
    except ValueError:
        return JSONResponse(status_code=413, content={"status": "file_too_large"})

    if not validate_image(path):
        if os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        # ---------- SEARCH BOOK ----------
        book, score = await asyncio.wait_for(
            asyncio.to_thread(search_book, path),
            timeout=12
        )

        # ============================================================
        # STRONG MATCH (definitely same book)
        # ============================================================
        if book is not None and score >= 0.82:
            title = book["title"]

            if user_has_book(uid, title):
                return {"status": "owned", "title": title}

            save_book_for_user(uid, title)

            return {
                "status": "matched",
                "title": title,
                "confidence": round(float(score), 3)
            }

        # ============================================================
        # WEAK MATCH (real-world variation: lighting/distance/angle)
        # DO NOT ADD AGAIN
        # ============================================================
        if book is not None and 0.70 <= score < 0.82:
            title = book["title"]

            if user_has_book(uid, title):
                return {"status": "owned", "title": title}

            save_book_for_user(uid, title)

            return {
                "status": "matched_weak",
                "title": title,
                "confidence": round(float(score), 3)
            }

        # ============================================================
        # NEW BOOK (only here we add)
        # ============================================================
        unique_title = f"Book_{uuid.uuid4().hex[:8]}"

        with index_lock:
            added = await asyncio.wait_for(
                asyncio.to_thread(add_book, path, unique_title),
                timeout=12
            )

        if not added:
            return {"status": "duplicate_prevented"}

        save_book_for_user(uid, unique_title)

        return {
            "status": "new_book",
            "title": unique_title
        }

    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content={"status": "timeout"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if path and os.path.exists(path):
            os.remove(path)