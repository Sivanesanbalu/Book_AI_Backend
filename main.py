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

# -------- FLOW 2 IMPORTS --------
from vision_ai.vision import detect_book
from vision_ai.book_fetcher import get_book_info
from vision_ai.ai_summary import summarize_book


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 4 * 1024 * 1024
index_lock = Lock()


# =========================================================
# 🖼 IMAGE VALIDATION
# =========================================================
def validate_image(path: str):
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except:
        return False


# =========================================================
# 💾 SAVE TEMP FILE
# =========================================================
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


# =========================================================
# 🔍 FLOW-1 → SCAN BOOK
# =========================================================
@app.post("/scan")
async def scan(uid: str = Query(...), file: UploadFile = File(...)):
    path = save_temp(file)

    if not validate_image(path):
        os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        book, score = await asyncio.to_thread(search_book, path)

        if book is None:
            return {"status": "not_found"}

        title = book["title"]

        if user_has_book(uid, title):
            return {"status": "owned", "title": title}

        return {"status": "found", "title": title, "confidence": round(float(score), 3)}

    finally:
        if os.path.exists(path):
            os.remove(path)


# =========================================================
# ➕ FLOW-1 → ADD BOOK
# =========================================================
@app.post("/add")
async def add(uid: str = Query(...), file: UploadFile = File(...)):
    path = save_temp(file)

    if not validate_image(path):
        os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        book, score = await asyncio.to_thread(search_book, path)

        if book is not None:
            title = book["title"]
            save_book_for_user(uid, title)
            return {"status": "saved_existing", "title": title}

        unique_title = f"Book_{uuid.uuid4().hex[:8]}"

        with index_lock:
            await asyncio.to_thread(add_book, path, unique_title)

        save_book_for_user(uid, unique_title)
        return {"status": "saved_new", "title": unique_title}

    finally:
        if os.path.exists(path):
            os.remove(path)


# =========================================================
# 🤖 FLOW-2 → ASK BOOK AI (VISION → SUMMARY)
# =========================================================
@app.post("/ask-book-ai")
async def ask_book_ai(file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{uuid.uuid4().hex}.jpg"

    with open(path, "wb") as f:
        f.write(await file.read())

    book_name = detect_book(path)
    book = get_book_info(book_name)

    if not book:
        return JSONResponse({"error": "Book not found"}, status_code=404)

    overview = summarize_book(book)

    os.remove(path)

    return {
        "title": book["title"],
        "overview": overview
    }