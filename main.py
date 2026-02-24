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
from assistant_api import router as assistant_router
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
app.include_router(assistant_router)
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


# =========================================================
# 🔍 SCAN BOOK  (SEARCH ONLY — NEVER SAVES)
# =========================================================
@app.post("/scan")
async def scan(uid: str = Query(...), file: UploadFile = File(...)):
    path = None

    try:
        path = save_temp(file)
    except ValueError:
        return JSONResponse(status_code=413, content={"status": "file_too_large"})

    if not validate_image(path):
        if os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        book, score = await asyncio.to_thread(search_book, path)

        if book is None:
            return {"status": "not_found"}

        title = book["title"]

        if user_has_book(uid, title):
            return {"status": "owned", "title": title}

        return {
            "status": "found",
            "title": title,
            "confidence": round(float(score), 3)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# =========================================================
# 📷 ADD BOOK  (FORCE SAVE / LEARN)
# =========================================================
@app.post("/add")
async def add(uid: str = Query(...), file: UploadFile = File(...)):
    path = None

    try:
        path = save_temp(file)
    except ValueError:
        return JSONResponse(status_code=413, content={"status": "file_too_large"})

    if not validate_image(path):
        if os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        # check already exists in AI DB
        book, score = await asyncio.to_thread(search_book, path)

        # ---------- ALREADY KNOWN BOOK ----------
        if book is not None:
            title = book["title"]

            if user_has_book(uid, title):
                return {"status": "already_saved", "title": title}

            save_book_for_user(uid, title)
            return {"status": "saved_existing", "title": title}

        # ---------- NEW BOOK ----------
        unique_title = f"Book_{uuid.uuid4().hex[:8]}"

        with index_lock:
            await asyncio.to_thread(add_book, path, unique_title)

        save_book_for_user(uid, unique_title)

        return {"status": "saved_new", "title": unique_title}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if path and os.path.exists(path):
            os.remove(path)