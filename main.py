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

# 🔐 FAISS safety (multiple users scan pannumboth crash aagadhu)
faiss_lock = Lock()


# ---------------- STARTUP ----------------
@app.on_event("startup")
async def boot():
    # model load block panna koodadhu
    await asyncio.to_thread(startup.start_ai)


# health check (Render ku important)
@app.get("/")
def root():
    return {"status": "alive"}


# ---------------- IMAGE VALIDATION ----------------
def validate_image(path: str):
    try:
        img = Image.open(path)
        img.verify()
        img.close()
        return True
    except Exception:
        return False


# ---------------- SAVE TEMP FILE ----------------
def save_temp(file: UploadFile):

    # only images allow
    if not file.content_type.startswith("image/"):
        raise ValueError("invalid_type")

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
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": str(e)})

    if not validate_image(path):
        if os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        # thread safe FAISS search
        with faiss_lock:
            book, score = await asyncio.to_thread(search_book, path)

        if book is None:
            return {"status": "not_found"}

        title = book["title"]

        owned = await asyncio.to_thread(user_has_book, uid, title)
        if owned:
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
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": str(e)})

    if not validate_image(path):
        if os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        with faiss_lock:
            book, score = await asyncio.to_thread(search_book, path)

        # ---------- ALREADY KNOWN BOOK ----------
        if book is not None:
            title = book["title"]

            owned = await asyncio.to_thread(user_has_book, uid, title)
            if owned:
                return {"status": "already_saved", "title": title}

            await asyncio.to_thread(save_book_for_user, uid, title)
            return {"status": "saved_existing", "title": title}

        # ---------- NEW BOOK ----------
        unique_title = f"Book_{uuid.uuid4().hex[:8]}"

        with index_lock:
            await asyncio.to_thread(add_book, path, unique_title)

        await asyncio.to_thread(save_book_for_user, uid, unique_title)

        return {"status": "saved_new", "title": unique_title}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if path and os.path.exists(path):
            os.remove(path)