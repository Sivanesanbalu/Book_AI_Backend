import os
import uuid
import shutil
import asyncio
from threading import Lock
from asyncio import Semaphore
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import startup
from image_search import search_book, add_book
from firebase_service import save_book_for_user, user_has_book
from assistant_api import router as assistant_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.include_router(assistant_router)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 4 * 1024 * 1024
index_lock = Lock()
faiss_lock = Lock()
gpu_semaphore = Semaphore(1)

# -------- AI READY FLAG --------
ai_ready = False

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def boot():
    global ai_ready
    await asyncio.to_thread(startup.start_ai)
    ai_ready = True

@app.get("/")
def root():
    return {"status": "alive"}

# ---------------- IMAGE VALIDATION ----------------
def validate_image(path: str):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

# ---------------- SAVE TEMP FILE ----------------
def save_temp(file: UploadFile):

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
        buffer.flush()
        os.fsync(buffer.fileno())

    file.file.close()
    return path

# =========================================================
# 🔍 SCAN BOOK
# =========================================================
@app.post("/scan")
async def scan(uid: str = Query(...), file: UploadFile = File(...)):

    if not ai_ready:
        return {"status": "ai_loading"}

    try:
        path = save_temp(file)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": str(e)})

    if not validate_image(path):
        os.remove(path)
        return {"status": "invalid_image"}

    try:
        async with gpu_semaphore:
            try:
                book, score = await asyncio.wait_for(
                    asyncio.to_thread(search_book, path),
                    timeout=15
                )
            except asyncio.TimeoutError:
                return {"status": "ai_busy"}
            except Exception:
                return {"status": "ai_error"}

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

    finally:
        if os.path.exists(path):
            os.remove(path)

# =========================================================
# 📷 ADD BOOK
# =========================================================
@app.post("/add")
async def add(uid: str = Query(...), file: UploadFile = File(...)):

    if not ai_ready:
        return {"status": "ai_loading"}

    try:
        path = save_temp(file)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": str(e)})

    if not validate_image(path):
        os.remove(path)
        return {"status": "invalid_image"}

    try:
        async with gpu_semaphore:
            try:
                book, score = await asyncio.wait_for(
                    asyncio.to_thread(search_book, path),
                    timeout=15
                )
            except asyncio.TimeoutError:
                return {"status": "ai_busy"}
            except Exception:
                return {"status": "ai_error"}

        if book is not None:
            title = book["title"]
            owned = await asyncio.to_thread(user_has_book, uid, title)

            if owned:
                return {"status": "already_saved", "title": title}

            await asyncio.to_thread(save_book_for_user, uid, title)
            return {"status": "saved_existing", "title": title}

        unique_title = f"Book_{uuid.uuid4().hex[:8]}"

        with index_lock:
            await asyncio.to_thread(add_book, path, unique_title)

        await asyncio.to_thread(save_book_for_user, uid, unique_title)
        return {"status": "saved_new", "title": unique_title}

    finally:
        if os.path.exists(path):
            os.remove(path)