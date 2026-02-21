import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from ocr import extract_text
from search_engine import search_book, add_book
from firebase_service import save_book_for_user, user_has_book
from title_memory import get_memory

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------
# SAVE TEMP IMAGE
# ------------------------------------------------
def save_temp_file(upload_file: UploadFile) -> str:
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path


# ------------------------------------------------
# NON BLOCKING OCR (THREAD SAFE)
# ------------------------------------------------
async def run_ocr(path: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_text, path)


# ------------------------------------------------
# FIND BEST MATCH FROM OCR CANDIDATES
# ------------------------------------------------
def best_match_from_candidates(candidates):

    best_book = None
    best_score = 0
    best_text = None

    for text in candidates:
        if not text or len(text) < 3:
            continue

        book, score = search_book(text)

        if score > best_score:
            best_book = book
            best_score = score
            best_text = text

    return best_book, best_score, best_text


# ==============================================================
# 🔎 SCAN  → ONLY DETECT (NO SAVE)
# ==============================================================
@app.post("/scan")
async def scan_book(uid: str = Query(...), file: UploadFile = File(...)):

    path = save_temp_file(file)

    try:
        titles = await run_ocr(path)

        if not titles:
            return {"status": "no_text"}

        book, score, detected = best_match_from_candidates(titles)

        # decide candidate title
        if book:
            candidate_title = book["title"]
            book_known = True
        elif detected:
            candidate_title = detected
            book_known = False
        else:
            return {"status": "no_text"}

        # -------- STABILITY MEMORY --------
        memory = get_memory(uid)
        stable_title = memory.update(candidate_title)

        # camera not steady yet
        if not stable_title:
            return {"status": "scanning"}

        # -------- FINAL RESULT --------
        if user_has_book(uid, stable_title):
            return {
                "status": "owned",
                "title": stable_title
            }

        if book_known:
            return {
                "status": "known_book",
                "title": stable_title,
                "confidence": round(float(score), 3)
            }

        return {
            "status": "detected",
            "title": stable_title,
            "confidence": round(float(score), 3),
            "note": "New book (not in database)"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)


# ==============================================================
# 📸 CAPTURE → ONLY SAVE (NO OCR AGAIN)
# ==============================================================
@app.post("/capture")
async def capture_book(uid: str = Query(...), file: UploadFile = File(...)):

    path = save_temp_file(file)

    try:
        memory = get_memory(uid)
        final_title = memory.confirm()

        # ❗ user pressed save before stable detection
        if not final_title:
            return {"status": "scan_first"}

        # already saved
        if user_has_book(uid, final_title):
            return {
                "status": "already_saved",
                "title": final_title
            }

        # add to global DB (FAISS incremental)
        add_book(final_title)

        # add to user collection
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