import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from image_search import search_book, add_book
from firebase_service import save_book_for_user, user_has_book

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------
# Save uploaded image safely
# ------------------------------------------------
def save_temp(file: UploadFile):
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()   # ⭐ VERY IMPORTANT (prevents memory leak)
    return path


# ------------------------------------------------
# CAPTURE ENDPOINT
# ------------------------------------------------
@app.post("/capture")
async def capture(uid: str = Query(...), file: UploadFile = File(...)):

    path = save_temp(file)

    try:
        # STEP 1 — Try match existing book
        book, score = search_book(path)

        if book is not None:

            title = book["title"]

            # Already owned
            if user_has_book(uid, title):
                return {"status": "owned", "title": title}

            # Save only ownership
            save_book_for_user(uid, title)

            return {
                "status": "matched",
                "title": title,
                "confidence": round(float(score), 3)
            }

        # STEP 2 — Unknown book → add safely
        unique_title = f"Book_{uuid.uuid4().hex[:8]}"

        try:
            add_book(path, unique_title)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "index_add_failed"}
            )

        save_book_for_user(uid, unique_title)

        return {
            "status": "new_book",
            "title": unique_title
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        if os.path.exists(path):
            os.remove(path)