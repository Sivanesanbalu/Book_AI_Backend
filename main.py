import os
import uuid
import shutil
import asyncio
import socket
from threading import Lock
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Request, Body, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from image_search import search_book, add_book
from firebase_service import (
    save_book_for_user,
    user_has_book,
    verify_user,
    get_user_books,
    remove_book_for_user,
    log_scan_event,
    get_user_profile,
    update_user_profile
)

# -------- FLOW 2 & VISION IMPORTS --------
from vision_ai.vision import detect_book
from vision_ai.book_fetcher import get_book_info
from vision_ai.ai_summary import summarize_book, generate_book_scenario, chat_with_book_ai

app = FastAPI(title="BookNest AI", version="2.0.0")

# Enable CORS for local testing and mobile web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
index_lock = Lock()


# =========================================================
# 🔐 AUTH HELPER
# =========================================================
async def get_uid(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return "default_user"

    token = auth_header.split(" ")[-1] if " " in auth_header else auth_header
    uid = verify_user(token)
    return uid or "default_user"


# =========================================================
# 🖼 IMAGE VALIDATION & TEMP STORAGE
# =========================================================
def validate_image(path: str):
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


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
# 🌐 STATIC ASSETS & FRONTEND SERVING
# =========================================================
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    return {"status": "running", "service": "BookNest AI Backend"}


@app.get("/manifest.json")
def manifest():
    if os.path.exists("static/manifest.json"):
        return FileResponse("static/manifest.json")
    return JSONResponse(status_code=404, content={"error": "manifest not found"})


@app.get("/download")
@app.get("/BookNestAI.apk")
def download_apk():
    apk_path = "static/BookNestAI.apk"
    if os.path.exists(apk_path):
        return FileResponse(
            apk_path,
            media_type="application/vnd.android.package-archive",
            filename="BookNestAI.apk"
        )
    return JSONResponse(status_code=404, content={"error": "APK not found"})


@app.get("/sw.js")
def service_worker():
    if os.path.exists("static/sw.js"):
        return FileResponse("static/sw.js", media_type="application/javascript")
    return JSONResponse(status_code=404, content={"error": "service worker not found"})


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "BookNest AI",
        "version": "2.0.0"
    }


# =========================================================
# 🔍 FLOW-1 → SCAN BOOK (COVER MATCH + DB IDENTIFICATION)
# =========================================================
@app.post("/scan")
async def scan(request: Request, file: UploadFile = File(...)):
    uid = await get_uid(request)
    path = None

    try:
        path = save_temp(file)
    except ValueError:
        return JSONResponse(status_code=413, content={"status": "file_too_large"})

    if not validate_image(path):
        if path and os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        # 1. Search vector database
        book, score = await asyncio.to_thread(search_book, path)
        title = None
        author = None
        cover_url = None
        genre = "General"
        overview = None
        scenario = None

        if book is not None:
            title = book.get("title")
            author = book.get("author", "")
            cover_url = book.get("cover_url", "")
            genre = book.get("genre", "Literature")
            overview = book.get("overview", "")

        # 2. If vector match not found or low confidence, trigger Vision AI
        if not title:
            detected_name = await asyncio.to_thread(detect_book, path)
            if detected_name:
                meta = get_book_info(detected_name)
                if meta:
                    title = meta.get("title")
                    author = meta.get("authors")
                    cover_url = meta.get("cover_url")
                    genre = meta.get("genre", "Literature")
                    overview = meta.get("description")
                    score = 0.94

        if not title:
            return {"status": "not_found"}

        # Generate scenario synopsis for the book
        book_dict = {
            "title": title,
            "authors": author or "Unknown Author",
            "description": overview or "",
            "genre": genre
        }
        scenario = generate_book_scenario(book_dict)

        # Check if user already owns this book in their personal database
        is_owned = user_has_book(uid, title)
        status_str = "owned" if is_owned else "found"

        # Log scan event
        log_scan_event(uid, title, float(score or 1.0), status_str)

        return {
            "status": status_str,
            "title": title,
            "author": author or "Author Unknown",
            "genre": genre,
            "cover_url": cover_url,
            "confidence": round(float(score or 0.92), 3),
            "overview": overview,
            "scenario": scenario
        }

    except Exception as e:
        print("🔥 Scan endpoint error:", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# =========================================================
# 📷 ADD BOOK (VECTOR INDEX + USER LIBRARY)
# =========================================================
@app.post("/add")
async def add(request: Request, file: UploadFile = File(...), title: str = None):
    uid = await get_uid(request)
    path = None

    try:
        path = save_temp(file)
    except ValueError:
        return JSONResponse(status_code=413, content={"status": "file_too_large"})

    if not validate_image(path):
        if path and os.path.exists(path):
            os.remove(path)
        return JSONResponse(status_code=400, content={"status": "invalid_image"})

    try:
        book, score = await asyncio.to_thread(search_book, path)

        if book is not None:
            found_title = book["title"]
            if user_has_book(uid, found_title):
                return {"status": "already_saved", "title": found_title}

            save_book_for_user(
                uid=uid,
                title=found_title,
                author=book.get("author", ""),
                genre=book.get("genre", "General"),
                cover_url=book.get("cover_url", ""),
                overview=book.get("overview", "")
            )
            return {"status": "saved_existing", "title": found_title}

        # New book registration
        final_title = title if title else f"Book_{uuid.uuid4().hex[:8]}"
        meta = get_book_info(final_title) or {}

        with index_lock:
            await asyncio.to_thread(
                add_book,
                path,
                final_title,
                meta.get("authors", ""),
                meta.get("cover_url", ""),
                meta.get("description", ""),
                meta.get("genre", "General")
            )

        save_book_for_user(
            uid=uid,
            title=final_title,
            author=meta.get("authors", ""),
            genre=meta.get("genre", "General"),
            cover_url=meta.get("cover_url", ""),
            overview=meta.get("description", "")
        )

        return {"status": "saved_new", "title": final_title}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# =========================================================
# 🤖 FLOW-2 → AI BOOK EXPLAIN
# =========================================================
@app.post("/ask-book-ai")
async def ask_book_ai(file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{uuid.uuid4().hex}.jpg"

    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty image uploaded"})

        with open(path, "wb") as f:
            f.write(contents)

        book_name = await asyncio.to_thread(detect_book, path)
        if not book_name:
            return JSONResponse(status_code=422, content={"error": "Could not identify book"})

        book = get_book_info(book_name)
        if not book:
            return JSONResponse(status_code=404, content={"error": f"No info found for '{book_name}'"})

        scenario = generate_book_scenario(book)
        overview = await asyncio.to_thread(summarize_book, book)

        return {
            "title": book["title"],
            "authors": book.get("authors", ""),
            "overview": overview,
            "scenario": scenario
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)


# =========================================================
# 💬 INTERACTIVE CHATBOT (MINI CHAT BOX & STORYTELLER)
# =========================================================
@app.post("/api/chat")
async def api_chat(payload: dict = Body(...)):
    book_title = payload.get("book_title", "Unknown Book")
    book_author = payload.get("book_author", "")
    book_description = payload.get("book_description", "")
    book_genre = payload.get("book_genre", "Literature")
    user_message = payload.get("user_message", "")
    history = payload.get("history", [])

    if not user_message:
        return JSONResponse(status_code=400, content={"error": "user_message is required"})

    book_dict = {
        "title": book_title,
        "authors": book_author,
        "description": book_description,
        "genre": book_genre
    }

    reply = await asyncio.to_thread(chat_with_book_ai, book_dict, user_message, history)
    return {"reply": reply}


# =========================================================
# 📚 USER LIBRARY API (CRUD)
# =========================================================
@app.get("/api/library")
async def api_get_library(request: Request):
    uid = await get_uid(request)
    books = get_user_books(uid)
    return books


@app.post("/api/library/add")
async def api_add_library(request: Request, payload: dict = Body(...)):
    uid = await get_uid(request)
    title = payload.get("title")
    if not title:
        return JSONResponse(status_code=400, content={"error": "title is required"})

    save_book_for_user(
        uid=uid,
        title=title,
        author=payload.get("author", ""),
        genre=payload.get("genre", "General"),
        cover_url=payload.get("cover_url", ""),
        overview=payload.get("overview", ""),
        confidence=float(payload.get("confidence", 1.0)),
        status=payload.get("status", "Reading"),
        rating=int(payload.get("rating", 5)),
        notes=payload.get("notes", "")
    )
    return {"status": "saved", "title": title}


@app.delete("/api/library/remove")
async def api_remove_library(request: Request, title: str = Query(...)):
    uid = await get_uid(request)
    remove_book_for_user(uid, title)
    return {"status": "removed", "title": title}


# =========================================================
# 👤 USER PROFILE API
# =========================================================
@app.get("/api/profile")
async def api_profile(request: Request):
    uid = await get_uid(request)
    profile = get_user_profile(uid)
    return profile


@app.post("/api/profile")
async def api_update_profile(request: Request, payload: dict = Body(...)):
    uid = await get_uid(request)
    profile = update_user_profile(
        uid=uid,
        name=payload.get("name"),
        email=payload.get("email"),
        reading_goal=payload.get("reading_goal"),
        favorite_genres=payload.get("favorite_genres")
    )
    return profile


# =========================================================
# 📲 PHONE CONNECT & NETWORK INFO
# =========================================================
@app.get("/api/network-info")
def get_network_info():
    """
    Returns the public HTTPS tunnel URL or Mac's local network IP for mobile phone access
    """
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "10.16.201.1"

    port = 8000
    public_url = None
    if os.path.exists("/tmp/cloudflared.log"):
        try:
            import re
            with open("/tmp/cloudflared.log") as f:
                content = f.read()
                matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if matches:
                    public_url = matches[-1]
        except Exception:
            pass

    mobile_url = public_url if public_url else f"http://{local_ip}:{port}"

    return {
        "local_ip": local_ip,
        "port": port,
        "public_url": public_url,
        "mobile_url": mobile_url,
        "instructions": "Scan the QR Code on your mobile phone or open the mobile URL in Safari/Chrome."
    }