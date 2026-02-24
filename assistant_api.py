from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from image_search import search_book
from llm_reply import generate_book_reply
import asyncio
import os
import uuid
import shutil

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ================= TEXT REQUEST =================
class AskRequest(BaseModel):
    question: str | None = None
    uid: str | None = None


@router.post("/ask")
async def ask_ai(data: AskRequest):

    question = data.question or "Explain this clearly in simple words"

    try:
        reply = await asyncio.to_thread(
            generate_book_reply,
            "General Book",
            "Education",
            question
        )

        return {"answer": reply}

    except Exception as e:
        print("AI TEXT ERROR:", e)
        return {"answer": "AI thinking... please try again"}


# ================= IMAGE REQUEST =================
@router.post("/ask-image")
async def ask_ai_image(
    question: str = Form(None),
    file: UploadFile = File(...)
):

    question = question or "Explain this book clearly in simple words"

    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")

    try:
        # ---------- SAVE IMAGE ----------
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # VERY IMPORTANT (Render bug fix)
        await file.close()

        # ---------- SEARCH BOOK ----------
        book_name = "Unknown Book"
        topic = "Education"

        try:
            book, score = await asyncio.to_thread(search_book, temp_path)

            if book:
                book_name = book.get("title", "Unknown Book")
                topic = book.get("topic", "Education")

        except Exception as e:
            print("FAISS SEARCH FAIL:", e)

        # ---------- GENERATE AI ----------
        try:
            reply = await asyncio.to_thread(
                generate_book_reply,
                book_name,
                topic,
                question
            )
        except Exception as e:
            print("LLM FAIL:", e)
            reply = f"I detected {book_name}. But AI failed to explain. Try again."

        return {
            "book": book_name,
            "topic": topic,
            "answer": reply
        }

    except Exception as e:
        print("IMAGE ASK FAIL:", e)
        return {"answer": "Image processing failed. Try another photo"}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)