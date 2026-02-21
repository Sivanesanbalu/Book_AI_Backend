import firebase_admin
from firebase_admin import credentials, firestore
import hashlib
import os

# ---------------- INIT FIREBASE ----------------
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ------------------------------------------------
# GENERATE SAFE BOOK ID (IMPORTANT)
# ------------------------------------------------
def book_id(title: str) -> str:
    """
    Create stable id independent of spaces / case / symbols
    """
    normalized = title.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


# ----------------------------------------
# CHECK USER ALREADY HAS BOOK
# ----------------------------------------
def user_has_book(uid: str, title: str) -> bool:
    try:
        doc_ref = (
            db.collection("users")
            .document(uid)
            .collection("books")
            .document(book_id(title))
        )
        return doc_ref.get().exists
    except Exception as e:
        print("🔥 Firebase check error:", e)
        return False


# ----------------------------------------
# SAVE BOOK FOR USER
# ----------------------------------------
def save_book_for_user(uid: str, title: str):
    try:
        doc_ref = (
            db.collection("users")
            .document(uid)
            .collection("books")
            .document(book_id(title))
        )

        # prevent overwrite spam
        if doc_ref.get().exists:
            return False

        doc_ref.set({
            "title": title,
            "createdAt": firestore.SERVER_TIMESTAMP
        })

        print(f"📘 Saved for user {uid}: {title}")
        return True

    except Exception as e:
        print("🔥 Firebase save error:", e)
        return False