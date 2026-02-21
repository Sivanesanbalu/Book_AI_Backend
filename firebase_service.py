import firebase_admin
from firebase_admin import credentials, firestore
import hashlib
import os
import json

# ---------------- INIT FIREBASE (ENV BASED) ----------------
if not firebase_admin._apps:
    firebase_json = os.environ.get("FIREBASE_KEY")

    if not firebase_json:
        raise RuntimeError("FIREBASE_KEY environment variable not set")

    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ------------------------------------------------
# GENERATE SAFE BOOK ID
# ------------------------------------------------
def book_id(title: str) -> str:
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