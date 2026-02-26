import firebase_admin
from firebase_admin import credentials, firestore, auth
import hashlib
import os
import json
import re
import time

# ---------------- INIT FIREBASE ----------------
if not firebase_admin._apps:
    firebase_json = os.environ.get("FIREBASE_KEY")

    if not firebase_json:
        raise RuntimeError("FIREBASE_KEY environment variable not set")

    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# =========================================================
# 🔐 TOKEN CACHE (CRITICAL RENDER FIX)
# prevents cold start verification delay
# =========================================================
_last_verify = {}
CACHE_SECONDS = 3600  # 1 hour


def verify_user(id_token: str):
    """
    Verify Firebase ID token from Flutter Authorization header
    Returns uid or None
    """

    if not id_token:
        return None

    # ---------- CACHE HIT ----------
    if id_token in _last_verify:
        uid, exp = _last_verify[id_token]
        if time.time() < exp:
            return uid

    # ---------- VERIFY ----------
    try:
        decoded = auth.verify_id_token(id_token)

        uid = decoded["uid"]

        # cache it
        _last_verify[id_token] = (uid, time.time() + CACHE_SECONDS)

        return uid

    except Exception as e:
        print("🔥 Token verification failed:", e)
        return None


# =========================================================
# 📚 NORMALIZE TITLE (PREVENT DUPLICATES)
# Harry Potter == harry-potter == HARRY POTTER
# =========================================================
def normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title


def book_id(title: str) -> str:
    normalized = normalize_title(title)
    return hashlib.md5(normalized.encode()).hexdigest()


# =========================================================
# 🔍 CHECK USER HAS BOOK
# =========================================================
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


# =========================================================
# 💾 SAVE BOOK FOR USER
# =========================================================
def save_book_for_user(uid: str, title: str):

    try:
        doc_ref = (
            db.collection("users")
            .document(uid)
            .collection("books")
            .document(book_id(title))
        )

        doc_ref.set({
            "title": title,
            "normalized": normalize_title(title),
            "createdAt": firestore.SERVER_TIMESTAMP
        }, merge=True)

        print(f"📘 Saved for user {uid}: {title}")
        return True

    except Exception as e:
        print("🔥 Firebase save error:", e)
        return False