import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime, timedelta
import re
from rapidfuzz import fuzz
from threading import Lock

# ---------------------------------------------------
# INIT FIREBASE
# ---------------------------------------------------
if not firebase_admin._apps:

    firebase_key = os.environ.get("FIREBASE_KEY")

    if firebase_key:
        cred_dict = json.loads(firebase_key)

    else:
        key_file = "serviceAccountKey.json"
        if not os.path.exists(key_file):
            raise Exception("Firebase key missing")

        with open(key_file, "r") as f:
            cred_dict = json.load(f)

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------------------------------------------
# NORMALIZE TITLE
# ---------------------------------------------------
def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# MEMORY SAFE TTL CACHE
# ---------------------------------------------------
CACHE_TTL = timedelta(minutes=5)
CACHE_LIMIT = 500   # prevents memory explosion
_user_cache = {}
cache_lock = Lock()


def load_user_books(user_id: str):

    now = datetime.utcnow()

    with cache_lock:

        # valid cache
        if user_id in _user_cache:
            titles, expire = _user_cache[user_id]
            if now < expire:
                return titles

        # fetch from firestore
        docs = db.collection("users").document(user_id).collection("books").stream()

        titles = []
        for doc in docs:
            data = doc.to_dict()
            title = normalize_title(data.get("title", ""))
            if title:
                titles.append(title)

        # cache cleanup (prevent memory leak)
        if len(_user_cache) > CACHE_LIMIT:
            _user_cache.clear()

        _user_cache[user_id] = (titles, now + CACHE_TTL)

        return titles


# ---------------------------------------------------
# CHECK USER OWNS BOOK (FAST)
# ---------------------------------------------------
def user_has_book(user_id: str, title: str) -> bool:

    clean_title = normalize_title(title)
    user_books = load_user_books(user_id)

    # fast exact match
    if clean_title in user_books:
        return True

    # fuzzy backup
    for saved in user_books:
        if fuzz.token_sort_ratio(clean_title, saved) >= 90:
            return True

    return False


# ---------------------------------------------------
# SAVE BOOK (ATOMIC SAFE)
# ---------------------------------------------------
def save_book_for_user(user_id: str, title: str):

    clean_title = normalize_title(title)

    # double check before write
    if user_has_book(user_id, clean_title):
        return False

    doc_ref = db.collection("users").document(user_id).collection("books")

    doc_ref.add({
        "title": clean_title,
        "original_title": title,
        "createdAt": datetime.utcnow()
    })

    # update cache safely
    with cache_lock:
        if user_id in _user_cache:
            titles, expire = _user_cache[user_id]
            titles.append(clean_title)
            _user_cache[user_id] = (titles, expire)

    return True