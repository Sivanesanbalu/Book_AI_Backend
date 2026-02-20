import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime, timedelta
import re
from rapidfuzz import fuzz

# ---------------------------------------------------
# INIT FIREBASE
# ---------------------------------------------------
if not firebase_admin._apps:
    firebase_key = os.environ.get("FIREBASE_KEY")

    if firebase_key is None:
        key_file = "serviceAccountKey.json"
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                cred_dict = json.load(f)
        else:
            raise Exception("Firebase key not found")
    else:
        cred_dict = json.loads(firebase_key)

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ---------------------------------------------------
# NORMALIZE TITLE
# ---------------------------------------------------
def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# TTL CACHE (AUTO MEMORY CONTROL)
# ---------------------------------------------------
CACHE_TTL = timedelta(minutes=5)
_user_cache = {}  # { user_id : (titles, expire_time) }


def load_user_books(user_id: str):

    now = datetime.utcnow()

    # return cache if valid
    if user_id in _user_cache:
        titles, expire = _user_cache[user_id]
        if now < expire:
            return titles

    # reload from firestore
    books_ref = db.collection("users").document(user_id).collection("books").stream()

    titles = []
    for book in books_ref:
        data = book.to_dict()
        titles.append(data.get("title", ""))

    # save with expiration
    _user_cache[user_id] = (titles, now + CACHE_TTL)

    return titles


# ---------------------------------------------------
# CHECK USER OWNS BOOK
# ---------------------------------------------------
def user_has_book(user_id: str, title: str) -> bool:

    clean_title = normalize_title(title)
    user_books = load_user_books(user_id)

    for saved in user_books:
        if fuzz.token_set_ratio(clean_title, saved) > 90:
            return True

    return False


# ---------------------------------------------------
# SAVE BOOK
# ---------------------------------------------------
def save_book_for_user(user_id: str, title: str):

    clean_title = normalize_title(title)

    if user_has_book(user_id, clean_title):
        return

    db.collection("users") \
      .document(user_id) \
      .collection("books") \
      .add({
        "title": clean_title,
        "original_title": title,
        "createdAt": datetime.utcnow()
      })

    # update cache safely
    if user_id in _user_cache:
        titles, expire = _user_cache[user_id]
        titles.append(clean_title)
        _user_cache[user_id] = (titles, expire)