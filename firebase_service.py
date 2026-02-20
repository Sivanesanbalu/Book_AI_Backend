import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime
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
# CACHE USER BOOKS (VERY IMPORTANT FOR SPEED)
# ---------------------------------------------------
_user_cache = {}

def load_user_books(user_id: str):
    if user_id in _user_cache:
        return _user_cache[user_id]

    books_ref = db.collection("users").document(user_id).collection("books").stream()

    titles = []
    for book in books_ref:
        data = book.to_dict()
        titles.append(data.get("title", ""))

    _user_cache[user_id] = titles
    return titles


# ---------------------------------------------------
# CHECK USER OWNS BOOK
# ---------------------------------------------------
def user_has_book(user_id: str, title: str) -> bool:

    clean_title = normalize_title(title)
    user_books = load_user_books(user_id)

    for saved in user_books:
        score = fuzz.token_set_ratio(clean_title, saved)

        if score > 90:
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

    # update cache instantly
    if user_id in _user_cache:
        _user_cache[user_id].append(clean_title)