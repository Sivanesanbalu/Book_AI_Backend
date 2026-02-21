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
# TTL CACHE (SET BASED = FAST)
# ---------------------------------------------------
CACHE_TTL = timedelta(minutes=5)
CACHE_LIMIT = 500
_user_cache = {}
cache_lock = Lock()


def load_user_books(user_id: str):

    now = datetime.utcnow()

    with cache_lock:

        if user_id in _user_cache:
            titles, expire = _user_cache[user_id]
            if now < expire:
                return titles

        docs = db.collection("users").document(user_id).collection("books").stream()

        titles = set()

        for doc in docs:
            data = doc.to_dict()
            title = normalize_title(data.get("title", ""))
            if title:
                titles.add(title)

        if len(_user_cache) > CACHE_LIMIT:
            _user_cache.clear()

        _user_cache[user_id] = (titles, now + CACHE_TTL)

        return titles


# ---------------------------------------------------
# MATCHING (OCR FRIENDLY)
# ---------------------------------------------------
def is_similar(a: str, b: str) -> bool:

    # fast length reject
    if abs(len(a) - len(b)) > 12:
        return False

    score1 = fuzz.token_set_ratio(a, b)
    score2 = fuzz.partial_ratio(a, b)

    return max(score1, score2) >= 85


def user_has_book(user_id: str, title: str) -> bool:

    clean_title = normalize_title(title)
    user_books = load_user_books(user_id)

    if clean_title in user_books:
        return True

    for saved in user_books:
        if is_similar(clean_title, saved):
            return True

    return False


# ---------------------------------------------------
# ATOMIC SAVE (NO DUPLICATES EVER)
# ---------------------------------------------------
def save_book_for_user(user_id: str, title: str):

    clean_title = normalize_title(title)

    user_ref = db.collection("users").document(user_id)
    books_ref = user_ref.collection("books")

    # deterministic doc id prevents duplicates
    doc_id = clean_title.replace(" ", "_")[:120]
    doc_ref = books_ref.document(doc_id)

    # transaction = atomic
    @firestore.transactional
    def txn(transaction):

        snapshot = doc_ref.get(transaction=transaction)
        if snapshot.exists:
            return False

        transaction.set(doc_ref, {
            "title": clean_title,
            "original_title": title,
            "createdAt": datetime.utcnow()
        })

        return True

    transaction = db.transaction()
    created = txn(transaction)

    # update cache safely
    if created:
        with cache_lock:
            if user_id in _user_cache:
                titles, expire = _user_cache[user_id]
                titles.add(clean_title)
                _user_cache[user_id] = (titles, expire)

    return created