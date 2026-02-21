import firebase_admin
from firebase_admin import credentials, firestore
import os, json, re, hashlib
from datetime import datetime, timedelta
from rapidfuzz import fuzz
from threading import Lock
from collections import OrderedDict

# ---------------------------------------------------
# INIT FIREBASE ADMIN
# ---------------------------------------------------
if not firebase_admin._apps:

    firebase_key = os.environ.get("FIREBASE_KEY")

    if firebase_key:
        cred_dict = json.loads(firebase_key)
    else:
        with open("serviceAccountKey.json") as f:
            cred_dict = json.load(f)

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------------------------------------------
# NORMALIZE TITLE (OCR FRIENDLY)
# ---------------------------------------------------
def normalize_title(text: str) -> str:
    text = text.lower()

    fixes = {
        "machne":"machine",
        "macnine":"machine",
        "lernng":"learning",
        "larnng":"learning",
        "inteligence":"intelligence",
        "artifcial":"artificial",
        "pyth0n":"python",
        "alg0rithm":"algorithm",
        "databse":"database",
        "powere":"power",
        "netwrok":"network",
        "computre":"computer"
    }

    for w,c in fixes.items():
        text = text.replace(w,c)

    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------
# BOOK FINGERPRINT (UNIQUE ID)
# ---------------------------------------------------
def book_fingerprint(title: str) -> str:
    words = normalize_title(title).split()
    words = sorted(set(words))[:6]
    key = " ".join(words)
    return hashlib.md5(key.encode()).hexdigest()

# ---------------------------------------------------
# CACHE (FAST CHECK)
# ---------------------------------------------------
CACHE_TTL = timedelta(minutes=5)
CACHE_LIMIT = 200
_user_cache = OrderedDict()
cache_lock = Lock()

def load_user_books(user_id: str):

    now = datetime.utcnow()

    with cache_lock:
        if user_id in _user_cache:
            titles, expire = _user_cache[user_id]
            if now < expire:
                _user_cache.move_to_end(user_id)
                return titles

    docs = db.collection("users").document(user_id).collection("books").stream()
    titles = set(normalize_title(doc.to_dict().get("title","")) for doc in docs)

    with cache_lock:
        _user_cache[user_id] = (titles, now + CACHE_TTL)
        if len(_user_cache) > CACHE_LIMIT:
            _user_cache.popitem(last=False)

    return titles

# ---------------------------------------------------
# SIMILARITY CHECK (OCR TOLERANT)
# ---------------------------------------------------
def is_similar(a,b):
    if abs(len(a)-len(b)) > 18:
        return False

    return max(
        fuzz.token_set_ratio(a,b),
        fuzz.partial_ratio(a,b)
    ) >= 78

# ---------------------------------------------------
# CHECK USER OWNS BOOK
# ---------------------------------------------------
def user_has_book(user_id: str, title: str):
    clean = normalize_title(title)
    books = load_user_books(user_id)

    for b in books:
        if is_similar(clean, b):
            return True

    return False

# ---------------------------------------------------
# SAVE BOOK (FINAL STABLE)
# ---------------------------------------------------
def save_book_for_user(user_id: str, title: str):

    clean = normalize_title(title)
    fingerprint = book_fingerprint(clean)

    user_ref = db.collection("users").document(user_id)
    doc_ref = user_ref.collection("books").document(fingerprint)

    print("🔥 Trying save:", clean)

    # Duplicate check (FAST CACHE FIRST)
    books = load_user_books(user_id)
    for b in books:
        if is_similar(clean, b):
            print("⚠️ Similar book exists — skip")
            return False

    # Firestore Transaction
    @firestore.transactional
    def txn(transaction):

        snapshot = doc_ref.get(transaction=transaction)
        if snapshot.exists:
            print("⚠️ Exact duplicate")
            return False

        transaction.set(doc_ref,{
            "title": clean,
            "original": title,
            "time": datetime.utcnow()
        })
        return True

    created = txn(db.transaction())

    if created:
        print("✅ FIREBASE SAVED:", clean)

        # FIXED CACHE UPDATE
        with cache_lock:
            if user_id in _user_cache:
                titles, expire = _user_cache[user_id]
                titles.add(clean)
                _user_cache[user_id] = (titles, expire)

    else:
        print("❌ SAVE FAILED")

    return created