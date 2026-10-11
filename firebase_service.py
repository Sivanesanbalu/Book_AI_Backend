import os
import json
import re
import time
import hashlib
import sqlite3
from threading import Lock

# Try initializing Firebase if credentials are provided
use_firestore = False
db = None
firebase_admin = None
auth = None
firestore = None

FIREBASE_KEY = os.environ.get("FIREBASE_KEY")

if FIREBASE_KEY:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, auth
        if not firebase_admin._apps:
            cred_dict = json.loads(FIREBASE_KEY)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        use_firestore = True
        print("✅ Firebase Firestore connected successfully.")
    except Exception as e:
        print(f"⚠️ Firebase initialization failed ({e}). Falling back to Local SQLite DB.")
        use_firestore = False
else:
    print("ℹ️ FIREBASE_KEY not set. Using Local SQLite Engine for library & user persistence.")

# ---------------- LOCAL SQLITE FALLBACK / STORAGE ----------------
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
SQLITE_PATH = os.path.join(DB_DIR, "library.db")
_db_lock = Lock()


def get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_local_db():
    with _db_lock:
        with get_sqlite_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                uid TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                reading_goal INTEGER DEFAULT 25,
                favorite_genres TEXT DEFAULT 'Fiction, Sci-Fi, Tech',
                avatar TEXT DEFAULT 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_books (
                uid TEXT,
                book_id TEXT,
                title TEXT,
                author TEXT,
                genre TEXT,
                cover_url TEXT,
                overview TEXT,
                confidence REAL DEFAULT 1.0,
                status TEXT DEFAULT 'Reading',
                rating INTEGER DEFAULT 5,
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (uid, book_id)
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT,
                title TEXT,
                confidence REAL,
                status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            # Ensure a default profile exists for the brother / primary user
            conn.execute("""
            INSERT OR IGNORE INTO users (uid, name, email, reading_goal, favorite_genres)
            VALUES ('default_user', 'Brother', 'brother@bookai.local', 25, 'Sci-Fi, AI & Tech, Fiction, Thriller')
            """)
            conn.commit()


init_local_db()

# ---------------- TOKEN CACHE ----------------
_last_verify = {}
CACHE_SECONDS = 3600


def verify_user(id_token: str) -> str:
    """
    Verify Firebase ID token or return local user uid
    """
    if not id_token:
        return "default_user"

    clean_token = id_token.strip()
    if clean_token.lower() in ("default", "guest", "demo", "brother", "user", "bearer demo"):
        return "default_user"

    if clean_token in _last_verify:
        uid, exp = _last_verify[clean_token]
        if time.time() < exp:
            return uid

    if use_firestore and auth:
        try:
            decoded = auth.verify_id_token(clean_token)
            uid = decoded["uid"]
            _last_verify[clean_token] = (uid, time.time() + CACHE_SECONDS)
            return uid
        except Exception as e:
            print("🔥 Firebase token verification failed:", e)
            # Graceful fallback to default user for local testing
            return "default_user"

    return "default_user"


# ---------------- NORMALIZATION ----------------
def normalize_title(title: str) -> str:
    if not title:
        return "unknown_book"
    t = title.lower().strip()
    t = re.sub(r'[^a-z0-9]', '', t)
    return t or "unknown_book"


def book_id(title: str) -> str:
    normalized = normalize_title(title)
    return hashlib.md5(normalized.encode()).hexdigest()


# ---------------- CHECK USER HAS BOOK ----------------
def user_has_book(uid: str, title: str) -> bool:
    b_id = book_id(title)
    if use_firestore and db:
        try:
            doc_ref = db.collection("users").document(uid).collection("books").document(b_id)
            return doc_ref.get().exists
        except Exception as e:
            print("🔥 Firebase check error:", e)

    # Local SQLite check
    with _db_lock:
        with get_sqlite_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM user_books WHERE uid = ? AND book_id = ?",
                (uid, b_id)
            ).fetchone()
            return row is not None


# ---------------- SAVE BOOK FOR USER ----------------
def save_book_for_user(uid: str, title: str, author: str = "", genre: str = "General",
                       cover_url: str = "", overview: str = "", confidence: float = 1.0,
                       status: str = "Reading", rating: int = 5, notes: str = ""):
    b_id = book_id(title)

    if use_firestore and db:
        try:
            doc_ref = db.collection("users").document(uid).collection("books").document(b_id)
            doc_ref.set({
                "title": title,
                "author": author,
                "genre": genre,
                "cover_url": cover_url,
                "overview": overview,
                "confidence": confidence,
                "status": status,
                "rating": rating,
                "notes": notes,
                "normalized": normalize_title(title),
                "createdAt": firestore.SERVER_TIMESTAMP if firestore else time.time()
            }, merge=True)
            print(f"📘 [Firestore] Saved for user {uid}: {title}")
        except Exception as e:
            print("🔥 Firebase save error:", e)

    # Always write to local SQLite as reliable mirror
    with _db_lock:
        with get_sqlite_conn() as conn:
            conn.execute("""
            INSERT INTO user_books (uid, book_id, title, author, genre, cover_url, overview, confidence, status, rating, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid, book_id) DO UPDATE SET
                title = excluded.title,
                author = COALESCE(NULLIF(excluded.author, ''), user_books.author),
                genre = COALESCE(NULLIF(excluded.genre, ''), user_books.genre),
                cover_url = COALESCE(NULLIF(excluded.cover_url, ''), user_books.cover_url),
                overview = COALESCE(NULLIF(excluded.overview, ''), user_books.overview),
                status = excluded.status,
                rating = excluded.rating
            """, (uid, b_id, title, author, genre, cover_url, overview, confidence, status, rating, notes))
            conn.commit()

    print(f"📘 Saved book: '{title}' for user: {uid}")
    return True


# ---------------- GET USER BOOKS ----------------
def get_user_books(uid: str):
    with _db_lock:
        with get_sqlite_conn() as conn:
            rows = conn.execute("""
            SELECT book_id, title, author, genre, cover_url, overview, confidence, status, rating, notes, created_at
            FROM user_books
            WHERE uid = ?
            ORDER BY created_at DESC
            """, (uid,)).fetchall()
            return [dict(r) for r in rows]


# ---------------- REMOVE BOOK ----------------
def remove_book_for_user(uid: str, title: str):
    b_id = book_id(title)
    if use_firestore and db:
        try:
            db.collection("users").document(uid).collection("books").document(b_id).delete()
        except Exception as e:
            print("🔥 Firebase delete error:", e)

    with _db_lock:
        with get_sqlite_conn() as conn:
            conn.execute("DELETE FROM user_books WHERE uid = ? AND book_id = ?", (uid, b_id))
            conn.commit()
    return True


# ---------------- LOG SCAN EVENT ----------------
def log_scan_event(uid: str, title: str, confidence: float, status: str):
    with _db_lock:
        with get_sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO scan_logs (uid, title, confidence, status) VALUES (?, ?, ?, ?)",
                (uid, title, confidence, status)
            )
            conn.commit()


# ---------------- USER PROFILE & STATS ----------------
def get_user_profile(uid: str):
    with _db_lock:
        with get_sqlite_conn() as conn:
            user_row = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()
            if not user_row:
                conn.execute(
                    "INSERT INTO users (uid, name, email) VALUES (?, 'Brother', 'brother@bookai.local')",
                    (uid,)
                )
                conn.commit()
                user_row = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()

            profile = dict(user_row)
            book_count = conn.execute("SELECT COUNT(*) as count FROM user_books WHERE uid = ?", (uid,)).fetchone()["count"]
            scan_count = conn.execute("SELECT COUNT(*) as count FROM scan_logs WHERE uid = ?", (uid,)).fetchone()["count"]

            profile["total_books"] = book_count
            profile["total_scans"] = scan_count
            profile["backend_mode"] = "Firestore Cloud" if use_firestore else "Local SQLite Engine"
            return profile


def update_user_profile(uid: str, name: str = None, email: str = None, reading_goal: int = None, favorite_genres: str = None):
    with _db_lock:
        with get_sqlite_conn() as conn:
            updates = []
            params = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if email is not None:
                updates.append("email = ?")
                params.append(email)
            if reading_goal is not None:
                updates.append("reading_goal = ?")
                params.append(reading_goal)
            if favorite_genres is not None:
                updates.append("favorite_genres = ?")
                params.append(favorite_genres)

            if updates:
                params.append(uid)
                sql = f"UPDATE users SET {', '.join(updates)} WHERE uid = ?"
                conn.execute(sql, params)
                conn.commit()

    return get_user_profile(uid)