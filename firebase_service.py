import os
import json
import re
import time
import hashlib
import sqlite3
from datetime import datetime
from threading import RLock

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
_db_lock = RLock()


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
                current_page INTEGER DEFAULT 0,
                total_pages INTEGER DEFAULT 300,
                reading_minutes INTEGER DEFAULT 0,
                last_read TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (uid, book_id)
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reading_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT,
                book_id TEXT,
                title TEXT,
                minutes INTEGER,
                pages_read INTEGER DEFAULT 0,
                start_page INTEGER DEFAULT 0,
                end_page INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            # Auto-migrate columns if missing in user_books & users
            cols = [r[1] for r in conn.execute("PRAGMA table_info(user_books)").fetchall()]
            if 'current_page' not in cols:
                conn.execute("ALTER TABLE user_books ADD COLUMN current_page INTEGER DEFAULT 0")
            if 'total_pages' not in cols:
                conn.execute("ALTER TABLE user_books ADD COLUMN total_pages INTEGER DEFAULT 300")
            if 'reading_minutes' not in cols:
                conn.execute("ALTER TABLE user_books ADD COLUMN reading_minutes INTEGER DEFAULT 0")
            if 'last_read' not in cols:
                conn.execute("ALTER TABLE user_books ADD COLUMN last_read TIMESTAMP")

            user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
            if 'last_drive_sync' not in user_cols:
                conn.execute("ALTER TABLE users ADD COLUMN last_drive_sync TIMESTAMP")
            if 'drive_sync_enabled' not in user_cols:
                conn.execute("ALTER TABLE users ADD COLUMN drive_sync_enabled INTEGER DEFAULT 1")
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
        return "guest_user"

    clean_token = id_token.strip()
    if clean_token.lower().startswith("bearer "):
        clean_token = clean_token[7:].strip()

    if clean_token.lower() in ("default", "demo", "brother", "user", "bearer demo"):
        return "guest_user"

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

    return clean_token or "guest_user"


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
                       status: str = "Reading", rating: int = 5, notes: str = "",
                       current_page: int = 0, total_pages: int = 300, reading_minutes: int = 0):
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
                "current_page": current_page,
                "total_pages": total_pages,
                "reading_minutes": reading_minutes,
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
            INSERT INTO user_books (uid, book_id, title, author, genre, cover_url, overview, confidence, status, rating, notes, current_page, total_pages, reading_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid, book_id) DO UPDATE SET
                title = excluded.title,
                author = COALESCE(NULLIF(excluded.author, ''), user_books.author),
                genre = COALESCE(NULLIF(excluded.genre, ''), user_books.genre),
                cover_url = COALESCE(NULLIF(excluded.cover_url, ''), user_books.cover_url),
                overview = COALESCE(NULLIF(excluded.overview, ''), user_books.overview),
                status = excluded.status,
                rating = excluded.rating,
                current_page = CASE WHEN excluded.current_page > 0 THEN excluded.current_page ELSE user_books.current_page END,
                total_pages = CASE WHEN excluded.total_pages > 0 THEN excluded.total_pages ELSE user_books.total_pages END,
                reading_minutes = user_books.reading_minutes + excluded.reading_minutes
            """, (uid, b_id, title, author, genre, cover_url, overview, confidence, status, rating, notes, current_page, total_pages, reading_minutes))
            conn.commit()

    print(f"📘 Saved book: '{title}' for user: {uid}")
    return True


# ---------------- GET USER BOOKS ----------------
def get_user_books(uid: str):
    with _db_lock:
        with get_sqlite_conn() as conn:
            rows = conn.execute("""
            SELECT book_id, title, author, genre, cover_url, overview, confidence, status, rating, notes, current_page, total_pages, reading_minutes, last_read, created_at
            FROM user_books
            WHERE uid = ?
            ORDER BY datetime(last_read) DESC, created_at DESC
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
                    "INSERT INTO users (uid, name, email, avatar, reading_goal) VALUES (?, 'Reader', 'reader@booknest.local', 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=240&q=80', 25)",
                    (uid,)
                )
                conn.commit()
                user_row = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()

            profile = dict(user_row)
            book_count = conn.execute("SELECT COUNT(*) as count FROM user_books WHERE uid = ?", (uid,)).fetchone()["count"]
            completed_count = conn.execute(
                "SELECT COUNT(*) as count FROM user_books WHERE uid = ? AND (LOWER(status) LIKE '%finish%' OR LOWER(status) LIKE '%complet%')",
                (uid,)
            ).fetchone()["count"]
            reading_count = conn.execute(
                "SELECT COUNT(*) as count FROM user_books WHERE uid = ? AND LOWER(status) LIKE '%read%' AND NOT LOWER(status) LIKE '%finish%' AND NOT LOWER(status) LIKE '%complet%'",
                (uid,)
            ).fetchone()["count"]
            wishlist_count = conn.execute(
                "SELECT COUNT(*) as count FROM user_books WHERE uid = ? AND (LOWER(status) LIKE '%wish%' OR LOWER(status) LIKE '%want%')",
                (uid,)
            ).fetchone()["count"]
            scan_count = conn.execute("SELECT COUNT(*) as count FROM scan_logs WHERE uid = ?", (uid,)).fetchone()["count"]

            # Distinct categories
            genres_rows = conn.execute("SELECT DISTINCT genre FROM user_books WHERE uid = ? AND genre IS NOT NULL AND genre != ''", (uid,)).fetchall()
            categories = [r["genre"] for r in genres_rows]

            profile["total_books"] = book_count
            profile["completed_books"] = completed_count
            profile["reading_books"] = reading_count
            profile["wishlist_books"] = wishlist_count
            profile["total_scans"] = scan_count
            profile["categories"] = categories
            profile["backend_mode"] = "Firestore Cloud" if use_firestore else "Local SQLite Engine"
            return profile


def update_user_profile(uid: str, name: str = None, email: str = None, reading_goal: int = None, favorite_genres: str = None, avatar: str = None):
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
            if avatar is not None:
                updates.append("avatar = ?")
                params.append(avatar)

            if updates:
                params.append(uid)
                sql = f"UPDATE users SET {', '.join(updates)} WHERE uid = ?"
                conn.execute(sql, params)
                conn.commit()

    return get_user_profile(uid)


def update_book_status(uid: str, title: str, status: str = None, rating: int = None, genre: str = None,
                       current_page: int = None, total_pages: int = None, add_minutes: int = 0):
    b_id = book_id(title)
    with _db_lock:
        with get_sqlite_conn() as conn:
            updates = ["last_read = CURRENT_TIMESTAMP"]
            params = []
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if rating is not None:
                updates.append("rating = ?")
                params.append(rating)
            if genre is not None:
                updates.append("genre = ?")
                params.append(genre)
            if current_page is not None:
                updates.append("current_page = ?")
                params.append(current_page)
            if total_pages is not None:
                updates.append("total_pages = ?")
                params.append(total_pages)
            if add_minutes > 0:
                updates.append("reading_minutes = reading_minutes + ?")
                params.append(add_minutes)

            params.extend([uid, b_id])
            sql = f"UPDATE user_books SET {', '.join(updates)} WHERE uid = ? AND book_id = ?"
            conn.execute(sql, params)
            conn.commit()
    return True


def log_reading_session(uid: str, title: str, minutes: int, start_page: int = 0, end_page: int = 0):
    b_id = book_id(title)
    pages_read = max(0, end_page - start_page) if end_page > start_page else 0
    with _db_lock:
        with get_sqlite_conn() as conn:
            conn.execute("""
            INSERT INTO reading_sessions (uid, book_id, title, minutes, pages_read, start_page, end_page)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (uid, b_id, title, minutes, pages_read, start_page, end_page))
            conn.commit()
    update_book_status(uid, title, current_page=end_page if end_page > 0 else None, add_minutes=minutes)
    return {"status": "logged", "minutes": minutes, "pages_read": pages_read}


def upsert_google_user(uid: str, name: str, email: str, avatar: str = None):
    with _db_lock:
        with get_sqlite_conn() as conn:
            existing = conn.execute("SELECT uid, created_at FROM users WHERE uid = ? OR (email IS NOT NULL AND LOWER(email) = LOWER(?))", (uid, email)).fetchone()
            is_new = (existing is None)
            conn.execute("""
            INSERT INTO users (uid, name, email, avatar, last_drive_sync)
            VALUES (?, ?, ?, COALESCE(?, 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80'), CURRENT_TIMESTAMP)
            ON CONFLICT(uid) DO UPDATE SET
                name = COALESCE(excluded.name, users.name),
                email = COALESCE(excluded.email, users.email),
                avatar = COALESCE(excluded.avatar, users.avatar),
                last_drive_sync = CURRENT_TIMESTAMP
            """, (uid, name, email, avatar))
            conn.commit()
    profile = get_user_profile(uid)
    return profile, is_new


def export_user_library_backup(uid: str) -> dict:
    """Exports full user library, reading sessions, and profile for Google Drive sync"""
    with _db_lock:
        with get_sqlite_conn() as conn:
            profile = get_user_profile(uid)
            books = [dict(r) for r in conn.execute("SELECT * FROM user_books WHERE uid = ?", (uid,)).fetchall()]
            sessions = [dict(r) for r in conn.execute("SELECT * FROM reading_sessions WHERE uid = ? ORDER BY timestamp DESC", (uid,)).fetchall()]
            return {
                "app": "BookNest AI",
                "version": "2.7.0",
                "exported_at": datetime.utcnow().isoformat(),
                "sync_target": "Google Drive Cloud Storage",
                "user": profile,
                "books": books,
                "reading_sessions": sessions
            }


def record_drive_sync(uid: str):
    """Updates the timestamp when the library was successfully backed up to Google Drive"""
    with _db_lock:
        with get_sqlite_conn() as conn:
            conn.execute("UPDATE users SET last_drive_sync = CURRENT_TIMESTAMP WHERE uid = ?", (uid,))
            conn.commit()


def restore_user_library_backup(uid: str, backup_data: dict) -> int:
    """Restores books and reading sessions from a Google Drive backup payload"""
    restored_count = 0
    books = backup_data.get("books", [])
    with _db_lock:
        with get_sqlite_conn() as conn:
            for b in books:
                book_id = b.get("book_id") or str(uuid.uuid4())[:8]
                conn.execute("""
                INSERT INTO user_books (uid, book_id, title, author, genre, cover_url, overview, status, rating, notes, current_page, total_pages, reading_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uid, book_id) DO UPDATE SET
                    title = excluded.title,
                    author = excluded.author,
                    status = excluded.status,
                    current_page = excluded.current_page,
                    reading_minutes = excluded.reading_minutes
                """, (
                    uid, book_id, b.get("title", "Untitled"), b.get("author", "Unknown"),
                    b.get("genre", "General"), b.get("cover_url", ""), b.get("overview", ""),
                    b.get("status", "Reading"), b.get("rating", 5), b.get("notes", ""),
                    b.get("current_page", 0), b.get("total_pages", 300), b.get("reading_minutes", 0)
                ))
                restored_count += 1
            conn.execute("UPDATE users SET last_drive_sync = CURRENT_TIMESTAMP WHERE uid = ?", (uid,))
            conn.commit()
    return restored_count
