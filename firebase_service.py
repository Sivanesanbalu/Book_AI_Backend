import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime
import re
from difflib import SequenceMatcher   # 🔥 NEW

# init only once
if not firebase_admin._apps:
    firebase_key = os.environ.get("FIREBASE_KEY")

    if firebase_key is None:
        key_file = "serviceAccountKey.json"
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                cred_dict = json.load(f)
        else:
            raise Exception("FIREBASE_KEY environment variable not found and serviceAccountKey.json file not found")
    else:
        cred_dict = json.loads(firebase_key)

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ---------------- NORMALIZE TITLE ----------------
def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- SIMILARITY CHECK ----------------
def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


####################################################
# Save book to USER LIBRARY
####################################################
def save_book_for_user(user_id: str, title: str):

    clean_title = normalize_title(title)

    # prevent duplicate save
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


####################################################
# Check user already owns book
####################################################
def user_has_book(user_id: str, title: str) -> bool:

    clean_title = normalize_title(title)

    books_ref = db.collection("users") \
                  .document(user_id) \
                  .collection("books") \
                  .stream()

    for book in books_ref:
        data = book.to_dict()
        saved_title = data.get("title","")

        # 🔥 FUZZY MATCH (IMPORTANT)
        if similar(saved_title, clean_title) > 0.88:
            return True

    return False

