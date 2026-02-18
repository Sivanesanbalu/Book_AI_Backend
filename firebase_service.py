import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime
import re

# init only once
if not firebase_admin._apps:
    firebase_key = os.environ.get("FIREBASE_KEY")

    if firebase_key is None:
        raise Exception("FIREBASE_KEY environment variable not found")

    cred_dict = json.loads(firebase_key)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ---------------- NORMALIZE TITLE ----------------
def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)   # remove symbols
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text


####################################################
# Save book to USER LIBRARY (capture only)
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
        if saved_title == clean_title:
            return True

    return False
