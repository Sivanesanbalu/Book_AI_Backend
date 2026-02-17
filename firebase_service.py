import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

# init only once
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

####################################################
# Save book to USER LIBRARY (capture only)
####################################################
def save_book_for_user(user_id: str, title: str):

    db.collection("users") \
      .document(user_id) \
      .collection("books") \
      .add({
        "title": title,
        "createdAt": datetime.utcnow()
      })

####################################################
# Check user already owns book
####################################################
def user_has_book(user_id: str, title: str) -> bool:

    books_ref = db.collection("users") \
                  .document(user_id) \
                  .collection("books") \
                  .stream()

    for book in books_ref:
        data = book.to_dict()
        if data.get("title","").lower() == title.lower():
            return True

    return False
