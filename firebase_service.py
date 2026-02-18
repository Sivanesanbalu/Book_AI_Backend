import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime

# init only once
if not firebase_admin._apps:
    firebase_key = os.environ.get("FIREBASE_KEY")

    if firebase_key is None:
        raise Exception("FIREBASE_KEY environment variable not found")

    cred_dict = json.loads(firebase_key)
    cred = credentials.Certificate(cred_dict)
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
