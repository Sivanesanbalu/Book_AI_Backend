import firebase_admin
from firebase_admin import credentials, firestore
import os

cred = credentials.Certificate(os.path.join("uploads", "serviceAccountKey.json"))
firebase_admin.initialize_app(cred)

db = firestore.client()

def save_book_to_cloud(title: str):
    db.collection("books").add({
        "title": title
    })
