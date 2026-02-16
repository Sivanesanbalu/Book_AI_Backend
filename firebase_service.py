import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("uploads\serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def save_book_to_cloud(title: str):
    db.collection("books").add({
        "title": title
    })
