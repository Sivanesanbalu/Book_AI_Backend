import requests
from .prompts import SYSTEM_PROMPT
def get_book_info(book_name):

    url = f"https://www.googleapis.com/books/v1/volumes?q={book_name}"
    r = requests.get(url).json()

    if "items" not in r:
        return None

    info = r["items"][0]["volumeInfo"]

    return {
        "title": info.get("title", ""),
        "authors": ", ".join(info.get("authors", [])),
        "description": info.get("description", "No description found")
    }