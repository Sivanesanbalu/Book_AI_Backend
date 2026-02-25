import requests, urllib.parse
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def fetch_book_details(title):

    query = urllib.parse.quote(f'intitle:"{title}"')
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"

    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None

    items = r.json().get("items")
    if not items:
        return None

    best, score = None, 0

    for item in items:
        info = item["volumeInfo"]
        s = similarity(title, info.get("title", ""))
        if s > score:
            score, best = s, info

    if score < 0.35:
        return None

    return {
        "title": best.get("title"),
        "authors": ", ".join(best.get("authors", ["Unknown"])),
        "description": best.get("description", "No description available"),
        "categories": ", ".join(best.get("categories", ["General"]))
    }