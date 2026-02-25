import requests
import urllib.parse
from difflib import SequenceMatcher


def clean_description(text: str):
    """Remove html + limit size so LLM won't hallucinate"""
    if not text:
        return "No description available"

    text = text.replace("<br>", " ").replace("<p>", " ").replace("</p>", " ")
    text = text.replace("\n", " ").strip()

    # limit length (important!)
    return text[:1200]


def similarity(a, b):
    """Fuzzy title matching"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fetch_book_details(title: str):

    try:
        # 🔥 intitle search gives accurate results
        query = urllib.parse.quote(f'intitle:"{title}"')
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"

        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None

        items = r.json().get("items")
        if not items:
            return None

        # ================= BEST MATCH SELECTION =================
        best = None
        best_score = 0.0

        for item in items:
            info = item.get("volumeInfo", {})
            book_title = info.get("title", "")

            score = similarity(title, book_title)

            if score > best_score:
                best_score = score
                best = info

        # if confidence too low → treat as not found
        if best_score < 0.35:
            return None

        # ================= RETURN CLEAN DATA =================
        return {
            "title": best.get("title", title),
            "authors": ", ".join(best.get("authors", ["Unknown Author"])),
            "description": clean_description(best.get("description")),
            "categories": ", ".join(best.get("categories", ["General"]))
        }

    except:
        return None