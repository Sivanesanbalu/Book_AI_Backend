import requests
import urllib.parse


def clean_description(text: str):
    """Remove html + limit size so LLM won't hallucinate"""
    if not text:
        return "No description available"

    text = text.replace("<br>", " ").replace("<p>", " ").replace("</p>", " ")
    text = text.replace("\n", " ").strip()

    # limit length (important!)
    return text[:1200]


def fetch_book_details(title: str):

    try:
        # 🔥 IMPORTANT: intitle search gives accurate results
        query = urllib.parse.quote(f'intitle:"{title}"')

        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
        r = requests.get(url, timeout=20)

        if r.status_code != 200:
            return None

        items = r.json().get("items")
        if not items:
            return None

        # choose BEST match
        best = None
        for item in items:
            info = item.get("volumeInfo", {})
            book_title = info.get("title", "").lower()

            # title similarity check
            if title.lower() in book_title:
                best = info
                break

        if not best:
            best = items[0]["volumeInfo"]

        return {
            "title": best.get("title", title),
            "authors": ", ".join(best.get("authors", ["Unknown Author"])),
            "description": clean_description(best.get("description")),
            "categories": ", ".join(best.get("categories", ["General"]))
        }

    except:
        return None