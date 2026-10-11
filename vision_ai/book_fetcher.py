import requests
import re
import urllib.parse


def clean_name(name: str):
    if not name:
        return ""
    name = name.replace("\n", " ").strip()
    name = re.sub(r'\s+', ' ', name)
    return name


def split_book(book_name: str):
    if not book_name:
        return "", ""
    if "-" in book_name:
        parts = book_name.split("-", 1)
        return clean_name(parts[0]), clean_name(parts[1])
    return clean_name(book_name), ""


# ---------------- GOOGLE BOOKS (RICH METADATA & COVERS) ----------------
def fetch_google_books(title: str, author: str = ""):
    try:
        query = f"intitle:{title}"
        if author:
            query += f"+inauthor:{author}"

        encoded_q = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_q}&maxResults=1"
        res = requests.get(url, timeout=8)

        if res.status_code != 200:
            return None

        data = res.json()
        if "items" not in data or len(data["items"]) == 0:
            # Fallback to general query without intitle prefix
            fallback_url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(title)}&maxResults=1"
            res = requests.get(fallback_url, timeout=8)
            if res.status_code != 200:
                return None
            data = res.json()
            if "items" not in data or len(data["items"]) == 0:
                return None

        info = data["items"][0].get("volumeInfo", {})

        # Extract cover image (prefer https)
        images = info.get("imageLinks", {})
        cover_url = images.get("extraLarge") or images.get("large") or images.get("medium") or images.get("thumbnail") or ""
        if cover_url.startswith("http://"):
            cover_url = "https://" + cover_url[7:]

        categories = info.get("categories", ["General Fiction"])
        genre = categories[0] if categories else "General Fiction"

        return {
            "title": info.get("title", title),
            "authors": ", ".join(info.get("authors", [author] if author else ["Unknown Author"])),
            "description": info.get("description", ""),
            "genre": genre,
            "cover_url": cover_url,
            "page_count": info.get("pageCount", 0),
            "published_date": info.get("publishedDate", "")
        }
    except Exception as e:
        print("⚠️ Google Books error:", e)
        return None


# ---------------- OPENLIBRARY FETCH ----------------
def fetch_openlibrary(title: str, author: str = ""):
    try:
        if author:
            url = f"https://openlibrary.org/search.json?title={urllib.parse.quote(title)}&author={urllib.parse.quote(author)}&limit=1"
        else:
            url = f"https://openlibrary.org/search.json?title={urllib.parse.quote(title)}&limit=1"

        r = requests.get(url, timeout=8).json()

        if "docs" not in r or len(r["docs"]) == 0:
            return None

        book = r["docs"][0]
        description = None

        if "first_sentence" in book:
            if isinstance(book["first_sentence"], dict):
                description = book["first_sentence"].get("value")
            elif isinstance(book["first_sentence"], list) and len(book["first_sentence"]) > 0:
                description = book["first_sentence"][0]
            else:
                description = str(book["first_sentence"])

        cover_i = book.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_i}-L.jpg" if cover_i else ""

        genres = book.get("subject", ["General"])
        genre = genres[0] if genres else "General"

        return {
            "title": book.get("title", title),
            "authors": ", ".join(book.get("author_name", [author] if author else ["Unknown Author"])),
            "description": description or "",
            "genre": genre,
            "cover_url": cover_url,
            "page_count": book.get("number_of_pages_median", 0),
            "published_date": str(book.get("first_publish_year", ""))
        }
    except Exception as e:
        print("⚠️ OpenLibrary error:", e)
        return None


# ---------------- WIKIPEDIA SUMMARY ----------------
def fetch_wikipedia(title: str):
    try:
        clean_t = re.sub(r'[^a-zA-Z0-9\s]', '', title).replace(' ', '_')
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_t)}"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "")
    except Exception as e:
        print("⚠️ Wikipedia error:", e)
    return ""


# ---------------- MASTER GET BOOK INFO ----------------
def get_book_info(book_name: str):
    if not book_name:
        return None

    title, author = split_book(book_name)

    # 1. Google Books (most detailed)
    book = fetch_google_books(title, author)

    # 2. OpenLibrary Fallback
    if not book:
        book = fetch_openlibrary(title, author)

    if not book:
        # Minimal info placeholder
        return {
            "title": title or "Unknown Book",
            "authors": author or "Author Unknown",
            "description": f"A captivating volume titled '{title}'. Explore its narrative and key themes.",
            "genre": "Literature",
            "cover_url": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=400&q=80",
            "page_count": 280,
            "published_date": "Recent"
        }

    # If missing description, fallback to Wikipedia
    if not book.get("description"):
        wiki_desc = fetch_wikipedia(title)
        book["description"] = wiki_desc if wiki_desc else f"An influential work in {book.get('genre', 'literature')} exploring fundamental themes and ideas."

    if not book.get("cover_url"):
        book["cover_url"] = "https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=400&q=80"

    return book