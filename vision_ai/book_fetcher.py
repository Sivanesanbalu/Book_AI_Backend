import requests
import re

# -------------------------------------------------
# Clean title  (remove extra symbols from LLM)
# -------------------------------------------------
def clean_name(name: str):
    name = name.replace("\n", " ").strip()
    name = re.sub(r'\s+', ' ', name)
    return name


# -------------------------------------------------
# Split "TITLE - AUTHOR"
# -------------------------------------------------
def split_book(book_name: str):
    if "-" in book_name:
        title, author = book_name.split("-", 1)
        return clean_name(title), clean_name(author)
    return clean_name(book_name), None


# -------------------------------------------------
# OPENLIBRARY FETCH (PRIMARY)
# -------------------------------------------------
def fetch_openlibrary(title, author=None):
    try:
        if author:
            url = f"https://openlibrary.org/search.json?title={title}&author={author}"
        else:
            url = f"https://openlibrary.org/search.json?title={title}"

        print("OPENLIB QUERY:", url)

        r = requests.get(url, timeout=10).json()

        if "docs" not in r or len(r["docs"]) == 0:
            return None

        book = r["docs"][0]

        description = None

        # sometimes description is inside "first_sentence"
        if "first_sentence" in book:
            if isinstance(book["first_sentence"], dict):
                description = book["first_sentence"].get("value")
            else:
                description = book["first_sentence"]

        return {
            "title": book.get("title", title),
            "authors": ", ".join(book.get("author_name", [])),
            "description": description
        }

    except Exception as e:
        print("OPENLIB ERROR:", e)
        return None


# -------------------------------------------------
# WIKIPEDIA FALLBACK (IF NO DESCRIPTION)
# -------------------------------------------------
def fetch_wikipedia(title):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        print("WIKI QUERY:", url)

        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()

        if "extract" not in data:
            return None

        return data["extract"]

    except Exception as e:
        print("WIKI ERROR:", e)
        return None


# -------------------------------------------------
# MAIN FUNCTION (USED BY FASTAPI)
# -------------------------------------------------
def get_book_info(book_name: str):

    title, author = split_book(book_name)

    # 1️⃣ Try OpenLibrary
    book = fetch_openlibrary(title, author)

    if not book:
        return None

    description = book.get("description")

    # 2️⃣ If no description → Wikipedia
    if not description:
        wiki_desc = fetch_wikipedia(title)
        if wiki_desc:
            description = wiki_desc
        else:
            description = "No description found"

    return {
        "title": book["title"],
        "authors": book["authors"],
        "description": description
    }