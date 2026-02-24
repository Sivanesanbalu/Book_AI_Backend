import requests
import re

def clean_text(text: str):
    if not text:
        return None

    # remove html tags
    text = re.sub(r'<.*?>', '', text)

    # remove long marketing sentences
    text = text.replace('\n', ' ')
    text = text.strip()

    # limit size (very important for RAG)
    return text[:1200]


def get_book_summary(title):

    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}"

    try:
        r = requests.get(url, timeout=10).json()

        if "items" not in r:
            return None

        info = r["items"][0]["volumeInfo"]

        desc = clean_text(info.get("description", None))

        if not desc:
            return None

        return {
            "title": info.get("title", title),
            "description": desc,
            "category": info.get("categories", ["Education"])[0]
        }

    except:
        return None