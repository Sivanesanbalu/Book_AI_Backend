import requests


def get_book_info(book_name: str):

    # 🔥 Split "TITLE - AUTHOR"
    if "-" in book_name:
        title, author = book_name.split("-", 1)
        title = title.strip()
        author = author.strip()
        query = f'intitle:{title}+inauthor:{author}'
    else:
        title = book_name.strip()
        query = f'intitle:{title}'

    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"

    print("GOOGLE QUERY:", url)  # debug

    r = requests.get(url).json()

    if "items" not in r:
        print("GOOGLE RESPONSE:", r)
        return None

    info = r["items"][0]["volumeInfo"]

    return {
        "title": info.get("title", title),
        "authors": ", ".join(info.get("authors", [])),
        "description": info.get("description", "No description found")
    }