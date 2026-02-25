import requests, urllib.parse
from difflib import SequenceMatcher

def sim(a,b): return SequenceMatcher(None,a.lower(),b.lower()).ratio()

def fetch_book_details(title):

    url=f"https://www.googleapis.com/books/v1/volumes?q=intitle:{urllib.parse.quote(title)}"

    r=requests.get(url,timeout=20)
    items=r.json().get("items")
    if not items: return None

    best,maxs=None,0
    for i in items:
        t=i["volumeInfo"].get("title","")
        s=sim(title,t)
        if s>maxs:
            maxs=s; best=i["volumeInfo"]

    if maxs<0.35: return None

    return {
        "title":best.get("title"),
        "authors":", ".join(best.get("authors",["Unknown"])),
        "description":best.get("description","No description"),
        "categories":", ".join(best.get("categories",["General"]))
    }