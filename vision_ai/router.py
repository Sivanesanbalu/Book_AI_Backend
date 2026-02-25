from fastapi import APIRouter,UploadFile,File,Form
import os,uuid,shutil
from .detect_book import detect_book_title
from .fetch_book import fetch_book_details
from .explain_book import explain_book

router=APIRouter(prefix="/ai")

@router.post("/understand-book")
async def understand_book(file:UploadFile=File(...),question:str=Form("")):

    temp=f"tmp/{uuid.uuid4().hex}.jpg"
    os.makedirs("tmp",exist_ok=True)

    with open(temp,"wb") as b:
        shutil.copyfileobj(file.file,b)

    title=detect_book_title(temp)

    if not title:
        return {"answer":"Could not identify book"}

    data=fetch_book_details(title)

    if not data:
        return {"book":title,"answer":"Book found but info unavailable"}

    ans=explain_book(title,data,question)

    return {"book":title,"answer":ans}