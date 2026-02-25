import os
import base64
import requests
import re
from PIL import Image, ExifTags, ImageEnhance, ImageFilter

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================================================
# FIX PHONE ROTATION
# =========================================================
def fix_rotation(img):
    try:
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == 'Orientation':
                orientation_key = k
                break

        exif = img._getexif()
        if exif and orientation_key in exif:
            val = exif[orientation_key]

            if val == 3:
                img = img.rotate(180, expand=True)
            elif val == 6:
                img = img.rotate(270, expand=True)
            elif val == 8:
                img = img.rotate(90, expand=True)

    except Exception:
        pass

    return img


# =========================================================
# PREPROCESS IMAGE FOR VISION MODEL
# =========================================================
def prepare_for_ai(path):
    try:
        img = Image.open(path)

        # rotate correctly
        img = fix_rotation(img)

        # convert to RGB
        img = img.convert("RGB")

        # center crop
        w, h = img.size
        crop = min(w, h)
        img = img.crop(((w-crop)//2, (h-crop)//2, (w+crop)//2, (h+crop)//2))

        # resize (very important for LLaVA)
        img = img.resize((768, 768), Image.LANCZOS)

        # enhance readability
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Sharpness(img).enhance(1.8)
        img = img.filter(ImageFilter.DETAIL)

        # save optimized jpeg
        img.save(path, "JPEG", quality=92, subsampling=0)

    except Exception as e:
        print("PREPROCESS ERROR:", e)


# =========================================================
# CLEAN TITLE OUTPUT
# =========================================================
def clean_title(text: str):

    if not text:
        return None

    text = text.replace("\n", " ").strip()

    # remove explanations
    text = re.split(r"by|—|-", text, maxsplit=1)[0]

    # remove weird chars
    text = re.sub(r"[^\w\s]", "", text)

    text = " ".join(text.split())

    if len(text) < 3:
        return None

    return text


# =========================================================
# MAIN DETECTION
# =========================================================
def detect_book_title(path):

    if not GROQ_API_KEY:
        print("❌ GROQ API KEY missing")
        return None

    # preprocess image
    prepare_for_ai(path)

    # encode
    try:
        with open(path, "rb") as f:
            img = base64.b64encode(f.read()).decode()
    except:
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"

    prompt = """
You are reading a real book cover photo.

TASK:
Extract the main book title and author.

Rules:
- Ignore logos and publisher marks
- Ignore subtitles
- Focus on biggest bold text
- Guess intelligently if partially visible

Return ONLY:
Title — Author
"""

    body = {
    "model": "llama-3.2-11b-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
You are reading a real book cover photo.

Return ONLY the book title and author.

Rules:
- Focus on big title text
- Ignore stickers, price tags
- Guess intelligently if partial visible
- Output format: Title — Author
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                }
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 80
}

    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=body,
            timeout=120
        )

        if r.status_code != 200:
            print("VISION ERROR:", r.text)
            return None

        raw = r.json()["choices"][0]["message"]["content"]
        print("VISION RAW:", raw)

        return clean_title(raw)

    except Exception as e:
        print("VISION REQUEST FAIL:", e)
        return None