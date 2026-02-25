import os, base64, requests, re
from PIL import Image, ExifTags, ImageEnhance, ImageFilter

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================================================
# FIX ORIENTATION
# =========================================================
def fix_rotation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif:
            val = exif.get(orientation)
            if val == 3:
                img = img.rotate(180, expand=True)
            elif val == 6:
                img = img.rotate(270, expand=True)
            elif val == 8:
                img = img.rotate(90, expand=True)
    except:
        pass
    return img


# =========================================================
# PREPROCESS FOR VISION MODEL (VERY IMPORTANT)
# =========================================================
def prepare_for_ai(path):

    img = Image.open(path)

    # 1) rotate correctly
    img = fix_rotation(img)

    # 2) convert to RGB (mobile images often RGBA/HEIC)
    img = img.convert("RGB")

    # 3) center crop (removes background noise)
    w, h = img.size
    crop = min(w, h)
    img = img.crop(((w-crop)//2, (h-crop)//2, (w+crop)//2, (h+crop)//2))

    # 4) resize to vision friendly resolution
    img = img.resize((768, 768), Image.LANCZOS)

    # 5) improve readability
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    img = img.filter(ImageFilter.DETAIL)

    # 6) save optimized jpeg
    img.save(path, "JPEG", quality=92, subsampling=0)


# =========================================================
# CLEAN TITLE
# =========================================================
def clean_title(text):
    text = text.replace("\n", " ").strip()
    text = re.split(r"—|-|by", text)[0]
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# =========================================================
# MAIN DETECTION
# =========================================================
def detect_book_title(path):

    if not GROQ_API_KEY:
        print("NO API KEY")
        return None

    # ⭐ THE MAGIC STEP
    prepare_for_ai(path)

    # encode image
    with open(path, "rb") as f:
        img = base64.b64encode(f.read()).decode()

    url = "https://api.groq.com/openai/v1/chat/completions"

    body = {
        "model": "llava-v1.5-7b-4096-preview",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                    "Read this book cover carefully. Return only: Title — Author"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                }
            ]
        }],
        "temperature": 0
    }

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json=body,
        timeout=120
    )

    if r.status_code != 200:
        print("VISION ERROR:", r.text)
        return None

    raw = r.json()["choices"][0]["message"]["content"]
    print("VISION RAW:", raw)

    title = clean_title(raw)

    if len(title) < 3:
        return None

    return title