FROM python:3.10-slim

WORKDIR /app

# ---------- SYSTEM DEPENDENCIES ----------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- PYTHON SETTINGS ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- INSTALL PYTHON PACKAGES ----------
COPY requirements.txt .

# install torch cpu first (stable)
RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ---------- COPY PROJECT ----------
COPY . .

# runtime folders
RUN mkdir -p uploads data

# debug (visible in render logs)
RUN tesseract --version

# ---------- START SERVER ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
