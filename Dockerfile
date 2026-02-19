FROM python:3.10-slim

WORKDIR /app

# ---------- SYSTEM DEPENDENCIES (VERY IMPORTANT) ----------
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

# ---------- PYTHON ----------
COPY requirements.txt .

# Install CPU Torch first (avoid CUDA huge install)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ---------- APP ----------
COPY . .

# Runtime folders
RUN mkdir -p uploads data

# (Optional debug – will show in logs)
RUN tesseract --version

# ---------- START SERVER ----------
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
