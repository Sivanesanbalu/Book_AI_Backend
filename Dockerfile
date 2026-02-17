FROM python:3.10-slim

WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python ----------
COPY requirements.txt .

# Install CPU Torch first (avoid CUDA huge install)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir -r requirements.txt

# ---------- App ----------
COPY . .

# Create runtime dirs
RUN mkdir -p uploads data

# ---------- Start Server ----------
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
