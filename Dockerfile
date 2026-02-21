# ---------- BASE ----------
FROM python:3.10-slim

WORKDIR /app

# ---------- SYSTEM LIBS (minimal but safe) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- ENV (prevents cpu spikes & tokenizer crash) ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ---------- INSTALL PYTHON DEPENDENCIES ----------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- COPY PROJECT ----------
COPY . .

# ---------- PRECREATE FOLDERS ----------
RUN mkdir -p uploads data

# ---------- PRELOAD MODEL (VERY IMPORTANT) ----------
# prevents first request timeout
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("Model cached successfully")
EOF

# ---------- START SERVER ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]