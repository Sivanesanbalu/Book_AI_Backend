FROM python:3.10-slim

WORKDIR /app

# -------------------------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# PYTHON SETTINGS
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads data


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]