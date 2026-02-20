FROM python:3.10-slim

WORKDIR /app

# -------------------------------------------------
# SYSTEM DEPENDENCIES (for opencv + paddleocr)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
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

# -------------------------------------------------
# INSTALL PYTHON PACKAGES
# -------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip

# paddle must install before requirements in many servers
RUN pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# COPY PROJECT FILES
# -------------------------------------------------
COPY . .

# runtime folders
RUN mkdir -p uploads data

# -------------------------------------------------
# START SERVER
# -------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]