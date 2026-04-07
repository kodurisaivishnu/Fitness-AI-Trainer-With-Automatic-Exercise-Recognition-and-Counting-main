FROM python:3.10-slim

# System dependencies for OpenCV, MediaPipe, pyttsx3, WebRTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    ffmpeg \
    espeak-ng \
    libespeak-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install deps - opencv-headless (smaller, no GUI)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Memory optimization env vars
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=-1
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=65536

# Render sets PORT env var
ENV PORT=10000
EXPOSE ${PORT}

CMD streamlit run main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.maxUploadSize=50
