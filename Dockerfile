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

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# opencv-python-headless for server (no GUI needed, smaller)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80 && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Render sets PORT env var; default to 10000 (Render's default)
ENV PORT=10000
EXPOSE ${PORT}

# Use shell form so $PORT is expanded at runtime
CMD streamlit run main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.maxUploadSize=50
