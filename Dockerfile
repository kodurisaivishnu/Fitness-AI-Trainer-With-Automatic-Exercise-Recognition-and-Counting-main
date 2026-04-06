FROM python:3.10-slim

# System dependencies for OpenCV, MediaPipe, pyttsx3, WebRTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
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
# Use opencv-python-headless on server (no GUI needed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=50"]
