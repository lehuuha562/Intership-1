# Dockerfile
FROM python:3.10-slim

# 1. Install System Drivers (Audio & Video)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install LIGHTWEIGHT PyTorch (CPU Only)
# This prevents downloading the 2GB+ NVIDIA CUDA drivers
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest of the Python libraries
RUN pip install --no-cache-dir \
    pymilvus \
    sentence-transformers \
    streamlit \
    pandas \
    rich \
    pillow \
    opencv-python-headless \
    moviepy \
    pydub \
    requests \
    scikit-learn \
    plotly \
    soundfile \
    librosa

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]