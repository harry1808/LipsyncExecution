# LipsyncExecution - Flask video dubbing app
FROM python:3.10-slim

# System deps: FFmpeg (for video/audio), git (for pip install parler-tts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install gunicorn for production server
RUN pip install --no-cache-dir gunicorn

# Application code
COPY . .

# Writable dirs: DB in /app/data (set via env), instance created at runtime if needed
RUN mkdir -p /app/data /app/instance/uploads /app/instance/outputs

# Default: run with gunicorn (bind 0.0.0.0 for Docker)
ENV FLASK_APP=flask_app:app
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "600", "flask_app:app"]
