# AI Video Dubbing - GPU-enabled Docker Image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python to not buffer output (better for Docker logs)
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy LatentSync requirements and install
RUN pip install --no-cache-dir -r LatentSync/requirements.txt || true

# Create necessary directories
RUN mkdir -p /app/instance/uploads /app/instance/outputs /app/instance/wav2lip_assets

# Expose port
EXPOSE 5000

# Set environment variables for production
ENV FLASK_ENV=production
ENV FLASK_SECRET_KEY=change-this-in-production

# Use gunicorn for production (more robust than Flask dev server)
RUN pip install --no-cache-dir gunicorn

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "600", "--threads", "2", "flask_app:app"]










