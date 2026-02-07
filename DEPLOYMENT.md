# 🚀 Deployment Guide - AI Video Dubbing

This guide covers all deployment options for your AI Video Dubbing application.

---

## 📋 Requirements

Your application needs:
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for LatentSync)
- **CUDA**: Version 12.1 or compatible
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ for models and dependencies
- **FFmpeg**: For video processing

---

## 🐳 Option 1: Docker Deployment (Recommended)

### Prerequisites
1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Quick Start

```bash
# 1. Clone your project (if not already done)
cd C:\Users\MURF-AI\Desktop\lipsyncExecution

# 2. Create environment file
copy env.example .env
# Edit .env with your secret key

# 3. Build and run with Docker Compose
docker-compose up -d --build

# 4. View logs
docker-compose logs -f

# 5. Access the app
# Open http://localhost:5000 in your browser
```

### Manual Docker Commands

```bash
# Build the image
docker build -t video-dubbing:latest .

# Run with GPU support
docker run -d \
  --gpus all \
  -p 5000:5000 \
  -v ./instance:/app/instance \
  -v ./LatentSync/checkpoints:/app/LatentSync/checkpoints \
  --shm-size=8g \
  -e FLASK_SECRET_KEY=your-secret-key \
  --name video-dubbing \
  video-dubbing:latest
```

---

## ☁️ Option 2: Cloud GPU Platforms

### A) RunPod (Budget-Friendly GPU Cloud)

1. **Create Account**: https://runpod.io

2. **Deploy GPU Pod**:
   - Select "Deploy" → "GPU Pods"
   - Choose template: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0`
   - Select GPU: RTX 3090 or A100 (8GB+ VRAM)
   - Storage: 50GB minimum

3. **Setup Commands**:
```bash
# Clone your project
git clone <your-repo-url>
cd lipsyncExecution

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Download model checkpoints (if not included)
# The app will download these automatically on first run

# Run production server
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 flask_app:app
```

4. **Expose Port**: RunPod automatically provides a public URL

### B) AWS EC2 with GPU

1. **Launch Instance**:
   - AMI: Deep Learning AMI (Ubuntu)
   - Instance type: `g4dn.xlarge` (cheapest GPU) or `p3.2xlarge`
   - Storage: 100GB EBS

2. **Security Group**:
   - Allow inbound TCP port 5000
   - Allow SSH (port 22)

3. **Setup**:
```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone project
git clone <your-repo-url>
cd lipsyncExecution

# Activate CUDA environment (pre-installed on Deep Learning AMI)
source activate pytorch

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Run with gunicorn (production)
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 flask_app:app

# Or run with nohup for background execution
nohup gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 flask_app:app &
```

### C) Google Cloud Platform (GCP)

1. **Create VM Instance**:
   - Machine type: `n1-standard-4` + 1x NVIDIA T4
   - Boot disk: Deep Learning VM (PyTorch)
   - Size: 100GB

2. **Setup**: Same as AWS above

### D) Vast.ai (Cheapest GPU Rentals)

1. **Sign up**: https://vast.ai

2. **Find Instance**:
   - Filter: PyTorch, CUDA 12.x, 8GB+ VRAM
   - Sort by price

3. **Launch and connect via SSH**

---

## 🤗 Option 3: Hugging Face Spaces

Hugging Face Spaces supports GPU-enabled Flask apps!

1. **Create Space**: https://huggingface.co/spaces

2. **Create `app.py`** (rename or copy flask_app.py):
```python
from webapp import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
```

3. **Create Space `README.md`**:
```yaml
---
title: AI Video Dubbing
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
hardware: t4-small
---
```

4. **Upload your Dockerfile and code**

5. **Hardware**: Request T4 GPU ($0.60/hour) in Space settings

---

## 🔮 Option 4: Replicate (Using Cog)

Your project already has `LatentSync/cog.yaml`! You can deploy the lip-sync component separately.

1. **Install Cog**:
```bash
# Linux/Mac
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
chmod +x /usr/local/bin/cog
```

2. **Push to Replicate**:
```bash
cd LatentSync
cog login
cog push r8.im/your-username/latentsync
```

---

## 🖥️ Option 5: Local Production Server

For running on your own GPU machine:

### Windows (PowerShell)

```powershell
# Activate virtual environment
.\lipsyncenv\Scripts\Activate

# Install production server
pip install waitress

# Run production server
waitress-serve --port=5000 flask_app:app
```

### Linux/Mac

```bash
# Activate virtual environment
source lipsyncenv/bin/activate

# Install production server
pip install gunicorn

# Run production server
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 flask_app:app
```

---

## 🔒 Production Checklist

Before deploying to production:

### Security
- [ ] Change `FLASK_SECRET_KEY` to a strong random value
- [ ] Enable HTTPS (use nginx/Caddy as reverse proxy)
- [ ] Set up proper authentication
- [ ] Limit upload file sizes if needed

### Performance
- [ ] Use gunicorn/waitress instead of Flask dev server
- [ ] Enable GPU caching for models
- [ ] Set appropriate worker count (2-4)

### Reliability
- [ ] Set up logging (file or cloud logging)
- [ ] Configure health checks
- [ ] Set up automatic restarts (systemd/docker restart policy)
- [ ] Monitor GPU memory usage

---

## 🌐 Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required
FLASK_SECRET_KEY=your-super-secret-key-change-this

# Optional
FLASK_ENV=production
NLLB_MODEL_NAME=facebook/nllb-200-distilled-600M
LIPSYNC_DEFAULT=0
HF_TOKEN=hf_your_token  # For gated models
```

---

## 📊 Cost Comparison

| Platform | GPU | Cost | Pros |
|----------|-----|------|------|
| RunPod | RTX 3090 | ~$0.40/hr | Cheap, easy |
| Vast.ai | Various | ~$0.20/hr | Cheapest |
| AWS g4dn.xlarge | T4 | ~$0.50/hr | Reliable |
| GCP | T4 | ~$0.35/hr | Good ML tools |
| HF Spaces | T4 | ~$0.60/hr | Easy deploy |
| Replicate | A40 | Pay per prediction | API-ready |

---

## 🆘 Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use smaller models
- Try `diffusers` with `torch.float16`

### Models Not Loading
- Ensure internet connection for first download
- Check disk space (models are 2-10GB each)
- Set `HF_TOKEN` for gated models

### Slow Performance
- Ensure GPU is being used: `nvidia-smi`
- Check CUDA version matches PyTorch
- Use production server (gunicorn), not Flask dev server

### Port Already in Use
```bash
# Find process using port 5000
netstat -tulpn | grep 5000
# Or on Windows
netstat -ano | findstr :5000
```

---

## 📞 Quick Commands Reference

```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run development server
python flask_app.py

# Run production server (Linux)
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 flask_app:app

# Run production server (Windows)
waitress-serve --port=5000 flask_app:app

# Docker build & run
docker-compose up -d --build

# View Docker logs
docker-compose logs -f
```

---

## 🎉 Success!

Once deployed, access your app at:
- Local: http://localhost:5000
- Docker: http://your-server-ip:5000
- Cloud: Your provider's URL

**Good luck with your deployment, Nitish! 🚀**










