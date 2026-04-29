# Deploy on another computer (remote or on-site)

Use this when the app should run on **someone else’s PC**, a **lab machine**, or any **second Windows/Linux host**—whether you configure it **in person** or **over a remote session** (screen sharing, SSH, RDP, etc.).

## What to do on the target machine

1. **Same setup as a normal install** — follow [README.md](README.md) (Python venv) **or** [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) (Docker).  
2. **Clone or copy** this repository onto that machine.  
3. **`.env`** — copy from `env.example`, set `FLASK_SECRET_KEY` and any model tokens you need.  
4. **`instance/`** — create `uploads`, `outputs`, `wav2lip_assets`; clone **eBack** under `instance/wav2lip_assets/eBack` and add **`wav2lip_gan.pth`** as in the Docker README.  
5. **FFmpeg** on PATH (or use the project’s `ffmpeg/bin` layout on Windows).  
6. **GPU (optional)** — install a CUDA-matched PyTorch build if you want GPU inference.

## Remote setup tips

- Prefer **Docker** on the target host if you want the fewest “which Python?” issues; the image pins **Python 3.10** inside the container.  
- For long model downloads, run **`docker compose build`** or **`pip install -r requirements.txt`** once while the network is stable.  
- If you use **screen sharing** or **remote desktop**, only the person at the keyboard needs to approve installers (Docker Desktop, Python, GPU drivers).

## Related docs

| Doc | Use when |
|-----|----------|
| [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) | Container-based run |
| [README.md](README.md) | Local venv, features, evaluation |
