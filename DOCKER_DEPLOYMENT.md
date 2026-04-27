# Deploy LipsyncExecution with Docker

This guide covers **Docker-only** deployment. You need Docker and Docker Compose on the host (your machine or professor's).

---

## Prerequisites

- **Docker** ([Install Docker Engine](https://docs.docker.com/engine/install/))
- **Docker Compose** ([Install Docker Compose](https://docs.docker.com/compose/install/))
- (Optional) **NVIDIA GPU**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

Check versions:

```bash
docker --version
docker compose version
```

> **Note:** If you have an older Docker setup, the command may be `docker-compose` (with a hyphen) instead of `docker compose`. Use whichever is installed.

---

## Step 1: Clone or copy the project

```bash
git clone https://github.com/harry1808/LipsyncExecution.git lipsyncExecution
cd lipsyncExecution
```

Or copy the project folder to the deployment machine and `cd` into it.

---

## Step 2: Create `.env` and set secret key

```bash
cp env.example .env
```

Edit `.env` and set a strong `FLASK_SECRET_KEY`:

```bash
# Linux/macOS
nano .env

# Or use any editor. Set at least:
# FLASK_SECRET_KEY=your-random-secret-key-here
```

Example:

```
FLASK_SECRET_KEY=my-super-secret-key-change-this
FLASK_ENV=production
NLLB_MODEL_NAME=facebook/nllb-200-distilled-600M
LIPSYNC_DEFAULT=0
```

Save and exit.

---

## Step 3: Prepare `instance` folder (uploads, outputs, Wav2Lip assets)

The app uses the host folder `./instance` for uploads, outputs, and Wav2Lip/eBack assets. Create the structure and eBack:

```bash
mkdir -p instance/uploads instance/outputs instance/wav2lip_assets
cd instance/wav2lip_assets
git clone https://github.com/LipSync-Edusync/eBack.git
cd ../..
```

Optional: download the Wav2Lip checkpoint so lip-sync works without first-run download:

```bash
cd instance/wav2lip_assets
wget -O wav2lip_gan.pth "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
cd ../..
```

(If you skip this, the app will try to download the checkpoint on first lip-sync use.)

---

## Step 4: Build and run with Docker Compose

From the **project root** (where `docker-compose.yml` and `Dockerfile` are):

```bash
docker compose up -d --build
```

- First run will build the image (can take several minutes).
- The app runs in the background. Logs:

```bash
docker compose logs -f
```

Stop when done with Ctrl+C. To stop the app:

```bash
docker compose down
```

---

## Step 5: Open the app

In a browser on the same machine:

**http://localhost:5000**

Register a user and upload a video to test. Uploads and outputs are stored in `./instance/uploads` and `./instance/outputs` on the host.

---

## Summary of what runs

| Item | Description |
|------|-------------|
| **Image** | Built from `Dockerfile`: Python 3.10, FFmpeg, your app + `requirements.txt`. |
| **Database** | SQLite file in a Docker volume `dbdata` at `/app/data/app.db` (persists across restarts). |
| **Instance** | Host folder `./instance` is mounted at `/app/instance` (uploads, outputs, `wav2lip_assets`). |
| **Port** | Container port 5000 is published as host port 5000. |

---

## Optional: Use GPU inside the container

If the host has an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed:

1. Edit `docker-compose.yml`.
2. Uncomment the `deploy` block under the `web` service (the `resources.reservations.devices` section for nvidia).
3. Recreate the container:

   ```bash
   docker compose up -d --build
   ```

---

## Troubleshooting

### Port 5000 already in use

Change the host port in `docker-compose.yml`:

```yaml
ports:
  - "5001:5000"
```

Then open **http://localhost:5001**.

### "unable to open database file" or SQLite errors

The app is configured to use `/app/data/app.db` inside the container, and the `dbdata` volume is mounted at `/app/data`. If you see DB errors:

- Ensure you didn’t override `SQLALCHEMY_DATABASE_URI` to a path that isn’t writable.
- Run without overriding: `docker compose down` then `docker compose up -d` and check logs: `docker compose logs -f`.

### eBack / Wav2Lip not found

- Confirm on the **host**: `instance/wav2lip_assets/eBack` exists (you ran `git clone` in Step 3).
- Restart: `docker compose restart`.

### Build fails (pip, MoviePy, etc.)

- Clear build cache and rebuild:  
  `docker compose build --no-cache`
- If a dependency fails, fix the version in `requirements.txt` and rebuild.

### First request is very slow

The first run may download NLLB, Whisper, and/or TTS models (several GB). Later requests use the cache.

### View logs

```bash
docker compose logs -f web
```

### Stop and remove containers/volumes

```bash
docker compose down
# To also remove the database volume:
docker compose down -v
```

---

## Quick reference

```bash
# First-time setup
cp env.example .env
# Edit .env: set FLASK_SECRET_KEY
mkdir -p instance/uploads instance/outputs instance/wav2lip_assets
cd instance/wav2lip_assets && git clone https://github.com/LipSync-Edusync/eBack.git && cd ../..

# Run
docker compose up -d --build

# Open browser
# http://localhost:5000

# Logs
docker compose logs -f

# Stop
docker compose down
```

That’s all you need for a Docker-only deployment.
