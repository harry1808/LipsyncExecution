# LipsyncExecution

**LipsyncExecution** is a Python-based toolkit for automatic video dubbing with accurate lip-sync. The system uses AI models to transcribe, translate, synthesize speech, and synchronize mouth movements with the new audio—producing dubbed videos in many target languages.

---

## Features

- **Automatic speech recognition (ASR):** Whisper transcribes the source video audio.
- **Translation:** NLLB (No Language Left Behind) translates the transcript to the target language.
- **Text-to-speech (TTS):** Indic Parler-TTS generates natural speech for the translation (supports 21+ Indic languages and English).
- **Lip-sync:** Wav2Lip (via [eBack](https://github.com/LipSync-Edusync/eBack)) aligns mouth movements with the new audio.
- **Web application:** Flask app with user accounts, upload, processing, and evaluation.
- **Evaluation pipeline:** ASR (WER, CER), translation (BLEU), and lip-sync metrics (LSE-D, LSE-C, AV offset, duration consistency) with composite quality score.
- **Batch processing:** Run dubbing and evaluation on multiple videos.
- **Fault tolerance:** Error handling, cleanup of temporary files, and optional retries.

---

## Project Structure

```
lipsyncExecution/
├── flask_app.py              # Flask application entry point
├── webapp/
│   ├── __init__.py           # App factory (create_app)
│   ├── dubbing.py            # Main dubbing pipeline (ASR → translate → TTS → lip-sync)
│   ├── lipsync.py            # Wav2Lip/eBack lip-sync interface
│   ├── lipsync_metrics.py     # Lip-sync evaluation (LSE-D, LSE-C, duration)
│   ├── syncnet_model.py      # SyncNet model for lip-sync scoring
│   ├── evaluation_metrics.py # BLEU, WER, CER, composite score
│   ├── evaluate_dubbing.py   # Full and batch evaluation (DubbingEvaluator)
│   ├── eback_pipeline/       # eBack Wav2Lip orchestration
│   ├── routes.py             # Web routes (dashboard, evaluate, download)
│   ├── models.py             # User, Activity (SQLAlchemy)
│   └── templates/            # HTML templates (dashboard, evaluation, results)
├── instance/
│   ├── wav2lip_assets/       # Wav2Lip/eBack repo + checkpoints (wav2lip_gan.pth, syncnet_v2.model)
│   ├── outputs/              # Processed videos
│   └── uploads/              # Uploaded source videos
├── test_dubbing.py           # Test script (process_video with sample_video.mp4)
├── run_test_lipsync.py       # End-to-end test with lip-sync
├── requirements.txt
├── env.example               # Copy to .env and configure
├── DEPLOYMENT.md             # Docker, cloud, and production deployment
├── INDIC_PARLER_TTS_INTEGRATION.md
└── QUICK_START_INDIC_TTS.md
```

---

## Requirements

- **Python 3.8+**
- **PyTorch** (with CUDA for GPU)
- **FFmpeg** (in PATH or under `ffmpeg/bin/`)
- **Wav2Lip assets:** eBack repo under `instance/wav2lip_assets/` and checkpoint `wav2lip_gan.pth`
- **Optional (lip-sync evaluation):** `syncnet_v2.model` in Wav2Lip assets for LSE-D/LSE-C metrics

See `requirements.txt` for Python dependencies (Whisper, transformers, Parler-TTS, Flask, etc.).

---

## Setup

### 1. Clone and virtual environment

```bash
git clone https://github.com/yourusername/LipsyncExecution.git
cd LipsyncExecution

python -m venv lipsyncenv
# Windows:
lipsyncenv\Scripts\activate
# Linux/macOS:
source lipsyncenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Wav2Lip / eBack assets

- Clone [eBack](https://github.com/LipSync-Edusync/eBack) into `instance/wav2lip_assets/eBack/`.
- Download the Wav2Lip checkpoint (e.g. `wav2lip_gan.pth`) and place it in `instance/wav2lip_assets/` (or the path referenced in code).
- For **lip-sync evaluation** (LSE-D, LSE-C): add `syncnet_v2.model` (e.g. under `instance/wav2lip_assets/Wav2Lip/evaluation/scores_LSE/data/` or project root). See [Wav2Lip evaluation](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation/scores_LSE).

### 4. Environment configuration

```bash
cp env.example .env
# Edit .env: set FLASK_SECRET_KEY, optionally LIPSYNC_DEFAULT=1, NLLB_MODEL_NAME, etc.
```

### 5. FFmpeg

- Install [FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH, or place `ffmpeg`/`ffprobe` under the project’s `ffmpeg/bin/` directory.

---

## Usage

### Web application

```bash
python flask_app.py
```

- Open **http://127.0.0.1:5000**
- Sign up / log in, upload a video, choose source and target languages and voice (male/female), optionally enable lip-sync.
- Processed videos appear on the dashboard; you can download them or run **Evaluation** (with ground-truth transcript and translation) to see ASR, translation, and lip-sync metrics.

### Programmatic: dubbing

```python
from webapp.dubbing import process_video
from pathlib import Path

final_path, transcript, translation = process_video(
    video_path=Path("input.mp4"),
    source_lang="en",
    dest_lang="hi",
    output_dir="./output",
    logger=my_logger,
    voice="female",
    enable_lipsync=True,
    lipsync_assets_dir="./instance/wav2lip_assets",
)
```

### Programmatic: evaluation

```python
from webapp.evaluate_dubbing import DubbingEvaluator

evaluator = DubbingEvaluator()
results = evaluator.evaluate_full_pipeline(
    video_path="test.mp4",
    source_lang="en",
    dest_lang="hi",
    ground_truth={
        "transcript": "Reference transcript text",
        "translation": "Reference translation text",
    },
    output_dir="./evaluation_output",
    enable_lipsync=True,
    lipsync_assets_dir="./instance/wav2lip_assets",
)
print(evaluator.generate_report(results))
```

### Test scripts

- **Quick pipeline test:** Put a video at `sample_video.mp4`, then:
  ```bash
  python test_dubbing.py
  ```
- **Lip-sync test:** Ensure `instance/uploads/` has sample videos and Wav2Lip assets are set up, then:
  ```bash
  python run_test_lipsync.py
  ```

---

## Evaluation Metrics

| Component    | Metrics |
|------------|---------|
| **ASR**     | WER (word error rate), CER (character error rate), accuracy |
| **Translation** | BLEU (1–4 grams), brevity penalty |
| **Lip-sync**   | Duration consistency (video vs audio length), LSE-D (lower is better), LSE-C (higher is better), AV offset (frames/ms). Optional SyncNet-based metrics when `syncnet_v2.model` is available. |
| **Overall** | Composite score (0–100) combining the above; used in the web UI and in `evaluate_full_pipeline`. |

In the **web app**, use **Evaluation** for a completed activity: enter ground-truth transcript and translation to get ASR, translation, and lip-sync results (when the output video exists and assets are configured).

---

## Supported Languages

The UI supports a subset of languages (e.g. English, French, Spanish, Hindi, Bengali, Telugu, Tamil, Malayalam, Kannada, Marathi, Gujarati, Punjabi, Urdu). Translation uses NLLB; TTS uses Indic Parler-TTS (see `INDIC_PARLER_TTS_INTEGRATION.md` and `webapp/language_support.py` for the full list and codes).

---

## Lip-Sync (Wav2Lip / eBack)

- Lip-sync is **optional**. Enable it in the web form or via `enable_lipsync=True` and `lipsync_assets_dir` in code.
- Pipeline: original video + new TTS audio → Wav2Lip (eBack) → output video with synced lips.
- Model files (`wav2lip_gan.pth`, etc.) are not included in the repo; download them as per eBack/Wav2Lip instructions.

---

## Configuration (`.env`)

| Variable           | Description |
|--------------------|-------------|
| `FLASK_SECRET_KEY` | Secret key for sessions (required in production). |
| `FLASK_ENV`        | `development` or `production`. |
| `NLLB_MODEL_NAME`  | Translation model (default: `facebook/nllb-200-distilled-600M`). |
| `LIPSYNC_DEFAULT`  | `1` to enable lip-sync by default in the web app, `0` to disable. |
| `WAV2LIP_ASSETS_DIR` | Set in app config if different from `instance/wav2lip_assets`. |

---

## Notes & Troubleshooting

- **Model files** (`.pth`, `.pt`, `syncnet_v2.model`) are not in the repo; download and place them in the paths expected by the code.
- **GPU:** Recommended for Whisper, NLLB, and Wav2Lip; CPU is slower.
- **FFmpeg:** Required for video/audio handling; ensure it is on PATH or under `ffmpeg/bin/`.
- **Lip-sync evaluation:** If SyncNet checkpoint is missing, only duration consistency is reported; LSE-D/LSE-C are skipped without failing the app.
- For deployment (Docker, cloud), see **DEPLOYMENT.md**.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Credits

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and [eBack](https://github.com/LipSync-Edusync/eBack) for lip-sync.
- [OpenAI Whisper](https://github.com/openai/whisper) for ASR.
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) and NLLB for translation.
- [Indic Parler-TTS](https://github.com/huggingface/parler-tts) for TTS.
- SyncNet/syncnet_python for lip-sync evaluation metrics (LSE-D, LSE-C).

For questions or contributions, please open an issue or pull request.
