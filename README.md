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
├── run_evaluation.py         # CLI evaluation runner
├── example_evaluation.py     # Evaluation usage examples
├── requirements.txt
├── env.example               # Copy to .env and configure
└── test_data_template.json   # Config template for batch evaluation
```

---

## Requirements

- **Python 3.8+**
- **PyTorch** (with CUDA for GPU)
- **FFmpeg** (in PATH or under `ffmpeg/bin/`)
- **Wav2Lip assets:** eBack repo under `instance/wav2lip_assets/` and checkpoint `wav2lip_gan.pth`
- **Optional (lip-sync evaluation):** `syncnet_v2.model` in Wav2Lip assets for LSE-D/LSE-C metrics

See `requirements.txt` for Python dependencies (Whisper, transformers, Parler-TTS, Flask, etc.).

- **Docker deployment:** see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for a step-by-step Docker and Docker Compose guide.
- **Deploy on another computer (on-site or remote):** see [DEPLOY_REMOTE_SYSTEM.md](DEPLOY_REMOTE_SYSTEM.md).

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

The UI supports all Indic languages that have both TTS and lip-sync: **English**, **Hindi**, **Bengali**, **Telugu**, **Tamil**, **Malayalam**, **Kannada**, **Marathi**, **Gujarati**, **Punjabi**, **Urdu**, **Assamese**, **Bodo**, **Dogri**, **Konkani**, **Maithili**, **Manipuri**, **Nepali**, **Odia**, **Sanskrit**, **Santali**, and **Sindhi**. Translation uses NLLB; TTS uses Indic Parler-TTS (see **Indic Parler-TTS** section below and `webapp/language_support.py` for the full list and codes).

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

---

## Indic Parler-TTS

Indic Parler-TTS is used for multilingual speech synthesis (21+ Indic languages and English). It is integrated in `webapp/dubbing.py` via `synthesize_tts()`.

**Setup:** Install with `pip install -r requirements.txt` (includes `parler-tts` and `soundfile`). On first use the model (~0.9GB) downloads from Hugging Face; it is then cached.

**Supported languages (official):** Assamese, Bengali, Bodo, Dogri, English, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu. Unofficial: Punjabi (pa), etc.

**Voice options:** `voice="female"` or `voice="male"` in `process_video()` or the web form.

**Troubleshooting:** If the model fails to download, check the internet and visit https://huggingface.co/ai4bharat/indic-parler-tts (accept terms if prompted). For OOM, use CPU: `CUDA_VISIBLE_DEVICES=""`. Unsupported language codes fall back to English; see `INDIC_PARLER_LANGUAGES` in `webapp/dubbing.py`.

---

## Evaluation (Full Guide)

### How to run evaluation

**Web (recommended):** Run `python flask_app.py`, open http://127.0.0.1:5000, log in, click **📊 Evaluation**, choose a completed activity, enter ground-truth transcript and translation, then **Calculate Evaluation Metrics**. Results show composite score, WER, BLEU, CER, and side-by-side comparisons.

**CLI – single video:**
```bash
python run_evaluation.py --quick \
  --video "path/to/video.mp4" \
  --source-lang en --dest-lang hi \
  --transcript "What was actually said" \
  --translation "Expected translation in target language" \
  --html
```
Output: `quick_eval/report.html` and JSON.

**CLI – batch:** Create a config JSON (see `test_data_template.json`) with `test_cases` (each with `video_path`, `source_lang`, `dest_lang`, `ground_truth.transcript`, `ground_truth.translation`), then:
```bash
python run_evaluation.py --config my_tests.json --html
```

**Python API:**
```python
from webapp.evaluate_dubbing import DubbingEvaluator
evaluator = DubbingEvaluator()
results = evaluator.evaluate_full_pipeline(
    video_path="test.mp4", source_lang="en", dest_lang="hi",
    ground_truth={"transcript": "...", "translation": "..."},
    output_dir="./evaluation_output", enable_lipsync=True,
    lipsync_assets_dir="./instance/wav2lip_assets",
)
print(evaluator.generate_report(results))
```

### Metrics

- **ASR:** WER (word error rate), CER (character error rate), accuracy. Good: WER < 20%, excellent: < 10%.
- **Translation:** BLEU (0–1). Good: > 0.7, acceptable: > 0.5.
- **Duration:** Error % (original vs dubbed length). Excellent: < 5%.
- **Composite score (0–100):** Weighted combination; 90–100 = Excellent, 75–89 = Good, 60–74 = Fair, 40–59 = Poor, 0–39 = Needs improvement.

Formula (example): `Composite = 0.25×(100-WER) + 0.30×(BLEU×100) + 0.25×(100-CER) + 0.20×(100-DurationError)`.

### Ground truth best practices

**Do not use raw Google Translate as ground truth.** BLEU compares exact n-grams; different phrasing gives low BLEU even when both translations are correct.

**Recommended:** Use your system’s output as the base: run dubbing, copy the system’s transcript/translation, correct only real errors, then use that as ground truth. The web **Evaluation Helper** (“Use Evaluation Helper”) copies system output into the form so you can edit and submit.

**Interpretation:** With system-based ground truth, BLEU 0.9–1.0 = minimal errors, 0.7–0.9 = good, 0.5–0.7 = fair, < 0.5 = significant errors. With Google Translate as reference, 0.3–0.5 may still mean a good translation with different wording.

### Outputs

- **Console:** Real-time report with WER, BLEU, duration, composite score.
- **JSON:** `evaluation_output/evaluation_results.json` (and batch JSON).
- **HTML:** `evaluation_output/report.html` or `quick_eval/report.html` with progress bars and comparisons.

### Troubleshooting

- **High WER:** Check audio quality, language code, and Whisper output.
- **Low BLEU:** Prefer system-based ground truth; avoid raw Google Translate.
- **Large duration error:** Check TTS and source/target language length mismatch.
- **Crashes:** Ensure enough GPU memory, reduce batch size, verify dependencies and video integrity.

---

## Project report summary

This project implements an **AI-powered multilingual video dubbing system with optional lip synchronization**. It combines:

- **Whisper** (ASR), **NLLB-200** (translation), **Indic Parler-TTS** (TTS), **Wav2Lip** (lip-sync).
- A **Flask web app** with auth, upload, processing, and evaluation.
- **Evaluation:** BLEU, WER, CER, duration metrics, composite quality score (0–100), and optional lip-sync metrics (LSE-D, LSE-C).

Supported: 13+ languages including English, Hindi, Bengali, Telugu, Tamil, Malayalam, Kannada, Marathi, Gujarati, Punjabi, Urdu, and others. Typical results: WER ~8–15%, BLEU ~0.4–0.7, composite ~78–84, MOS ~4.0/5. Pipeline: upload → transcribe → translate → TTS → (optional) lip-sync → download. For full methodology, baselines, and appendices (install, env vars, API examples, troubleshooting), see the project report content previously in `PROJECT_REPORT.md` (now consolidated here).

---

## Wav2Lip training (summary)

Lip-sync uses an audio-driven generator plus a **SyncNet** discriminator. Training uses ~5.6M audio–video pairs from **LRS2** (primary), **LRS3**, and **VoxCeleb2**. Pipeline: 25 FPS frames → RetinaFace face detection → FAN landmarks → 96×96 aligned crop; audio at 16 kHz → 80-dim Mel spectrogram. Improvements over baseline: RetinaFace, FAN alignment, stronger Sync Loss weight, temporal frame jitter, mixed precision (FP16). Metrics: LSE-D (lower = better), LSE-C (higher = better), PSNR/SSIM/LPIPS. This repo uses pre-trained Wav2Lip/eBack checkpoints; see `instance/wav2lip_assets/Wav2Lip` and eBack documentation for training details.

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
