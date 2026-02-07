# Quick Start: Indic Parler-TTS Integration

## ✅ Integration Complete!

Indic Parler-TTS has been successfully integrated into your LipsyncExecution pipeline. Here's what was done:

### Changes Made

1. ✅ Added `parler-tts` and `soundfile` to `requirements.txt`
2. ✅ Created `synthesize_tts()` function in `webapp/dubbing.py`
3. ✅ Updated `process_video()` to automatically generate TTS audio
4. ✅ Added language code mappings for Indic languages
5. ✅ Implemented voice selection (male/female)

## 🚀 Quick Setup Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `git+https://github.com/huggingface/parler-tts.git`
- `soundfile`

### Step 2: Verify Installation

Run this Python command to test:

```python
python -c "from parler_tts import ParlerTTSForConditionalGeneration; print('✓ Indic Parler-TTS installed!')"
```

### Step 3: Test the Integration

The TTS will automatically work when you:
1. Upload a video through the web interface
2. Select source and target languages
3. Choose voice type (male/female)
4. Process the video

The system will:
- Extract audio from video
- Transcribe using Whisper
- Translate using NLLB
- **Generate TTS audio using Indic Parler-TTS** ← NEW!
- Apply lip-sync (if enabled)
- Create final dubbed video

## 📝 Usage

### Through Web Interface

1. Go to `/dashboard`
2. Upload a video
3. Select languages (e.g., English → Hindi)
4. Choose voice (male/female)
5. Enable lip-sync if needed
6. Submit

### Programmatic Usage

```python
from webapp.dubbing import process_video

final_path, transcript, translation = process_video(
    video_path="input.mp4",
    source_lang="en",
    dest_lang="hi",
    output_dir="./outputs",
    voice="female",  # or "male"
    enable_lipsync=True,
    lipsync_assets_dir="./instance/wav2lip_assets"
)
```

## 🌍 Supported Languages

The following languages are now supported for TTS:

- ✅ English (en)
- ✅ Hindi (hi)
- ✅ Bengali (bn)
- ✅ Telugu (te)
- ✅ Tamil (ta)
- ✅ Malayalam (ml)
- ✅ Kannada (kn)
- ✅ Marathi (mr)
- ✅ Gujarati (gu)
- ✅ Urdu (ur)
- ✅ Assamese (as)
- ✅ Odia (or)
- ✅ Punjabi (pa) - unofficial
- ✅ Nepali (ne)
- ✅ Sanskrit (sa)

## ⚙️ How It Works

1. **Model Loading**: On first use, the model (~0.9GB) downloads from Hugging Face
2. **Caching**: Model is cached in memory for subsequent calls
3. **Device**: Automatically uses GPU if available, otherwise CPU
4. **Voice**: Uses natural language descriptions for voice control

## 🔧 Troubleshooting

### Model Download Issues

If the model fails to download:
1. Check internet connection
2. Visit https://huggingface.co/ai4bharat/indic-parler-tts
3. Accept the model's terms of use (if prompted)
4. Try again

### Memory Issues

If you get out-of-memory errors:
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python your_script.py
```

### Language Not Working

- Check if language code is in `INDIC_PARLER_LANGUAGES` dictionary
- Unsupported languages will fallback to English
- You can add custom mappings in `webapp/dubbing.py`

## 📚 More Information

See `INDIC_PARLER_TTS_INTEGRATION.md` for detailed documentation.

## 🎯 What's Next?

The integration is complete and ready to use! Just install the dependencies and start processing videos. The TTS will work automatically.

---

**Note**: The first run will be slower as it downloads the model. Subsequent runs will be faster due to caching.

