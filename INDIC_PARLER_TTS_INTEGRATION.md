# Indic Parler-TTS Integration Guide

This document explains how Indic Parler-TTS has been integrated into the LipsyncExecution pipeline and how to use it.

## Overview

Indic Parler-TTS is a multilingual text-to-speech model that supports 21 Indic languages and English. It has been integrated into the dubbing pipeline to replace the previous TTS system.

## Installation

### Step 1: Install Dependencies

The required dependencies have been added to `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

This will install:
- `parler-tts` (from GitHub)
- `soundfile` (for audio file handling)

### Step 2: Verify Installation

You can verify the installation by running:

```python
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

print("Indic Parler-TTS installed successfully!")
```

## Supported Languages

Indic Parler-TTS officially supports the following languages:

1. **Assamese** (as)
2. **Bengali** (bn)
3. **Bodo** (brx)
4. **Dogri** (doi)
5. **English** (en)
6. **Gujarati** (gu)
7. **Hindi** (hi)
8. **Kannada** (kn)
9. **Konkani** (kok)
10. **Maithili** (mai)
11. **Malayalam** (ml)
12. **Manipuri** (mni)
13. **Marathi** (mr)
14. **Nepali** (ne)
15. **Odia** (or)
16. **Sanskrit** (sa)
17. **Santali** (sat)
18. **Sindhi** (sd)
19. **Tamil** (ta)
20. **Telugu** (te)
21. **Urdu** (ur)

**Unofficial support** (may work but not extensively tested):
- Chhattisgarhi
- Kashmiri
- Punjabi (pa)

## How It Works

### Integration Flow

1. **Video Processing**: The video is loaded and audio is extracted
2. **Speech Recognition**: Whisper transcribes the original audio
3. **Translation**: NLLB translates the transcript to the target language
4. **TTS Synthesis**: Indic Parler-TTS generates speech from the translated text
5. **Lip Sync** (optional): Wav2Lip synchronizes mouth movements with the new audio
6. **Video Assembly**: Final video is created with the new audio track

### Code Structure

The TTS functionality is implemented in `webapp/dubbing.py`:

- `synthesize_tts()`: Main TTS function that generates audio from text
- `_load_indic_parler_tts()`: Cached model loader for efficiency
- `INDIC_PARLER_LANGUAGES`: Language code mapping dictionary
- `VOICE_DESCRIPTIONS`: Voice style descriptions for male/female voices

### Usage Example

```python
from webapp.dubbing import synthesize_tts

# Generate TTS audio
audio_path = synthesize_tts(
    text="Hello, how are you?",
    language_code="hi",  # Hindi
    voice="female",
    output_path="output.wav"
)
```

## Voice Options

The system supports two voice types:

- **female**: Female speaker with clear, natural voice
- **male**: Male speaker with clear, natural voice

You can customize the voice description in `VOICE_DESCRIPTIONS` dictionary for more control over:
- Pitch (high/low/balanced)
- Speaking rate (slow/moderate/fast)
- Expressivity (monotone/expressive)
- Background noise level
- Reverberation

## Configuration

### Model Loading

The model is loaded lazily and cached using `@lru_cache`. The first TTS call will download the model (~0.9GB) from Hugging Face. Subsequent calls will reuse the cached model.

### Device Selection

The system automatically uses GPU if available (`cuda:0`), otherwise falls back to CPU. CPU inference is slower but works without GPU.

## Troubleshooting

### Issue: ModuleNotFoundError for parler_tts

**Solution**: Install the package:
```bash
pip install git+https://github.com/huggingface/parler-tts.git soundfile
```

### Issue: Model download fails

**Solution**: 
- Check your internet connection
- Ensure you have enough disk space (~1GB for the model)
- Try accessing the model page: https://huggingface.co/ai4bharat/indic-parler-tts
- You may need to accept the model's terms of use on Hugging Face

### Issue: Out of memory errors

**Solution**:
- Use CPU instead of GPU (set `CUDA_VISIBLE_DEVICES=""` before running)
- Process shorter text segments
- Close other applications using GPU memory

### Issue: Language not supported

**Solution**:
- Check if the language code is in `INDIC_PARLER_LANGUAGES`
- For unsupported languages, the system will fallback to English
- You can add custom language mappings in the dictionary

### Issue: Audio quality issues

**Solution**:
- Try different voice descriptions (modify `VOICE_DESCRIPTIONS`)
- Ensure the input text is clean and properly formatted
- For better quality, use GPU inference

## Performance Considerations

- **First Run**: Slow due to model download (~1GB)
- **GPU Inference**: Fast (~1-2 seconds per sentence)
- **CPU Inference**: Slower (~5-10 seconds per sentence)
- **Model Size**: ~0.9GB (loaded into memory)

## Advanced Usage

### Custom Voice Descriptions

You can customize voice characteristics by modifying the description:

```python
custom_description = (
    "A female speaker with a high-pitched, fast-paced voice "
    "delivers cheerful speech. The recording is of very high quality, "
    "with no background noise."
)
```

### Batch Processing

For processing multiple texts, reuse the loaded model:

```python
model, tokenizer, desc_tokenizer, device = _load_indic_parler_tts()

# Process multiple texts without reloading model
for text in texts:
    # Generate audio...
```

## API Compatibility

The integration maintains backward compatibility:
- If `audio_path` is provided, TTS is skipped (uses provided audio)
- If `audio_path` is `None`, TTS is automatically generated
- The `voice` parameter now controls TTS voice selection

## References

- [Indic Parler-TTS Model Card](https://huggingface.co/ai4bharat/indic-parler-tts)
- [Parler-TTS GitHub](https://github.com/huggingface/parler-tts)
- [Original Paper](https://arxiv.org/abs/2402.01912)

## License

Indic Parler-TTS is licensed under Apache 2.0, compatible with this project.

