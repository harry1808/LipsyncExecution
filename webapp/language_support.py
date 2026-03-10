from typing import Sequence

# All languages that support both TTS (Indic Parler-TTS) and lip-sync (Wav2Lip is language-agnostic)
SUPPORTED_LANGS = [
    {"code": "en", "label": "English"},
    {"code": "hi", "label": "Hindi"},
    {"code": "bn", "label": "Bengali"},
    {"code": "te", "label": "Telugu"},
    {"code": "ta", "label": "Tamil"},
    {"code": "ml", "label": "Malayalam"},
    {"code": "kn", "label": "Kannada"},
    {"code": "mr", "label": "Marathi"},
    {"code": "gu", "label": "Gujarati"},
    {"code": "pa", "label": "Punjabi"},
    {"code": "ur", "label": "Urdu"},
    {"code": "as", "label": "Assamese"},
    {"code": "brx", "label": "Bodo"},
    {"code": "doi", "label": "Dogri"},
    {"code": "kok", "label": "Konkani"},
    {"code": "mai", "label": "Maithili"},
    {"code": "mni", "label": "Manipuri"},
    {"code": "ne", "label": "Nepali"},
    {"code": "or", "label": "Odia"},
    {"code": "sa", "label": "Sanskrit"},
    {"code": "sat", "label": "Santali"},
    {"code": "sd", "label": "Sindhi"},
]

SUPPORTED_CODES: Sequence[str] = tuple(lang["code"] for lang in SUPPORTED_LANGS)
