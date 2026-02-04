from typing import Sequence

SUPPORTED_LANGS = [
    {"code": "en", "label": "English"},
    {"code": "fr", "label": "French"},
    {"code": "es", "label": "Spanish"},
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
]

SUPPORTED_CODES: Sequence[str] = tuple(lang["code"] for lang in SUPPORTED_LANGS)

