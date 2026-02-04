"""
eBack LipSync Pipeline Integration

This module integrates the advanced lip sync pipeline from the LipSync-Edusync/eBack project.
Key features:
- Face detection intervals using MediaPipe
- Segment-based video processing
- Intelligent Wav2Lip application (only where faces are detected)
- Integration with existing Indic-Parler-TTS for high-quality audio

Repository: https://github.com/LipSync-Edusync/eBack
"""

from .orchestrator import process_video_with_lipsync
from .face_detection import detect_face_intervals, merge_intervals
from .video_utils import split_video, split_audio, process_videos, remove_audio, merge_audio_with_silent_video
from .audio_utils import convert_mp4_to_wav, detect_non_silent_regions

__all__ = [
    'process_video_with_lipsync',
    'detect_face_intervals',
    'merge_intervals',
    'split_video',
    'split_audio',
    'process_videos',
    'remove_audio',
    'merge_audio_with_silent_video',
    'convert_mp4_to_wav',
    'detect_non_silent_regions',
]

