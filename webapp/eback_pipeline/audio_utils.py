"""
Audio Processing Utilities for LipSync Pipeline

Handles audio extraction, silence detection, and audio manipulation.
"""

import os
import logging
import subprocess
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub import silence as pydub_silence
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

logger = logging.getLogger("eback_pipeline.audio_utils")


def get_ffmpeg_path():
    """Get the path to ffmpeg executable."""
    # Check common locations
    possible_paths = [
        "ffmpeg",  # System PATH
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except FileNotFoundError:
            continue
    
    return "ffmpeg"  # Default, hope it's in PATH


def convert_mp4_to_wav(video_path, output_dir):
    """
    Extract audio from video and convert to WAV format.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save the output WAV file
    
    Returns:
        Path to the output WAV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = Path(video_path)
    base = video_path.stem
    out_path = os.path.join(output_dir, f"{base}.wav")
    
    ffmpeg_path = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None
    
    return out_path


def detect_non_silent_regions(audio_input, silence_thresh=-60, min_silence_len=1000):
    """
    Detect non-silent regions in audio.
    
    Args:
        audio_input: Either path to WAV file or AudioSegment object
        silence_thresh: Silence threshold in dBFS (default: -60)
        min_silence_len: Minimum silence length in ms (default: 1000)
    
    Returns:
        List of (start_seconds, end_seconds) tuples for non-silent regions
    """
    if not HAS_PYDUB:
        logger.warning("pydub not available, returning entire audio as non-silent")
        return [(0, float('inf'))]
    
    if isinstance(audio_input, AudioSegment):
        audio = audio_input
    else:
        audio = AudioSegment.from_wav(str(audio_input))
    
    silent_ranges = pydub_silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    
    # Convert to seconds
    silent_ranges = [
        (start / 1000.0, end / 1000.0) for start, end in silent_ranges
    ]
    
    non_silent_ranges = []
    prev = 0.0
    duration = len(audio) / 1000.0
    
    for start, end in silent_ranges:
        if start > prev:
            non_silent_ranges.append((prev, start))
        prev = end
    
    if prev < duration:
        non_silent_ranges.append((prev, duration))
    
    return non_silent_ranges


def load_audio_segment(audio_path):
    """Load an audio file as a pydub AudioSegment."""
    if not HAS_PYDUB:
        raise ImportError("pydub is required for audio processing")
    
    return AudioSegment.from_file(str(audio_path))


def export_audio_segment(audio_segment, output_path, format="wav"):
    """Export an AudioSegment to a file."""
    audio_segment.export(str(output_path), format=format)
    return output_path

