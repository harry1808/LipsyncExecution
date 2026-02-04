"""
Video Processing Utilities for LipSync Pipeline

Handles video splitting, merging, and processing operations.
"""

import os
import logging
import subprocess
from pathlib import Path

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

logger = logging.getLogger("eback_pipeline.video_utils")


def get_ffmpeg_path():
    """Get the path to ffmpeg executable."""
    possible_paths = [
        "ffmpeg",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except FileNotFoundError:
            continue
    
    return "ffmpeg"


def remove_audio(input_video, output_dir):
    """
    Remove audio from a video file.
    
    Args:
        input_video: Path to the input video
        output_dir: Directory to save the output video
    
    Returns:
        Path to the silent video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    input_video = Path(input_video)
    base_name = input_video.stem
    output_video = os.path.join(output_dir, f"{base_name}_no_audio.mp4")
    
    ffmpeg_path = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_video),
        "-c", "copy",
        "-an",
        str(output_video)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error removing audio: {e}")
        return None
    
    return output_video


def merge_audio_with_silent_video(silent_video, input_audio, output_video):
    """
    Merge audio with a silent video.
    
    Args:
        silent_video: Path to the video without audio
        input_audio: Path to the audio file
        output_video: Path for the output video
    """
    ffmpeg_path = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(silent_video),
        "-i", str(input_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        str(output_video)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to merge audio with video: {result.stderr}")


def split_video(video_path, face_intervals, output_prefix):
    """
    Split video based on face intervals.
    
    Args:
        video_path: Path to the input video
        face_intervals: List of (start, duration, has_face) tuples
        output_prefix: Prefix for output files
    
    Returns:
        List of paths to split video files
    """
    ffmpeg_path = get_ffmpeg_path()
    split_files = []
    
    for i, (start, duration, value) in enumerate(face_intervals):
        output_path = f"{output_prefix}_fiv-{value}_{i}.mp4"
        
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"FFmpeg warning for segment {i}: {result.stderr}")
        
        split_files.append(output_path)
    
    return split_files


def split_audio(audio_path, face_intervals, output_prefix):
    """
    Split audio based on face intervals.
    
    Args:
        audio_path: Path to the input audio (WAV)
        face_intervals: List of (start, duration, has_face) tuples
        output_prefix: Prefix for output files
    
    Returns:
        List of paths to split audio files
    """
    if not HAS_PYDUB:
        raise ImportError("pydub is required for audio splitting")
    
    audio = AudioSegment.from_wav(str(audio_path))
    split_files = []
    
    for i, (start, duration, value) in enumerate(face_intervals):
        output_path = f"{output_prefix}_fiv-{value}_{i}.wav"
        end = start + duration
        
        segment = audio[int(start * 1000):int(end * 1000)]
        segment.export(str(output_path), format="wav")
        
        split_files.append(output_path)
    
    return split_files


def process_videos(video_files, utc_str, output_dir):
    """
    Concatenate multiple video files into one.
    
    Args:
        video_files: List of video file paths to concatenate
        utc_str: Unique identifier string
        output_dir: Directory for output file
    
    Returns:
        Path to the concatenated video file
    """
    ffmpeg_path = get_ffmpeg_path()
    
    # Filter out invalid files
    valid_files = []
    for video in video_files:
        if os.path.exists(video) and os.path.getsize(video) > 0:
            valid_files.append(video)
    
    if not valid_files:
        logger.error("No valid video files to concatenate")
        return None
    
    concat_file_path = os.path.join(output_dir, f"merged_final_video_{utc_str}.mp4")
    concat_list_path = os.path.join(output_dir, f"concat_list_{utc_str}.txt")
    
    # Write concat list file
    with open(concat_list_path, "w") as f:
        for file in valid_files:
            # Escape single quotes and use absolute path
            abs_path = os.path.abspath(file).replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        str(concat_file_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up concat list
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg concat error: {result.stderr}")
        return None
    
    return concat_file_path


def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    ffmpeg_path = get_ffmpeg_path().replace("ffmpeg", "ffprobe")
    
    cmd = [
        ffmpeg_path,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Error getting video duration: {e}")
    
    return None

