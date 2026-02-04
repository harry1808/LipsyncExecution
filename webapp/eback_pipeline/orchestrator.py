"""
eBack LipSync Pipeline Orchestrator

Uses the Wav2Lip implementation from the LipSync-Edusync/eBack repository.
Repository: https://github.com/LipSync-Edusync/eBack

Processes the FULL video with FULL audio to maintain voice consistency.
"""

import os
import sys
import time
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from .face_detection import detect_face_intervals, merge_intervals
from .video_utils import get_ffmpeg_path

logger = logging.getLogger("eback_pipeline.orchestrator")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _get_eback_paths(assets_dir):
    """
    Get paths to eBack's Wav2Lip inference script and checkpoint.
    
    Uses the Wav2Lip from: https://github.com/LipSync-Edusync/eBack
    """
    assets_dir = Path(assets_dir)
    
    # eBack repo location
    eback_wav2lip = assets_dir / "eBack" / "api" / "pipeline" / "wav2lip"
    
    if not eback_wav2lip.exists():
        raise FileNotFoundError(
            f"eBack repository not found at {eback_wav2lip}. "
            f"Please clone https://github.com/LipSync-Edusync/eBack to {assets_dir / 'eBack'}"
        )
    
    inference_script = eback_wav2lip / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"inference.py not found at {inference_script}")
    
    # Get checkpoint from lipsync module
    from ..lipsync import _ensure_checkpoint
    checkpoint_path = _ensure_checkpoint(assets_dir, logger)
    
    return eback_wav2lip, inference_script, checkpoint_path


def _run_wav2lip_full(video_path, audio_path, output_path, assets_dir, logger=None):
    """
    Run Wav2Lip on the FULL video with FULL audio using eBack's implementation.
    This maintains voice consistency throughout the video.
    """
    wav2lip_dir, inference_script, checkpoint_path = _get_eback_paths(assets_dir)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate audio file exists and has content
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    audio_size = audio_path.stat().st_size
    if audio_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    if logger:
        logger.info(f"Audio file validated: {audio_path} ({audio_size} bytes)")
    
    # Create temp directory for Wav2Lip output
    temp_dir = wav2lip_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Use absolute paths for better reliability
    video_path = Path(video_path)
    abs_video = str(video_path.resolve() if video_path.is_absolute() else video_path.absolute())
    abs_audio = str(audio_path.resolve() if audio_path.is_absolute() else audio_path.absolute())
    abs_output = str(output_path.resolve() if output_path.is_absolute() else output_path.absolute())
    
    command = [
        sys.executable,
        str(inference_script),
        "--checkpoint_path", str(checkpoint_path),
        "--face", abs_video,
        "--audio", abs_audio,
        "--outfile", abs_output,
        "--face_det_batch_size", "4",  # Reduced for memory efficiency
        "--resize_factor", "2",  # Resize for faster processing
    ]
    
    if logger:
        logger.info(f"Running eBack Wav2Lip on full video...")
        logger.info(f"  Video: {abs_video}")
        logger.info(f"  Audio: {abs_audio} ({audio_size} bytes)")
        logger.info(f"  Output: {abs_output}")
        logger.info(f"  Working directory: {wav2lip_dir}")
        logger.info(f"  Command: {' '.join(command)}")
    
    print(f"[eBack] Starting Wav2Lip process...", flush=True)
    print(f"[eBack] Video: {abs_video}", flush=True)
    print(f"[eBack] Audio: {abs_audio} ({audio_size} bytes)", flush=True)
    print(f"[eBack] Output: {abs_output}", flush=True)
    print(f"[eBack] Command: {' '.join(command)}", flush=True)
    
    # Merge stderr into stdout to avoid deadlock (subprocess blocks if stderr buffer fills)
    process = subprocess.Popen(
        command,
        cwd=str(wav2lip_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout to prevent deadlock
        text=True,
        bufsize=1,
    )
    
    output_lines = []
    
    # Read combined stdout+stderr in real-time
    for line in process.stdout:
        line = line.strip()
        if line:
            output_lines.append(line)
            if logger:
                logger.info(f"[Wav2Lip] {line}")
            print(f"[Wav2Lip] {line}", flush=True)
    
    return_code = process.wait()
    
    if return_code != 0:
        error_msg = "\n".join(output_lines[-20:]) if output_lines else "No error details"
        if logger:
            logger.error(f"[Wav2Lip] Failed with code {return_code}")
            logger.error(f"[Wav2Lip] Error output:\n{error_msg}")
        raise RuntimeError(f"Wav2Lip inference failed with code {return_code}:\n{error_msg}")
    
    if not output_path.exists():
        raise RuntimeError(f"Wav2Lip did not produce output file: {output_path}")
    
    return output_path


def _get_media_duration(file_path):
    """Get duration of a video or audio file using ffprobe."""
    ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    return None


def _merge_audio_video(video_path, audio_path, output_path, logger=None):
    """
    Merge audio with video without lip sync.
    Removes any existing audio from video first, then adds new audio.
    """
    ffmpeg_path = get_ffmpeg_path()
    
    # First, verify audio file exists and has content
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    audio_size = audio_path.stat().st_size
    if audio_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    if logger:
        logger.info(f"Merging audio with video (no lip sync needed)...")
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Audio: {audio_path} ({audio_size} bytes)")
        logger.info(f"  Output: {output_path}")
    
    # Use explicit mapping and ensure we remove any existing audio
    # Map only video from first input, ignore any existing audio
    # Map audio from second input
    # Try with copy first (faster), but if that fails, re-encode
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",      # Map only video stream from first input
        "-map", "1:a:0",      # Map audio stream from second input
        "-c:v", "libx264",    # Re-encode video to ensure compatibility (more reliable than copy)
        "-preset", "medium",  # Balance between speed and quality
        "-crf", "23",         # Good quality
        "-c:a", "aac",        # Encode audio as AAC
        "-b:a", "192k",       # Set audio bitrate
        "-ar", "44100",       # Set audio sample rate
        "-ac", "2",           # Set audio channels (stereo)
        "-movflags", "+faststart",  # Enable fast start for web playback
        "-shortest",          # Match to shortest stream
        str(output_path)
    ]
    
    if logger:
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        if logger:
            logger.error(f"FFmpeg merge failed: {error_msg}")
        raise RuntimeError(f"FFmpeg merge failed: {error_msg}")
    
    # Verify output was created
    output_path = Path(output_path)
    if not output_path.exists():
        raise RuntimeError(f"Output file was not created: {output_path}")
    
    output_size = output_path.stat().st_size
    if output_size == 0:
        raise RuntimeError(f"Output file is empty: {output_path}")
    
    if logger:
        logger.info(f"Merge completed: {output_path} ({output_size} bytes)")
    
    return output_path


def process_video_with_lipsync(
    video_path,
    translated_audio_path,
    output_dir,
    assets_dir,
    logger=None,
    cleanup_temp=True,
):
    """
    Process a video with lip sync using the eBack Wav2Lip implementation.
    
    This function:
    1. Detects if the video contains faces
    2. If faces found: applies Wav2Lip to the FULL video with FULL audio
    3. If no faces: simply merges the audio with the video
    
    Uses: https://github.com/LipSync-Edusync/eBack
    
    Args:
        video_path: Path to the input video
        translated_audio_path: Path to the translated audio (single TTS output)
        output_dir: Directory for output files
        assets_dir: Directory containing eBack repo
        logger: Optional logger
        cleanup_temp: Whether to clean up temporary files
    
    Returns:
        Path to the final video
    """
    if logger:
        logger.info("[eBack] Starting lip sync with eBack Wav2Lip...")
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    utc_str = uuid4().hex[:12]
    output_filename = f"lipsync_{utc_str}.mp4"
    final_path = output_dir / output_filename
    
    temp_dir = tempfile.mkdtemp(prefix=f"eback_{utc_str}_")
    
    try:
        t0 = time.time()
        
        # Step 1: Quick face detection
        if logger:
            logger.info("[eBack] Checking for faces in video...")
        
        face_intervals = detect_face_intervals(str(video_path), fps_sample=2)
        merged_intervals = merge_intervals(face_intervals)
        
        has_faces = any(interval[2] == 1 for interval in merged_intervals)
        
        face_duration = sum(interval[1] for interval in merged_intervals if interval[2] == 1)
        total_duration = sum(interval[1] for interval in merged_intervals) if merged_intervals else 0
        
        if logger:
            if has_faces:
                logger.info(f"[eBack] Faces detected in {face_duration:.1f}s of {total_duration:.1f}s")
            else:
                logger.info("[eBack] No faces detected")
        
        # Step 2: Process
        if has_faces:
            # Apply Wav2Lip to FULL video with FULL audio (one consistent voice)
            _run_wav2lip_full(
                video_path=video_path,
                audio_path=translated_audio_path,
                output_path=final_path,
                assets_dir=assets_dir,
                logger=logger
            )
        else:
            # No faces - just merge audio
            _merge_audio_video(
                video_path=video_path,
                audio_path=translated_audio_path,
                output_path=final_path,
                logger=logger
            )
        
        elapsed = time.time() - t0
        if logger:
            logger.info(f"[eBack] Complete! Time: {elapsed:.2f}s")
            logger.info(f"[eBack] Output: {final_path}")
        
        # Verify the output file has audio
        if logger:
            logger.info(f"[eBack] Verifying output file has audio...")
        
        try:
            ffprobe_path = get_ffmpeg_path().replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
            verify_cmd = [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,codec_type",
                "-of", "json",
                str(final_path)
            ]
            
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            if verify_result.returncode == 0:
                import json
                probe_data = json.loads(verify_result.stdout)
                streams = probe_data.get('streams', [])
                audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
                
                if audio_streams:
                    audio_codec = audio_streams[0].get('codec_name', 'unknown')
                    # Also check audio duration and bitrate to ensure it's not empty
                    duration_cmd = [
                        ffprobe_path,
                        "-v", "error",
                        "-select_streams", "a:0",
                        "-show_entries", "stream=duration,bit_rate",
                        "-of", "json",
                        str(final_path)
                    ]
                    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    if duration_result.returncode == 0:
                        duration_data = json.loads(duration_result.stdout)
                        duration_streams = duration_data.get('streams', [])
                        if duration_streams:
                            duration = duration_streams[0].get('duration')
                            bit_rate = duration_streams[0].get('bit_rate')
                            if logger:
                                logger.info(f"[eBack] [OK] Output file has audio: codec={audio_codec}, duration={duration}s, bitrate={bit_rate}")
                            if duration and float(duration) == 0:
                                if logger:
                                    logger.warning(f"[eBack] [WARNING] Audio stream exists but duration is 0!")
                        else:
                            if logger:
                                logger.info(f"[eBack] [OK] Output file has audio: codec={audio_codec}")
                    else:
                        if logger:
                            logger.info(f"[eBack] [OK] Output file has audio: codec={audio_codec}")
                else:
                    if logger:
                        logger.warning(f"[eBack] [WARNING] Output file has NO audio stream! Attempting to merge audio again...")
                    # Try to merge audio again using the simple merge function
                    temp_output = final_path.parent / f"temp_{final_path.name}"
                    _merge_audio_video(
                        video_path=final_path,
                        audio_path=translated_audio_path,
                        output_path=temp_output,
                        logger=logger
                    )
                    # Replace the original with the merged version
                    if temp_output.exists():
                        final_path.unlink()  # Remove original
                        temp_output.rename(final_path)  # Rename merged to original
                        if logger:
                            logger.info(f"[eBack] [OK] Audio re-merged successfully")
            else:
                if logger:
                    logger.warning(f"[eBack] Could not verify audio (ffprobe failed), but file exists")
        except Exception as e:
            if logger:
                logger.warning(f"[eBack] Could not verify audio: {e}")
        
        return final_path
        
    except Exception as e:
        if logger:
            logger.error(f"[eBack] Error: {e}")
        raise
    finally:
        if cleanup_temp and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
