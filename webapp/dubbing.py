import logging
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

try:  # pragma: no cover - import guard for optional deps
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError("torch is required for NLLB translation.") from exc

import whisper
from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips

import numpy as np

try:  # pragma: no cover - import guard for optional deps
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError("transformers is required for NLLB translation.") from exc


def _log(logger, level, message):
    if logger:
        logger.log(level, message)


# SD Resolution for faster processing
SD_RESOLUTION = (640, 480)


def _get_ffmpeg_path():
    """Get path to ffmpeg executable."""
    import subprocess
    # Check project's ffmpeg first
    project_ffmpeg = Path(__file__).parent.parent / "ffmpeg" / "bin" / "ffmpeg.exe"
    if project_ffmpeg.exists():
        return str(project_ffmpeg)
    return "ffmpeg"


def _resize_video_to_sd(input_path, output_path, logger=None):
    """
    Resize video to SD resolution (640x480) for faster processing.
    
    Args:
        input_path: Path to input video
        output_path: Path for resized video
        logger: Optional logger
    
    Returns:
        Path to resized video
    """
    import subprocess
    
    ffmpeg_path = _get_ffmpeg_path()
    
    _log(logger, logging.INFO, f"Resizing video to SD ({SD_RESOLUTION[0]}x{SD_RESOLUTION[1]}) for faster processing...")
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-vf", f"scale={SD_RESOLUTION[0]}:{SD_RESOLUTION[1]}",
        "-c:a", "copy",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        _log(logger, logging.WARNING, f"Video resize failed: {result.stderr}")
        # Return original path if resize fails
        return input_path
    
    _log(logger, logging.INFO, f"Video resized successfully to {output_path}")
    return output_path


NLLB_MODEL_NAME = os.environ.get("NLLB_MODEL_NAME", "facebook/nllb-200-distilled-600M")
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}



def _get_nllb_code(code):
    lang_code = NLLB_LANGUAGE_CODES.get(code.lower())
    if not lang_code:
        raise ValueError(f"Language '{code}' is not supported by the NLLB mapping.")
    return lang_code


def _get_token_id_for_lang(tokenizer, lang_code):
    """Resolve the BOS token id for a target language."""
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if mapping and lang_code in mapping:
        return mapping[lang_code]

    mapping = getattr(tokenizer, "lang_code_to_token_id", None)
    if mapping and lang_code in mapping:
        return mapping[lang_code]

    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if token_id != tokenizer.unk_token_id:
        return token_id

    raise ValueError(f"Tokenizer missing BOS token for language '{lang_code}'.")


@lru_cache(maxsize=1)
def _load_nllb_model():
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device




# TTS functionality has been removed - integrate your own TTS system here
# You can add a function like:
# def synthesize_tts(text, language_code, voice=None, output_path=None, logger=None):
#     # Your TTS implementation here
#     # Should return path to generated audio file
#     pass


def recognize_speech(audio_path, source_language, logger=None):
    lang_map = {
        "en": "english",
        "bn": "bengali",
        "hi": "hindi",
        "te": "telugu",
        "ta": "tamil",
        "ml": "malayalam",
        "kn": "kannada",
        "mr": "marathi",
        "gu": "gujarati",
        "pa": "punjabi",
        "ur": "urdu",
    }
    whisper_lang = lang_map.get(source_language.lower(), "english")

    _log(logger, logging.INFO, f"Loading Whisper model for {whisper_lang} transcription.")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=whisper_lang)
    text = result["text"].strip()
    _log(logger, logging.INFO, "Transcription finished.")
    return text


def translate_text(text, source_language, destination_language, logger=None):
    src = source_language.lower()
    dest = destination_language.lower()

    src_lang_code = _get_nllb_code(src)
    dest_lang_code = _get_nllb_code(dest)

    tokenizer, model, device = _load_nllb_model()
    tokenizer.src_lang = src_lang_code

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  # Increased for longer transcripts
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    forced_bos_token_id = _get_token_id_for_lang(tokenizer, dest_lang_code)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_length=2048,  # Increased for longer translations
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    _log(logger, logging.INFO, "Translation finished via NLLB.")
    return translation


def process_video(
    video_path,
    source_lang,
    dest_lang,
    output_dir,
    logger=None,
    voice="female",
    enable_lipsync=False,
    lipsync_assets_dir=None,
    lipsync_method="wav2lip",  # kept for API compatibility
    lipsync_options=None,  # kept for API compatibility
    audio_path=None,  # Optional: path to pre-generated audio file (TTS removed)
):
    """
    Dub the supplied video and return final path plus transcripts.
    
    Args:
        video_path: Path to input video
        source_lang: Source language code (e.g., "en", "hi")
        dest_lang: Destination language code
        output_dir: Output directory for dubbed video
        logger: Optional logger instance
        voice: Voice type ("female" or "male") - kept for API compatibility, not used
        enable_lipsync: Whether to apply lip synchronization (uses Wav2Lip)
        lipsync_assets_dir: Directory with lip-sync assets (eBack repo location)
        audio_path: Optional path to pre-generated audio file (TTS functionality removed)
    
    Returns:
        tuple: (final_path, transcript, translation)
    
    Note:
        TTS functionality has been removed. You must provide audio_path with a pre-generated
        audio file, or integrate your own TTS system before calling this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Use provided audio_path or raise error
        if audio_path is None:
            raise ValueError(
                "TTS functionality has been removed. Please provide audio_path parameter "
                "with a path to a pre-generated audio file, or integrate your own TTS system."
            )
        
        tts_audio_path = Path(audio_path)
        if not tts_audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        original_audio_path = temp_dir / "original_audio.mp3"
        
        # Auto-resize video to SD for faster lip sync processing
        if enable_lipsync:
            sd_video_path = temp_dir / "video_sd.mp4"
            video_path_for_lipsync = _resize_video_to_sd(video_path, sd_video_path, logger)
        else:
            video_path_for_lipsync = video_path

        video_clip = None
        audio_clip = None
        new_audio_clip = None
        final_clip = None
        freeze_frame = None
        original_video_clip = None  # Keep reference for cleanup

        try:
            video_clip = VideoFileClip(str(video_path), target_resolution=None)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(str(original_audio_path), logger=None)
            if audio_clip:
                audio_clip.close()
                audio_clip = None  # Already closed

            transcript = recognize_speech(str(original_audio_path), source_lang, logger)
            translation = translate_text(transcript, source_lang, dest_lang, logger)
            
            # TTS removed - using provided audio_path
            _log(logger, logging.INFO, f"Using provided audio file: {tts_audio_path}")

            output_filename = f"dubbed_{uuid4().hex}.mp4"
            final_path = output_dir / output_filename

            if enable_lipsync:
                # Use eBack pipeline directly for wav2lip (original working method)
                from .eback_pipeline import process_video_with_lipsync
                
                if not lipsync_assets_dir:
                    raise ValueError("lipsync_assets_dir is required for lip-sync")
                
                _log(logger, logging.INFO, "Starting Wav2Lip lip-sync with eBack pipeline...")
                
                lipsync_result = process_video_with_lipsync(
                    video_path=video_path_for_lipsync,
                    translated_audio_path=tts_audio_path,
                    output_dir=temp_dir,
                    assets_dir=lipsync_assets_dir,
                    logger=logger,
                    cleanup_temp=True,
                )
                
                shutil.move(str(lipsync_result), str(final_path))
                _log(logger, logging.INFO, f"Lip-sync video rendered via Wav2Lip (SD resolution).")
                return final_path, transcript, translation

            new_audio_clip = AudioFileClip(str(tts_audio_path))
            if new_audio_clip.duration > video_clip.duration:
                # Calculate safe freeze time - use a small offset from end to avoid EOF issues
                fps = video_clip.fps or 24
                freeze_time = max(video_clip.duration - (2.0 / fps), 0)
                
                # Try to get the last frame with fallback to earlier frames
                frame_array = None
                for attempt_offset in [2, 5, 10, 20]:  # Try progressively earlier frames
                    try:
                        attempt_time = max(video_clip.duration - (attempt_offset / fps), 0)
                        frame_array = video_clip.get_frame(attempt_time)
                        break
                    except (IOError, OSError) as e:
                        _log(logger, logging.WARNING, f"Failed to get frame at {attempt_time:.2f}s, trying earlier...")
                        continue
                
                if frame_array is None:
                    # Last resort: try frame at 0
                    try:
                        frame_array = video_clip.get_frame(0)
                        _log(logger, logging.WARNING, "Using first frame as freeze frame fallback")
                    except (IOError, OSError):
                        raise IOError(
                            f"Cannot read any frames from video. The file may be corrupted: {video_path}"
                        )
                freeze_frame = ImageClip(frame_array).set_duration(
                    new_audio_clip.duration - video_clip.duration
                )
                if video_clip.fps:
                    freeze_frame = freeze_frame.set_fps(video_clip.fps)
                
                # Keep reference to original clip for cleanup
                original_video_clip = video_clip
                # Create extended clip but DON'T close original yet
                extended_clip = concatenate_videoclips([video_clip, freeze_frame])
                video_clip = extended_clip

            # Use FFmpeg directly for more reliable audio merging instead of MoviePy
            _log(logger, logging.INFO, "Merging audio with video using FFmpeg...")
            
            # First, write the video without audio to a temp file
            temp_video_path = temp_dir / "temp_video_no_audio.mp4"
            video_clip.write_videofile(
                str(temp_video_path),
                codec="libx264",
                audio=False,  # No audio in video
                temp_audiofile=str(temp_dir / "temp-audio.m4a"),
                remove_temp=True,
                logger=None,
            )
            
            # Close clips to free resources
            video_clip.close()
            if 'original_video_clip' in locals() and original_video_clip:
                original_video_clip.close()
            
            # Now merge audio using FFmpeg directly for reliability
            from .eback_pipeline.video_utils import get_ffmpeg_path
            import subprocess
            
            ffmpeg_path = get_ffmpeg_path()
            merge_cmd = [
                ffmpeg_path,
                "-y",
                "-i", str(temp_video_path),
                "-i", str(tts_audio_path),
                "-map", "0:v:0",  # Map video from first input
                "-map", "1:a:0",  # Map audio from second input
                "-c:v", "copy",   # Copy video codec (no re-encoding)
                "-c:a", "aac",    # Encode audio as AAC
                "-b:a", "192k",   # Audio bitrate
                "-shortest",      # Match to shortest stream
                str(final_path)
            ]
            
            _log(logger, logging.INFO, f"Merging with FFmpeg: {' '.join(merge_cmd)}")
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                _log(logger, logging.ERROR, f"FFmpeg merge failed: {error_msg}")
                raise RuntimeError(f"Failed to merge audio with video: {error_msg}")
            
            # Verify the output has audio
            try:
                ffprobe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
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
                        _log(logger, logging.INFO, f"✓ Output file has audio: codec={audio_codec}")
                    else:
                        _log(logger, logging.WARNING, "⚠ Output file has NO audio stream!")
                else:
                    _log(logger, logging.WARNING, "Could not verify audio (ffprobe failed)")
            except Exception as e:
                _log(logger, logging.WARNING, f"Could not verify audio: {e}")
            
            # Clean up temp video file
            if temp_video_path.exists():
                temp_video_path.unlink()
            
            _log(logger, logging.INFO, "Video dubbing finished.")
            return final_path, transcript, translation
        finally:
            # Clean up all clips in reverse order of creation
            clips_to_close = []
            if 'final_clip' in locals() and final_clip is not None:
                clips_to_close.append(final_clip)
            if 'new_audio_clip' in locals() and new_audio_clip is not None:
                clips_to_close.append(new_audio_clip)
            if 'freeze_frame' in locals() and freeze_frame is not None:
                clips_to_close.append(freeze_frame)
            if 'video_clip' in locals() and video_clip is not None:
                clips_to_close.append(video_clip)
            if 'original_video_clip' in locals() and original_video_clip is not None:
                clips_to_close.append(original_video_clip)
            if 'audio_clip' in locals() and audio_clip is not None:
                clips_to_close.append(audio_clip)
            
            for clip in clips_to_close:
                try:
                    clip.close()
                except:
                    pass  # Ignore errors during cleanup

