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

try:  # pragma: no cover - import guard for optional deps
    from parler_tts import ParlerTTSForConditionalGeneration
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - import guard
    ParlerTTSForConditionalGeneration = None
    sf = None


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




# Indic Parler-TTS language mapping
# Maps language codes to Indic Parler-TTS supported language names
INDIC_PARLER_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ur": "Urdu",
    "as": "Assamese",
    "or": "Odia",
    "pa": "Punjabi",  # Unofficial support
    "ne": "Nepali",
    "sa": "Sanskrit",
    "fr": "English",  # Fallback to English for non-Indic languages
    "es": "English",  # Fallback to English for non-Indic languages
    # Add more mappings as needed
}

# Voice descriptions for Indic Parler-TTS
VOICE_DESCRIPTIONS = {
    "female": "A female speaker with a clear and natural voice delivers expressive speech with moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and close up.",
    "male": "A male speaker with a clear and natural voice delivers expressive speech with moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and close up.",
}


@lru_cache(maxsize=1)
def _load_indic_parler_tts():
    """Load Indic Parler-TTS model and tokenizers."""
    if ParlerTTSForConditionalGeneration is None:
        raise RuntimeError(
            "parler-tts is required for TTS. Install it with: "
            "pip install git+https://github.com/huggingface/parler-tts.git soundfile"
        )
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _log(None, logging.INFO, f"Loading Indic Parler-TTS model on {device}...")
    
    # Use slow tokenizer with low memory flags to avoid RAM blowups on low-spec machines
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/indic-parler-tts",
        use_fast=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    
    # Get description tokenizer from model config
    description_tokenizer_path = model.config.text_encoder._name_or_path
    description_tokenizer = AutoTokenizer.from_pretrained(
        description_tokenizer_path,
        use_fast=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    _log(None, logging.INFO, "Indic Parler-TTS model loaded successfully.")
    return model, tokenizer, description_tokenizer, device


def _split_text_segments(text, max_chars=512):
    """
    Split text into segments for TTS processing.
    Tries to split at sentence boundaries first, then falls back to character limits.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per segment (default 512 for tokenizer limit)
    
    Returns:
        List of text segments
    """
    if not text or len(text.strip()) == 0:
        return [""]
    
    # Split on sentence boundaries (periods, question marks, exclamation marks, and Indic punctuation)
    # Include common Indic sentence endings: । (Devanagari danda), । (Bengali), etc.
    sentences = re.split(r'(?<=[।.!?।।])\s+', text.strip())
    
    segments = []
    current = ""
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        tentative = f"{current} {sentence}".strip() if current else sentence.strip()
        
        # If tentative segment fits, continue building
        if len(tentative) <= max_chars:
            current = tentative
        else:
            # Save current segment and start new one
            if current:
                segments.append(current)
            
            # Handle sentences longer than max_chars - split by words or characters
            if len(sentence) > max_chars:
                # Try to split by words first
                words = sentence.split()
                current_word_chunk = ""
                for word in words:
                    if len(current_word_chunk) + len(word) + 1 <= max_chars:
                        current_word_chunk = f"{current_word_chunk} {word}".strip() if current_word_chunk else word
                    else:
                        if current_word_chunk:
                            segments.append(current_word_chunk)
                        # Handle very long words by splitting them
                        if len(word) > max_chars:
                            # Split long word into chunks
                            for i in range(0, len(word), max_chars):
                                word_chunk = word[i:i+max_chars]
                                if word_chunk:
                                    segments.append(word_chunk)
                            current_word_chunk = ""
                        else:
                            current_word_chunk = word
                
                if current_word_chunk:
                    current = current_word_chunk
                else:
                    current = ""
            else:
                current = sentence.strip()
    
    # Add the last segment
    if current:
        segments.append(current)
    
    # If no segments were created (shouldn't happen), return the original text
    if not segments:
        segments = [text]
    
    # Verify all characters are preserved
    total_seg_chars = sum(len(seg) for seg in segments)
    original_chars = len(text.strip())
    
    # If we lost characters, try to recover them
    if total_seg_chars < original_chars:
        # Find missing characters and add to last segment
        missing_start = total_seg_chars
        if missing_start < original_chars and segments:
            missing_text = text[missing_start:original_chars].strip()
            if missing_text:
                segments[-1] = segments[-1] + " " + missing_text if segments[-1] else missing_text
    
    return segments


def synthesize_tts(text, language_code, voice="female", output_path=None, logger=None):
    """
    Synthesize speech using Indic Parler-TTS.
    Handles long texts by splitting into segments and concatenating audio.
    
    Args:
        text: Text to synthesize (can be long - will be split into segments)
        language_code: Language code (e.g., "en", "hi", "bn")
        voice: Voice type ("female" or "male")
        output_path: Path to save the audio file
        logger: Optional logger instance
    
    Returns:
        Path to generated audio file
    """
    if ParlerTTSForConditionalGeneration is None:
        raise RuntimeError(
            "parler-tts is required for TTS. Install it with: "
            "pip install git+https://github.com/huggingface/parler-tts.git soundfile"
        )
    
    # Check if language is supported
    lang_name = INDIC_PARLER_LANGUAGES.get(language_code.lower())
    if not lang_name:
        _log(logger, logging.WARNING, 
             f"Language '{language_code}' may not be officially supported by Indic Parler-TTS. "
             f"Attempting with English fallback.")
        lang_name = "English"
    
    # Get voice description
    voice_desc = VOICE_DESCRIPTIONS.get(voice.lower(), VOICE_DESCRIPTIONS["female"])
    
    # Enhance description with language context for better quality
    if language_code.lower() in ["en"]:
        # For English, specify Indian accent for better compatibility
        if "Indian" not in voice_desc:
            voice_desc = voice_desc.replace("speaker", "speaker with an Indian accent")
    
    _log(logger, logging.INFO, f"Generating TTS for {lang_name} ({language_code}) with {voice} voice...")
    _log(logger, logging.INFO, f"Input text length: {len(text)} characters")
    
    try:
        model, tokenizer, description_tokenizer, device = _load_indic_parler_tts()
        
        # Prepare voice description inputs ONCE (same for all segments to ensure voice consistency)
        description_input_ids = description_tokenizer(voice_desc, return_tensors="pt").to(device)
        
        # Split text into segments to handle long translations
        text_segments = _split_text_segments(text, max_chars=512)
        _log(logger, logging.INFO, f"Split text into {len(text_segments)} segment(s) for TTS processing")
        
        # Log segment details for debugging
        total_chars = sum(len(seg) for seg in text_segments)
        _log(logger, logging.INFO, f"Total characters in segments: {total_chars} (original: {len(text)})")
        if total_chars != len(text):
            _log(logger, logging.WARNING, 
                 f"Character count mismatch! Segments: {total_chars}, Original: {len(text)}")
            # Try to fix by adding missing character
            if total_chars < len(text):
                missing = text[total_chars:]
                if text_segments:
                    text_segments[-1] += missing
                    _log(logger, logging.INFO, f"Added missing {len(missing)} characters to last segment")
        
        all_audio_chunks = []
        sampling_rate = model.config.sampling_rate
        silence_duration = 0.12  # 120ms pause between segments
        silence_samples = int(sampling_rate * silence_duration)
        
        processed_segments = 0
        skipped_segments = 0
        
        # Use fixed random seed for voice consistency across segments
        # This ensures the same voice characteristics for all segments
        torch.manual_seed(42)  # Fixed seed for consistency
        
        # Generate audio for each segment
        for idx, segment in enumerate(text_segments):
            if not segment.strip():
                skipped_segments += 1
                _log(logger, logging.WARNING, f"Skipping empty segment {idx + 1}/{len(text_segments)}")
                continue
                
            processed_segments += 1
            _log(logger, logging.INFO, f"Processing segment {idx + 1}/{len(text_segments)} ({len(segment)} chars): {segment[:100]}..." if len(segment) > 100 else f"Processing segment {idx + 1}/{len(text_segments)} ({len(segment)} chars): {segment}")
            
            # Tokenize the segment with truncation to handle edge cases
            prompt_input_ids = tokenizer(
                segment,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Generate audio for this segment
            # Calculate max_new_tokens based on input length to ensure full audio generation
            # Use a more generous calculation to ensure complete audio
            input_length = prompt_input_ids.input_ids.shape[-1]
            # Increase multiplier to ensure full audio generation (was 32, now 40)
            max_new_tokens = min(4096, max(2048, 40 * input_length))  # More generous, higher cap
            
            _log(logger, logging.INFO, 
                 f"Generating audio for segment {idx + 1} with max_new_tokens={max_new_tokens} "
                 f"(input_length={input_length}, segment_chars={len(segment)})")
            
            # Use same voice description and consistent generation parameters for all segments
            with torch.no_grad():
                generation = model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    do_sample=True,
                    temperature=0.7,  # Slightly lower for more consistency
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1  # Ensure single output
                )
            
            # Extract audio waveform - handle both tensor and tuple outputs
            if isinstance(generation, tuple):
                audio_chunk = generation[0].cpu().numpy().astype(np.float32).squeeze()
            elif isinstance(generation, torch.Tensor):
                audio_chunk = generation.cpu().numpy().astype(np.float32).squeeze()
            else:
                # Try to get the first element if it's a list or dict-like
                audio_chunk = generation[0] if hasattr(generation, '__getitem__') else generation
                if isinstance(audio_chunk, torch.Tensor):
                    audio_chunk = audio_chunk.cpu().numpy().astype(np.float32).squeeze()
                else:
                    audio_chunk = np.array(audio_chunk).astype(np.float32).squeeze()
            
            # Ensure audio_chunk is 1D
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()
            
            chunk_duration = len(audio_chunk) / sampling_rate
            _log(logger, logging.INFO, 
                 f"Segment {idx + 1} generated: {len(audio_chunk)} samples ({chunk_duration:.2f}s)")
            
            all_audio_chunks.append(audio_chunk)
            
            # Add silence between segments (except after the last one)
            if idx < len(text_segments) - 1:
                silence = np.zeros(silence_samples, dtype=audio_chunk.dtype)
                all_audio_chunks.append(silence)
        
        # Concatenate all audio chunks
        if not all_audio_chunks:
            raise RuntimeError("No audio was generated from the text segments")
        
        _log(logger, logging.INFO, 
             f"TTS processing complete: {processed_segments} segments processed, "
             f"{skipped_segments} segments skipped, {len(all_audio_chunks)} audio chunks created")
        
        audio_arr = np.concatenate(all_audio_chunks)
        total_duration = len(audio_arr) / sampling_rate
        _log(logger, logging.INFO, f"Generated audio duration: {total_duration:.2f} seconds ({len(audio_arr)} samples at {sampling_rate}Hz)")
        
        # Estimate expected duration (rough: ~150 words per minute, ~5 chars per word)
        estimated_chars_per_second = 12.5  # Rough estimate
        estimated_duration = len(text) / estimated_chars_per_second
        _log(logger, logging.INFO, 
             f"Estimated duration for {len(text)} chars: ~{estimated_duration:.2f}s, "
             f"Actual: {total_duration:.2f}s")
        
        if total_duration < estimated_duration * 0.5:
            _log(logger, logging.WARNING, 
                 f"Generated audio duration ({total_duration:.2f}s) is significantly shorter than "
                 f"estimated ({estimated_duration:.2f}s). Some content may be missing!")
        
        # Create output path if not provided
        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / f"tts_{uuid4().hex}.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio file
        sf.write(str(output_path), audio_arr, sampling_rate)
        _log(logger, logging.INFO, f"TTS audio saved to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        _log(logger, logging.ERROR, f"TTS generation failed: {e}")
        raise RuntimeError(f"Failed to generate TTS audio: {e}") from e


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
    audio_path=None,  # Optional: path to pre-generated audio file (if provided, skips TTS)
):
    """
    Dub the supplied video and return final path plus transcripts.
    
    Args:
        video_path: Path to input video
        source_lang: Source language code (e.g., "en", "hi")
        dest_lang: Destination language code
        output_dir: Output directory for dubbed video
        logger: Optional logger instance
        voice: Voice type ("female" or "male") for TTS synthesis
        enable_lipsync: Whether to apply lip synchronization (uses Wav2Lip)
        lipsync_assets_dir: Directory with lip-sync assets (eBack repo location)
        lipsync_method: Lip-sync method (kept for API compatibility)
        lipsync_options: Lip-sync options (kept for API compatibility)
        audio_path: Optional path to pre-generated audio file (if provided, skips TTS)
    
    Returns:
        tuple: (final_path, transcript, translation)
    
    Note:
        Uses Indic Parler-TTS for text-to-speech synthesis if audio_path is not provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        original_audio_path = temp_dir / "original_audio.mp3"
        
        # Use original video for lip sync processing (no resizing to preserve quality)
        # IMPORTANT: Use original video path directly - do NOT resize to SD
        video_path_for_lipsync = str(video_path)  # Ensure it's the original video path
        _log(logger, logging.INFO, f"Using original video for lip sync: {video_path_for_lipsync} (no resizing)")

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
            
            # Generate transcript and translation
            transcript = recognize_speech(str(original_audio_path), source_lang, logger)
            _log(logger, logging.INFO, f"Transcript length: {len(transcript)} characters")
            _log(logger, logging.INFO, f"Transcript preview: {transcript[:200]}..." if len(transcript) > 200 else f"Transcript: {transcript}")
            
            if not transcript or len(transcript.strip()) == 0:
                raise ValueError("Transcript is empty! Cannot proceed with translation.")
            
            translation = translate_text(transcript, source_lang, dest_lang, logger)
            _log(logger, logging.INFO, f"Translation length: {len(translation)} characters")
            _log(logger, logging.INFO, f"Translation preview: {translation[:200]}..." if len(translation) > 200 else f"Translation: {translation}")
            
            if not translation or len(translation.strip()) == 0:
                raise ValueError("Translation is empty! Cannot proceed with TTS.")
            
            # Validate translation completeness
            # Translation should typically be similar length to transcript (within reasonable range)
            transcript_words = len(transcript.split())
            translation_words = len(translation.split())
            word_ratio = translation_words / transcript_words if transcript_words > 0 else 0
            
            _log(logger, logging.INFO, 
                 f"Transcript: {transcript_words} words, Translation: {translation_words} words "
                 f"(ratio: {word_ratio:.2f})")
            
            if word_ratio < 0.3:
                _log(logger, logging.WARNING, 
                     f"Translation appears incomplete! Word ratio ({word_ratio:.2f}) is very low. "
                     f"Expected ratio typically between 0.5-2.0 for most language pairs.")
            elif word_ratio > 3.0:
                _log(logger, logging.WARNING, 
                     f"Translation appears unusually long! Word ratio ({word_ratio:.2f}) is very high.")
            
            # Generate TTS audio if not provided
            if audio_path is None:
                # Generate TTS using Indic Parler-TTS
                _log(logger, logging.INFO, f"Generating TTS audio for translation in {dest_lang}...")
                _log(logger, logging.INFO, f"Full translation text ({len(translation)} chars) will be processed by TTS")
                tts_audio_path = temp_dir / "tts_audio.wav"
                synthesize_tts(
                    text=translation,
                    language_code=dest_lang,
                    voice=voice,
                    output_path=str(tts_audio_path),
                    logger=logger
                )
                tts_audio_path = Path(tts_audio_path)
            else:
                # Use provided audio_path
                tts_audio_path = Path(audio_path)
                if not tts_audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
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
                _log(logger, logging.INFO, f"Lip-sync video rendered via Wav2Lip (original resolution preserved).")
                return final_path, transcript, translation

            new_audio_clip = AudioFileClip(str(tts_audio_path))
            audio_needs_extension = False
            extended_audio_path = None
            audio_duration = new_audio_clip.duration
            video_duration = video_clip.duration
            
            if audio_duration < video_duration:
                # Audio is shorter than video - extend audio by looping to match video duration
                _log(logger, logging.INFO, 
                     f"Audio duration ({audio_duration:.2f}s) is shorter than video duration ({video_duration:.2f}s). "
                     f"Extending audio to match video...")
                
                # Use FFmpeg to loop audio efficiently
                from .eback_pipeline.video_utils import get_ffmpeg_path
                import subprocess
                
                ffmpeg_path = get_ffmpeg_path()
                extended_audio_path = temp_dir / "extended_audio.wav"
                
                # Calculate how many times we need to loop (with some buffer)
                loops_needed = int(np.ceil(video_duration / audio_duration)) + 1
                
                # Use FFmpeg to loop the audio and trim to exact video duration
                loop_cmd = [
                    ffmpeg_path,
                    "-y",
                    "-stream_loop", str(loops_needed),
                    "-i", str(tts_audio_path),
                    "-t", str(video_duration),  # Trim to exact video duration
                    "-c:a", "pcm_s16le",  # Use uncompressed PCM for intermediate file
                    str(extended_audio_path)
                ]
                
                _log(logger, logging.INFO, f"Looping audio {loops_needed} times using FFmpeg to match video duration")
                loop_result = subprocess.run(loop_cmd, capture_output=True, text=True)
                
                if loop_result.returncode != 0:
                    error_msg = loop_result.stderr if loop_result.stderr else loop_result.stdout
                    _log(logger, logging.ERROR, f"FFmpeg audio looping failed: {error_msg}")
                    raise RuntimeError(f"Failed to loop audio: {error_msg}")
                
                audio_needs_extension = True
                _log(logger, logging.INFO, f"Extended audio saved to {extended_audio_path}")
                
            elif audio_duration > video_duration:
                # Audio is longer than video - extend video with freeze frame
                _log(logger, logging.INFO,
                     f"Audio duration ({audio_duration:.2f}s) is longer than video duration ({video_duration:.2f}s). "
                     f"Extending video with freeze frame...")
                # Calculate safe freeze time - use a small offset from end to avoid EOF issues
                fps = video_clip.fps or 24
                freeze_time = max(video_duration - (2.0 / fps), 0)
                
                # Try to get the last frame with fallback to earlier frames
                frame_array = None
                for attempt_offset in [2, 5, 10, 20]:  # Try progressively earlier frames
                    try:
                        attempt_time = max(video_duration - (attempt_offset / fps), 0)
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
                    audio_duration - video_duration
                )
                if video_clip.fps:
                    freeze_frame = freeze_frame.set_fps(video_clip.fps)
                
                # Keep reference to original clip for cleanup
                original_video_clip = video_clip
                # Create extended clip but DON'T close original yet
                extended_clip = concatenate_videoclips([video_clip, freeze_frame])
                video_clip = extended_clip
            else:
                # Durations match - no extension needed
                _log(logger, logging.INFO,
                     f"Audio duration ({audio_duration:.2f}s) matches video duration ({video_duration:.2f}s). No extension needed.")

            # Use FFmpeg directly for more reliable audio merging and to preserve original video quality
            _log(logger, logging.INFO, "Merging audio with video using FFmpeg...")
            
            # Get original video properties to preserve resolution and quality
            original_video_clip_for_info = VideoFileClip(str(video_path), target_resolution=None)
            original_size = original_video_clip_for_info.size  # (width, height)
            original_fps = original_video_clip_for_info.fps
            original_duration = original_video_clip_for_info.duration
            original_video_clip_for_info.close()
            
            _log(logger, logging.INFO, 
                 f"Original video: {original_size[0]}x{original_size[1]}, FPS: {original_fps}, Duration: {original_duration:.2f}s")
            
            # Determine which video to use (original or extended)
            # If video was extended (audio > video), we need to save the extended video first
            video_needs_saving = audio_duration > video_duration
            
            if video_needs_saving:
                # Save extended video to temp file, preserving original resolution
                temp_extended_video = temp_dir / "extended_video.mp4"
                _log(logger, logging.INFO, f"Saving extended video to {temp_extended_video}...")
                video_clip.write_videofile(
                    str(temp_extended_video),
                    codec="libx264",
                    audio=False,
                    temp_audiofile=str(temp_dir / "temp-audio.m4a"),
                    remove_temp=True,
                    logger=None,
                    preset="medium",
                    ffmpeg_params=["-crf", "18", "-vf", f"scale={original_size[0]}:{original_size[1]}"]  # Preserve resolution
                )
                video_to_merge = str(temp_extended_video)
            else:
                # Use original video directly (no extension needed)
                video_to_merge = str(video_path)
            
            # Close clips to free resources before FFmpeg processing
            if video_needs_saving:
                video_clip.close()
            if 'original_video_clip' in locals() and original_video_clip:
                original_video_clip.close()
            
            # Use FFmpeg directly to merge audio with video, preserving original quality
            from .eback_pipeline.video_utils import get_ffmpeg_path
            import subprocess
            
            ffmpeg_path = get_ffmpeg_path()
            
            # Use extended audio if it was created, otherwise use original TTS audio
            audio_to_merge = str(extended_audio_path) if audio_needs_extension else str(tts_audio_path)
            
            # Verify audio file exists and get its duration
            if not Path(audio_to_merge).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_to_merge}")
            
            # Get audio duration using ffprobe
            ffprobe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
            audio_duration_cmd = [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_to_merge
            ]
            audio_dur_result = subprocess.run(audio_duration_cmd, capture_output=True, text=True)
            if audio_dur_result.returncode == 0:
                try:
                    audio_dur = float(audio_dur_result.stdout.strip())
                    _log(logger, logging.INFO, f"Audio duration: {audio_dur:.2f}s, Video duration: {video_duration:.2f}s")
                except:
                    pass
            
            # Merge audio with video using FFmpeg, preserving original video quality
            # Use -c:v copy when using original video (no re-encoding), or libx264 for extended video
            video_codec = "copy" if not video_needs_saving else "libx264"
            merge_cmd = [
                ffmpeg_path,
                "-y",
                "-i", video_to_merge,  # Use original or extended video
                "-ss", "0",  # Start audio from beginning (no offset)
                "-i", audio_to_merge,
                "-map", "0:v:0",  # Map video from first input
                "-map", "1:a:0",  # Map audio from second input
                "-c:v", video_codec,   # Copy codec for original (preserves quality) or encode for extended
            ]
            
            # Add quality parameters if re-encoding
            if video_needs_saving:
                merge_cmd.extend([
                    "-preset", "medium",
                    "-crf", "18",  # High quality
                    "-vf", f"scale={original_size[0]}:{original_size[1]}",  # Preserve resolution
                ])
            
            merge_cmd.extend([
                "-c:a", "aac",    # Encode audio as AAC
                "-b:a", "192k",   # Audio bitrate
                "-shortest",  # Match to shortest stream (but we've ensured they match)
                "-avoid_negative_ts", "make_zero",  # Ensure audio starts at 0
                str(final_path)
            ])
            
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
            
            # Clean up temp video file if it was created
            if video_needs_saving and 'temp_extended_video' in locals():
                temp_extended_video_path = Path(temp_extended_video)
                if temp_extended_video_path.exists():
                    temp_extended_video_path.unlink()
            
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

