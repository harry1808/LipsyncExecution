"""
LipSync Module - Multi-Method Support

This module provides lip synchronization with multiple methods:

1. Wav2Lip (default) - Using eBack Repository
   - Repository: https://github.com/LipSync-Edusync/eBack
   - Faster processing, lower VRAM (4GB+)
   - Good quality for most use cases

2. LatentSync - ByteDance's Diffusion-based Lip Sync
   - Repository: https://github.com/bytedance/LatentSync
   - Higher quality lip-sync using Stable Diffusion
   - Requires more VRAM (8GB+ for v1.5, 18GB+ for v1.6)
   - Slower but more accurate results

Set the method via the `lipsync_method` parameter: "wav2lip" or "latentsync"
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from enum import Enum
from typing import Optional, Union

import requests
from huggingface_hub import hf_hub_download


class LipSyncMethod(Enum):
    """Available lip-sync methods."""
    WAV2LIP = "wav2lip"
    LATENTSYNC = "latentsync"


# Checkpoint download settings (checkpoint file not in eBack repo)
HF_WAV2LIP_REPO_ID = os.environ.get("WAV2LIP_HF_REPO", "camenduru/Wav2Lip")
HF_WAV2LIP_FILENAME = os.environ.get("WAV2LIP_HF_FILENAME", "checkpoints/wav2lip_gan.pth")

# Default lip-sync method
DEFAULT_LIPSYNC_METHOD = os.environ.get("DEFAULT_LIPSYNC_METHOD", "wav2lip")


def _log(logger, level, message):
    if logger:
        logger.log(level, message)


def _download_file(url, destination, logger=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    _log(logger, 20, f"Downloading {url}...")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as target:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    target.write(chunk)
    _log(logger, 20, f"Saved file to {destination}")


def _get_eback_wav2lip_dir(assets_dir):
    """Get path to eBack's Wav2Lip directory."""
    assets_dir = Path(assets_dir)
    eback_wav2lip = assets_dir / "eBack" / "api" / "pipeline" / "wav2lip"
    
    if eback_wav2lip.exists():
        return eback_wav2lip
    
    # Fallback to old Wav2Lip location
    old_wav2lip = assets_dir / "Wav2Lip"
    if old_wav2lip.exists():
        return old_wav2lip
    
    raise FileNotFoundError(
        f"eBack Wav2Lip not found. Please clone https://github.com/LipSync-Edusync/eBack "
        f"to {assets_dir / 'eBack'}"
    )


def _ensure_checkpoint(assets_dir, logger=None):
    """Ensure Wav2Lip checkpoint exists, download if needed."""
    assets_dir = Path(assets_dir)
    
    # Check multiple possible locations
    possible_paths = [
        assets_dir / "wav2lip_gan.pth",
        assets_dir / "eBack" / "api" / "pipeline" / "wav2lip" / "checkpoints" / "wav2lip_gan.pth",
        assets_dir / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth",
    ]
    
    for path in possible_paths:
        if path.exists():
            _log(logger, logging.INFO, f"Using existing checkpoint: {path}")
            return path
    
    # Download checkpoint
    checkpoint_path = assets_dir / "wav2lip_gan.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_url = os.environ.get("WAV2LIP_CHECKPOINT_URL")
    if not checkpoint_url:
        try:
            _log(logger, logging.INFO, "Downloading Wav2Lip checkpoint from HuggingFace...")
            hf_path = hf_hub_download(
                repo_id=HF_WAV2LIP_REPO_ID,
                filename=HF_WAV2LIP_FILENAME,
                repo_type="model",
            )
            shutil.copy(hf_path, checkpoint_path)
            _log(logger, logging.INFO, f"Checkpoint saved to: {checkpoint_path}")
            return checkpoint_path
        except Exception as exc:
            _log(logger, logging.WARNING, f"HuggingFace download failed ({exc}), trying direct URL...")
            checkpoint_url = (
                os.environ.get("WAV2LIP_CHECKPOINT_URL")
                or "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
            )
    
    _download_file(checkpoint_url, checkpoint_path, logger)
    return checkpoint_path


def ensure_wav2lip_assets(base_dir, logger=None):
    """Ensure eBack Wav2Lip and checkpoint are available."""
    base_dir = Path(base_dir)
    wav2lip_dir = _get_eback_wav2lip_dir(base_dir)
    checkpoint_path = _ensure_checkpoint(base_dir, logger)
    return wav2lip_dir, checkpoint_path


def run_wav2lip(face_video, audio_path, output_path, assets_dir, logger=None):
    """
    Run Wav2Lip lip synchronization using the eBack pipeline.
    
    Uses the Wav2Lip implementation from:
    https://github.com/LipSync-Edusync/eBack
    
    Args:
        face_video: Path to the input video
        audio_path: Path to the audio file (translated TTS output)
        output_path: Path for the output video
        assets_dir: Directory containing eBack repo and checkpoint
        logger: Optional logger instance
    
    Returns:
        Path to the lip-synced video
    """
    from .eback_pipeline import process_video_with_lipsync
    
    _log(logger, 20, "Running lip-sync with eBack pipeline (Wav2Lip)...")
    
    output_path = Path(output_path)
    output_dir = output_path.parent
    
    result_path = process_video_with_lipsync(
        video_path=face_video,
        translated_audio_path=audio_path,
        output_dir=output_dir,
        assets_dir=assets_dir,
        logger=logger,
        cleanup_temp=True,
    )
    
    # Rename to expected output path if different
    if result_path != output_path:
        shutil.move(str(result_path), str(output_path))
    
    _log(logger, 20, "Wav2Lip lip-sync complete.")
    return output_path


def run_latentsync(
    face_video,
    audio_path,
    output_path,
    logger=None,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    seed: int = 1247,
    use_512_resolution: bool = True,
):
    """
    Run LatentSync lip synchronization using ByteDance's diffusion model.
    
    Uses LatentSync from:
    https://github.com/bytedance/LatentSync
    
    Args:
        face_video: Path to the input video
        audio_path: Path to the audio file (translated TTS output)
        output_path: Path for the output video
        logger: Optional logger instance
        inference_steps: Number of diffusion steps (20-50)
        guidance_scale: Guidance scale (1.0-3.0)
        seed: Random seed for reproducibility
        use_512_resolution: Use 512x512 for higher quality
    
    Returns:
        Path to the lip-synced video
    """
    from .latentsync_pipeline import run_latentsync as _run_latentsync
    
    _log(logger, 20, "Running lip-sync with LatentSync (Diffusion-based)...")
    
    result_path = _run_latentsync(
        face_video=face_video,
        audio_path=audio_path,
        output_path=output_path,
        logger=logger,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        use_512_resolution=use_512_resolution,
    )
    
    _log(logger, 20, "LatentSync lip-sync complete.")
    return result_path


def run_lipsync(
    face_video: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    method: Union[str, LipSyncMethod] = None,
    assets_dir: Optional[Union[str, Path]] = None,
    logger=None,
    **kwargs
) -> Path:
    """
    Unified lip-sync interface supporting multiple methods.
    
    Args:
        face_video: Path to the input video
        audio_path: Path to the audio file (translated TTS output)
        output_path: Path for the output video
        method: Lip-sync method - "wav2lip" or "latentsync" (default: wav2lip)
        assets_dir: Directory containing assets (required for wav2lip)
        logger: Optional logger instance
        **kwargs: Additional method-specific arguments
            For LatentSync:
                - inference_steps: int (20-50)
                - guidance_scale: float (1.0-3.0)
                - seed: int
                - use_512_resolution: bool
    
    Returns:
        Path to the lip-synced video
    """
    # Determine method
    if method is None:
        method = DEFAULT_LIPSYNC_METHOD
    
    if isinstance(method, LipSyncMethod):
        method = method.value
    
    method = method.lower().strip()
    
    _log(logger, logging.INFO, f"Using lip-sync method: {method}")
    
    if method == LipSyncMethod.WAV2LIP.value or method == "wav2lip":
        if assets_dir is None:
            raise ValueError("assets_dir is required for Wav2Lip method")
        return run_wav2lip(
            face_video=face_video,
            audio_path=audio_path,
            output_path=output_path,
            assets_dir=assets_dir,
            logger=logger,
        )
    
    elif method == LipSyncMethod.LATENTSYNC.value or method == "latentsync":
        return run_latentsync(
            face_video=face_video,
            audio_path=audio_path,
            output_path=output_path,
            logger=logger,
            inference_steps=kwargs.get("inference_steps", 20),
            guidance_scale=kwargs.get("guidance_scale", 1.5),
            seed=kwargs.get("seed", 1247),
            use_512_resolution=kwargs.get("use_512_resolution", True),
        )
    
    else:
        raise ValueError(
            f"Unknown lip-sync method: {method}. "
            f"Supported methods: {[m.value for m in LipSyncMethod]}"
        )


def get_available_methods() -> list:
    """Get list of available lip-sync methods."""
    return [m.value for m in LipSyncMethod]


def get_method_info(method: str) -> dict:
    """
    Get information about a lip-sync method.
    
    Args:
        method: Method name ("wav2lip" or "latentsync")
    
    Returns:
        dict with method info including requirements and features
    """
    info = {
        "wav2lip": {
            "name": "Wav2Lip",
            "description": "Fast lip-sync using GAN-based approach",
            "repository": "https://github.com/LipSync-Edusync/eBack",
            "min_vram_gb": 4,
            "speed": "fast",
            "quality": "good",
            "requirements": ["eBack repository", "wav2lip_gan.pth checkpoint"],
        },
        "latentsync": {
            "name": "LatentSync",
            "description": "High-quality lip-sync using Stable Diffusion",
            "repository": "https://github.com/bytedance/LatentSync",
            "min_vram_gb": 8,
            "recommended_vram_gb": 18,
            "speed": "slow",
            "quality": "excellent",
            "requirements": [
                "LatentSync repository",
                "latentsync_unet.pt checkpoint",
                "Whisper tiny.pt checkpoint"
            ],
            "parameters": {
                "inference_steps": "20-50 (higher = better quality, slower)",
                "guidance_scale": "1.0-3.0 (higher = better sync, may distort)",
                "use_512_resolution": "True for better quality, requires more VRAM",
            }
        }
    }
    return info.get(method.lower(), {})
