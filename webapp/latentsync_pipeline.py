"""
LatentSync Integration Module

Integrates ByteDance's LatentSync (https://github.com/bytedance/LatentSync)
as an alternative lip-sync method using audio-conditioned latent diffusion models.

LatentSync offers higher quality lip-sync compared to Wav2Lip, but requires:
- More VRAM (8GB min for v1.5, 18GB for v1.6)
- Longer processing time
- GPU with float16 support for optimal performance

Version 1.6 uses 512x512 resolution for better quality.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

# Add LatentSync to path
LATENTSYNC_DIR = Path(__file__).parent.parent / "LatentSync"
if str(LATENTSYNC_DIR) not in sys.path:
    sys.path.insert(0, str(LATENTSYNC_DIR))


def _log(logger, level, message):
    if logger:
        logger.log(level, message)


def _get_latentsync_dir():
    """Get path to LatentSync directory."""
    latentsync_dir = LATENTSYNC_DIR
    if latentsync_dir.exists():
        return latentsync_dir
    raise FileNotFoundError(
        f"LatentSync not found at {latentsync_dir}. "
        "Please clone https://github.com/bytedance/LatentSync"
    )


def _ensure_checkpoints(latentsync_dir: Path, logger=None) -> tuple:
    """
    Ensure LatentSync checkpoints exist, download if needed.
    
    Returns:
        tuple: (unet_checkpoint_path, whisper_checkpoint_path)
    """
    from huggingface_hub import hf_hub_download
    
    checkpoints_dir = latentsync_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    unet_checkpoint = checkpoints_dir / "latentsync_unet.pt"
    whisper_dir = checkpoints_dir / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    whisper_checkpoint = whisper_dir / "tiny.pt"
    
    # Download UNet checkpoint if not present
    if not unet_checkpoint.exists():
        _log(logger, logging.INFO, "Downloading LatentSync UNet checkpoint from HuggingFace...")
        try:
            hf_path = hf_hub_download(
                repo_id="ByteDance/LatentSync-1.6",
                filename="latentsync_unet.pt",
                repo_type="model",
            )
            shutil.copy(hf_path, unet_checkpoint)
            _log(logger, logging.INFO, f"UNet checkpoint saved to: {unet_checkpoint}")
        except Exception as e:
            _log(logger, logging.ERROR, f"Failed to download UNet checkpoint: {e}")
            raise RuntimeError(
                f"Failed to download LatentSync checkpoint. Error: {e}\n"
                "Please manually download from https://huggingface.co/ByteDance/LatentSync-1.6"
            )
    
    # Download Whisper checkpoint if not present
    if not whisper_checkpoint.exists():
        _log(logger, logging.INFO, "Downloading Whisper tiny checkpoint...")
        try:
            hf_path = hf_hub_download(
                repo_id="ByteDance/LatentSync-1.6",
                filename="whisper/tiny.pt",
                repo_type="model",
            )
            shutil.copy(hf_path, whisper_checkpoint)
            _log(logger, logging.INFO, f"Whisper checkpoint saved to: {whisper_checkpoint}")
        except Exception as e:
            _log(logger, logging.WARNING, f"Failed to download Whisper checkpoint: {e}")
            # Try alternative download
            try:
                import urllib.request
                whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
                urllib.request.urlretrieve(whisper_url, str(whisper_checkpoint))
                _log(logger, logging.INFO, f"Whisper checkpoint saved to: {whisper_checkpoint}")
            except Exception as e2:
                _log(logger, logging.ERROR, f"Failed to download Whisper checkpoint: {e2}")
                raise RuntimeError(f"Failed to download Whisper checkpoint: {e2}")
    
    return unet_checkpoint, whisper_checkpoint


def check_latentsync_requirements(logger=None) -> bool:
    """
    Check if LatentSync requirements are met.
    
    Returns:
        bool: True if requirements are met (has CUDA)
    """
    import torch
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        _log(logger, logging.WARNING, "LatentSync requires CUDA. Running on CPU will be very slow.")
        print("[LatentSync] ⚠️  No CUDA GPU detected - processing will be EXTREMELY slow", flush=True)
        
        # Check available RAM
        try:
            import psutil
            available_ram = psutil.virtual_memory().available / (1024**3)
            total_ram = psutil.virtual_memory().total / (1024**3)
            print(f"[LatentSync] System RAM: {available_ram:.1f}GB available / {total_ram:.1f}GB total", flush=True)
            if available_ram < 16:
                print("[LatentSync] ⚠️  WARNING: Less than 16GB RAM available. Model loading may fail!", flush=True)
                _log(logger, logging.WARNING, f"Low RAM: {available_ram:.1f}GB available. LatentSync needs ~16GB")
        except ImportError:
            pass
        
        return False
    
    # Check VRAM (minimum 8GB recommended)
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[LatentSync] ✓ CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB VRAM)", flush=True)
        if gpu_memory < 8:
            _log(logger, logging.WARNING, 
                 f"LatentSync recommends 8GB+ VRAM. Detected: {gpu_memory:.1f}GB")
            print(f"[LatentSync] ⚠️  Low VRAM ({gpu_memory:.1f}GB). May run out of memory.", flush=True)
    except Exception:
        pass
    
    return True


def run_latentsync(
    face_video: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    logger=None,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    seed: int = 1247,
    use_512_resolution: bool = True,
) -> Path:
    """
    Run LatentSync lip synchronization.
    
    Args:
        face_video: Path to the input video
        audio_path: Path to the audio file (translated TTS output)
        output_path: Path for the output video
        logger: Optional logger instance
        inference_steps: Number of diffusion steps (20-50, higher = better quality)
        guidance_scale: Guidance scale (1.0-3.0, higher = better lip-sync but may distort)
        seed: Random seed for reproducibility
        use_512_resolution: Use 512x512 resolution (better quality, more VRAM)
    
    Returns:
        Path to the lip-synced video
    """
    import torch
    from omegaconf import OmegaConf
    
    _log(logger, logging.INFO, "Starting LatentSync lip-sync processing...")
    
    # Check requirements
    check_latentsync_requirements(logger)
    
    # Get LatentSync directory and ensure checkpoints
    latentsync_dir = _get_latentsync_dir()
    unet_checkpoint, whisper_checkpoint = _ensure_checkpoints(latentsync_dir, logger)
    
    # Convert paths
    face_video = Path(face_video).absolute()
    audio_path = Path(audio_path).absolute()
    output_path = Path(output_path).absolute()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Change to LatentSync directory for proper imports
    original_cwd = os.getcwd()
    os.chdir(str(latentsync_dir))
    
    try:
        # Import LatentSync components (must be done after changing directory)
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature
        from accelerate.utils import set_seed
        
        # Try to import DeepCache for faster inference
        try:
            from DeepCache import DeepCacheSDHelper
            use_deepcache = True
        except ImportError:
            use_deepcache = False
            _log(logger, logging.WARNING, "DeepCache not available, inference will be slower")
        
        # Load config
        config_name = "stage2_512.yaml" if use_512_resolution else "stage2.yaml"
        config_path = latentsync_dir / "configs" / "unet" / config_name
        
        if not config_path.exists():
            # Fall back to available config
            config_path = latentsync_dir / "configs" / "unet" / "stage2.yaml"
            use_512_resolution = False
            
        config = OmegaConf.load(str(config_path))
        
        _log(logger, logging.INFO, f"Using config: {config_path.name}")
        _log(logger, logging.INFO, f"Resolution: {'512x512' if use_512_resolution else '256x256'}")
        
        # Check GPU and set dtype
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        
        _log(logger, logging.INFO, f"Using dtype: {dtype}")
        _log(logger, logging.INFO, f"Input video: {face_video}")
        _log(logger, logging.INFO, f"Input audio: {audio_path}")
        
        # Load scheduler
        scheduler = DDIMScheduler.from_pretrained(str(latentsync_dir / "configs"))
        
        # Determine whisper model based on config
        if config.model.cross_attention_dim == 768:
            whisper_model_path = str(latentsync_dir / "checkpoints" / "whisper" / "small.pt")
        else:
            whisper_model_path = str(whisper_checkpoint)
        
        # Load audio encoder
        _log(logger, logging.INFO, "Loading Whisper audio encoder...")
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )
        
        # Load VAE
        _log(logger, logging.INFO, "Loading VAE...")
        print("[LatentSync] Loading VAE...", flush=True)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        print("[LatentSync] VAE loaded successfully", flush=True)
        
        # Load UNet - This is the slowest step (5GB model)
        _log(logger, logging.INFO, "Loading UNet model (5GB - this may take several minutes on CPU)...")
        print("[LatentSync] Loading UNet model (5GB)...", flush=True)
        print("[LatentSync] ⚠️  WARNING: Running on CPU - this will be VERY slow!", flush=True)
        print("[LatentSync] ⚠️  Expected time: 5-15 minutes to load, 30-60+ minutes to process", flush=True)
        print("[LatentSync] ⚠️  For faster processing, use a CUDA-enabled GPU or switch to Wav2Lip", flush=True)
        
        import time
        load_start = time.time()
        
        try:
            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(config.model),
                str(unet_checkpoint),
                device="cpu",
            )
            load_time = time.time() - load_start
            print(f"[LatentSync] UNet loaded in {load_time:.1f}s", flush=True)
        except Exception as e:
            print(f"[LatentSync] ERROR loading UNet: {e}", flush=True)
            _log(logger, logging.ERROR, f"Failed to load UNet: {e}")
            raise RuntimeError(f"Failed to load UNet model: {e}. This may be due to insufficient RAM.")
        
        print("[LatentSync] Converting UNet to dtype...", flush=True)
        unet = unet.to(dtype=dtype)
        print("[LatentSync] UNet ready", flush=True)
        
        # Create pipeline
        _log(logger, logging.INFO, "Creating LatentSync pipeline...")
        print("[LatentSync] Creating pipeline...", flush=True)
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )
        
        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Enable DeepCache for faster inference
        if use_deepcache:
            try:
                helper = DeepCacheSDHelper(pipe=pipeline)
                helper.set_params(cache_interval=3, cache_branch_id=0)
                helper.enable()
                _log(logger, logging.INFO, "DeepCache enabled for faster inference")
            except Exception as e:
                _log(logger, logging.WARNING, f"Failed to enable DeepCache: {e}")
        
        # Set seed
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()
        
        _log(logger, logging.INFO, f"Starting inference with {inference_steps} steps...")
        
        # Create temp directory for LatentSync
        temp_dir = tempfile.mkdtemp(prefix="latentsync_")
        
        try:
            # Run pipeline
            pipeline(
                video_path=str(face_video),
                audio_path=str(audio_path),
                video_out_path=str(output_path),
                num_frames=config.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=dtype,
                width=config.data.resolution,
                height=config.data.resolution,
                mask_image_path=str(latentsync_dir / config.data.mask_image_path),
                temp_dir=temp_dir,
            )
            
            _log(logger, logging.INFO, f"LatentSync processing complete: {output_path}")
            
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Clean up GPU memory
        del pipeline, unet, vae, audio_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        _log(logger, logging.ERROR, f"LatentSync processing failed: {e}")
        raise
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def ensure_latentsync_assets(logger=None) -> tuple:
    """
    Ensure LatentSync is properly set up.
    
    Returns:
        tuple: (latentsync_dir, unet_checkpoint_path)
    """
    latentsync_dir = _get_latentsync_dir()
    unet_checkpoint, whisper_checkpoint = _ensure_checkpoints(latentsync_dir, logger)
    return latentsync_dir, unet_checkpoint


# Alias for consistency with wav2lip module
def ensure_latentsync_ready(logger=None) -> bool:
    """Check if LatentSync is ready to use."""
    try:
        latentsync_dir = _get_latentsync_dir()
        unet_checkpoint, whisper_checkpoint = _ensure_checkpoints(latentsync_dir, logger)
        return unet_checkpoint.exists() and whisper_checkpoint.exists()
    except Exception as e:
        _log(logger, logging.ERROR, f"LatentSync not ready: {e}")
        return False

