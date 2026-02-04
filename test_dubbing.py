"""
Test script to run the video dubbing pipeline with Wav2Lip lip-sync.
Run this script directly to test the full pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    INPUT_VIDEO = project_root / "LatentSync" / "assets" / "demo1_video.mp4"
    OUTPUT_DIR = project_root / "test_output"
    WAV2LIP_ASSETS_DIR = project_root / "instance" / "wav2lip_assets"
    
    # Language settings
    SOURCE_LANG = "en"  # English input
    TARGET_LANG = "hi"  # Hindi output
    VOICE = "female"
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check if input video exists
    if not INPUT_VIDEO.exists():
        logger.error(f"Input video not found: {INPUT_VIDEO}")
        logger.info("Available sample videos:")
        assets_dir = project_root / "LatentSync" / "assets"
        if assets_dir.exists():
            for f in assets_dir.glob("*.mp4"):
                logger.info(f"  - {f}")
        return 1
    
    # Check if wav2lip assets exist
    if not WAV2LIP_ASSETS_DIR.exists():
        logger.error(f"Wav2Lip assets directory not found: {WAV2LIP_ASSETS_DIR}")
        return 1
    
    # Check for checkpoint
    checkpoint_path = WAV2LIP_ASSETS_DIR / "wav2lip_gan.pth"
    if not checkpoint_path.exists():
        logger.error(f"Wav2Lip checkpoint not found: {checkpoint_path}")
        return 1
    
    logger.info("=" * 60)
    logger.info("VIDEO DUBBING TEST")
    logger.info("=" * 60)
    logger.info(f"Input video: {INPUT_VIDEO}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Source language: {SOURCE_LANG}")
    logger.info(f"Target language: {TARGET_LANG}")
    logger.info(f"Voice: {VOICE}")
    logger.info(f"Wav2Lip assets: {WAV2LIP_ASSETS_DIR}")
    logger.info(f"Lip-sync: ENABLED (wav2lip)")
    logger.info("=" * 60)
    
    try:
        from webapp.dubbing import process_video
        
        logger.info("Starting video processing...")
        
        final_path, transcript, translation = process_video(
            video_path=INPUT_VIDEO,
            source_lang=SOURCE_LANG,
            dest_lang=TARGET_LANG,
            output_dir=OUTPUT_DIR,
            logger=logger,
            voice=VOICE,
            enable_lipsync=True,
            lipsync_assets_dir=WAV2LIP_ASSETS_DIR,
            lipsync_method="wav2lip",
        )
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Output video: {final_path}")
        logger.info(f"Original transcript: {transcript}")
        logger.info(f"Translation: {translation}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

