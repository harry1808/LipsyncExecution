"""
End-to-end test script for video dubbing with Wav2Lip lip synchronization.
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add webapp to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    from webapp.dubbing import process_video
    
    # Configuration
    PROJECT_ROOT = Path(__file__).parent
    
    # Use a sample video with faces for lip sync testing
    # sample1 videos have faces, sample/sample2 may not
    sample_videos = list((PROJECT_ROOT / "instance" / "uploads").glob("*sample*.mp4"))
    
    if not sample_videos:
        logger.error("No sample videos found in instance/uploads/")
        return 1
    
    # Pick the smallest video for testing (reduces memory usage)
    input_video = min(sample_videos, key=lambda p: p.stat().st_size)
    logger.info(f"Selected test video: {input_video.name} ({input_video.stat().st_size / 1024:.1f} KB)")
    
    # Output directory
    output_dir = PROJECT_ROOT / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Wav2Lip assets directory
    wav2lip_assets = PROJECT_ROOT / "instance" / "wav2lip_assets"
    
    # Check if assets exist
    if not wav2lip_assets.exists():
        logger.error(f"Wav2Lip assets not found at: {wav2lip_assets}")
        return 1
    
    checkpoint = wav2lip_assets / "wav2lip_gan.pth"
    if not checkpoint.exists():
        logger.error(f"Wav2Lip checkpoint not found at: {checkpoint}")
        return 1
    
    logger.info(f"Using Wav2Lip assets from: {wav2lip_assets}")
    logger.info(f"Checkpoint found: {checkpoint}")
    
    # Run the pipeline
    logger.info("=" * 60)
    logger.info("Starting Video Dubbing Pipeline with Wav2Lip")
    logger.info("=" * 60)
    logger.info(f"Input: {input_video}")
    logger.info(f"Source Language: English (en)")
    logger.info(f"Target Language: Hindi (hi)")
    logger.info(f"Lip-Sync: Enabled (Wav2Lip)")
    logger.info("=" * 60)
    
    try:
        final_path, transcript, translation = process_video(
            video_path=input_video,
            source_lang="en",
            dest_lang="hi",
            output_dir=output_dir,
            logger=logger,
            voice="female",
            enable_lipsync=True,
            lipsync_assets_dir=str(wav2lip_assets),
            lipsync_method="wav2lip",
        )
        
        logger.info("=" * 60)
        logger.info("SUCCESS! Pipeline completed.")
        logger.info("=" * 60)
        logger.info(f"Output video: {final_path}")
        logger.info(f"Transcript: {transcript[:100]}..." if len(transcript) > 100 else f"Transcript: {transcript}")
        logger.info(f"Translation: {translation[:100]}..." if len(translation) > 100 else f"Translation: {translation}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

