"""
Lip-sync evaluation metrics.

Uses the same pipeline as your lip-sync: Wav2Lip/eBack. SyncNet checkpoint is
resolved from wav2lip_assets_dir (same assets used for dubbing).

Provides:
- Duration consistency (no extra deps): video vs audio length match.
- SyncNet-based metrics (optional): LSE-D, LSE-C, AV offset when checkpoint is available.

Standard metrics (from Wav2Lip evaluation):
- LSE-D (Lip-Sync Error - Distance): embedding distance; lower is better.
- LSE-C (Lip-Sync Error - Confidence): sync confidence; higher is better.
- AV offset: temporal offset in frames/ms; 0 is best.
"""

import glob
import logging
import math
import subprocess
import tempfile
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional

from .evaluation_metrics import (
    calculate_lipsync_duration_consistency,
    lipsync_score_from_syncnet_metrics,
)


def _get_video_and_audio_durations(video_path: str) -> Optional[tuple]:
    """Return (video_duration_sec, audio_duration_sec) using moviepy or ffprobe."""
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        if clip.audio is not None:
            audio_duration = clip.audio.duration
        else:
            audio_duration = video_duration
        clip.close()
        return (video_duration, audio_duration)
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            d = float(result.stdout.strip())
            return (d, d)
    except Exception:
        pass
    return None


def _resolve_syncnet_checkpoint(
    wav2lip_assets_dir: Optional[str] = None,
    syncnet_checkpoint_path: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve path to syncnet_v2.model. Prefer Wav2Lip/eBack assets (same as your lip-sync pipeline).
    """
    if syncnet_checkpoint_path and Path(syncnet_checkpoint_path).exists():
        return str(Path(syncnet_checkpoint_path).resolve())
    if wav2lip_assets_dir:
        root = Path(wav2lip_assets_dir)
        candidates = [
            root / "Wav2Lip" / "evaluation" / "scores_LSE" / "data" / "syncnet_v2.model",
            root / "syncnet_v2.model",
            root / "eBack" / "api" / "pipeline" / "wav2lip" / "syncnet_v2.model",
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())
    return None


def _run_syncnet_eval(
    video_path: str,
    checkpoint_path: str,
    temp_dir: str,
    batch_size: int = 20,
    vshift: int = 15,
):
    """
    Run SyncNet evaluation (same logic as Wav2Lip evaluation / syncnet_python).
    Returns (av_offset_frames, min_dist, conf) or raises.
    """
    import numpy
    import torch
    from scipy.io import wavfile
    import cv2
    import python_speech_features

    from .syncnet_model import SyncNetS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SyncNetS(num_layers_in_fc_layers=1024).to(device)
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    model.load_state_dict(loaded, strict=False)
    model.eval()

    if Path(temp_dir).exists():
        rmtree(temp_dir)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", video_path, "-f", "image2",
         str(Path(temp_dir) / "%06d.jpg")],
        check=True, capture_output=True, timeout=120,
    )
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", video_path, "-ac", "1", "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", str(Path(temp_dir) / "audio.wav")],
        check=True, capture_output=True, timeout=60,
    )

    flist = sorted(glob.glob(str(Path(temp_dir) / "*.jpg")))
    images = []
    for fname in flist:
        img = cv2.imread(fname)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
    if not images:
        raise FileNotFoundError("No frames extracted from video")

    im = numpy.stack(images, axis=3)
    im = numpy.expand_dims(im, axis=0)
    im = numpy.transpose(im, (0, 3, 4, 1, 2))
    imtv = torch.from_numpy(im.astype(numpy.float32)).float().to(device)

    sample_rate, audio = wavfile.read(str(Path(temp_dir) / "audio.wav"))
    mfcc = list(zip(*python_speech_features.mfcc(audio, sample_rate)))
    mfcc = numpy.stack([numpy.array(i) for i in mfcc])
    cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
    cct = torch.from_numpy(cc.astype(numpy.float32)).float().to(device)

    min_length = min(len(images), math.floor(len(audio) / 640))
    lastframe = min_length - 5
    if lastframe <= 0:
        raise ValueError("Video or audio too short for SyncNet")

    im_feat_list, cc_feat_list = [], []
    for i in range(0, lastframe, batch_size):
        im_batch = [imtv[:, :, vf : vf + 5, :, :] for vf in range(i, min(lastframe, i + batch_size))]
        im_in = torch.cat(im_batch, 0)
        with torch.no_grad():
            im_out = model.forward_lip(im_in)
        im_feat_list.append(im_out.cpu())
        cc_batch = [cct[:, :, :, vf * 4 : vf * 4 + 20] for vf in range(i, min(lastframe, i + batch_size))]
        cc_in = torch.cat(cc_batch, 0)
        with torch.no_grad():
            cc_out = model.forward_aud(cc_in)
        cc_feat_list.append(cc_out.cpu())

    im_feat = torch.cat(im_feat_list, 0)
    cc_feat = torch.cat(cc_feat_list, 0)

    # Pairwise distances over vshift window (same as Wav2Lip)
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(cc_feat, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(im_feat)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                im_feat[i : i + 1].repeat(win_size, 1),
                feat2p[i : i + win_size, :],
            )
        )
    mean_dists = torch.mean(torch.stack(dists, 1), 1)
    min_dist, minidx = torch.min(mean_dists, 0)
    av_offset = (vshift - minidx).item()
    conf = (torch.median(mean_dists) - min_dist).item()
    min_dist_val = min_dist.item()

    if Path(temp_dir).exists():
        rmtree(temp_dir)
    return av_offset, min_dist_val, conf


def evaluate_lipsync_duration(video_path: str) -> Optional[Dict]:
    """
    Evaluate lip-sync duration consistency (video vs audio length).
    No heavy dependencies; uses moviepy if available else ffprobe.
    """
    try:
        durations = _get_video_and_audio_durations(video_path)
        if durations is None:
            return None
        video_d, audio_d = durations
        return calculate_lipsync_duration_consistency(video_d, audio_d)
    except Exception:
        return None


def evaluate_lipsync_syncnet(
    video_path: str,
    syncnet_checkpoint_path: Optional[str] = None,
    wav2lip_assets_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    fps: float = 25.0,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run SyncNet-based lip-sync evaluation (same model as Wav2Lip evaluation).

    Checkpoint is resolved from wav2lip_assets_dir first (same assets as your
    Wav2Lip/eBack pipeline), then from explicit syncnet_checkpoint_path.

    Args:
        video_path: Path to the lip-synced video (with audio).
        syncnet_checkpoint_path: Optional explicit path to syncnet_v2.model.
        wav2lip_assets_dir: Directory containing Wav2Lip/eBack assets; used to
            find syncnet_v2.model (e.g. instance/wav2lip_assets).
        temp_dir: Temporary directory for frame extraction.
        fps: Frames per second for AV offset conversion to ms.
        logger: Optional logger.

    Returns:
        Dict with status, lse_d, lse_c, av_offset_frames, av_offset_ms, lipsync_score.
    """
    log = logger or logging.getLogger(__name__)
    out = {
        "status": "unavailable",
        "lse_d": None,
        "lse_c": None,
        "av_offset_frames": None,
        "av_offset_ms": None,
        "lipsync_score": None,
    }

    checkpoint = _resolve_syncnet_checkpoint(wav2lip_assets_dir, syncnet_checkpoint_path)
    if not checkpoint:
        log.info(
            "Lip-sync SyncNet eval: no checkpoint found. Set wav2lip_assets_dir (or syncnet_checkpoint_path) "
            "to the directory containing syncnet_v2.model (e.g. Wav2Lip/evaluation/scores_LSE/data/)."
        )
        return out

    video_path = str(Path(video_path).resolve())
    if not Path(video_path).exists():
        out["status"] = "error"
        out["error"] = f"Video not found: {video_path}"
        return out

    temp_dir = temp_dir or tempfile.mkdtemp(prefix="lipsync_eval_")

    try:
        av_offset_frames, min_dist, conf = _run_syncnet_eval(
            video_path=video_path,
            checkpoint_path=checkpoint,
            temp_dir=temp_dir,
        )
        av_offset_ms = (av_offset_frames / fps) * 1000.0
        out["status"] = "success"
        out["lse_d"] = float(min_dist)
        out["lse_c"] = float(conf)
        out["av_offset_frames"] = int(av_offset_frames)
        out["av_offset_ms"] = av_offset_ms
        out["lipsync_score"] = lipsync_score_from_syncnet_metrics(
            lse_d=out["lse_d"],
            lse_c=out["lse_c"],
            av_offset_frames=out["av_offset_frames"],
            fps=fps,
        )
        log.info(
            "Lip-sync metrics (Wav2Lip SyncNet): LSE-D=%.4f, LSE-C=%.4f, AV offset=%d frames (%.0f ms), score=%.1f",
            out["lse_d"], out["lse_c"], out["av_offset_frames"], out["av_offset_ms"], out["lipsync_score"]
        )
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        log.warning("Lip-sync SyncNet evaluation failed: %s", e)

    return out


def evaluate_lipsync(
    video_path: str,
    run_syncnet: bool = True,
    syncnet_checkpoint_path: Optional[str] = None,
    wav2lip_assets_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run all available lip-sync evaluations on the given video.

    Uses Wav2Lip/eBack assets for SyncNet when wav2lip_assets_dir is set (same as your lip-sync pipeline).

    Args:
        video_path: Path to the lip-synced video.
        run_syncnet: Whether to run SyncNet-based metrics (requires syncnet_v2.model in wav2lip_assets).
        syncnet_checkpoint_path: Optional explicit path to syncnet_v2.model.
        wav2lip_assets_dir: Directory with Wav2Lip/eBack assets (used to find syncnet_v2.model).
        logger: Optional logger.

    Returns:
        Dict with duration_consistency, LSE-D/C, AV offset, lipsync_score when available.
    """
    log = logger or logging.getLogger(__name__)
    result = {
        "lipsync_duration_consistency": None,
        "lse_d": None,
        "lse_c": None,
        "av_offset_frames": None,
        "av_offset_ms": None,
        "lipsync_score": None,
        "lipsync_eval_status": None,
    }

    dur = evaluate_lipsync_duration(video_path)
    if dur:
        result["lipsync_duration_consistency"] = dur
        result["lipsync_score"] = dur.get("consistency_score")

    if run_syncnet:
        syncnet_out = evaluate_lipsync_syncnet(
            video_path=video_path,
            syncnet_checkpoint_path=syncnet_checkpoint_path,
            wav2lip_assets_dir=wav2lip_assets_dir,
            logger=log,
        )
        result["lipsync_eval_status"] = syncnet_out.get("status")
        if syncnet_out.get("status") == "success":
            result["lse_d"] = syncnet_out.get("lse_d")
            result["lse_c"] = syncnet_out.get("lse_c")
            result["av_offset_frames"] = syncnet_out.get("av_offset_frames")
            result["av_offset_ms"] = syncnet_out.get("av_offset_ms")
            result["lipsync_score"] = syncnet_out.get("lipsync_score")
        elif syncnet_out.get("status") == "error":
            result["lipsync_eval_error"] = syncnet_out.get("error", "Unknown error")

    return result
