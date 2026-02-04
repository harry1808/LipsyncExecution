"""
Face Detection Module for LipSync Pipeline

Uses OpenCV Haar Cascade for face detection to determine which segments
of a video contain faces and should be processed with Wav2Lip.

Note: MediaPipe was removed due to protobuf version conflicts with parler-tts.
OpenCV Haar Cascade provides reliable face detection without dependency issues.
"""

import logging
import numpy as np
import cv2

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

logger = logging.getLogger("eback_pipeline.face_detection")


def detect_face_intervals(video_path, fps_sample=1):
    """
    Detect face intervals in a video using OpenCV Haar Cascade.
    
    Args:
        video_path: Path to the video file
        fps_sample: How many frames per second to sample (default: 1)
    
    Returns:
        List of tuples: (start_time, duration, has_face)
        where has_face is 1 if face detected, 0 otherwise
    """
    clip = VideoFileClip(str(video_path))
    face_intervals = []
    
    duration = clip.duration
    face_detected = None
    start_time = 0
    
    buffer_size = 7
    buffer = []
    
    # Use OpenCV Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    def process_frame(frame, t):
        nonlocal face_detected, start_time, buffer
        
        # Convert RGB to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        buffer.append(len(faces) > 0)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        
        detected = np.mean(buffer) > 0.25
        
        if detected:
            if face_detected is False or face_detected is None:
                if face_detected is False:
                    face_intervals.append(
                        (start_time, t - start_time, 0)
                    )
                start_time = t
                face_detected = True
        else:
            if face_detected is True or face_detected is None:
                if face_detected is True:
                    face_intervals.append(
                        (start_time, t - start_time, 1)
                    )
                start_time = t
                face_detected = False
    
    for t, frame in clip.iter_frames(fps=fps_sample, with_times=True, dtype="uint8"):
        process_frame(frame, t)
    
    # Add final interval
    if face_detected is not None:
        face_intervals.append(
            (
                start_time,
                duration - start_time,
                1 if face_detected else 0,
            )
        )
    
    clip.close()
    return face_intervals


def merge_intervals(intervals, gap=1.0, min_duration=0.3):
    """
    Merge adjacent face intervals with the same value.
    
    Args:
        intervals: List of (start, duration, has_face) tuples
        gap: Maximum gap between intervals to merge (seconds)
        min_duration: Minimum duration for an interval to be kept (seconds)
    
    Returns:
        Merged list of intervals
    """
    if not intervals:
        return []
    
    intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = []
    
    cur_start, cur_dur, cur_val = intervals[0]
    cur_end = cur_start + cur_dur
    
    for start, duration, val in intervals[1:]:
        end = start + duration
        
        if val == cur_val and start <= cur_end + gap:
            cur_end = max(cur_end, end)
        else:
            if (cur_end - cur_start) >= min_duration:
                merged.append((cur_start, cur_end - cur_start, cur_val))
            
            cur_start, cur_end, cur_val = start, end, val
    
    if (cur_end - cur_start) >= min_duration:
        merged.append((cur_start, cur_end - cur_start, cur_val))
    
    return merged
