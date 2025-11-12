"""
Utility Functions
=================
Common utilities for logging, timing, and data processing.
"""

import logging
import time
from functools import wraps
from typing import Callable
import sys


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def time_it(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @time_it
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        
        return result
    
    return wrapper


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp in seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "01:23")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_fps(num_frames: int, elapsed_time: float) -> float:
    """
    Calculate frames per second.
    
    Args:
        num_frames: Number of frames processed
        elapsed_time: Time elapsed in seconds
        
    Returns:
        FPS as float
    """
    if elapsed_time == 0:
        return 0.0
    return num_frames / elapsed_time


def normalize_pose(landmarks: 'np.ndarray') -> 'np.ndarray':
    """
    Normalize pose landmarks for scale/translation invariance.
    Centers pose at origin and scales to unit distance.
    
    Args:
        landmarks: Pose landmarks array (N, 3)
        
    Returns:
        Normalized landmarks
    """
    import numpy as np
    
    # Center at origin (using hip midpoint)
    center = landmarks.mean(axis=0)
    centered = landmarks - center
    
    # Scale to unit distance
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    return normalized


def smooth_landmarks(
    landmark_history: list,
    window_size: int = 5,
) -> 'np.ndarray':
    """
    Apply temporal smoothing to reduce jitter in pose tracking.
    Uses simple moving average.
    
    Args:
        landmark_history: List of recent landmark arrays
        window_size: Size of smoothing window
        
    Returns:
        Smoothed landmarks
    """
    import numpy as np
    
    if len(landmark_history) < window_size:
        return landmark_history[-1]
    
    recent = landmark_history[-window_size:]
    smoothed = np.mean(recent, axis=0)
    
    return smoothed


class PerformanceTimer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with PerformanceTimer("My operation"):
            # code to time
            ...
    """
    
    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} took {elapsed:.3f}s")


def get_video_info(video_path: str) -> dict:
    """
    Get basic video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }
    
    cap.release()
    
    return info
