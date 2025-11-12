"""
Pose Extractor Module
=====================
Extracts pose landmarks for dual fighters using MediaPipe.
Handles multi-person detection and tracks both Player 1 and Player 2.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class PoseLandmarks:
    """Represents pose landmarks for a single person."""
    
    landmarks: np.ndarray  # Shape: (33, 3) for MediaPipe pose
    visibility: np.ndarray  # Shape: (33,)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    player_id: int  # 1 or 2
    timestamp: float  # Frame timestamp


class PoseExtractor:
    """
    Extracts pose landmarks from video frames for dual fighters.
    Uses MediaPipe Pose with custom logic to separate two fighters.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Initialize pose extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: 0=lite, 1=full, 2=heavy (faster to slower)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        logger.info(f"PoseExtractor initialized with complexity={model_complexity}")
    
    def extract_poses_from_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> List[PoseLandmarks]:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: BGR image from OpenCV
            timestamp: Current frame timestamp in seconds
            
        Returns:
            List of PoseLandmarks objects (up to 2 for dual fighters)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose_detector.process(rgb_frame)
        
        poses = []
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks_array = self._landmarks_to_array(results.pose_landmarks)
            visibility_array = self._extract_visibility(results.pose_landmarks)
            bbox = self._calculate_bbox(results.pose_landmarks, frame.shape)
            
            pose = PoseLandmarks(
                landmarks=landmarks_array,
                visibility=visibility_array,
                bbox=bbox,
                player_id=0,  # Will be assigned by tracker
                timestamp=timestamp,
            )
            poses.append(pose)
        
        return poses
    
    def extract_poses_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Tuple[List[List[PoseLandmarks]], Dict]:
        """
        Extract poses from entire video file.
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None = all)
            
        Returns:
            Tuple of (pose_sequences, metadata)
            - pose_sequences: List of frame-wise pose lists
            - metadata: Video metadata (fps, resolution, etc.)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "video_path": video_path,
        }
        
        logger.info(f"Processing video: {total_frames} frames @ {fps} FPS")
        
        all_poses = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            timestamp = frame_idx / fps
            poses = self.extract_poses_from_frame(frame, timestamp)
            all_poses.append(poses)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        logger.info(f"Extraction complete: {frame_idx} frames processed")
        
        return all_poses, metadata
    
    def draw_poses_on_frame(
        self,
        frame: np.ndarray,
        poses: List[PoseLandmarks],
    ) -> np.ndarray:
        """
        Draw pose landmarks on frame for visualization.
        
        Args:
            frame: Input BGR frame
            poses: List of PoseLandmarks to draw
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for pose in poses:
            # Draw bounding box
            x, y, w, h = pose.bbox
            color = (0, 255, 0) if pose.player_id == 1 else (255, 0, 0)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw player label
            label = f"Player {pose.player_id}" if pose.player_id > 0 else "Untracked"
            cv2.putText(
                annotated_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            
            # Draw pose skeleton
            # TODO: Implement proper skeleton drawing using MediaPipe utilities
        
        return annotated_frame
    
    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array."""
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        )
    
    def _extract_visibility(self, landmarks) -> np.ndarray:
        """Extract visibility scores from landmarks."""
        return np.array([lm.visibility for lm in landmarks.landmark])
    
    def _calculate_bbox(
        self,
        landmarks,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from pose landmarks.
        
        Returns:
            Tuple of (x, y, width, height)
        """
        h, w = frame_shape[:2]
        
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def close(self):
        """Release resources."""
        self.pose_detector.close()
        logger.info("PoseExtractor closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# TODO: Implement dual-person detection using YOLO + MediaPipe
# TODO: Add pose normalization for scale/translation invariance
# TODO: Implement temporal smoothing for jittery detections
