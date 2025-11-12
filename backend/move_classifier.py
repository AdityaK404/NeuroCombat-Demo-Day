"""
Move Classifier Module
======================
Classifies MMA moves from pose sequences using temporal features.
Supports: Cross, Jab, Front Kick, Roundhouse Kick, Uppercut, Neutral.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
from .pose_extractor import PoseLandmarks
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class MoveClassification:
    """Represents a classified move."""
    
    move_name: str
    confidence: float
    player_id: int
    timestamp: float
    duration: float = 0.0


class MoveClassifier:
    """
    Classifies MMA moves from pose sequence features.
    Uses temporal window + feature engineering + lightweight ML model.
    """
    
    MOVE_CLASSES = [
        "neutral",
        "jab",
        "cross",
        "uppercut",
        "front_kick",
        "roundhouse_kick",
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = 15,  # Number of frames to analyze
        confidence_threshold: float = 0.6,
        use_mock: bool = True,  # Use mock predictions for rapid prototyping
    ):
        """
        Initialize move classifier.
        
        Args:
            model_path: Path to trained classifier model (.pkl)
            window_size: Number of frames in temporal window
            confidence_threshold: Minimum confidence for valid prediction
            use_mock: If True, use mock predictions (for hackathon demo)
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.use_mock = use_mock
        
        # Temporal buffers for each player
        self.pose_buffers: Dict[int, deque] = {
            1: deque(maxlen=window_size),
            2: deque(maxlen=window_size),
        }
        
        self.model = None
        if model_path and Path(model_path).exists() and not use_mock:
            self.model = self._load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("Running in MOCK mode - using rule-based predictions")
        
        logger.info(f"MoveClassifier initialized with window_size={window_size}")
    
    def classify_move(
        self,
        pose: PoseLandmarks,
    ) -> Optional[MoveClassification]:
        """
        Classify move for a single pose (with temporal context).
        
        Args:
            pose: Current pose landmarks
            
        Returns:
            MoveClassification if confident prediction, None otherwise
        """
        player_id = pose.player_id
        
        if player_id not in self.pose_buffers:
            return None
        
        # Add pose to buffer
        self.pose_buffers[player_id].append(pose)
        
        # Need full window for classification
        if len(self.pose_buffers[player_id]) < self.window_size:
            return None
        
        # Extract features from pose sequence
        features = self._extract_features(list(self.pose_buffers[player_id]))
        
        # Classify
        if self.use_mock or self.model is None:
            move_name, confidence = self._mock_classify(features, pose)
        else:
            move_name, confidence = self._model_classify(features)
        
        # Return if confident
        if confidence >= self.confidence_threshold:
            return MoveClassification(
                move_name=move_name,
                confidence=confidence,
                player_id=player_id,
                timestamp=pose.timestamp,
            )
        
        return None
    
    def classify_sequence(
        self,
        pose_sequence: List[PoseLandmarks],
    ) -> List[MoveClassification]:
        """
        Classify moves from a complete pose sequence.
        
        Args:
            pose_sequence: List of poses for one player
            
        Returns:
            List of classified moves
        """
        classifications = []
        
        for pose in pose_sequence:
            classification = self.classify_move(pose)
            if classification:
                classifications.append(classification)
        
        # Post-process to remove duplicates and merge consecutive same moves
        classifications = self._post_process_classifications(classifications)
        
        return classifications
    
    def _extract_features(self, pose_window: List[PoseLandmarks]) -> np.ndarray:
        """
        Extract temporal features from pose sequence window.
        
        Features include:
        - Joint velocities (frame-to-frame changes)
        - Joint accelerations
        - Key angles (elbow, knee, hip)
        - Spatial relationships
        - Movement magnitude
        
        Args:
            pose_window: Sequence of poses (length = window_size)
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract velocity features (change in position)
        for i in range(1, len(pose_window)):
            prev_pose = pose_window[i - 1].landmarks
            curr_pose = pose_window[i].landmarks
            
            velocity = curr_pose - prev_pose
            features.extend([
                np.mean(np.abs(velocity)),  # Average movement
                np.max(np.abs(velocity)),   # Max movement
                np.std(velocity),           # Movement variability
            ])
        
        # Extract key angles from latest pose
        latest_pose = pose_window[-1].landmarks
        
        # Right arm angle (shoulder-elbow-wrist)
        right_shoulder = latest_pose[12]  # MediaPipe landmark indices
        right_elbow = latest_pose[14]
        right_wrist = latest_pose[16]
        right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Left arm angle
        left_shoulder = latest_pose[11]
        left_elbow = latest_pose[13]
        left_wrist = latest_pose[15]
        left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Right leg angle (hip-knee-ankle)
        right_hip = latest_pose[24]
        right_knee = latest_pose[26]
        right_ankle = latest_pose[28]
        right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Left leg angle
        left_hip = latest_pose[23]
        left_knee = latest_pose[25]
        left_ankle = latest_pose[27]
        left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        
        features.extend([
            right_arm_angle,
            left_arm_angle,
            right_leg_angle,
            left_leg_angle,
        ])
        
        # Wrist height relative to shoulder (for uppercut detection)
        wrist_height_right = right_wrist[1] - right_shoulder[1]
        wrist_height_left = left_wrist[1] - left_shoulder[1]
        features.extend([wrist_height_right, wrist_height_left])
        
        # Foot height (for kick detection)
        foot_height_right = right_ankle[1] - right_hip[1]
        foot_height_left = left_ankle[1] - left_hip[1]
        features.extend([foot_height_right, foot_height_left])
        
        return np.array(features)
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at point p2 formed by p1-p2-p3.
        
        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _mock_classify(
        self,
        features: np.ndarray,
        pose: PoseLandmarks,
    ) -> Tuple[str, float]:
        """
        Mock classifier using simple heuristics (for rapid prototyping).
        
        Returns:
            Tuple of (move_name, confidence)
        """
        landmarks = pose.landmarks
        
        # Extract key points
        right_wrist = landmarks[16]
        left_wrist = landmarks[15]
        right_shoulder = landmarks[12]
        left_shoulder = landmarks[11]
        right_ankle = landmarks[28]
        left_ankle = landmarks[27]
        right_hip = landmarks[24]
        left_hip = landmarks[23]
        
        # Simple rule-based classification
        
        # Check for kicks (foot raised high)
        right_foot_height = right_hip[1] - right_ankle[1]
        left_foot_height = left_hip[1] - left_ankle[1]
        
        if right_foot_height > 0.3 or left_foot_height > 0.3:
            # Determine kick type by horizontal movement
            if abs(right_ankle[0] - right_hip[0]) < 0.2:
                return "front_kick", 0.85
            else:
                return "roundhouse_kick", 0.80
        
        # Check for uppercut (wrist moving upward rapidly)
        right_wrist_height = right_shoulder[1] - right_wrist[1]
        left_wrist_height = left_shoulder[1] - left_wrist[1]
        
        if right_wrist_height > 0.2 or left_wrist_height > 0.2:
            return "uppercut", 0.75
        
        # Check for jab/cross (hand extended forward)
        # Use z-coordinate or x-extension as proxy
        right_extension = abs(right_wrist[0] - right_shoulder[0])
        left_extension = abs(left_wrist[0] - left_shoulder[0])
        
        if right_extension > 0.3:
            return "cross", 0.70
        elif left_extension > 0.3:
            return "jab", 0.70
        
        # Default to neutral
        return "neutral", 0.95
    
    def _model_classify(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify using trained ML model.
        
        Returns:
            Tuple of (move_name, confidence)
        """
        # Reshape for model input
        features_reshaped = features.reshape(1, -1)
        
        # Get predictions
        probabilities = self.model.predict_proba(features_reshaped)[0]
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        move_name = self.MOVE_CLASSES[predicted_idx]
        
        return move_name, confidence
    
    def _post_process_classifications(
        self,
        classifications: List[MoveClassification],
    ) -> List[MoveClassification]:
        """
        Remove duplicate consecutive classifications and merge similar events.
        """
        if not classifications:
            return []
        
        filtered = [classifications[0]]
        
        for current in classifications[1:]:
            previous = filtered[-1]
            
            # Merge if same move within short time window
            if (current.move_name == previous.move_name and
                current.timestamp - previous.timestamp < 1.0):
                # Update duration
                previous.duration = current.timestamp - previous.timestamp
            elif current.move_name != "neutral":  # Don't spam neutral states
                filtered.append(current)
        
        return filtered
    
    def _load_model(self, model_path: str):
        """Load trained classifier from pickle file."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def reset(self):
        """Clear all pose buffers."""
        for buffer in self.pose_buffers.values():
            buffer.clear()
        logger.info("MoveClassifier reset")


# TODO: Train actual classifier on Harmony4D + local MMA dataset
# TODO: Implement LSTM or Transformer for better temporal modeling
# TODO: Add move transition matrix for contextual priors
# TODO: Implement confidence calibration
