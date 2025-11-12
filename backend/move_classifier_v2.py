"""
Move Classification Engine for NeuroCombat - V2 Enhanced
========================================================

Classifies MMA moves from pose sequences using engineered motion features.
Supports: Cross, Jab, Front Kick, Roundhouse Kick, Uppercut, Neutral stance.

Features:
- 23 engineered motion features (angles, velocities, extensions)
- Temporal smoothing for stable predictions
- Real-time inference (<100ms per frame)
- Mock classifier for demo without trained model
- Full integration with pose_extractor_v2.py output

Author: NeuroCombat Team
Date: November 12, 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Move labels
MOVE_CLASSES = [
    "neutral",
    "jab",
    "cross",
    "uppercut",
    "front_kick",
    "roundhouse_kick"
]

# MediaPipe pose landmark indices
LANDMARK_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class MoveFeatures:
    """Engineered motion features for move classification"""
    # Arm angles (degrees)
    left_elbow_angle: float
    right_elbow_angle: float
    left_shoulder_angle: float
    right_shoulder_angle: float
    
    # Leg angles
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    
    # Extension metrics (normalized distances)
    left_arm_extension: float
    right_arm_extension: float
    left_leg_extension: float
    right_leg_extension: float
    
    # Velocity features (change from previous frame)
    left_wrist_velocity: float
    right_wrist_velocity: float
    left_ankle_velocity: float
    right_ankle_velocity: float
    
    # Height metrics
    wrist_height_left: float
    wrist_height_right: float
    knee_height_left: float
    knee_height_right: float
    
    # Cross-body features
    arm_cross_distance: float
    stance_width: float
    center_of_mass_y: float


class MoveClassifier:
    """
    Classifies MMA moves from pose sequences using engineered features
    """
    
    def __init__(
        self,
        model_path: str = "models/move_classifier.pkl",
        confidence_threshold: float = 0.5,
        use_temporal_smoothing: bool = True,
        smoothing_alpha: float = 0.6
    ):
        """
        Initialize move classifier
        
        Args:
            model_path: Path to trained model (.pkl file)
            confidence_threshold: Minimum confidence for predictions
            use_temporal_smoothing: Enable exponential smoothing
            smoothing_alpha: Smoothing factor (0=no smooth, 1=full history)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_temporal_smoothing = use_temporal_smoothing
        self.smoothing_alpha = smoothing_alpha
        
        # Load or create model
        self.model = self._load_or_create_model()
        self.scaler = StandardScaler()
        
        # Temporal smoothing state
        self.prediction_history = {
            "player_1": deque(maxlen=5),
            "player_2": deque(maxlen=5)
        }
        
        # Pose history for kick heuristic detection (sliding window)
        self.pose_history = {
            "player_1": deque(maxlen=5),
            "player_2": deque(maxlen=5)
        }
        
        # Previous frame poses for velocity calculation
        self.prev_poses = {"player_1": None, "player_2": None}
        
        # Kick fallback tracking
        self.enable_kick_fallback = True
        self.fallback_events = []  # Log fallback detections
        
    def _load_or_create_model(self):
        """Load trained model or create mock classifier"""
        if self.model_path.exists():
            print(f"📦 Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"⚠️  Model not found at {self.model_path}")
            print(f"💡 Using mock classifier for demonstration")
            return self._create_mock_classifier()
    
    def _create_mock_classifier(self):
        """Create a mock classifier for demo purposes"""
        # Simple rule-based mock for demonstration
        class MockClassifier:
            def predict_proba(self, X):
                """Mock predictions based on simple heuristics"""
                predictions = []
                for features in X:
                    # Simple heuristic rules
                    if features[8] > 0.7 or features[9] > 0.7:  # Arm extension
                        if features[16] > 0.5:  # High wrist
                            pred = [0.1, 0.6, 0.2, 0.05, 0.025, 0.025]  # Jab
                        else:
                            pred = [0.1, 0.2, 0.6, 0.05, 0.025, 0.025]  # Cross
                    elif features[10] > 0.6 or features[11] > 0.6:  # Leg extension
                        if features[18] > 0.5:  # High knee
                            pred = [0.1, 0.05, 0.05, 0.025, 0.7, 0.075]  # Front kick
                        else:
                            pred = [0.1, 0.05, 0.05, 0.025, 0.075, 0.7]  # Roundhouse
                    elif features[0] < 60 or features[1] < 60:  # Bent elbows
                        pred = [0.1, 0.1, 0.1, 0.65, 0.025, 0.025]  # Uppercut
                    else:
                        pred = [0.8, 0.05, 0.05, 0.025, 0.025, 0.05]  # Neutral
                    
                    predictions.append(pred)
                
                return np.array(predictions)
        
        return MockClassifier()
    
    def classify_from_json(
        self,
        pose_json_path: str,
        output_path: str = None
    ) -> Dict:
        """
        Classify moves from pose JSON file
        
        Args:
            pose_json_path: Path to pose data JSON from pose_extractor_v2.py
            output_path: Path to save classification results (auto if None)
            
        Returns:
            Dictionary containing frame-wise move predictions
        """
        pose_json_path = Path(pose_json_path)
        if not pose_json_path.exists():
            raise FileNotFoundError(f"Pose JSON not found: {pose_json_path}")
        
        print(f"\n🧠 Loading pose data from: {pose_json_path}")
        with open(pose_json_path, 'r') as f:
            pose_data = json.load(f)
        
        frames = pose_data.get("frames", {})
        if not frames:
            raise ValueError("No frames found in pose data")
        
        print(f"📊 Processing {len(frames)} frames...")
        
        # Reset state
        self.prediction_history = {
            "player_1": deque(maxlen=5),
            "player_2": deque(maxlen=5)
        }
        self.prev_poses = {"player_1": None, "player_2": None}
        
        # Classify each frame
        classified_moves = {
            "metadata": {
                "source": str(pose_json_path),
                "total_frames": len(frames),
                "move_classes": MOVE_CLASSES
            },
            "frames": {},
            "statistics": {
                "player_1": {move: 0 for move in MOVE_CLASSES},
                "player_2": {move: 0 for move in MOVE_CLASSES}
            }
        }
        
        for frame_key in sorted(frames.keys()):
            frame_data = frames[frame_key]
            frame_predictions = {}
            
            for player_id in ["player_1", "player_2"]:
                if player_id in frame_data:
                    pose = frame_data[player_id]
                    
                    # Extract features
                    features = self._pose_to_features(
                        pose["keypoints"],
                        self.prev_poses[player_id]
                    )
                    
                    # Update pose history for kick heuristic
                    self.pose_history[player_id].append(pose["keypoints"])
                    
                    # Classify with ML model
                    move, confidence = self._classify_single_pose(features, player_id)
                    
                    # Kick heuristic fallback for low-confidence predictions
                    if self.enable_kick_fallback and confidence < 0.6 and len(self.pose_history[player_id]) >= 3:
                        heuristic_move, heuristic_score = self._detect_kick_heuristic(
                            list(self.pose_history[player_id])
                        )
                        
                        if heuristic_move and heuristic_score > 0.7:
                            # Log fallback event
                            fallback_info = {
                                "frame": frame_key,
                                "player": player_id,
                                "original_move": move,
                                "original_conf": confidence,
                                "heuristic_move": heuristic_move,
                                "heuristic_score": heuristic_score
                            }
                            self.fallback_events.append(fallback_info)
                            
                            print(f"[FALLBACK] {frame_key}: {player_id} → {heuristic_move} "
                                  f"(score={heuristic_score:.2f}, was {move}@{confidence:.2f})")
                            
                            # Use heuristic prediction
                            move = heuristic_move
                            confidence = heuristic_score
                    
                    frame_predictions[player_id] = {
                        "move": move,
                        "confidence": round(float(confidence), 3)
                    }
                    
                    # Update statistics
                    classified_moves["statistics"][player_id][move] += 1
                    
                    # Update previous pose
                    self.prev_poses[player_id] = pose["keypoints"]
            
            classified_moves["frames"][frame_key] = frame_predictions
        
        # Calculate percentages
        for player_id in ["player_1", "player_2"]:
            total = sum(classified_moves["statistics"][player_id].values())
            if total > 0:
                for move in MOVE_CLASSES:
                    count = classified_moves["statistics"][player_id][move]
                    classified_moves["statistics"][player_id][f"{move}_pct"] = \
                        round(count / total * 100, 2)
        
        # Save results
        if output_path is None:
            output_path = f"data/processed/moves_{pose_json_path.stem.replace('poses_', '')}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(classified_moves, f, indent=2)
        
        print(f"\n💾 Classification saved → {output_path}")
        self._print_summary(classified_moves["statistics"])
        
        # Print fallback summary if kicks were detected
        if self.fallback_events:
            print(f"\n🦵 Kick Heuristic Fallbacks: {len(self.fallback_events)} detections")
            kick_counts = {"front_kick": 0, "roundhouse_kick": 0}
            for event in self.fallback_events:
                kick_counts[event["heuristic_move"]] += 1
            print(f"   • Front kicks: {kick_counts['front_kick']}")
            print(f"   • Roundhouse kicks: {kick_counts['roundhouse_kick']}")
        
        return classified_moves
    
    def _pose_to_features(
        self,
        keypoints: List,
        prev_keypoints: Optional[List] = None
    ) -> np.ndarray:
        """
        Extract engineered motion features from pose keypoints
        
        Args:
            keypoints: List of [x, y, visibility] for 33 landmarks
            prev_keypoints: Previous frame keypoints for velocity (optional)
            
        Returns:
            Feature vector as numpy array (30 features):
            - 8 angle features (elbow, shoulder, knee, hip, leg angles)
            - 4 extension features (arm, leg extension ratios)
            - 4 velocity features (wrist, ankle movements)
            - 4 height features (wrist, knee elevations)
            - 3 cross-body features (arm distance, stance, COM)
            - 7 NEW leg-based features for kick detection:
              * left_leg_angle, right_leg_angle (hip-knee-ankle)
              * left_foot_raise, right_foot_raise (foot elevation)
              * hip_rotation (lateral movement)
              * left_ankle_vel_y, right_ankle_vel_y (vertical velocity)
        """
        # Convert to numpy array
        kpts = np.array(keypoints)
        
        # Extract key landmarks
        def get_point(name):
            idx = LANDMARK_INDICES[name]
            return kpts[idx][:2]  # x, y only
        
        # Calculate angles
        left_elbow_angle = self._calculate_angle(
            get_point("left_shoulder"),
            get_point("left_elbow"),
            get_point("left_wrist")
        )
        
        right_elbow_angle = self._calculate_angle(
            get_point("right_shoulder"),
            get_point("right_elbow"),
            get_point("right_wrist")
        )
        
        left_knee_angle = self._calculate_angle(
            get_point("left_hip"),
            get_point("left_knee"),
            get_point("left_ankle")
        )
        
        right_knee_angle = self._calculate_angle(
            get_point("right_hip"),
            get_point("right_knee"),
            get_point("right_ankle")
        )
        
        left_shoulder_angle = self._calculate_angle(
            get_point("left_elbow"),
            get_point("left_shoulder"),
            get_point("left_hip")
        )
        
        right_shoulder_angle = self._calculate_angle(
            get_point("right_elbow"),
            get_point("right_shoulder"),
            get_point("right_hip")
        )
        
        left_hip_angle = self._calculate_angle(
            get_point("left_knee"),
            get_point("left_hip"),
            get_point("right_hip")
        )
        
        right_hip_angle = self._calculate_angle(
            get_point("right_knee"),
            get_point("right_hip"),
            get_point("left_hip")
        )
        
        # NEW: Leg angles for kick detection (hip-knee-ankle)
        left_leg_angle = self._calculate_angle(
            get_point("left_hip"),
            get_point("left_knee"),
            get_point("left_ankle")
        )
        
        right_leg_angle = self._calculate_angle(
            get_point("right_hip"),
            get_point("right_knee"),
            get_point("right_ankle")
        )
        
        # Calculate extensions (normalized)
        left_arm_extension = np.linalg.norm(
            get_point("left_wrist") - get_point("left_shoulder")
        ) / 100.0  # Normalize
        
        right_arm_extension = np.linalg.norm(
            get_point("right_wrist") - get_point("right_shoulder")
        ) / 100.0
        
        # NEW: Enhanced leg extension ratios for kick detection
        # Calculate (ankle-hip) / (knee-hip) to detect leg straightening
        left_ankle_hip_dist = np.linalg.norm(get_point("left_ankle") - get_point("left_hip"))
        left_knee_hip_dist = np.linalg.norm(get_point("left_knee") - get_point("left_hip"))
        left_leg_extension = (left_ankle_hip_dist / left_knee_hip_dist) if left_knee_hip_dist > 0 else 1.0
        
        right_ankle_hip_dist = np.linalg.norm(get_point("right_ankle") - get_point("right_hip"))
        right_knee_hip_dist = np.linalg.norm(get_point("right_knee") - get_point("right_hip"))
        right_leg_extension = (right_ankle_hip_dist / right_knee_hip_dist) if right_knee_hip_dist > 0 else 1.0
        
        # Calculate velocities (if previous frame available)
        if prev_keypoints is not None:
            prev_kpts = np.array(prev_keypoints)
            
            def get_prev_point(name):
                idx = LANDMARK_INDICES[name]
                return prev_kpts[idx][:2]
            
            left_wrist_velocity = np.linalg.norm(
                get_point("left_wrist") - get_prev_point("left_wrist")
            )
            right_wrist_velocity = np.linalg.norm(
                get_point("right_wrist") - get_prev_point("right_wrist")
            )
            
            # NEW: Ankle velocities for kick detection (vertical component)
            left_ankle_vel_y = get_prev_point("left_ankle")[1] - get_point("left_ankle")[1]  # Negative = upward
            right_ankle_vel_y = get_prev_point("right_ankle")[1] - get_point("right_ankle")[1]
            
            left_ankle_velocity = np.linalg.norm(
                get_point("left_ankle") - get_prev_point("left_ankle")
            )
            right_ankle_velocity = np.linalg.norm(
                get_point("right_ankle") - get_prev_point("right_ankle")
            )
            
            # NEW: Hip rotation (lateral hip movement for roundhouse kicks)
            left_hip_x_delta = abs(get_point("left_hip")[0] - get_prev_point("left_hip")[0])
            right_hip_x_delta = abs(get_point("right_hip")[0] - get_prev_point("right_hip")[0])
            hip_rotation = (left_hip_x_delta + right_hip_x_delta) / 2.0
        else:
            left_wrist_velocity = 0.0
            right_wrist_velocity = 0.0
            left_ankle_velocity = 0.0
            right_ankle_velocity = 0.0
            left_ankle_vel_y = 0.0
            right_ankle_vel_y = 0.0
            hip_rotation = 0.0
        
        # Calculate height metrics (normalized by hip height)
        hip_y = (get_point("left_hip")[1] + get_point("right_hip")[1]) / 2
        
        wrist_height_left = (hip_y - get_point("left_wrist")[1]) / 100.0
        wrist_height_right = (hip_y - get_point("right_wrist")[1]) / 100.0
        knee_height_left = (hip_y - get_point("left_knee")[1]) / 100.0
        knee_height_right = (hip_y - get_point("right_knee")[1]) / 100.0
        
        # NEW: Foot raise metrics (hip_y - ankle_y) for kick detection
        left_foot_raise = (hip_y - get_point("left_ankle")[1]) / 100.0
        right_foot_raise = (hip_y - get_point("right_ankle")[1]) / 100.0
        
        # Cross-body features
        arm_cross_distance = abs(
            get_point("left_wrist")[0] - get_point("right_wrist")[0]
        ) / 100.0
        
        stance_width = np.linalg.norm(
            get_point("left_ankle") - get_point("right_ankle")
        ) / 100.0
        
        center_of_mass_y = hip_y / 100.0
        
        # Assemble feature vector (28 features total: 23 original + 5 new leg features)
        features = np.array([
            # Original 8 angle features
            left_elbow_angle,
            right_elbow_angle,
            left_shoulder_angle,
            right_shoulder_angle,
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            # Original 4 extension features
            left_arm_extension,
            right_arm_extension,
            left_leg_extension,  # NOW: extension ratio for kicks
            right_leg_extension,  # NOW: extension ratio for kicks
            # Original 4 velocity features
            left_wrist_velocity,
            right_wrist_velocity,
            left_ankle_velocity,
            right_ankle_velocity,
            # Original 4 height features
            wrist_height_left,
            wrist_height_right,
            knee_height_left,
            knee_height_right,
            # Original 3 cross-body features
            arm_cross_distance,
            stance_width,
            center_of_mass_y,
            # NEW: 5 leg-based features for kick detection
            left_leg_angle,       # Feature 23: hip-knee-ankle angle (left)
            right_leg_angle,      # Feature 24: hip-knee-ankle angle (right)
            left_foot_raise,      # Feature 25: foot elevation (left)
            right_foot_raise,     # Feature 26: foot elevation (right)
            hip_rotation,         # Feature 27: hip lateral movement
            left_ankle_vel_y,     # Feature 28: ankle vertical velocity (left)
            right_ankle_vel_y     # Feature 29: ankle vertical velocity (right)
        ])
        
        return features
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3
        
        Returns: Angle in degrees (0-180)
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _classify_single_pose(
        self,
        features: np.ndarray,
        player_id: str
    ) -> Tuple[str, float]:
        """
        Classify a single pose into a move category
        
        Args:
            features: Engineered feature vector
            player_id: Player identifier for temporal smoothing
            
        Returns:
            (move_name, confidence) tuple
        """
        # Reshape for prediction
        X = features.reshape(1, -1)
        
        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        
        # Apply temporal smoothing
        if self.use_temporal_smoothing:
            self.prediction_history[player_id].append(probs)
            if len(self.prediction_history[player_id]) > 1:
                # Exponential moving average
                smoothed_probs = np.zeros_like(probs)
                weights = np.array([self.smoothing_alpha ** i 
                                   for i in range(len(self.prediction_history[player_id]))])
                weights = weights / weights.sum()
                
                for i, hist_probs in enumerate(self.prediction_history[player_id]):
                    smoothed_probs += weights[-(i+1)] * hist_probs
                
                probs = smoothed_probs
        
        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        move_name = MOVE_CLASSES[pred_idx]
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            move_name = "neutral"
            confidence = 1.0 - confidence
        
        return move_name, confidence
    
    def _detect_kick_heuristic(
        self,
        pose_window: List[List],
        fps: float = 25.0
    ) -> Tuple[Optional[str], float]:
        """
        Rule-based kick detector using leg biomechanics
        
        Analyzes leg extension, ankle velocity, and hip rotation over a sliding
        window to detect front kicks and roundhouse kicks when ML classifier
        confidence is low.
        
        Args:
            pose_window: List of keypoint arrays (last N frames, typically 5)
            fps: Video frame rate for velocity normalization
            
        Returns:
            (move_label, confidence_score) or (None, 0.0) if no kick detected
            
        Heuristics:
            - Front Kick: High ankle upward velocity + leg extension > 1.3x
            - Roundhouse: Front kick criteria + hip rotation > threshold
        """
        if len(pose_window) < 3:
            return None, 0.0
        
        # Thresholds (tuned for normalized coordinates)
        VEL_THRESH = 0.01      # Minimum upward ankle velocity
        EXT_THRESH = 1.3       # Minimum leg extension ratio
        ROT_THRESH = 0.05      # Minimum hip rotation for roundhouse
        
        # Extract features from recent frames
        left_extensions = []
        right_extensions = []
        left_ankle_vels = []
        right_ankle_vels = []
        hip_rotations = []
        foot_raises = []
        
        for i, kpts in enumerate(pose_window):
            kpts_array = np.array(kpts)
            
            # Get key points
            left_hip = kpts_array[LANDMARK_INDICES["left_hip"]][:2]
            right_hip = kpts_array[LANDMARK_INDICES["right_hip"]][:2]
            left_knee = kpts_array[LANDMARK_INDICES["left_knee"]][:2]
            right_knee = kpts_array[LANDMARK_INDICES["right_knee"]][:2]
            left_ankle = kpts_array[LANDMARK_INDICES["left_ankle"]][:2]
            right_ankle = kpts_array[LANDMARK_INDICES["right_ankle"]][:2]
            
            # Calculate leg extension ratios
            left_ankle_hip_dist = np.linalg.norm(left_ankle - left_hip)
            left_knee_hip_dist = np.linalg.norm(left_knee - left_hip)
            left_ext = (left_ankle_hip_dist / left_knee_hip_dist) if left_knee_hip_dist > 0 else 1.0
            left_extensions.append(left_ext)
            
            right_ankle_hip_dist = np.linalg.norm(right_ankle - right_hip)
            right_knee_hip_dist = np.linalg.norm(right_knee - right_hip)
            right_ext = (right_ankle_hip_dist / right_knee_hip_dist) if right_knee_hip_dist > 0 else 1.0
            right_extensions.append(right_ext)
            
            # Calculate foot raise (hip_y - ankle_y, positive = foot above hip)
            hip_y = (left_hip[1] + right_hip[1]) / 2
            left_foot_raise = (hip_y - left_ankle[1]) / 100.0
            right_foot_raise = (hip_y - right_ankle[1]) / 100.0
            foot_raises.append((left_foot_raise, right_foot_raise))
            
            # Calculate ankle velocities (if not first frame)
            if i > 0:
                prev_kpts = np.array(pose_window[i-1])
                prev_left_ankle = prev_kpts[LANDMARK_INDICES["left_ankle"]][:2]
                prev_right_ankle = prev_kpts[LANDMARK_INDICES["right_ankle"]][:2]
                
                # Vertical velocity (negative = upward in image coordinates)
                left_vel_y = prev_left_ankle[1] - left_ankle[1]
                right_vel_y = prev_right_ankle[1] - right_ankle[1]
                left_ankle_vels.append(left_vel_y)
                right_ankle_vels.append(right_vel_y)
                
                # Hip rotation
                prev_left_hip = prev_kpts[LANDMARK_INDICES["left_hip"]][:2]
                prev_right_hip = prev_kpts[LANDMARK_INDICES["right_hip"]][:2]
                left_hip_x_delta = abs(left_hip[0] - prev_left_hip[0])
                right_hip_x_delta = abs(right_hip[0] - prev_right_hip[0])
                hip_rot = (left_hip_x_delta + right_hip_x_delta) / 2.0
                hip_rotations.append(hip_rot)
        
        # Compute average metrics over window
        avg_left_ext = np.mean(left_extensions)
        avg_right_ext = np.mean(right_extensions)
        avg_left_vel = np.mean(left_ankle_vels) if left_ankle_vels else 0.0
        avg_right_vel = np.mean(right_ankle_vels) if right_ankle_vels else 0.0
        avg_hip_rot = np.mean(hip_rotations) if hip_rotations else 0.0
        max_left_raise = max([fr[0] for fr in foot_raises]) if foot_raises else 0.0
        max_right_raise = max([fr[1] for fr in foot_raises]) if foot_raises else 0.0
        
        # Detect kicks (check both legs)
        detected_move = None
        confidence = 0.0
        
        # Left leg kick detection
        if avg_left_vel > VEL_THRESH and avg_left_ext > EXT_THRESH and max_left_raise > 0.3:
            if avg_hip_rot > ROT_THRESH:
                detected_move = "roundhouse_kick"
                confidence = min(0.75 + (avg_hip_rot / ROT_THRESH) * 0.15, 0.95)
            else:
                detected_move = "front_kick"
                confidence = min(0.70 + (avg_left_vel / VEL_THRESH) * 0.15, 0.90)
        
        # Right leg kick detection (prioritize if stronger signal)
        if avg_right_vel > VEL_THRESH and avg_right_ext > EXT_THRESH and max_right_raise > 0.3:
            right_conf = 0.0
            if avg_hip_rot > ROT_THRESH:
                detected_move = "roundhouse_kick"
                right_conf = min(0.75 + (avg_hip_rot / ROT_THRESH) * 0.15, 0.95)
            else:
                detected_move = "front_kick"
                right_conf = min(0.70 + (avg_right_vel / VEL_THRESH) * 0.15, 0.90)
            
            # Use the stronger signal
            if right_conf > confidence:
                confidence = right_conf
        
        return detected_move, confidence
    
    def _print_summary(self, statistics: Dict):
        """Print classification summary"""
        print("\n" + "="*70)
        print("✅ MOVE CLASSIFICATION COMPLETE")
        print("="*70)
        
        for player_id in ["player_1", "player_2"]:
            print(f"\n📊 {player_id.replace('_', ' ').title()} Statistics:")
            stats = statistics[player_id]
            
            for move in MOVE_CLASSES:
                count = stats[move]
                pct = stats.get(f"{move}_pct", 0)
                print(f"   {move:15s}: {count:4d} frames ({pct:5.1f}%)")
        
        print("="*70 + "\n")


# High-level API functions

def classify_moves(
    pose_json_path: str,
    model_path: str = "models/move_classifier.pkl",
    output_path: str = None
) -> Dict:
    """
    High-level API: Classify moves from pose JSON
    
    Args:
        pose_json_path: Path to pose data JSON from pose extraction
        model_path: Path to trained classifier model
        output_path: Path to save results (auto-generated if None)
        
    Returns:
        Dictionary containing move predictions for both players
        
    Example:
        >>> moves = classify_moves("data/processed/poses_fight.json")
        >>> print(moves["frames"]["frame_000001"]["player_1"]["move"])
        "jab"
    """
    classifier = MoveClassifier(
        model_path=model_path,
        confidence_threshold=0.5,
        use_temporal_smoothing=True
    )
    
    return classifier.classify_from_json(pose_json_path, output_path)


def pose_to_features(pose_keypoints: List) -> np.ndarray:
    """
    Convert raw pose keypoints to engineered features
    
    Args:
        pose_keypoints: List of [x, y, visibility] for 33 landmarks
        
    Returns:
        Feature vector (23 features)
        
    Features include:
        - 8 joint angles (elbows, knees, shoulders, hips)
        - 4 extension metrics (arms, legs)
        - 4 velocity features (wrists, ankles)
        - 4 height metrics
        - 3 cross-body features
    """
    classifier = MoveClassifier()
    return classifier._pose_to_features(pose_keypoints, None)


def smooth_predictions(
    predictions: Dict,
    alpha: float = 0.6
) -> Dict:
    """
    Apply exponential smoothing to stabilize predictions
    
    Args:
        predictions: Raw frame-wise predictions
        alpha: Smoothing factor (0=no smooth, 1=full history)
        
    Returns:
        Smoothed predictions dictionary
    """
    # Note: Smoothing is already integrated in MoveClassifier
    # This function is provided for API compatibility
    return predictions


def save_classified_moves(move_dict: Dict, output_path: str):
    """
    Save classification results to JSON
    
    Args:
        move_dict: Classification results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(move_dict, f, indent=2)
    
    print(f"✅ Moves saved to {output_path}")


def train_classifier(
    training_data_path: str,
    output_model_path: str = "models/move_classifier.pkl",
    n_estimators: int = 100
):
    """
    Train a new move classifier from labeled data
    
    Args:
        training_data_path: Path to labeled training data JSON
        output_model_path: Path to save trained model
        n_estimators: Number of trees in random forest
        
    Note:
        Training data format:
        {
          "samples": [
            {"features": [...], "label": "jab"},
            {"features": [...], "label": "cross"},
            ...
          ]
        }
    """
    print(f"\n🎓 Training classifier from {training_data_path}")
    
    # Load training data
    with open(training_data_path, 'r') as f:
        data = json.load(f)
    
    samples = data["samples"]
    X = np.array([s["features"] for s in samples])
    y = np.array([s["label"] for s in samples])
    
    print(f"📊 Training samples: {len(X)}")
    print(f"📊 Classes: {np.unique(y)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save model
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model trained and saved to {output_path}")
    
    # Print feature importances
    print("\n📊 Top 10 Feature Importances:")
    feature_names = [
        "left_elbow_angle", "right_elbow_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_arm_ext", "right_arm_ext",
        "left_leg_ext", "right_leg_ext",
        "left_wrist_vel", "right_wrist_vel",
        "left_ankle_vel", "right_ankle_vel",
        "wrist_h_left", "wrist_h_right",
        "knee_h_left", "knee_h_right",
        "arm_cross_dist", "stance_width", "com_y"
    ]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]:20s}: {importances[idx]:.3f}")
