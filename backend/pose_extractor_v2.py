"""
Pose Extraction Module for NeuroCombat - V2 Enhanced
====================================================

Production-ready dual-fighter pose extraction with:
- MediaPipe Pose integration with person detection
- Hungarian algorithm player tracking
- Real-time overlay visualization
- JSON export for downstream processing

Author: NeuroCombat Team
Date: November 12, 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def to_py(obj):
    """
    Universal sanitizer: Convert NumPy types to native Python types for JSON serialization.
    
    Handles np.int64, np.float32, np.ndarray, and nested structures (lists, tuples, dicts).
    Compatible with NumPy 1.x and 2.x.
    """
    # Check for NumPy integer types (np.int64, np.int32, etc.)
    if isinstance(obj, np.integer):
        return int(obj)
    # Check for NumPy floating types (np.float64, np.float32, etc.)
    if isinstance(obj, np.floating):
        return float(obj)
    # Check for NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle tuples recursively
    if isinstance(obj, tuple):
        return tuple(to_py(x) for x in obj)
    # Handle lists recursively
    if isinstance(obj, list):
        return [to_py(x) for x in obj]
    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    # Return primitives as-is
    return obj


@dataclass
class Pose:
    """Represents a single pose detection with keypoints"""
    keypoints: List[List[float]]  # List of [x, y, visibility] for 33 keypoints
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    confidence: float


class PoseExtractor:
    """Extracts and tracks poses for two fighters in MMA videos with enhanced dual detection"""
    
    # MediaPipe Pose landmark connections for drawing
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
    
    def __init__(self, confidence_threshold: float = 0.5, verbose_logging: bool = False):
        """
        Initialize pose extractor with MediaPipe and person detection
        
        Args:
            confidence_threshold: Minimum confidence for pose detection
            verbose_logging: Enable detailed frame-by-frame logging for debugging
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.confidence_threshold = confidence_threshold
        self.verbose_logging = verbose_logging
        
        # Initialize two separate pose detectors for better dual tracking
        self.pose_detector_1 = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        self.pose_detector_2 = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Player tracking state
        self.player_positions = {"player_1": None, "player_2": None}
        self.frame_count = 0
        
        # Detection metrics for analysis
        self.detection_log = {
            "frame_pose_counts": [],  # Number of poses detected per frame
            "frame_confidences": [],  # Average confidence per frame
            "spatial_strategy_used": []  # Track when spatial splitting was used
        }
        
        # Coordinate smoothing state (EMA filter with alpha=0.6)
        self.smoothed_keypoints = {"player_1": None, "player_2": None}
        self.smoothing_alpha = 0.6
        
    def extract_poses_from_video(
        self, 
        video_path: str,
        output_json: str = None,
        overlay_video: str = None,
        display: bool = False
    ) -> Dict:
        """
        Extract poses from video file with dual-fighter tracking
        
        Args:
            video_path: Path to input video
            output_json: Path to save pose data JSON (auto-generated if None)
            overlay_video: Path to save visualization video (optional)
            display: Show real-time overlay window
            
        Returns:
            Dictionary containing frame-wise pose data and metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🎥 Processing video: {video_path.name}")
        print(f"📊 Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")
        
        # Setup video writer for overlay
        writer = None
        if overlay_video:
            overlay_path = Path(overlay_video)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
        
        # Data storage
        pose_data = {
            "metadata": {
                "video_name": video_path.name,
                "resolution": [width, height],
                "fps": fps,
                "total_frames": total_frames
            },
            "frames": {}
        }
        
        # Statistics
        stats = {
            "processed_frames": 0,
            "dual_detections": 0,
            "single_detections": 0,
            "no_detections": 0,
            "avg_keypoints_p1": [],
            "avg_keypoints_p2": []
        }
        
        # Reset tracking state
        self.player_positions = {"player_1": None, "player_2": None}
        self.frame_count = 0
        
        # Reset detection metrics
        self.detection_log = {
            "frame_pose_counts": [],
            "frame_confidences": [],
            "spatial_strategy_used": []
        }
        
        # Process video frame by frame
        with tqdm(total=total_frames, desc="Extracting poses", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                frame_key = f"frame_{self.frame_count:06d}"
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect all poses in frame
                poses = self._detect_poses_in_frame(rgb_frame, frame.shape)
                
                # Track and assign player identities
                player_poses = self._track_players(poses)
                
                # Store pose data
                frame_data = {}
                for player_id, pose in player_poses.items():
                    if pose:
                        frame_data[player_id] = {
                            "keypoints": pose.keypoints,
                            "bbox": pose.bbox,
                            "centroid": pose.centroid,
                            "confidence": pose.confidence
                        }
                        
                        # Update statistics
                        visible_kpts = sum(1 for kpt in pose.keypoints if kpt[2] > 0.5)
                        if player_id == "player_1":
                            stats["avg_keypoints_p1"].append(visible_kpts)
                        else:
                            stats["avg_keypoints_p2"].append(visible_kpts)
                
                pose_data["frames"][frame_key] = frame_data
                
                # Update statistics
                stats["processed_frames"] += 1
                if len(frame_data) == 2:
                    stats["dual_detections"] += 1
                elif len(frame_data) == 1:
                    stats["single_detections"] += 1
                else:
                    stats["no_detections"] += 1
                
                # Draw overlay
                if writer or display:
                    overlay_frame = self._draw_overlay(frame, player_poses)
                    
                    if writer:
                        writer.write(overlay_frame)
                    
                    if display:
                        cv2.imshow('NeuroCombat - Pose Extraction', overlay_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        dual_detection_rate = round(stats["dual_detections"] / stats["processed_frames"] * 100, 2) if stats["processed_frames"] > 0 else 0
        spatial_split_usage = round(sum(self.detection_log["spatial_strategy_used"]) / len(self.detection_log["spatial_strategy_used"]) * 100, 2) if self.detection_log["spatial_strategy_used"] else 0
        avg_confidence = round(np.mean(self.detection_log["frame_confidences"]), 3) if self.detection_log["frame_confidences"] else 0
        
        pose_data["statistics"] = {
            "processed_frames": stats["processed_frames"],
            "dual_detections": stats["dual_detections"],
            "single_detections": stats["single_detections"],
            "no_detections": stats["no_detections"],
            "detection_rate": dual_detection_rate,
            "avg_keypoints_p1": round(np.mean(stats["avg_keypoints_p1"]), 2) if stats["avg_keypoints_p1"] else 0,
            "avg_keypoints_p2": round(np.mean(stats["avg_keypoints_p2"]), 2) if stats["avg_keypoints_p2"] else 0,
            # Enhanced metrics
            "spatial_split_usage_rate": spatial_split_usage,
            "avg_detection_confidence": avg_confidence,
            "frame_pose_distribution": {
                "0_poses": self.detection_log["frame_pose_counts"].count(0),
                "1_pose": self.detection_log["frame_pose_counts"].count(1),
                "2_poses": self.detection_log["frame_pose_counts"].count(2)
            }
        }
        
        # Save JSON
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Debug: Check for non-serializable types before sanitization
            for k, v in pose_data.items():
                if isinstance(v, np.generic):
                    print(f"⚠️  Non-serializable type found: {type(v)} at {k}")
            
            # Sanitize all NumPy types to native Python types
            sanitized = to_py(pose_data)
            
            with open(output_path, 'w') as f:
                json.dump(sanitized, f, indent=2)
            print(f"\n💾 Pose data saved → {output_path}")
        
        if overlay_video:
            print(f"🎥 Overlay video saved → {overlay_video}")
        
        # Print summary
        self._print_summary(pose_data["statistics"])
        
        return pose_data
    
    def _detect_poses_in_frame(self, rgb_frame: np.ndarray, frame_shape: tuple) -> List[Pose]:
        """
        Detect all poses in a single frame using optimized dual detection strategy
        
        Strategy: Try full frame first. If single detection, split frame to find second person.
        This reduces overhead from 3x to 1-2x per frame.
        
        Args:
            rgb_frame: RGB image array
            frame_shape: Original frame dimensions (h, w, c)
            
        Returns:
            List of Pose objects detected in frame (up to 2)
        """
        h, w = frame_shape[:2]
        used_spatial_split = False  # Track if we used spatial splitting
        
        # FIX P3.2: Optimized detection strategy
        # Step 1: Try detecting from full frame first
        full_frame_poses = self._detect_single_pose(rgb_frame, (0, 0), frame_shape, (w, h))
        
        # If we found 2 poses in full frame, we're done (unlikely with MediaPipe but possible)
        if len(full_frame_poses) >= 2:
            unique_poses = full_frame_poses[:2]
        else:
            # Step 2: If we found 0 or 1 pose, try spatial splitting to find missing person(s)
            # Only perform additional detections if needed
            used_spatial_split = True
            mid_x = w // 2
            overlap = 50  # 50px overlap to catch people near center
            
            # Left half detection
            left_crop = rgb_frame[:, :mid_x + overlap]
            left_poses = self._detect_single_pose(left_crop, (0, 0), (h, mid_x + overlap, 3), (w, h))
            
            # Right half detection
            right_crop = rgb_frame[:, mid_x - overlap:]
            right_poses = self._detect_single_pose(right_crop, (0, mid_x - overlap), (h, w - mid_x + overlap, 3), (w, h))
            
            # Merge all detections
            all_poses = full_frame_poses + left_poses + right_poses
            
            # FIX P4.2: Optimized duplicate removal using single-pass algorithm
            # Sort by confidence first (highest to lowest)
            all_poses.sort(key=lambda p: p.confidence, reverse=True)
            
            unique_poses = []
            duplicate_threshold = 100  # 100px centroid distance threshold
            near_duplicate_threshold = 0.05  # Normalized distance for near-duplicates (5% of frame)
            
            for pose in all_poses:
                is_duplicate = False
                for existing in unique_poses:
                    # Pixel distance check
                    dist = np.sqrt((pose.centroid[0] - existing.centroid[0])**2 + 
                                 (pose.centroid[1] - existing.centroid[1])**2)
                    
                    # Normalized distance check (for single-fighter near-duplicates)
                    normalized_dist = dist / w  # Normalize by frame width
                    
                    if dist < duplicate_threshold or normalized_dist < near_duplicate_threshold:
                        is_duplicate = True
                        break  # Already have better pose (since sorted by confidence)
                
                if not is_duplicate:
                    unique_poses.append(pose)
                    
                # Early exit: we only need max 2 poses
                if len(unique_poses) >= 2:
                    break
        
        # SINGLE-FIGHTER EDGE CASE: Check if two detected poses are actually same person
        # If centroids are very close (< 5% of frame width), drop the second pose
        if len(unique_poses) == 2:
            centroid_dist = np.sqrt((unique_poses[0].centroid[0] - unique_poses[1].centroid[0])**2 + 
                                   (unique_poses[0].centroid[1] - unique_poses[1].centroid[1])**2)
            normalized_centroid_dist = centroid_dist / w
            
            if normalized_centroid_dist < near_duplicate_threshold:
                # Very close centroids - likely same person detected twice
                unique_poses = [unique_poses[0]]  # Keep only the higher confidence pose
        
        # Log detection metrics
        self.detection_log["frame_pose_counts"].append(len(unique_poses))
        self.detection_log["spatial_strategy_used"].append(used_spatial_split)
        if unique_poses:
            avg_conf = np.mean([p.confidence for p in unique_poses])
            self.detection_log["frame_confidences"].append(float(avg_conf))
            
            if self.verbose_logging and self.frame_count % 30 == 0:  # Log every 30 frames
                print(f"  Frame {self.frame_count}: {len(unique_poses)} poses detected "
                      f"(conf: {avg_conf:.3f}, spatial: {used_spatial_split})")
        else:
            self.detection_log["frame_confidences"].append(0.0)
        
        return unique_poses
    
    def _detect_single_pose(self, rgb_crop: np.ndarray, offset: Tuple[int, int], 
                           crop_shape: tuple, frame_dims: Tuple[int, int]) -> List[Pose]:
        """
        Detect a single pose in a cropped region
        
        Args:
            rgb_crop: RGB crop of frame
            offset: (x_offset, y_offset) of crop in original frame
            crop_shape: Shape of crop (h, w, c) - NOT USED, gets actual dimensions from rgb_crop
            frame_dims: (width, height) of the full original frame for proper coordinate bounds
            
        Returns:
            List containing detected Pose (empty if none)
        """
        # Alternate between detectors to prevent tracking lockup
        detector = self.pose_detector_1 if self.frame_count % 2 == 0 else self.pose_detector_2
        results = detector.process(rgb_crop)
        
        if not results.pose_landmarks:
            return []
        
        landmarks = results.pose_landmarks.landmark
        
        # Get actual crop dimensions from the cropped frame
        crop_h, crop_w = rgb_crop.shape[:2]
        x_offset, y_offset = offset
        
        # Get actual full frame dimensions
        full_w, full_h = frame_dims
        
        # Convert normalized landmarks to pixel coordinates (with offset)
        keypoints = []
        coord_overflow_count = 0
        
        for idx, lm in enumerate(landmarks):
            # Scale by actual crop dimensions, then add offset to map to original frame
            x = int(lm.x * crop_w) + x_offset
            y = int(lm.y * crop_h) + y_offset
            
            # Clip coordinates to frame bounds and log overflow
            if x < 0 or x >= full_w or y < 0 or y >= full_h:
                coord_overflow_count += 1
                x = np.clip(x, 0, full_w - 1)
                y = np.clip(y, 0, full_h - 1)
            
            visibility = lm.visibility
            keypoints.append([float(x), float(y), float(visibility)])
        
        # Log coordinate overflow warning if detected
        if coord_overflow_count > 0 and self.verbose_logging:
            print(f"  ⚠️  Frame {self.frame_count}: {coord_overflow_count}/33 keypoints clipped to frame bounds")
        
        # Calculate bounding box and centroid (using actual full frame dimensions)
        bbox = self._calculate_bbox(keypoints, full_w, full_h)
        centroid = self._calculate_centroid(keypoints)
        
        # Calculate average confidence
        avg_confidence = np.mean([lm.visibility for lm in landmarks])
        
        pose = Pose(
            keypoints=keypoints,
            bbox=bbox,
            centroid=centroid,
            confidence=float(avg_confidence)
        )
        
        return [pose]  # Return as list for consistency
    
    def _calculate_bbox(self, keypoints: List, width: int, height: int) -> Tuple[int, int, int, int]:
        """Calculate bounding box from pose landmarks"""
        visible_points = [kp for kp in keypoints if kp[2] > 0.5]
        
        if not visible_points:
            return (0, 0, 0, 0)
        
        xs = [kp[0] for kp in visible_points]
        ys = [kp[1] for kp in visible_points]
        
        x_min, x_max = max(0, min(xs)), min(width, max(xs))
        y_min, y_max = max(0, min(ys)), min(height, max(ys))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        # Ensure native Python ints for JSON serialization
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _calculate_centroid(self, keypoints: List) -> Tuple[float, float]:
        """Calculate centroid from pose landmarks (hip midpoint for stability)"""
        # Use hip keypoints (indices 23, 24) for stable centroid
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        
        if left_hip[2] > 0.5 and right_hip[2] > 0.5:
            cx = (left_hip[0] + right_hip[0]) / 2
            cy = (left_hip[1] + right_hip[1]) / 2
        else:
            # Fallback to all visible points
            visible_points = [kp for kp in keypoints if kp[2] > 0.5]
            if visible_points:
                cx = np.mean([kp[0] for kp in visible_points])
                cy = np.mean([kp[1] for kp in visible_points])
            else:
                cx, cy = 0, 0
        
        # Ensure native Python floats for JSON serialization
        return (float(cx), float(cy))
    
    def _track_players(self, poses: List[Pose]) -> Dict[str, Optional[Pose]]:
        """
        Assign consistent player IDs using Hungarian algorithm
        
        Args:
            poses: List of detected poses in current frame
            
        Returns:
            Dictionary mapping player_id to Pose object
        """
        result = {"player_1": None, "player_2": None}
        
        if len(poses) == 0:
            return result
        
        if self.player_positions["player_1"] is None and self.player_positions["player_2"] is None:
            # First frame - initialize players
            if len(poses) >= 2:
                poses_sorted = sorted(poses, key=lambda p: p.centroid[0])
                result["player_1"] = poses_sorted[0]
                result["player_2"] = poses_sorted[1]
            elif len(poses) == 1:
                result["player_1"] = poses[0]
            
            self.player_positions = result
            return result
        
        # Build cost matrix for Hungarian algorithm
        prev_centroids = []
        prev_ids = []
        
        for player_id in ["player_1", "player_2"]:
            if self.player_positions[player_id] is not None:
                prev_centroids.append(self.player_positions[player_id].centroid)
                prev_ids.append(player_id)
        
        if not prev_centroids:
            # Lost all tracking - reinitialize
            if len(poses) >= 2:
                poses_sorted = sorted(poses, key=lambda p: p.centroid[0])
                result["player_1"] = poses_sorted[0]
                result["player_2"] = poses_sorted[1]
            elif len(poses) == 1:
                result["player_1"] = poses[0]
            self.player_positions = result
            return result
        
        curr_centroids = [p.centroid for p in poses]
        
        # Calculate cost matrix (Euclidean distances)
        cost_matrix = np.zeros((len(prev_centroids), len(curr_centroids)))
        for i, prev_c in enumerate(prev_centroids):
            for j, curr_c in enumerate(curr_centroids):
                dist = np.sqrt((prev_c[0] - curr_c[0])**2 + (prev_c[1] - curr_c[1])**2)
                cost_matrix[i, j] = dist
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Threshold for maximum tracking distance
        MAX_TRACKING_DISTANCE = 300
        
        for prev_idx, curr_idx in zip(row_ind, col_ind):
            if cost_matrix[prev_idx, curr_idx] < MAX_TRACKING_DISTANCE:
                player_id = prev_ids[prev_idx]
                result[player_id] = poses[curr_idx]
        
        # Handle new detections
        assigned_indices = set(col_ind)
        for i, pose in enumerate(poses):
            if i not in assigned_indices:
                if result["player_1"] is None:
                    result["player_1"] = pose
                elif result["player_2"] is None:
                    result["player_2"] = pose
        
        self.player_positions = result
        return result
    
    def _draw_overlay(self, frame: np.ndarray, player_poses: Dict[str, Optional[Pose]]) -> np.ndarray:
        """Draw colored skeleton overlay on frame with coordinate validation and smoothing"""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Player colors: Player 1 = 🔴 Red, Player 2 = 🔵 Blue
        colors = {
            "player_1": (0, 0, 255),    # Red (BGR format)
            "player_2": (255, 0, 0)     # Blue (BGR format)
        }
        
        for player_id, pose in player_poses.items():
            if pose is None:
                continue
            
            color = colors[player_id]
            raw_keypoints = pose.keypoints
            
            # Apply exponential moving average (EMA) smoothing for stable rendering
            if self.smoothed_keypoints[player_id] is None:
                # Initialize smoothed state
                smoothed_kpts = [[kp[0], kp[1], kp[2]] for kp in raw_keypoints]
            else:
                # Apply EMA: smoothed = alpha * new + (1-alpha) * smoothed
                smoothed_kpts = []
                prev_smooth = self.smoothed_keypoints[player_id]
                for i, (x, y, vis) in enumerate(raw_keypoints):
                    if vis > 0.5:  # Only smooth visible keypoints
                        prev_x, prev_y, _ = prev_smooth[i]
                        smooth_x = self.smoothing_alpha * x + (1 - self.smoothing_alpha) * prev_x
                        smooth_y = self.smoothing_alpha * y + (1 - self.smoothing_alpha) * prev_y
                        smoothed_kpts.append([smooth_x, smooth_y, vis])
                    else:
                        smoothed_kpts.append([x, y, vis])
            
            self.smoothed_keypoints[player_id] = smoothed_kpts
            
            # Clip all coordinates to frame bounds
            clipped_keypoints = []
            for kp in smoothed_kpts:
                x_clip = int(np.clip(kp[0], 0, w - 1))
                y_clip = int(np.clip(kp[1], 0, h - 1))
                clipped_keypoints.append([x_clip, y_clip, kp[2]])
            
            # Draw connections (skeleton lines)
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(clipped_keypoints) and end_idx < len(clipped_keypoints):
                    start_point = clipped_keypoints[start_idx]
                    end_point = clipped_keypoints[end_idx]
                    
                    # Only draw if both endpoints are visible
                    if start_point[2] > 0.5 and end_point[2] > 0.5:
                        cv2.line(
                            overlay,
                            (start_point[0], start_point[1]),
                            (end_point[0], end_point[1]),
                            color,
                            2
                        )
            
            # Draw keypoints (joint circles)
            for kp in clipped_keypoints:
                if kp[2] > 0.5:  # Only draw visible keypoints
                    cv2.circle(overlay, (kp[0], kp[1]), 3, color, -1)
            
            # Draw bounding box with bounds checking
            x, y, bbox_w, bbox_h = pose.bbox
            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))
            x2 = int(np.clip(x + bbox_w, 0, w))
            y2 = int(np.clip(y + bbox_h, 0, h))
            cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
            
            # Draw player label with background for better visibility
            label = f"{player_id.replace('_', ' ').title()} ({pose.confidence:.2f})"
            label_y = max(y - 10, 15)  # Prevent label from going off-screen
            cv2.putText(overlay, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Frame counter
        cv2.putText(overlay, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return overlay
    
    def _print_summary(self, stats: Dict):
        """Print extraction summary statistics with enhanced metrics"""
        print("\n" + "="*70)
        print("✅ POSE EXTRACTION COMPLETE")
        print("="*70)
        print(f"📊 Processed frames: {stats['processed_frames']}")
        print(f"🎯 Dual detections: {stats['dual_detections']} ({stats['detection_rate']}%)")
        
        # Target validation
        target_met = "✅ TARGET MET" if stats['detection_rate'] >= 50 else "❌ BELOW TARGET"
        print(f"   └─ Target (≥50%): {target_met}")
        
        print(f"⚠️  Single detections: {stats['single_detections']}")
        print(f"❌ No detections: {stats['no_detections']}")
        print(f"🔢 Avg keypoints P1: {stats['avg_keypoints_p1']} / 33")
        print(f"🔢 Avg keypoints P2: {stats['avg_keypoints_p2']} / 33")
        
        # Enhanced metrics
        print(f"\n📈 Enhanced Detection Metrics:")
        print(f"   • Spatial split usage: {stats.get('spatial_split_usage_rate', 0)}% of frames")
        print(f"   • Avg detection confidence: {stats.get('avg_detection_confidence', 0):.3f}")
        
        dist = stats.get('frame_pose_distribution', {})
        print(f"   • Frame distribution:")
        print(f"     - 0 poses: {dist.get('0_poses', 0)} frames")
        print(f"     - 1 pose:  {dist.get('1_pose', 0)} frames")
        print(f"     - 2 poses: {dist.get('2_poses', 0)} frames")
        
        print("="*70 + "\n")
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'pose_detector_1'):
            self.pose_detector_1.close()
        if hasattr(self, 'pose_detector_2'):
            self.pose_detector_2.close()


def extract_poses(video_path: str, display: bool = True) -> dict:
    """
    High-level API: Extract poses from MMA fight video
    
    Args:
        video_path: Path to input video file
        display: Show real-time visualization
        
    Returns:
        Dictionary containing frame-wise pose data for both fighters
    """
    video_path = Path(video_path)
    output_json = f"data/processed/poses_{video_path.stem}.json"
    overlay_video = f"data/processed/overlay_{video_path.stem}.mp4"
    
    extractor = PoseExtractor(confidence_threshold=0.5)
    pose_data = extractor.extract_poses_from_video(
        video_path=str(video_path),
        output_json=output_json,
        overlay_video=overlay_video,
        display=display
    )
    
    return pose_data


def save_pose_json(pose_dict: dict, output_path: str):
    """Save extracted pose data as JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(pose_dict, f, indent=2)
    
    print(f"✅ Pose data saved to {output_path}")
