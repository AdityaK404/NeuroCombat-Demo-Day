"""
Player Tracker Module
=====================
Tracks player identities (Player 1 vs Player 2) across video frames.
Uses spatial proximity and bounding box IoU for consistent player assignment.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from .pose_extractor import PoseLandmarks
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class PlayerTrack:
    """Represents a tracked player over time."""
    
    player_id: int
    bbox_history: deque  # Recent bounding boxes
    last_seen_frame: int
    confidence: float


class PlayerTracker:
    """
    Tracks player identities across frames using spatial consistency.
    Auto-assigns Player 1 (left) and Player 2 (right) based on initial positions.
    """
    
    def __init__(
        self,
        max_missing_frames: int = 30,
        iou_threshold: float = 0.3,
        history_size: int = 10,
    ):
        """
        Initialize player tracker.
        
        Args:
            max_missing_frames: Max frames before considering a player lost
            iou_threshold: Minimum IoU for bbox matching
            history_size: Number of historical bboxes to maintain
        """
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold
        self.history_size = history_size
        
        self.tracks: Dict[int, PlayerTrack] = {}
        self.next_player_id = 1
        self.current_frame = 0
        
        logger.info(f"PlayerTracker initialized with iou_threshold={iou_threshold}")
    
    def update(
        self,
        poses: List[PoseLandmarks],
        frame_idx: int,
    ) -> List[PoseLandmarks]:
        """
        Update tracking with new frame detections.
        
        Args:
            poses: List of detected poses in current frame
            frame_idx: Current frame index
            
        Returns:
            List of poses with updated player_id assignments
        """
        self.current_frame = frame_idx
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        # Match detections to existing tracks
        matched_poses = []
        
        if not self.tracks:
            # Initialize tracks for first frame
            matched_poses = self._initialize_tracks(poses)
        else:
            matched_poses = self._match_poses_to_tracks(poses)
        
        return matched_poses
    
    def _initialize_tracks(self, poses: List[PoseLandmarks]) -> List[PoseLandmarks]:
        """
        Initialize player tracks for first detection.
        Assigns Player 1 (left) and Player 2 (right) based on x-position.
        """
        if len(poses) == 0:
            return []
        
        # Sort by x-coordinate (left to right)
        sorted_poses = sorted(poses, key=lambda p: p.bbox[0])
        
        for idx, pose in enumerate(sorted_poses[:2]):  # Max 2 players
            player_id = idx + 1
            pose.player_id = player_id
            
            self.tracks[player_id] = PlayerTrack(
                player_id=player_id,
                bbox_history=deque([pose.bbox], maxlen=self.history_size),
                last_seen_frame=self.current_frame,
                confidence=1.0,
            )
            
            logger.info(f"Initialized Player {player_id} at bbox {pose.bbox}")
        
        return sorted_poses[:2]
    
    def _match_poses_to_tracks(
        self,
        poses: List[PoseLandmarks],
    ) -> List[PoseLandmarks]:
        """
        Match detected poses to existing player tracks using IoU.
        """
        matched_poses = []
        unmatched_poses = poses.copy()
        
        # For each track, find best matching pose
        for player_id, track in self.tracks.items():
            if not unmatched_poses:
                break
            
            # Get predicted bbox from history
            predicted_bbox = track.bbox_history[-1]
            
            # Find best matching pose
            best_iou = 0.0
            best_pose_idx = -1
            
            for idx, pose in enumerate(unmatched_poses):
                iou = self._calculate_iou(predicted_bbox, pose.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_pose_idx = idx
            
            # Update track if match found
            if best_pose_idx >= 0:
                matched_pose = unmatched_poses.pop(best_pose_idx)
                matched_pose.player_id = player_id
                
                track.bbox_history.append(matched_pose.bbox)
                track.last_seen_frame = self.current_frame
                track.confidence = min(1.0, track.confidence + 0.1)
                
                matched_poses.append(matched_pose)
        
        # Handle unmatched poses (new players or tracking failures)
        for pose in unmatched_poses:
            # Try to reassign to missing tracks
            assigned = False
            for player_id in [1, 2]:
                if player_id not in self.tracks:
                    pose.player_id = player_id
                    self.tracks[player_id] = PlayerTrack(
                        player_id=player_id,
                        bbox_history=deque([pose.bbox], maxlen=self.history_size),
                        last_seen_frame=self.current_frame,
                        confidence=0.5,
                    )
                    matched_poses.append(pose)
                    assigned = True
                    logger.info(f"Reassigned Player {player_id}")
                    break
            
            if not assigned:
                pose.player_id = 0  # Untracked
        
        return matched_poses
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        stale_ids = []
        
        for player_id, track in self.tracks.items():
            frames_missing = self.current_frame - track.last_seen_frame
            if frames_missing > self.max_missing_frames:
                stale_ids.append(player_id)
        
        for player_id in stale_ids:
            del self.tracks[player_id]
            logger.warning(f"Lost track of Player {player_id}")
    
    @staticmethod
    def _calculate_iou(bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1, bbox2: Bounding boxes in format (x, y, width, height)
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_player_positions(self) -> Dict[int, Tuple[int, int]]:
        """
        Get current center positions of tracked players.
        
        Returns:
            Dict mapping player_id to (x, y) center coordinates
        """
        positions = {}
        
        for player_id, track in self.tracks.items():
            if track.bbox_history:
                bbox = track.bbox_history[-1]
                x, y, w, h = bbox
                center_x = x + w // 2
                center_y = y + h // 2
                positions[player_id] = (center_x, center_y)
        
        return positions
    
    def reset(self):
        """Reset all tracking state."""
        self.tracks.clear()
        self.next_player_id = 1
        self.current_frame = 0
        logger.info("PlayerTracker reset")


# TODO: Implement Kalman filtering for smoother tracking
# TODO: Add re-identification using pose similarity when tracks are lost
# TODO: Handle occlusions and fighter crossovers
