"""
NeuroCombat Main Orchestration Script
======================================
Command-line interface for processing MMA fight videos.

Usage:
    python main.py --video path/to/fight.mp4 --output results/
    python main.py --video fight.mp4 --realtime --display
"""

import argparse
import cv2
import json
from pathlib import Path
from typing import Optional
import sys

from backend.pose_extractor import PoseExtractor
from backend.tracker import PlayerTracker
from backend.move_classifier import MoveClassifier
from backend.commentary_engine import CommentaryEngine
from backend.utils import setup_logging, PerformanceTimer, format_timestamp

logger = setup_logging(__name__)


class NeuroCombatPipeline:
    """
    Main pipeline orchestrator for NeuroCombat system.
    Coordinates pose extraction, tracking, classification, and commentary.
    """
    
    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        classifier_window: int = 15,
        use_mock_classifier: bool = True,
        commentary_interval: float = 2.0,
    ):
        """
        Initialize the NeuroCombat pipeline.
        
        Args:
            detection_confidence: Pose detection confidence threshold
            tracking_confidence: Pose tracking confidence threshold
            classifier_window: Number of frames for move classification
            use_mock_classifier: Use mock predictions (True for demo)
            commentary_interval: Min seconds between commentary lines
        """
        logger.info("Initializing NeuroCombat Pipeline...")
        
        self.pose_extractor = PoseExtractor(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        
        self.tracker = PlayerTracker(
            iou_threshold=0.3,
            max_missing_frames=30,
        )
        
        self.move_classifier = MoveClassifier(
            window_size=classifier_window,
            confidence_threshold=0.6,
            use_mock=use_mock_classifier,
        )
        
        self.commentary_engine = CommentaryEngine(
            min_time_between_comments=commentary_interval,
        )
        
        logger.info("✅ Pipeline initialized successfully")
    
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        display: bool = False,
        save_video: bool = True,
    ) -> dict:
        """
        Process a complete video file.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results (None = don't save)
            display: Show real-time visualization
            save_video: Save annotated video output
            
        Returns:
            Dictionary with processing results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"📹 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {total_frames} frames @ {fps} FPS ({width}x{height})")
        
        # Setup output video writer if needed
        video_writer = None
        if save_video and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps,
                (width, height)
            )
            logger.info(f"💾 Saving output to: {output_video_path}")
        
        # Processing loop
        commentary_events = []
        frame_idx = 0
        
        with PerformanceTimer("Video processing", logger):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Step 1: Extract poses
                poses = self.pose_extractor.extract_poses_from_frame(frame, timestamp)
                
                # Step 2: Track players
                tracked_poses = self.tracker.update(poses, frame_idx)
                
                # Step 3: Classify moves and generate commentary
                for pose in tracked_poses:
                    classification = self.move_classifier.classify_move(pose)
                    
                    if classification:
                        commentary = self.commentary_engine.generate_commentary(classification)
                        
                        if commentary:
                            commentary_events.append(commentary)
                            logger.info(f"[{format_timestamp(timestamp)}] {commentary.text}")
                
                # Step 4: Annotate frame
                annotated_frame = self.pose_extractor.draw_poses_on_frame(
                    frame,
                    tracked_poses
                )
                
                # Add latest commentary overlay
                if commentary_events:
                    latest = commentary_events[-1]
                    self._draw_commentary_overlay(annotated_frame, latest.text)
                
                # Display if requested
                if display:
                    cv2.imshow('NeuroCombat - Live Analysis', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted processing")
                        break
                
                # Save frame
                if video_writer:
                    video_writer.write(annotated_frame)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Get fight statistics
        fight_stats = self.commentary_engine.get_fight_summary()
        
        # Save results
        results = {
            "video_path": str(video_path),
            "total_frames": frame_idx,
            "fps": fps,
            "commentary_events": [
                {
                    "timestamp": event.timestamp,
                    "text": event.text,
                    "type": event.event_type,
                    "players": event.players_involved,
                }
                for event in commentary_events
            ],
            "fight_stats": fight_stats,
        }
        
        if output_dir:
            results_path = output_dir / f"{video_path.stem}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved to: {results_path}")
        
        logger.info(f"✅ Processing complete: {len(commentary_events)} commentary events")
        
        return results
    
    def _draw_commentary_overlay(self, frame, text: str):
        """Draw commentary text overlay on frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    def close(self):
        """Release all resources."""
        self.pose_extractor.close()
        logger.info("Pipeline closed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroCombat - AI-Powered MMA Fight Commentary System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to input MMA fight video",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--display",
        "-d",
        action="store_true",
        help="Display real-time visualization",
    )
    
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Don't save annotated video (faster processing)",
    )
    
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Pose detection confidence threshold (0.0-1.0)",
    )
    
    parser.add_argument(
        "--commentary-interval",
        type=float,
        default=2.0,
        help="Minimum seconds between commentary lines",
    )
    
    parser.add_argument(
        "--use-ml-model",
        action="store_true",
        help="Use trained ML model instead of mock classifier",
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("🥊 NeuroCombat - AI Fight Commentary System 🥊")
    print("=" * 60)
    print()
    
    try:
        # Initialize pipeline
        pipeline = NeuroCombatPipeline(
            detection_confidence=args.detection_confidence,
            use_mock_classifier=not args.use_ml_model,
            commentary_interval=args.commentary_interval,
        )
        
        # Process video
        results = pipeline.process_video(
            video_path=args.video,
            output_dir=args.output,
            display=args.display,
            save_video=not args.no_save_video,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 FIGHT ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Commentary Lines: {len(results['commentary_events'])}")
        print(f"Player 1 Total Moves: {results['fight_stats']['total_moves'].get(1, 0)}")
        print(f"Player 2 Total Moves: {results['fight_stats']['total_moves'].get(2, 0)}")
        print("=" * 60)
        
        pipeline.close()
        
        print("\n✅ Processing complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
