"""
Standalone Pose Extraction Script for NeuroCombat
==================================================

Quick CLI tool to extract poses from MMA videos.

Usage:
    python run_pose_extraction.py --video data/raw/sample.mp4
    python run_pose_extraction.py --video fight.mp4 --display
    python run_pose_extraction.py --video fight.mp4 --no-overlay

Author: NeuroCombat Team
Date: November 12, 2025
"""

import argparse
import sys
from pathlib import Path
from backend.pose_extractor_v2 import extract_poses, PoseExtractor


def main():
    parser = argparse.ArgumentParser(
        description="NeuroCombat Pose Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract poses with real-time display
  python run_pose_extraction.py --video data/raw/fight.mp4 --display
  
  # Extract poses without display (faster)
  python run_pose_extraction.py --video data/raw/fight.mp4
  
  # Custom output paths
  python run_pose_extraction.py --video fight.mp4 --output-json results/poses.json --output-video results/overlay.mp4
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input MMA fight video (.mp4, .avi, etc.)"
    )
    
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show real-time pose visualization (press 'q' to quit)"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save pose data JSON (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path to save overlay video (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip generating overlay video (faster processing)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Pose detection confidence threshold (0.0-1.0, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Setup output paths
    if args.output_json is None:
        args.output_json = f"data/processed/poses_{video_path.stem}.json"
    
    if args.output_video is None and not args.no_overlay:
        args.output_video = f"data/processed/overlay_{video_path.stem}.mp4"
    elif args.no_overlay:
        args.output_video = None
    
    print("\n" + "="*70)
    print("🥊 NEUROCOMBAT - POSE EXTRACTION MODULE")
    print("="*70)
    print(f"📹 Input video: {video_path}")
    print(f"💾 Output JSON: {args.output_json}")
    if args.output_video:
        print(f"🎥 Output video: {args.output_video}")
    else:
        print(f"🎥 Output video: DISABLED")
    print(f"👁️  Real-time display: {'ENABLED' if args.display else 'DISABLED'}")
    print(f"🎯 Confidence threshold: {args.confidence}")
    print("="*70 + "\n")
    
    try:
        # Create extractor
        extractor = PoseExtractor(confidence_threshold=args.confidence)
        
        # Extract poses
        pose_data = extractor.extract_poses_from_video(
            video_path=str(video_path),
            output_json=args.output_json,
            overlay_video=args.output_video,
            display=args.display
        )
        
        # Success message
        print("\n" + "="*70)
        print("✅ EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Total frames: {pose_data['statistics']['processed_frames']}")
        print(f"🎯 Dual detections: {pose_data['statistics']['dual_detections']}")
        print(f"📈 Detection rate: {pose_data['statistics']['detection_rate']}%")
        print("\n💡 Next steps:")
        print(f"   1. Review pose data: {args.output_json}")
        if args.output_video:
            print(f"   2. Watch overlay video: {args.output_video}")
        print(f"   3. Run move classification: python main.py --stage classify")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
