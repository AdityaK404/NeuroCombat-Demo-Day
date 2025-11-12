"""
Standalone Move Classification Script for NeuroCombat
======================================================

Quick CLI tool to classify moves from pose JSON.

Usage:
    python run_move_classification.py --input data/processed/poses_sample.json
    python run_move_classification.py --input poses.json --model models/custom.pkl

Author: NeuroCombat Team
Date: November 12, 2025
"""

import argparse
import sys
from pathlib import Path
from backend.move_classifier_v2 import classify_moves, MoveClassifier, MOVE_CLASSES


def main():
    parser = argparse.ArgumentParser(
        description="NeuroCombat Move Classification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify moves from pose JSON
  python run_move_classification.py --input data/processed/poses_fight.json
  
  # Use custom model
  python run_move_classification.py --input poses.json --model models/custom.pkl
  
  # Custom output path
  python run_move_classification.py --input poses.json --output results/moves.json
  
  # Adjust confidence threshold
  python run_move_classification.py --input poses.json --confidence 0.7
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to pose data JSON from pose extraction"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/move_classifier.pkl",
        help="Path to trained classifier model (default: models/move_classifier.pkl)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save classification results (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable temporal smoothing (less stable but faster)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Setup output path
    if args.output is None:
        args.output = f"data/processed/moves_{input_path.stem.replace('poses_', '')}.json"
    
    print("\n" + "="*70)
    print("🧠 NEUROCOMBAT - MOVE CLASSIFICATION MODULE")
    print("="*70)
    print(f"📥 Input pose JSON: {input_path}")
    print(f"🤖 Model: {args.model}")
    print(f"💾 Output: {args.output}")
    print(f"🎯 Confidence threshold: {args.confidence}")
    print(f"📊 Temporal smoothing: {'DISABLED' if args.no_smoothing else 'ENABLED'}")
    print(f"📋 Move classes: {', '.join(MOVE_CLASSES)}")
    print("="*70 + "\n")
    
    try:
        # Create classifier
        classifier = MoveClassifier(
            model_path=args.model,
            confidence_threshold=args.confidence,
            use_temporal_smoothing=not args.no_smoothing
        )
        
        # Classify moves
        move_data = classifier.classify_from_json(
            pose_json_path=str(input_path),
            output_path=args.output
        )
        
        # Success message
        print("\n" + "="*70)
        print("✅ CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Total frames: {move_data['metadata']['total_frames']}")
        print(f"💾 Results saved: {args.output}")
        
        # Show sample predictions
        print("\n📋 Sample Predictions (first 5 frames):")
        print("-"*70)
        for i, (frame_key, frame_data) in enumerate(sorted(move_data["frames"].items())[:5]):
            print(f"\n{frame_key}:")
            for player_id in ["player_1", "player_2"]:
                if player_id in frame_data:
                    pred = frame_data[player_id]
                    print(f"  {player_id}: {pred['move']:15s} (confidence: {pred['confidence']:.3f})")
        
        print("\n💡 Next steps:")
        print(f"   1. Review classification: {args.output}")
        print(f"   2. Generate commentary: python main.py --stage commentary")
        print(f"   3. View in UI: python app.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Classification interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
