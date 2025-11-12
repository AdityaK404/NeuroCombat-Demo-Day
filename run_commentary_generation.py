"""
Run Commentary Generation - NeuroCombat
========================================

Standalone script for generating AI fight commentary from move classification.

Usage:
    python run_commentary_generation.py --input artifacts/moves_fight1.json
    python run_commentary_generation.py -i artifacts/moves_fight1.json --tts
    python run_commentary_generation.py -i data.json --fps 30 --output results/

Author: NeuroCombat Team
Date: November 12, 2025
"""

import argparse
import sys
from pathlib import Path
import json
import time
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backend.commentary_engine_v2 import CommentaryEngine, CommentaryLine


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def validate_input_file(input_path: str) -> bool:
    """
    Validate that input file exists and is valid JSON.
    
    Args:
        input_path: Path to input JSON file
    
    Returns:
        True if valid, False otherwise
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return False
    
    if input_file.suffix != '.json':
        print(f"⚠️  Warning: Input file is not .json: {input_path}")
    
    # Try to load JSON
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if 'frames' not in data:
            print(f"❌ Error: Invalid JSON structure. Missing 'frames' key.")
            return False
        
        if not data['frames']:
            print(f"❌ Error: No frames found in input file.")
            return False
        
        # Validate at least one frame has correct structure
        first_frame = next(iter(data['frames'].values()))
        if 'player_1' not in first_frame or 'player_2' not in first_frame:
            print(f"❌ Error: Invalid frame structure. Missing player data.")
            return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating input: {e}")
        return False


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 70)
    print("🎙️  NEUROCOMBAT - AI FIGHT COMMENTARY GENERATOR")
    print("=" * 70 + "\n")


def print_commentary_preview(commentary_lines: list, max_lines: int = 10):
    """
    Print preview of generated commentary.
    
    Args:
        commentary_lines: List of CommentaryLine objects
        max_lines: Maximum lines to display
    """
    print("\n📢 Commentary Preview:")
    print("-" * 70)
    
    for i, line in enumerate(commentary_lines[:max_lines]):
        # Color code by player
        if line.player == 1:
            prefix = "🔴"
        elif line.player == 2:
            prefix = "🔵"
        else:
            prefix = "⚪"
        
        print(f"{prefix} {line}")
    
    if len(commentary_lines) > max_lines:
        print(f"   ... and {len(commentary_lines) - max_lines} more lines")
    
    print("-" * 70)


def print_statistics(commentary_lines: list, elapsed_time: float):
    """
    Print generation statistics.
    
    Args:
        commentary_lines: List of CommentaryLine objects
        elapsed_time: Time taken to generate
    """
    if not commentary_lines:
        return
    
    # Count by event type
    event_counts = {}
    player1_actions = 0
    player2_actions = 0
    
    for line in commentary_lines:
        event_counts[line.event_type] = event_counts.get(line.event_type, 0) + 1
        if line.player == 1:
            player1_actions += 1
        elif line.player == 2:
            player2_actions += 1
    
    # Calculate stats
    total_duration = commentary_lines[-1].timestamp if commentary_lines else 0
    avg_confidence = sum(l.confidence for l in commentary_lines) / len(commentary_lines)
    
    print("\n📊 Generation Statistics:")
    print("-" * 70)
    print(f"  Total Commentary Lines: {len(commentary_lines)}")
    print(f"  Fight Duration:         {total_duration:.1f}s")
    print(f"  Generation Time:        {elapsed_time:.2f}s")
    print(f"  Average Confidence:     {avg_confidence:.2%}")
    print(f"\n  Player 1 Actions:       {player1_actions}")
    print(f"  Player 2 Actions:       {player2_actions}")
    print(f"\n  Event Breakdown:")
    for event_type, count in sorted(event_counts.items()):
        print(f"    {event_type.title():12} {count:3d}")
    print("-" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate AI commentary from move classification data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_commentary_generation.py --input artifacts/moves_fight1.json
  
  # With text-to-speech
  python run_commentary_generation.py -i artifacts/moves.json --tts
  
  # Custom FPS and output location
  python run_commentary_generation.py -i data.json --fps 30 --output results/
  
  # Adjust confidence threshold
  python run_commentary_generation.py -i data.json --min-confidence 0.7
  
  # Verbose output for debugging
  python run_commentary_generation.py -i data.json --verbose

Output:
  Creates two files at output location:
  - commentary_<name>.json  (structured data)
  - commentary_<name>.txt   (human-readable)
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to move classification JSON file (from move_classifier_v2.py)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        default='artifacts',
        help='Output directory for commentary files (default: artifacts/)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Frames per second of source video (default: 25)'
    )
    
    parser.add_argument(
        '--tts',
        action='store_true',
        help='Enable text-to-speech output (requires pyttsx3)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum confidence for detailed commentary (default: 0.6)'
    )
    
    parser.add_argument(
        '--context-window',
        type=int,
        default=5,
        help='Number of recent moves to track for variety (default: 5)'
    )
    
    parser.add_argument(
        '--neutral-threshold',
        type=int,
        default=10,
        help='Neutral frames before "pause" comment (default: 10)'
    )
    
    parser.add_argument(
        '--preview-lines',
        type=int,
        default=10,
        help='Number of commentary lines to preview (default: 10)'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable commentary preview output'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Print banner
    print_banner()
    
    # Validate input
    print(f"📂 Input File: {args.input}")
    if not validate_input_file(args.input):
        sys.exit(1)
    
    print("✅ Input validation passed\n")
    
    # Prepare output path
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_name = f"commentary_{input_path.stem.replace('moves_', '')}"
    output_path = output_dir / output_name
    
    print(f"📤 Output Location: {output_path}.json / .txt")
    print(f"🎬 Video FPS: {args.fps}")
    print(f"🎚️  Min Confidence: {args.min_confidence}")
    
    if args.tts:
        print("🔊 Text-to-Speech: ENABLED")
        try:
            import pyttsx3
            print("   ✅ TTS engine available")
        except ImportError:
            print("   ⚠️  pyttsx3 not installed, TTS disabled")
            print("   Install with: pip install pyttsx3")
            args.tts = False
    
    print("\n" + "-" * 70)
    print("⚙️  Initializing Commentary Engine...")
    
    # Initialize commentary engine
    try:
        engine = CommentaryEngine(
            fps=args.fps,
            context_window=args.context_window,
            min_confidence=args.min_confidence,
            neutral_threshold=args.neutral_threshold,
            enable_tts=args.tts
        )
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        sys.exit(1)
    
    print("✅ Engine initialized successfully")
    print("\n🎙️  Generating commentary...")
    
    # Generate commentary with timing
    start_time = time.time()
    
    try:
        commentary_lines = engine.generate_commentary(
            moves_json_path=str(input_path),
            output_path=str(output_path)
        )
        
        elapsed_time = time.time() - start_time
        
    except Exception as e:
        print(f"\n❌ Error generating commentary: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Display results
    print(f"\n✅ Commentary generation complete!")
    
    # Show preview
    if not args.no_preview and commentary_lines:
        print_commentary_preview(commentary_lines, args.preview_lines)
    
    # Show statistics
    print_statistics(commentary_lines, elapsed_time)
    
    # Success message
    print("\n🎉 Success! Commentary files saved:")
    print(f"   📄 {output_path}.json")
    print(f"   📄 {output_path}.txt")
    
    if args.tts:
        print("\n🔊 TTS playback completed")
    
    print("\n💡 Next Steps:")
    print("   1. Review commentary in text file")
    print("   2. Use JSON file for programmatic access")
    print("   3. Integrate with Streamlit UI (app_v2.py)")
    print("   4. Run: streamlit run app_v2.py")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
