"""
Commentary Engine V2 - NeuroCombat Real-Time Fight Commentary
==============================================================

Generates natural, context-aware commentary for MMA fight analysis.

Features:
- Template-based generation with Markov-style variety
- Clash detection and defensive phase recognition
- Confidence-based phrasing
- Timestamp calculation from frame numbers
- Optional Text-to-Speech (TTS) integration
- JSON and text export

Author: NeuroCombat Team
Date: November 12, 2025
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import deque
import logging

# Optional TTS support
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 not available. TTS functionality disabled.")


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class CommentaryLine:
    """Represents a single line of commentary."""
    timestamp: float          # Time in seconds
    frame_number: int         # Frame index
    text: str                 # Commentary text
    event_type: str          # "action", "clash", "defensive", "analysis"
    player: Optional[int]    # Which player (1 or 2), None for both
    confidence: float        # Average confidence of moves involved
    
    def __str__(self) -> str:
        """Format for display."""
        return f"[{self.timestamp:.1f}s] {self.text}"


@dataclass
class CommentaryContext:
    """Tracks recent fight context to avoid repetitive commentary."""
    recent_moves_p1: deque    # Last N moves by player 1
    recent_moves_p2: deque    # Last N moves by player 2
    last_templates_used: deque  # Last N templates to avoid repetition
    consecutive_neutrals: int   # Track boring stretches
    last_clash_frame: int      # Prevent spam on multi-frame clashes


# ==============================================================================
# COMMENTARY ENGINE
# ==============================================================================

class CommentaryEngine:
    """
    Generates natural, varied commentary for MMA fight sequences.
    
    Uses template-based generation with contextual awareness to produce
    engaging, non-repetitive commentary that adapts to fight dynamics.
    """
    
    # Commentary templates organized by move type
    MOVE_TEMPLATES = {
        "jab": [
            "Player {p} throws a quick jab!",
            "Player {p} fires off a sharp jab!",
            "Player {p} probes with the jab.",
            "A clean jab from Player {p}!",
            "Player {p} snaps out a jab!",
            "Player {p} lands a crisp jab!"
        ],
        "cross": [
            "Player {p} throws a powerful cross!",
            "Player {p} unleashes a heavy cross!",
            "A devastating cross from Player {p}!",
            "Player {p} connects with a straight right!",
            "Boom! Big cross by Player {p}!",
            "Player {p} counters with a massive cross!"
        ],
        "front_kick": [
            "Player {p} attempts a front kick!",
            "Player {p} goes for the front kick!",
            "A powerful front kick from Player {p}!",
            "Player {p} pushes forward with a front kick!",
            "Player {p} measures distance with a front kick!",
            "Front kick! Player {p} keeps the pressure on!"
        ],
        "roundhouse_kick": [
            "Player {p} swings a roundhouse kick!",
            "Player {p} goes high with a roundhouse!",
            "Devastating roundhouse from Player {p}!",
            "Player {p} unleashes a spinning roundhouse!",
            "Watch out! Roundhouse kick by Player {p}!",
            "Player {p} commits to the roundhouse kick!"
        ],
        "uppercut": [
            "Player {p} fires an uppercut!",
            "Player {p} digs deep with an uppercut!",
            "An explosive uppercut from Player {p}!",
            "Player {p} goes to the body with an uppercut!",
            "Uppercut! Player {p} finds the opening!",
            "Player {p} lands a crushing uppercut!"
        ],
        "neutral": [
            "Player {p} holds stance, reading the opponent.",
            "Player {p} maintains distance.",
            "Player {p} circles carefully.",
            "Player {p} stays defensive.",
            "Player {p} resets position."
        ]
    }
    
    # Clash commentary (when both attack simultaneously)
    CLASH_TEMPLATES = [
        "Both fighters exchange blows!",
        "What a clash! Both players engage!",
        "Simultaneous strikes! The intensity is rising!",
        "Both fighters commit to the attack!",
        "A fierce exchange between both players!",
        "They're trading shots! This is heating up!",
        "Both players refuse to back down!",
        "An explosive exchange! The crowd would be on their feet!"
    ]
    
    # Defensive phase commentary (one attacking, one neutral)
    DEFENSIVE_TEMPLATES = [
        "Player {atk} presses forward while Player {def} stays cautious.",
        "Player {atk} on the offensive, Player {def} looking to counter.",
        "Player {def} defends well against Player {atk}'s aggression.",
        "Player {atk} applies pressure, but Player {def} holds strong.",
        "Patient defense from Player {def} against Player {atk}'s assault."
    ]
    
    # Low confidence commentary
    LOW_CONFIDENCE_PHRASES = [
        "possibly a {move}",
        "attempting a {move}",
        "looks like a {move}",
        "maybe a {move}",
        "transitioning with a {move}"
    ]
    
    # Combo recognition patterns
    COMBO_PATTERNS = {
        ("jab", "cross"): "Player {p} executes the classic jab-cross combo!",
        ("jab", "jab", "cross"): "Player {p} sets up the cross with a double jab!",
        ("cross", "uppercut"): "Player {p} follows the cross with a devastating uppercut!",
        ("front_kick", "roundhouse_kick"): "Player {p} chains kicks beautifully!",
        ("jab", "front_kick"): "Player {p} mixes striking levels with a jab-kick combo!"
    }
    
    # Neutral stretch commentary (when both neutral for many frames)
    NEUTRAL_STRETCH_TEMPLATES = [
        "Both fighters take a moment to reset...",
        "A brief tactical pause as both players assess...",
        "They're sizing each other up...",
        "The pace slows as both fighters regroup...",
        "Strategic positioning from both sides..."
    ]
    
    def __init__(
        self,
        fps: int = 25,
        context_window: int = 5,
        min_confidence: float = 0.6,
        neutral_threshold: int = 10,
        enable_tts: bool = False
    ):
        """
        Initialize the Commentary Engine.
        
        Args:
            fps: Frames per second of source video
            context_window: Number of recent moves to track for variety
            min_confidence: Minimum confidence to generate detailed commentary
            neutral_threshold: Consecutive neutral frames before "pause" comment
            enable_tts: Enable text-to-speech output
        """
        self.fps = fps
        self.context_window = context_window
        self.min_confidence = min_confidence
        self.neutral_threshold = neutral_threshold
        self.enable_tts = enable_tts
        
        # Initialize TTS engine if available and requested
        self.tts_engine = None
        if enable_tts and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure TTS properties
                self.tts_engine.setProperty('rate', 175)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                logging.info("TTS engine initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize TTS: {e}")
                self.tts_engine = None
        elif enable_tts and not TTS_AVAILABLE:
            logging.warning("TTS requested but pyttsx3 not installed")
        
        # Initialize context tracking
        self.context = CommentaryContext(
            recent_moves_p1=deque(maxlen=context_window),
            recent_moves_p2=deque(maxlen=context_window),
            last_templates_used=deque(maxlen=10),
            consecutive_neutrals=0,
            last_clash_frame=-100  # Start far in past
        )
        
        self.logger = logging.getLogger(__name__)
    
    def generate_commentary(
        self,
        moves_json_path: str,
        output_path: Optional[str] = None
    ) -> List[CommentaryLine]:
        """
        Generate commentary from move classification JSON.
        
        Args:
            moves_json_path: Path to classified moves JSON file
            output_path: Optional path to save commentary (JSON and TXT)
        
        Returns:
            List of CommentaryLine objects with timestamps
        """
        self.logger.info(f"Generating commentary from: {moves_json_path}")
        
        # Load move classifications
        with open(moves_json_path, 'r') as f:
            moves_data = json.load(f)
        
        commentary_lines: List[CommentaryLine] = []
        
        # Extract metadata
        metadata = moves_data.get('metadata', {})
        fps = metadata.get('fps', self.fps)
        
        # SINGLE-FIGHTER MODE DETECTION
        # Count frames with Player 2 data to determine if dual-fighter or single-fighter
        frames = moves_data.get('frames', {})
        total_frames = len(frames)
        player2_frames = sum(1 for frame_data in frames.values() if 'player_2' in frame_data)
        player2_ratio = player2_frames / total_frames if total_frames > 0 else 0
        
        # If Player 2 appears in < 20% of frames, treat as single-fighter mode
        single_fighter_mode = player2_ratio < 0.20
        
        if single_fighter_mode:
            self.logger.info(f"Single-fighter mode detected (Player 2: {player2_ratio*100:.1f}% of frames)")
            # Relax confidence threshold for single-fighter commentary
            original_min_confidence = self.min_confidence
            self.min_confidence = 0.4  # Lower threshold for single-fighter
        else:
            self.logger.info(f"Dual-fighter mode (Player 2: {player2_ratio*100:.1f}% of frames)")
            single_fighter_mode = False
        
        # Store mode for later use
        self.single_fighter_mode = single_fighter_mode
        
        # Process each frame
        frame_keys = sorted(frames.keys(), key=lambda x: int(x.split('_')[1]))
        
        for frame_key in frame_keys:
            frame_data = frames[frame_key]
            frame_num = int(frame_key.split('_')[1])
            timestamp = frame_num / fps
            
            # Extract player moves and confidences
            p1_data = frame_data.get('player_1', {})
            p2_data = frame_data.get('player_2', {})
            
            p1_move = p1_data.get('move', 'neutral')
            p2_move = p2_data.get('move', 'neutral')
            p1_conf = p1_data.get('confidence', 0.0)
            p2_conf = p2_data.get('confidence', 0.0)
            
            # Generate commentary for this frame
            line = self._generate_frame_commentary(
                frame_num=frame_num,
                timestamp=timestamp,
                p1_move=p1_move,
                p2_move=p2_move,
                p1_conf=p1_conf,
                p2_conf=p2_conf
            )
            
            if line:
                commentary_lines.append(line)
                
                # Speak commentary if TTS enabled
                if self.tts_engine:
                    self._speak_commentary(line.text)
        
        self.logger.info(f"Generated {len(commentary_lines)} commentary lines")
        
        # Save commentary if output path provided
        if output_path:
            self.save_commentary(commentary_lines, output_path)
        
        # Restore original min_confidence if we modified it
        if single_fighter_mode:
            self.min_confidence = original_min_confidence
        
        return commentary_lines
    
    def _generate_frame_commentary(
        self,
        frame_num: int,
        timestamp: float,
        p1_move: str,
        p2_move: str,
        p1_conf: float,
        p2_conf: float
    ) -> Optional[CommentaryLine]:
        """
        Generate commentary for a single frame.
        
        Args:
            frame_num: Frame number
            timestamp: Time in seconds
            p1_move: Player 1's move
            p2_move: Player 2's move
            p1_conf: Player 1's confidence
            p2_conf: Player 2's confidence
        
        Returns:
            CommentaryLine or None if no commentary needed
        """
        # Update context
        self.context.recent_moves_p1.append(p1_move)
        self.context.recent_moves_p2.append(p2_move)
        
        # Check for both neutral (boring stretch)
        if p1_move == "neutral" and p2_move == "neutral":
            self.context.consecutive_neutrals += 1
            
            # Only comment on long neutral stretches
            if self.context.consecutive_neutrals == self.neutral_threshold:
                text = random.choice(self.NEUTRAL_STRETCH_TEMPLATES)
                return CommentaryLine(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    text=text,
                    event_type="analysis",
                    player=None,
                    confidence=(p1_conf + p2_conf) / 2
                )
            return None
        else:
            self.context.consecutive_neutrals = 0
        
        # SINGLE-FIGHTER MODE: Only generate commentary for Player 1
        if getattr(self, 'single_fighter_mode', False):
            # In single-fighter mode, ignore Player 2 completely
            if p1_move != "neutral" and p1_conf >= self.min_confidence:
                # Check for combo
                combo_text = self._check_for_combo(1)
                if combo_text:
                    return CommentaryLine(
                        timestamp=timestamp,
                        frame_number=frame_num,
                        text=combo_text,
                        event_type="action",
                        player=1,
                        confidence=p1_conf
                    )
                
                # Regular move commentary
                text = self._get_move_commentary(1, p1_move, p1_conf)
                return CommentaryLine(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    text=text,
                    event_type="action",
                    player=1,
                    confidence=p1_conf
                )
            return None
        
        # DUAL-FIGHTER MODE: Use existing clash/defense logic
        # Check for clash (both attacking)
        if p1_move != "neutral" and p2_move != "neutral":
            # Avoid commenting on every frame of a multi-frame clash
            if frame_num - self.context.last_clash_frame > 5:
                self.context.last_clash_frame = frame_num
                text = random.choice(self.CLASH_TEMPLATES)
                return CommentaryLine(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    text=text,
                    event_type="clash",
                    player=None,
                    confidence=(p1_conf + p2_conf) / 2
                )
            return None
        
        # Check for defensive phase (one attacking, one neutral)
        if (p1_move != "neutral" and p2_move == "neutral") or \
           (p1_move == "neutral" and p2_move != "neutral"):
            
            atk_player = 1 if p1_move != "neutral" else 2
            def_player = 2 if atk_player == 1 else 1
            conf = p1_conf if atk_player == 1 else p2_conf
            
            # Only comment if attacker has good confidence
            if conf >= self.min_confidence:
                # Check for combo
                combo_text = self._check_for_combo(atk_player)
                if combo_text:
                    return CommentaryLine(
                        timestamp=timestamp,
                        frame_number=frame_num,
                        text=combo_text,
                        event_type="action",
                        player=atk_player,
                        confidence=conf
                    )
                
                # Regular move commentary
                move = p1_move if atk_player == 1 else p2_move
                text = self._get_move_commentary(atk_player, move, conf)
                
                return CommentaryLine(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    text=text,
                    event_type="action",
                    player=atk_player,
                    confidence=conf
                )
        
        return None
    
    def _check_for_combo(self, player: int) -> Optional[str]:
        """
        Check if recent moves form a recognized combo pattern.
        
        Args:
            player: Player number (1 or 2)
        
        Returns:
            Combo commentary text or None
        """
        recent = list(self.context.recent_moves_p1 if player == 1 
                     else self.context.recent_moves_p2)
        
        # Only check combos if we have enough history
        if len(recent) < 2:
            return None
        
        # Check 3-move combos first
        if len(recent) >= 3:
            last_3 = tuple(recent[-3:])
            if last_3 in self.COMBO_PATTERNS:
                return self.COMBO_PATTERNS[last_3].format(p=player)
        
        # Check 2-move combos
        last_2 = tuple(recent[-2:])
        if last_2 in self.COMBO_PATTERNS:
            # Ensure we haven't just used this combo template
            template = self.COMBO_PATTERNS[last_2]
            if template not in self.context.last_templates_used:
                self.context.last_templates_used.append(template)
                return template.format(p=player)
        
        return None
    
    def _get_move_commentary(
        self,
        player: int,
        move: str,
        confidence: float
    ) -> str:
        """
        Generate commentary for a single move.
        
        Args:
            player: Player number (1 or 2)
            move: Move name
            confidence: Classification confidence
        
        Returns:
            Commentary text
        """
        # Get templates for this move
        templates = self.MOVE_TEMPLATES.get(move, self.MOVE_TEMPLATES["neutral"])
        
        # Filter out recently used templates for variety
        available_templates = [
            t for t in templates 
            if t not in self.context.last_templates_used
        ]
        
        # If all templates used recently, reset
        if not available_templates:
            available_templates = templates
        
        # Select random template
        template = random.choice(available_templates)
        self.context.last_templates_used.append(template)
        
        # Generate base text
        text = template.format(p=player)
        
        # Add low confidence phrasing if needed
        if confidence < self.min_confidence:
            low_conf_phrase = random.choice(self.LOW_CONFIDENCE_PHRASES)
            text = f"Player {player} {low_conf_phrase.format(move=move)}."
        
        return text
    
    def _speak_commentary(self, text: str):
        """
        Speak commentary text using TTS.
        
        Args:
            text: Commentary text to speak
        """
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.logger.warning(f"TTS error: {e}")
    
    def save_commentary(
        self,
        commentary_lines: List[CommentaryLine],
        output_path: str
    ):
        """
        Save commentary to JSON and text files.
        
        Args:
            commentary_lines: List of CommentaryLine objects
            output_path: Base output path (extensions added automatically)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        json_data = {
            'metadata': {
                'total_lines': len(commentary_lines),
                'fps': self.fps,
                'generated_at': str(Path(output_path).stem)
            },
            'commentary': [asdict(line) for line in commentary_lines]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Saved JSON commentary to: {json_path}")
        
        # Save as text
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NEUROCOMBAT - AI FIGHT COMMENTARY\n")
            f.write("=" * 70 + "\n\n")
            
            for line in commentary_lines:
                f.write(f"{line}\n")
        
        self.logger.info(f"Saved text commentary to: {txt_path}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def generate_commentary(
    moves_json_path: str,
    fps: int = 25,
    tts: bool = False,
    output_path: Optional[str] = None
) -> List[CommentaryLine]:
    """
    Convenience function to generate commentary from move classification JSON.
    
    Args:
        moves_json_path: Path to classified moves JSON
        fps: Frames per second of video
        tts: Enable text-to-speech
        output_path: Optional output path for saving commentary
    
    Returns:
        List of CommentaryLine objects
    
    Example:
        >>> commentary = generate_commentary(
        ...     "artifacts/moves_fight1.json",
        ...     fps=30,
        ...     tts=True,
        ...     output_path="artifacts/commentary_fight1"
        ... )
        >>> for line in commentary:
        ...     print(line)
    """
    engine = CommentaryEngine(fps=fps, enable_tts=tts)
    return engine.generate_commentary(moves_json_path, output_path)


def get_commentary_for_frame(
    p1_move: str,
    p2_move: str,
    p1_conf: float,
    p2_conf: float,
    frame_num: int = 0,
    fps: int = 25
) -> Optional[str]:
    """
    Generate a single line of commentary for a specific frame.
    
    Args:
        p1_move: Player 1's move
        p2_move: Player 2's move
        p1_conf: Player 1's confidence
        p2_conf: Player 2's confidence
        frame_num: Frame number
        fps: Frames per second
    
    Returns:
        Commentary text or None
    
    Example:
        >>> text = get_commentary_for_frame(
        ...     "jab", "neutral", 0.92, 0.85, 150, 30
        ... )
        >>> print(text)
        "Player 1 throws a quick jab!"
    """
    engine = CommentaryEngine(fps=fps)
    timestamp = frame_num / fps
    
    line = engine._generate_frame_commentary(
        frame_num, timestamp, p1_move, p2_move, p1_conf, p2_conf
    )
    
    return line.text if line else None


# ==============================================================================
# MAIN EXECUTION (for testing)
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("COMMENTARY ENGINE V2 - Test Mode")
    print("=" * 70)
    
    # Test with sample data
    sample_moves = {
        "metadata": {
            "video_name": "test_fight.mp4",
            "total_frames": 10,
            "fps": 25
        },
        "frames": {
            "frame_001": {
                "player_1": {"move": "neutral", "confidence": 0.85},
                "player_2": {"move": "neutral", "confidence": 0.82}
            },
            "frame_002": {
                "player_1": {"move": "jab", "confidence": 0.92},
                "player_2": {"move": "neutral", "confidence": 0.88}
            },
            "frame_003": {
                "player_1": {"move": "cross", "confidence": 0.89},
                "player_2": {"move": "neutral", "confidence": 0.85}
            },
            "frame_004": {
                "player_1": {"move": "neutral", "confidence": 0.83},
                "player_2": {"move": "front_kick", "confidence": 0.91}
            },
            "frame_005": {
                "player_1": {"move": "jab", "confidence": 0.87},
                "player_2": {"move": "cross", "confidence": 0.90}
            }
        }
    }
    
    # Save sample data
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_moves, f)
        sample_path = f.name
    
    # Generate commentary
    print("\n🎙️  Generating commentary...\n")
    commentary = generate_commentary(sample_path, fps=25, tts=False)
    
    print("\n📢 Generated Commentary:\n")
    for line in commentary:
        print(f"  {line}")
    
    print(f"\n✅ Successfully generated {len(commentary)} commentary lines!")
    print("=" * 70)
