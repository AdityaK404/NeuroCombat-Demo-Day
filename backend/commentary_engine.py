"""
Commentary Engine Module
=========================
Generates real-time AI commentary for MMA fight actions.
Combines template-based generation with contextual awareness.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from .move_classifier import MoveClassification
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class CommentaryEvent:
    """Represents a single commentary line."""
    
    text: str
    timestamp: float
    event_type: str  # "move", "combo", "exchange", "analysis"
    players_involved: List[int]


class CommentaryEngine:
    """
    Generates real-time fight commentary using templates and context.
    Tracks fight flow and generates diverse, engaging commentary.
    """
    
    # Commentary templates organized by move type
    MOVE_TEMPLATES = {
        "jab": [
            "Player {player} fires off a quick jab!",
            "A sharp jab from Player {player}!",
            "Player {player} probes with the jab.",
            "Clean jab by Player {player}.",
        ],
        "cross": [
            "Player {player} throws a powerful cross!",
            "Hard cross from Player {player}!",
            "Player {player} connects with a straight right!",
            "Boom! Big cross by Player {player}!",
        ],
        "uppercut": [
            "Player {player} goes to the body with an uppercut!",
            "Devastating uppercut from Player {player}!",
            "Player {player} lands a clean uppercut!",
            "That's a picture-perfect uppercut by Player {player}!",
        ],
        "front_kick": [
            "Player {player} delivers a front kick!",
            "Front kick to the midsection by Player {player}!",
            "Player {player} keeps distance with a front kick.",
            "Nice teep kick from Player {player}!",
        ],
        "roundhouse_kick": [
            "Player {player} unleashes a roundhouse kick!",
            "Powerful roundhouse from Player {player}!",
            "Player {player} goes high with the roundhouse!",
            "That's a Thai-style roundhouse from Player {player}!",
        ],
        "neutral": [
            "Both fighters are circling, looking for openings.",
            "The fighters reset to their stances.",
            "Tactical moment here as they both assess.",
        ],
    }
    
    # Contextual commentary for combinations and exchanges
    COMBO_TEMPLATES = [
        "Player {player} is putting together a nice combination!",
        "Multiple strikes from Player {player}!",
        "Player {player} is on the attack!",
        "Excellent combination work by Player {player}!",
    ]
    
    EXCHANGE_TEMPLATES = [
        "We've got an exchange! Both fighters trading blows!",
        "They're going blow-for-blow!",
        "What an exchange! Both fighters landing!",
        "This is heating up!",
    ]
    
    OPENING_TEMPLATES = [
        "The fight is underway! Both fighters looking sharp.",
        "Here we go! Let's see who makes the first move.",
        "And we're live! Both fighters establishing their range.",
    ]
    
    def __init__(
        self,
        min_time_between_comments: float = 2.0,
        combo_window: float = 3.0,
        exchange_window: float = 2.0,
    ):
        """
        Initialize commentary engine.
        
        Args:
            min_time_between_comments: Minimum seconds between commentary lines
            combo_window: Time window to detect combinations (seconds)
            exchange_window: Time window to detect exchanges (seconds)
        """
        self.min_time_between_comments = min_time_between_comments
        self.combo_window = combo_window
        self.exchange_window = exchange_window
        
        self.last_comment_time = 0.0
        self.recent_moves: List[MoveClassification] = []
        self.fight_started = False
        
        self.move_counts = {1: {}, 2: {}}  # Track move statistics per player
        self.commentary_history: List[CommentaryEvent] = []
        
        logger.info("CommentaryEngine initialized")
    
    def generate_commentary(
        self,
        move: MoveClassification,
    ) -> Optional[CommentaryEvent]:
        """
        Generate commentary for a single move.
        
        Args:
            move: Classified move to comment on
            
        Returns:
            CommentaryEvent if commentary generated, None otherwise
        """
        # Opening commentary for first move
        if not self.fight_started:
            self.fight_started = True
            commentary = random.choice(self.OPENING_TEMPLATES)
            return self._create_event(commentary, move.timestamp, "opening", [1, 2])
        
        # Skip neutral moves and enforce minimum time between comments
        if move.move_name == "neutral":
            return None
        
        if move.timestamp - self.last_comment_time < self.min_time_between_comments:
            return None
        
        # Add to recent moves for context
        self.recent_moves.append(move)
        self._update_statistics(move)
        
        # Check for combinations (multiple moves from same player)
        if self._is_combo(move):
            commentary = random.choice(self.COMBO_TEMPLATES).format(player=move.player_id)
            event = self._create_event(commentary, move.timestamp, "combo", [move.player_id])
        
        # Check for exchanges (both players active)
        elif self._is_exchange():
            commentary = random.choice(self.EXCHANGE_TEMPLATES)
            event = self._create_event(commentary, move.timestamp, "exchange", [1, 2])
        
        # Standard move commentary
        else:
            templates = self.MOVE_TEMPLATES.get(move.move_name, [])
            if templates:
                commentary = random.choice(templates).format(player=move.player_id)
                event = self._create_event(commentary, move.timestamp, "move", [move.player_id])
            else:
                return None
        
        self.last_comment_time = move.timestamp
        self.commentary_history.append(event)
        
        # Clean up old moves from buffer
        self._cleanup_recent_moves(move.timestamp)
        
        return event
    
    def generate_batch_commentary(
        self,
        moves: List[MoveClassification],
    ) -> List[CommentaryEvent]:
        """
        Generate commentary for a batch of moves (e.g., full video).
        
        Args:
            moves: List of classified moves
            
        Returns:
            List of commentary events
        """
        commentary_events = []
        
        for move in moves:
            event = self.generate_commentary(move)
            if event:
                commentary_events.append(event)
        
        return commentary_events
    
    def _is_combo(self, current_move: MoveClassification) -> bool:
        """
        Check if current move is part of a combination.
        Multiple moves from same player in short time window.
        """
        recent_same_player = [
            m for m in self.recent_moves
            if m.player_id == current_move.player_id
            and current_move.timestamp - m.timestamp <= self.combo_window
            and m.move_name != "neutral"
        ]
        
        return len(recent_same_player) >= 2
    
    def _is_exchange(self) -> bool:
        """
        Check if recent moves constitute an exchange.
        Both players active within short time window.
        """
        if len(self.recent_moves) < 2:
            return False
        
        latest_time = self.recent_moves[-1].timestamp
        
        recent_window = [
            m for m in self.recent_moves
            if latest_time - m.timestamp <= self.exchange_window
            and m.move_name != "neutral"
        ]
        
        # Check if both players have moves in window
        players_active = set(m.player_id for m in recent_window)
        
        return len(players_active) == 2 and len(recent_window) >= 3
    
    def _cleanup_recent_moves(self, current_time: float):
        """Remove moves outside the context window."""
        cutoff_time = current_time - max(self.combo_window, self.exchange_window)
        self.recent_moves = [
            m for m in self.recent_moves
            if m.timestamp > cutoff_time
        ]
    
    def _update_statistics(self, move: MoveClassification):
        """Track move counts for each player."""
        player_id = move.player_id
        move_name = move.move_name
        
        if move_name not in self.move_counts[player_id]:
            self.move_counts[player_id][move_name] = 0
        
        self.move_counts[player_id][move_name] += 1
    
    def _create_event(
        self,
        text: str,
        timestamp: float,
        event_type: str,
        players: List[int],
    ) -> CommentaryEvent:
        """Create a commentary event."""
        return CommentaryEvent(
            text=text,
            timestamp=timestamp,
            event_type=event_type,
            players_involved=players,
        )
    
    def get_fight_summary(self) -> Dict:
        """
        Generate summary statistics for the fight.
        
        Returns:
            Dictionary with fight statistics
        """
        summary = {
            "total_commentary_lines": len(self.commentary_history),
            "player_1_moves": self.move_counts[1],
            "player_2_moves": self.move_counts[2],
            "total_moves": {
                1: sum(self.move_counts[1].values()),
                2: sum(self.move_counts[2].values()),
            },
        }
        
        return summary
    
    def reset(self):
        """Reset engine state for new fight."""
        self.last_comment_time = 0.0
        self.recent_moves.clear()
        self.fight_started = False
        self.move_counts = {1: {}, 2: {}}
        self.commentary_history.clear()
        logger.info("CommentaryEngine reset")


# TODO: Add TTS integration (pyttsx3 or gTTS)
# TODO: Implement LLM-based dynamic commentary for variety
# TODO: Add fight momentum tracking for contextual commentary
# TODO: Implement post-fight analysis generation
