"""
Configuration Module
====================
Central configuration for NeuroCombat system.
"""

from pathlib import Path
from dataclasses import dataclass


# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class PoseConfig:
    """Configuration for pose extraction."""
    
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0=lite, 1=full, 2=heavy
    
    # MediaPipe pose model has 33 landmarks
    num_landmarks: int = 33
    landmark_dimensions: int = 3  # (x, y, z)


@dataclass
class TrackerConfig:
    """Configuration for player tracking."""
    
    iou_threshold: float = 0.3
    max_missing_frames: int = 30
    history_size: int = 10
    
    # Initial player assignment strategy
    assignment_strategy: str = "left_right"  # left=Player1, right=Player2


@dataclass
class ClassifierConfig:
    """Configuration for move classification."""
    
    window_size: int = 15  # Number of frames for temporal analysis
    confidence_threshold: float = 0.6
    use_mock: bool = True  # Use mock predictions (set False for trained model)
    
    # Move classes
    move_classes: list = None
    
    def __post_init__(self):
        if self.move_classes is None:
            self.move_classes = [
                "neutral",
                "jab",
                "cross",
                "uppercut",
                "front_kick",
                "roundhouse_kick",
            ]
    
    # Model paths
    model_path: str = str(MODELS_DIR / "move_classifier.pkl")


@dataclass
class CommentaryConfig:
    """Configuration for commentary generation."""
    
    min_time_between_comments: float = 2.0  # Seconds
    combo_window: float = 3.0  # Seconds to detect combinations
    exchange_window: float = 2.0  # Seconds to detect exchanges
    
    # TTS settings (optional)
    enable_tts: bool = False
    tts_engine: str = "pyttsx3"  # or "gtts"
    tts_rate: int = 150  # Words per minute


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    # Output video settings
    output_codec: str = "mp4v"
    output_fps: int = 30
    
    # Display settings
    window_name: str = "NeuroCombat - Live Analysis"
    display_scale: float = 1.0  # Scale factor for display
    
    # Frame processing
    skip_frames: int = 0  # Process every Nth frame (0 = all frames)


@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    
    page_title: str = "NeuroCombat - AI Fight Commentary"
    page_icon: str = "🥊"
    layout: str = "wide"
    
    # Theming
    primary_color: str = "#FF6B6B"
    secondary_color: str = "#4ECDC4"
    background_color: str = "#0E1117"
    
    # Upload limits
    max_upload_size_mb: int = 500
    allowed_video_formats: list = None
    
    def __post_init__(self):
        if self.allowed_video_formats is None:
            self.allowed_video_formats = ["mp4", "avi", "mov", "mkv"]


# Global configuration instances
POSE_CONFIG = PoseConfig()
TRACKER_CONFIG = TrackerConfig()
CLASSIFIER_CONFIG = ClassifierConfig()
COMMENTARY_CONFIG = CommentaryConfig()
VIDEO_CONFIG = VideoConfig()
UI_CONFIG = UIConfig()


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(OUTPUT_DIR / "neurocombat.log"),
            "mode": "a",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}


def get_config_summary() -> dict:
    """
    Get a summary of all configuration settings.
    
    Returns:
        Dictionary with configuration values
    """
    return {
        "pose": POSE_CONFIG.__dict__,
        "tracker": TRACKER_CONFIG.__dict__,
        "classifier": CLASSIFIER_CONFIG.__dict__,
        "commentary": COMMENTARY_CONFIG.__dict__,
        "video": VIDEO_CONFIG.__dict__,
        "ui": UI_CONFIG.__dict__,
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "output_dir": str(OUTPUT_DIR),
        },
    }


def save_config(output_path: str = None):
    """
    Save current configuration to JSON file.
    
    Args:
        output_path: Path to save config (default: output/config.json)
    """
    import json
    
    if output_path is None:
        output_path = OUTPUT_DIR / "config.json"
    
    config_dict = get_config_summary()
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to: {output_path}")


if __name__ == "__main__":
    # Print configuration summary
    import json
    
    print("=" * 60)
    print("NeuroCombat Configuration Summary")
    print("=" * 60)
    print(json.dumps(get_config_summary(), indent=2))
    print("=" * 60)
