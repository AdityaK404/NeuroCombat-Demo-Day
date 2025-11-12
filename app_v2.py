"""
NeuroCombat V2 - AI-Powered MMA Fight Commentary System
========================================================

Modern, production-ready Streamlit UI for real-time fight analysis.

Complete Pipeline:
1. Upload MMA video
2. Extract dual-fighter poses
3. Classify moves (6 classes)
4. Generate AI commentary
5. Watch synchronized playback

Run with: streamlit run app_v2.py

Author: NeuroCombat Team
Date: November 12, 2025
"""

import streamlit as st
import cv2
import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import asdict

# Backend imports
try:
    from backend.pose_extractor_v2 import PoseExtractor
    from backend.move_classifier_v2 import MoveClassifier
    from backend.commentary_engine_v2 import CommentaryEngine, CommentaryLine
except ImportError:
    st.error("❌ Failed to import backend modules. Ensure backend/*.py files exist.")
    st.stop()


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="NeuroCombat - AI Fight Analyst",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CUSTOM STYLING
# ==============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #0E1117;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #FF4B4B, #FF8C00, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    
    /* Commentary box styling */
    .commentary-line {
        background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
        border-left: 4px solid;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 1rem;
        animation: slideIn 0.3s ease-out;
    }
    
    .commentary-player1 {
        border-left-color: #FF4B4B;
    }
    
    .commentary-player2 {
        border-left-color: #4B9BFF;
    }
    
    .commentary-both {
        border-left-color: #FFD700;
    }
    
    .commentary-analysis {
        border-left-color: #888;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Stats box styling */
    .stats-box {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
    }
    
    /* Progress indicators */
    .progress-stage {
        background-color: #2A2A2A;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid #555;
    }
    
    .progress-stage.active {
        border-left-color: #FFD700;
        background-color: #3A3A1A;
    }
    
    .progress-stage.complete {
        border-left-color: #4CAF50;
        background-color: #1A3A1A;
    }
    
    /* Video player container */
    .video-container {
        border: 2px solid #333;
        border-radius: 12px;
        padding: 0.5rem;
        background-color: #000;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        background-color: #333;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #FFD700, #FF4B4B);
        transition: width 0.3s ease;
    }
    
    /* Action buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C00 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Upload section */
    .upload-section {
        border: 2px dashed #555;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background-color: #1A1A1A;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'uploaded_video_path': None,
        'pose_data_path': None,
        'moves_data_path': None,
        'commentary_data': None,
        'processing_stage': 'idle',  # idle, pose, classify, commentary, complete
        'pipeline_stats': {},
        'video_metadata': {},
        'current_commentary_index': 0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Path to saved file
    """
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def get_video_metadata(video_path: str) -> Dict:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    metadata = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0.0
    }
    
    if metadata['fps'] > 0:
        metadata['duration'] = metadata['total_frames'] / metadata['fps']
    
    cap.release()
    return metadata


def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def render_commentary_line(line: CommentaryLine, index: int):
    """
    Render a single commentary line with styling.
    
    Args:
        line: CommentaryLine object
        index: Line index for animation
    """
    # Determine CSS class based on player
    if line.player == 1:
        css_class = "commentary-player1"
        emoji = "🔴"
    elif line.player == 2:
        css_class = "commentary-player2"
        emoji = "🔵"
    elif line.event_type == "clash":
        css_class = "commentary-both"
        emoji = "⚡"
    else:
        css_class = "commentary-analysis"
        emoji = "💭"
    
    # Render with HTML
    st.markdown(f"""
    <div class="commentary-line {css_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{emoji}</span>
                <span style="font-weight: 500;">{line.text}</span>
            </div>
            <div style="color: #888; font-size: 0.85rem;">
                {line.timestamp:.1f}s
            </div>
        </div>
        <div class="confidence-bar" style="margin-top: 0.5rem;">
            <div class="confidence-fill" style="width: {line.confidence * 100}%;"></div>
        </div>
        <div style="color: #666; font-size: 0.75rem; margin-top: 0.25rem;">
            Confidence: {line.confidence:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_progress_stage(stage_name: str, status: str, message: str):
    """
    Render a pipeline stage progress indicator.
    
    Args:
        stage_name: Name of the stage
        status: 'pending', 'active', or 'complete'
        message: Status message
    """
    icons = {
        'pending': '⚪',
        'active': '🟡',
        'complete': '✅'
    }
    
    css_class = "progress-stage"
    if status == 'active':
        css_class += " active"
    elif status == 'complete':
        css_class += " complete"
    
    st.markdown(f"""
    <div class="{css_class}">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 1rem;">{icons[status]}</span>
            <div>
                <div style="font-weight: bold; font-size: 1.1rem;">{stage_name}</div>
                <div style="color: #888; font-size: 0.9rem;">{message}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PIPELINE EXECUTION FUNCTIONS
# ==============================================================================

def run_pose_extraction(video_path: str, progress_bar, status_text) -> str:
    """
    Run pose extraction stage.
    
    Args:
        video_path: Path to input video
        progress_bar: Streamlit progress bar
        status_text: Streamlit text element for status
    
    Returns:
        Path to pose JSON file
    """
    status_text.text("🔧 Initializing pose extraction...")
    
    # Initialize extractor
    extractor = PoseExtractor(
        confidence_threshold=0.5
    )
    
    # Create output path
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    video_name = Path(video_path).stem
    output_json = output_dir / f"poses_{video_name}.json"
    output_video = output_dir / f"poses_{video_name}_overlay.mp4"
    
    status_text.text("🎬 Processing video frames...")
    
    # Extract poses
    result = extractor.extract_poses_from_video(
        video_path=video_path,
        output_json=str(output_json),
        overlay_video=str(output_video),
        display=False
    )
    
    progress_bar.progress(1.0)
    status_text.text("✅ Pose extraction complete!")
    
    # Update session state
    st.session_state.video_metadata = {
        'total_frames': result['metadata']['total_frames'],
        'fps': result['metadata']['fps'],
        'dual_detections': result['statistics']['dual_detections']
    }
    st.session_state.overlay_video_path = str(output_video)  # Store overlay video path
    
    return str(output_json)


def run_move_classification(pose_json_path: str, progress_bar, status_text) -> str:
    """
    Run move classification stage.
    
    Args:
        pose_json_path: Path to pose JSON file
        progress_bar: Streamlit progress bar
        status_text: Streamlit text element for status
    
    Returns:
        Path to moves JSON file
    """
    status_text.text("🔧 Initializing move classifier...")
    
    # Initialize classifier (will use mock model if trained model doesn't exist)
    classifier = MoveClassifier(
        model_path="models/move_classifier.pkl",
        confidence_threshold=0.5
    )
    
    # Create output path
    output_dir = Path("artifacts")
    pose_name = Path(pose_json_path).stem.replace('poses_', '')
    output_json = output_dir / f"moves_{pose_name}.json"
    
    status_text.text("🥊 Classifying moves...")
    
    # Classify moves
    result = classifier.classify_from_json(
        pose_json_path=str(pose_json_path),
        output_path=str(output_json)
    )
    
    progress_bar.progress(1.0)
    status_text.text("✅ Move classification complete!")
    
    return str(output_json)


def run_commentary_generation(
    moves_json_path: str,
    fps: int,
    progress_bar,
    status_text
) -> Tuple[List[CommentaryLine], bool]:
    """
    Run commentary generation stage.
    
    Args:
        moves_json_path: Path to moves JSON file
        fps: Video FPS
        progress_bar: Streamlit progress bar
        status_text: Streamlit text element for status
    
    Returns:
        Tuple of (List of CommentaryLine objects, single_fighter_mode boolean)
    """
    status_text.text("🔧 Initializing commentary engine...")
    
    # Initialize engine
    engine = CommentaryEngine(fps=fps, enable_tts=False)
    
    # Create output path
    output_dir = Path("artifacts")
    moves_name = Path(moves_json_path).stem.replace('moves_', '')
    output_path = output_dir / f"commentary_{moves_name}"
    
    status_text.text("🎙️ Generating commentary...")
    
    # Generate commentary
    commentary_lines = engine.generate_commentary(
        moves_json_path=str(moves_json_path),
        output_path=str(output_path)
    )
    
    # Check if single-fighter mode was detected
    single_fighter_mode = getattr(engine, 'single_fighter_mode', False)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Commentary generation complete!")
    
    return commentary_lines, single_fighter_mode


def run_full_pipeline(video_path: str):
    """
    Run the complete analysis pipeline.
    
    Args:
        video_path: Path to uploaded video
    """
    # Get video metadata
    metadata = get_video_metadata(video_path)
    st.session_state.video_metadata = metadata
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Processing Pipeline")
        
        # Stage 1: Pose Extraction
        stage1_progress = st.progress(0.0)
        stage1_status = st.empty()
        
        with st.spinner("Extracting poses..."):
            try:
                pose_json = run_pose_extraction(video_path, stage1_progress, stage1_status)
                st.session_state.pose_data_path = pose_json
                st.session_state.processing_stage = 'pose'
                time.sleep(0.5)
            except Exception as e:
                st.error(f"❌ Pose extraction failed: {e}")
                return
        
        # Stage 2: Move Classification
        stage2_progress = st.progress(0.0)
        stage2_status = st.empty()
        
        with st.spinner("Classifying moves..."):
            try:
                moves_json = run_move_classification(pose_json, stage2_progress, stage2_status)
                st.session_state.moves_data_path = moves_json
                st.session_state.processing_stage = 'classify'
                time.sleep(0.5)
            except Exception as e:
                st.error(f"❌ Move classification failed: {e}")
                return
        
        # Stage 3: Commentary Generation
        stage3_progress = st.progress(0.0)
        stage3_status = st.empty()
        
        with st.spinner("Generating commentary..."):
            try:
                commentary, single_fighter_mode = run_commentary_generation(
                    moves_json,
                    metadata['fps'],
                    stage3_progress,
                    stage3_status
                )
                st.session_state.commentary_data = commentary
                st.session_state.single_fighter_mode = single_fighter_mode
                st.session_state.processing_stage = 'complete'
                time.sleep(0.5)
            except Exception as e:
                st.error(f"❌ Commentary generation failed: {e}")
                return
        
        st.success("🎉 Pipeline complete! Results ready below.")


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🥊 NEUROCOMBAT</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-Powered MMA Fight Commentary System</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### Processing Options")
        min_confidence = st.slider(
            "Detection Confidence",
            min_value=0.3,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Minimum confidence for pose detection"
        )
        
        display_overlay = st.checkbox(
            "Generate Overlay Video",
            value=True,
            help="Create video with pose overlay"
        )
        
        enable_tts = st.checkbox(
            "Enable Text-to-Speech",
            value=False,
            help="Speak commentary (requires pyttsx3)"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Pipeline Status")
        
        stages = [
            ("Pose Extraction", "pose"),
            ("Move Classification", "classify"),
            ("Commentary Generation", "complete")
        ]
        
        current_stage = st.session_state.processing_stage
        
        for stage_name, stage_key in stages:
            if current_stage == 'idle':
                status = 'pending'
            elif current_stage == stage_key:
                status = 'complete'
            elif stages.index((stage_name, stage_key)) < [s[1] for s in stages].index(current_stage):
                status = 'complete'
            else:
                status = 'pending'
            
            if status == 'complete':
                st.markdown(f"✅ {stage_name}")
            elif status == 'active':
                st.markdown(f"🟡 {stage_name}")
            else:
                st.markdown(f"⚪ {stage_name}")
        
        st.markdown("---")
        
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        1. **Upload** an MMA video file
        2. **Process** through AI pipeline
        3. **Watch** synchronized playback
        4. **Review** generated commentary
        5. **Download** results
        """)
        
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🎬 Results & Playback", "📊 Statistics"])
    
    # TAB 1: Upload & Process
    with tab1:
        st.markdown("## Upload Your MMA Fight Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload an MMA fight video for AI analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            if st.session_state.uploaded_video_path is None:
                with st.spinner("Saving video..."):
                    video_path = save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_video_path = video_path
                    
                    # Get metadata
                    metadata = get_video_metadata(video_path)
                    st.session_state.video_metadata = metadata
            
            # Display video info
            metadata = st.session_state.video_metadata
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Duration", format_duration(metadata.get('duration', 0)))
            with col2:
                st.metric("FPS", metadata.get('fps', 'N/A'))
            with col3:
                st.metric("Frames", f"{metadata.get('total_frames', 0):,}")
            with col4:
                st.metric("Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}")
            
            st.markdown("---")
            
            # Process button
            if st.session_state.processing_stage == 'idle':
                if st.button("🚀 Start AI Analysis", use_container_width=True, type="primary"):
                    run_full_pipeline(st.session_state.uploaded_video_path)
                    st.rerun()
            else:
                st.success("✅ Processing complete! Check the 'Results & Playback' tab.")
                
                if st.button("🔄 Process Another Video", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
    
    # TAB 2: Results & Playback
    with tab2:
        if st.session_state.processing_stage == 'complete':
            st.markdown("## 🎬 Analysis Results")
            
            # Display single-fighter mode indicator
            if st.session_state.get('single_fighter_mode', False):
                st.info("ℹ️ **Single-fighter mode detected** - Commentary focused on Player 1 with relaxed confidence thresholds")
            
            # Two-column layout
            col_video, col_commentary = st.columns([1.2, 1])
            
            with col_video:
                st.markdown("### Video with Pose Overlay")
                
                # Use stored overlay video path
                if 'overlay_video_path' in st.session_state and Path(st.session_state.overlay_video_path).exists():
                    overlay_video = st.session_state.overlay_video_path
                    st.video(overlay_video)
                else:
                    # Fallback: try to construct path
                    video_name = Path(st.session_state.uploaded_video_path).stem
                    overlay_video = Path("artifacts") / f"poses_{video_name}_overlay.mp4"
                    
                    if overlay_video.exists():
                        st.video(str(overlay_video))
                    else:
                        st.warning("⚠️ Overlay video not found. Showing original video.")
                        st.video(st.session_state.uploaded_video_path)
            
            with col_commentary:
                st.markdown("### 🎙️ Live Commentary Feed")
                
                commentary_data = st.session_state.commentary_data
                
                if commentary_data:
                    # Commentary controls
                    show_all = st.checkbox("Show All Commentary", value=False)
                    
                    if show_all:
                        display_lines = commentary_data
                    else:
                        display_lines = commentary_data[:10]
                    
                    # Scrollable commentary container
                    commentary_container = st.container()
                    
                    with commentary_container:
                        for i, line in enumerate(display_lines):
                            render_commentary_line(line, i)
                    
                    if not show_all and len(commentary_data) > 10:
                        st.info(f"Showing 10 of {len(commentary_data)} lines. Check 'Show All' to see more.")
                else:
                    st.warning("No commentary generated.")
        else:
            st.info("👈 Upload and process a video first to see results here.")
    
    # TAB 3: Statistics
    with tab3:
        if st.session_state.processing_stage == 'complete':
            st.markdown("## 📊 Fight Analysis Statistics")
            
            commentary_data = st.session_state.commentary_data
            
            if commentary_data:
                # Calculate statistics
                total_lines = len(commentary_data)
                p1_actions = sum(1 for l in commentary_data if l.player == 1)
                p2_actions = sum(1 for l in commentary_data if l.player == 2)
                clashes = sum(1 for l in commentary_data if l.event_type == "clash")
                avg_confidence = sum(l.confidence for l in commentary_data) / total_lines
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="stats-box">
                        <div class="stat-value">{}</div>
                        <div class="stat-label">Total Commentary Lines</div>
                    </div>
                    """.format(total_lines), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="stats-box">
                        <div class="stat-value" style="color: #FF4B4B;">{}</div>
                        <div class="stat-label">🔴 Player 1 Actions</div>
                    </div>
                    """.format(p1_actions), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="stats-box">
                        <div class="stat-value" style="color: #4B9BFF;">{}</div>
                        <div class="stat-label">🔵 Player 2 Actions</div>
                    </div>
                    """.format(p2_actions), unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                    <div class="stats-box">
                        <div class="stat-value" style="color: #FFD700;">{}</div>
                        <div class="stat-label">⚡ Clash Events</div>
                    </div>
                    """.format(clashes), unsafe_allow_html=True)
                
                # Event breakdown chart
                st.markdown("---")
                st.markdown("### Event Type Distribution")
                
                event_counts = {}
                for line in commentary_data:
                    event_counts[line.event_type] = event_counts.get(line.event_type, 0) + 1
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.bar_chart(event_counts)
                
                with col_chart2:
                    st.markdown(f"""
                    **Average Confidence:** {avg_confidence:.1%}
                    
                    **Fight Duration:** {commentary_data[-1].timestamp:.1f}s
                    
                    **Commentary Density:** {total_lines / (commentary_data[-1].timestamp / 60):.1f} lines/min
                    
                    **Player Balance:** {(min(p1_actions, p2_actions) / max(p1_actions, p2_actions) * 100) if max(p1_actions, p2_actions) > 0 else 0:.0f}%
                    """)
                
                # Download section
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    if st.session_state.pose_data_path:
                        with open(st.session_state.pose_data_path, 'r') as f:
                            st.download_button(
                                "📄 Download Pose Data",
                                data=f.read(),
                                file_name="poses.json",
                                mime="application/json"
                            )
                
                with col_dl2:
                    if st.session_state.moves_data_path:
                        with open(st.session_state.moves_data_path, 'r') as f:
                            st.download_button(
                                "🥊 Download Move Data",
                                data=f.read(),
                                file_name="moves.json",
                                mime="application/json"
                            )
                
                with col_dl3:
                    # Create commentary text
                    commentary_text = "\n".join([str(line) for line in commentary_data])
                    st.download_button(
                        "🎙️ Download Commentary",
                        data=commentary_text,
                        file_name="commentary.txt",
                        mime="text/plain"
                    )
        else:
            st.info("👈 Process a video first to see statistics here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>NeuroCombat</strong> - AI Fight Analyst of the Future 🥊</p>
        <p style="font-size: 0.9rem;">Powered by MediaPipe • Scikit-learn • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
