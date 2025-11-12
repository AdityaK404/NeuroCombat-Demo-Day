"""
NeuroCombat - AI-Powered MMA Fight Commentary System
=====================================================
Streamlit UI for real-time fight analysis and commentary generation.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import List, Optional
import time

# Import backend modules
from backend.pose_extractor import PoseExtractor
from backend.tracker import PlayerTracker
from backend.move_classifier import MoveClassifier
from backend.commentary_engine import CommentaryEngine
from backend.utils import setup_logging, format_timestamp

logger = setup_logging(__name__)


# Page configuration
st.set_page_config(
    page_title="NeuroCombat - AI Fight Commentary",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .commentary-box {
        background-color: #1E1E1E;
        border-left: 4px solid #FF6B6B;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        color: #FFFFFF;
    }
    .stats-box {
        background-color: #2E2E2E;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    if 'commentary_events' not in st.session_state:
        st.session_state.commentary_events = []
    if 'fight_stats' not in st.session_state:
        st.session_state.fight_stats = {}
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False


def process_video(
    video_path: str,
    progress_bar,
    status_text,
) -> tuple:
    """
    Process uploaded video through the full pipeline.
    
    Args:
        video_path: Path to uploaded video
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
        
    Returns:
        Tuple of (commentary_events, fight_stats, annotated_video_path)
    """
    # Initialize pipeline components
    status_text.text("🔧 Initializing AI components...")
    
    pose_extractor = PoseExtractor(
        min_detection_confidence=0.5,
        model_complexity=1,
    )
    tracker = PlayerTracker(
        iou_threshold=0.3,
        max_missing_frames=30,
    )
    move_classifier = MoveClassifier(
        window_size=15,
        confidence_threshold=0.6,
        use_mock=True,  # Use mock for demo
    )
    commentary_engine = CommentaryEngine(
        min_time_between_comments=2.0,
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    commentary_events = []
    frame_idx = 0
    
    status_text.text("🎥 Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = (frame_idx + 1) / total_frames
        progress_bar.progress(progress)
        
        timestamp = frame_idx / fps
        
        # Extract poses
        poses = pose_extractor.extract_poses_from_frame(frame, timestamp)
        
        # Track players
        tracked_poses = tracker.update(poses, frame_idx)
        
        # Classify moves
        for pose in tracked_poses:
            classification = move_classifier.classify_move(pose)
            
            if classification:
                # Generate commentary
                commentary = commentary_engine.generate_commentary(classification)
                if commentary:
                    commentary_events.append(commentary)
        
        # Annotate frame
        annotated_frame = pose_extractor.draw_poses_on_frame(frame, tracked_poses)
        
        # Add commentary overlay
        if commentary_events:
            latest = commentary_events[-1]
            cv2.putText(
                annotated_frame,
                latest.text,
                (50, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        
        out.write(annotated_frame)
        frame_idx += 1
        
        # Update status every 30 frames
        if frame_idx % 30 == 0:
            status_text.text(f"🎥 Processed {frame_idx}/{total_frames} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    pose_extractor.close()
    
    # Get fight statistics
    fight_stats = commentary_engine.get_fight_summary()
    
    status_text.text("✅ Processing complete!")
    
    return commentary_events, fight_stats, output_path


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    st.sidebar.markdown("### Detection Settings")
    detection_confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for pose detection"
    )
    
    st.sidebar.markdown("### Classifier Settings")
    classifier_mode = st.sidebar.selectbox(
        "Classifier Mode",
        options=["Mock (Rule-based)", "ML Model"],
        index=0,
        help="Use mock predictions or trained ML model"
    )
    
    st.sidebar.markdown("### Commentary Settings")
    commentary_speed = st.sidebar.slider(
        "Commentary Frequency",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Minimum seconds between commentary lines"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info(
        "**NeuroCombat** uses AI to analyze MMA fights in real-time, "
        "detecting fighter poses, classifying moves, and generating "
        "dynamic commentary."
    )
    
    return {
        "detection_confidence": detection_confidence,
        "classifier_mode": classifier_mode,
        "commentary_speed": commentary_speed,
    }


def render_main_content():
    """Render main content area."""
    # Header
    st.markdown('<h1 class="main-header">🥊 NeuroCombat 🥊</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #888;'>"
        "AI-Powered Real-Time MMA Fight Commentary System"
        "</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Video upload section
    st.markdown("## 📤 Upload Fight Video")
    uploaded_file = st.file_uploader(
        "Choose an MMA fight video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file of an MMA fight"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Preview original video
        st.markdown("### 🎬 Original Video")
        st.video(video_path)
        
        # Process button
        if st.button("🚀 Analyze Fight & Generate Commentary", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process video
                    commentary_events, fight_stats, output_path = process_video(
                        video_path,
                        progress_bar,
                        status_text,
                    )
                    
                    # Store in session state
                    st.session_state.commentary_events = commentary_events
                    st.session_state.fight_stats = fight_stats
                    st.session_state.processed_video = output_path
                    st.session_state.processing_complete = True
                    
                    st.success("🎉 Fight analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)
    
    # Display results if processing is complete
    if st.session_state.processing_complete:
        render_results()


def render_results():
    """Render analysis results."""
    st.markdown("---")
    st.markdown("## 📊 Fight Analysis Results")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎥 Analyzed Video")
        if st.session_state.processed_video:
            st.video(st.session_state.processed_video)
    
    with col2:
        st.markdown("### 📈 Fight Statistics")
        stats = st.session_state.fight_stats
        
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Total Commentary Lines", stats.get("total_commentary_lines", 0))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Player 1 Moves", stats.get("total_moves", {}).get(1, 0))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Player 2 Moves", stats.get("total_moves", {}).get(2, 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Commentary timeline
    st.markdown("### 💬 Live Commentary")
    
    if st.session_state.commentary_events:
        for event in st.session_state.commentary_events:
            timestamp_str = format_timestamp(event.timestamp)
            
            st.markdown(
                f'<div class="commentary-box">'
                f'<strong>[{timestamp_str}]</strong> {event.text}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No commentary generated. Try adjusting detection settings.")
    
    # Download options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Annotated Video"):
            st.info("Video download will be implemented in production version.")
    
    with col2:
        if st.button("📄 Download Commentary Transcript"):
            st.info("Commentary export will be implemented in production version.")


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Render sidebar
    config = render_sidebar()
    
    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
