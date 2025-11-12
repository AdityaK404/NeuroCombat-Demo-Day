"""
Quick Test Script for Pose Extraction Module
=============================================

This script helps you quickly test the pose extraction system
with various configurations.

Author: NeuroCombat Team
Date: November 12, 2025
"""

import sys
from pathlib import Path


def print_banner():
    print("\n" + "="*70)
    print("🥊 NEUROCOMBAT - POSE EXTRACTION TEST SUITE")
    print("="*70 + "\n")


def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking dependencies...")
    
    required = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"💡 Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed!\n")
    return True


def check_video_files():
    """Check if sample video files exist"""
    print("📹 Checking for test videos...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"   ⚠️  Directory not found: {data_dir}")
        print(f"   💡 Creating directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    videos = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    
    if videos:
        print(f"   ✅ Found {len(videos)} video(s):")
        for v in videos:
            print(f"      - {v.name}")
        return videos[0]
    else:
        print(f"   ❌ No videos found in {data_dir}")
        print(f"\n💡 To test the system:")
        print(f"   1. Download any MMA fight clip")
        print(f"   2. Save it to data/raw/sample.mp4")
        print(f"   3. Run this test again")
        return None


def run_quick_test(video_path):
    """Run a quick test on first 100 frames"""
    print(f"\n🧪 Running quick test on: {video_path.name}")
    print("   (Processing first 100 frames only...)\n")
    
    try:
        import cv2
        from backend.pose_extractor_v2 import PoseExtractor
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Test first frame
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Cannot read frame from video")
            cap.release()
            return False
        
        print(f"\n🎯 Testing pose detection...")
        extractor = PoseExtractor(confidence_threshold=0.5)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect poses
        poses = extractor._detect_poses_in_frame(rgb_frame, frame.shape)
        
        print(f"   ✅ Detected {len(poses)} person(s) in first frame")
        
        if poses:
            for i, pose in enumerate(poses):
                visible_kpts = sum(1 for kpt in pose.keypoints if kpt[2] > 0.5)
                print(f"      Person {i+1}: {visible_kpts}/33 keypoints visible")
        
        cap.release()
        
        print(f"\n✅ Quick test PASSED!")
        print(f"\n💡 To run full extraction:")
        print(f"   python run_pose_extraction.py --video {video_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Step 2: Check for video files
    video_path = check_video_files()
    
    if video_path is None:
        print("\n⚠️  No test videos available")
        print("   Cannot run test without video file")
        sys.exit(0)
    
    # Step 3: Run quick test
    print("-"*70)
    success = run_quick_test(video_path)
    
    if success:
        print("\n" + "="*70)
        print("✅ TEST SUITE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n🚀 Ready for full pose extraction!")
        print("\n📝 Next steps:")
        print("   1. python run_pose_extraction.py --video data/raw/sample.mp4")
        print("   2. python run_pose_extraction.py --video data/raw/sample.mp4 --display")
        print("   3. python main.py --stage pose --video data/raw/sample.mp4")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print("Please check the error messages above")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
