import json

# Load the pose data
with open('artifacts/test_dual_poses.json', 'r') as f:
    data = json.load(f)

stats = data['statistics']

print("=" * 70)
print("📊 ENHANCED DUAL DETECTION METRICS")
print("=" * 70)
print(f"\n🎯 Detection Performance:")
print(f"  Initial Detections (from detector):")
dist = stats.get('frame_pose_distribution', {})
print(f"    • 0 poses: {dist.get('0_poses', 0)} frames")
print(f"    • 1 pose:  {dist.get('1_pose', 0)} frames")
print(f"    • 2 poses: {dist.get('2_poses', 0)} frames ({dist.get('2_poses', 0)/stats['processed_frames']*100:.1f}%)")

print(f"\n  Final Tracked Output (after Hungarian matching):")
print(f"    • Dual detections: {stats.get('dual_detections', 0)} frames ({stats.get('detection_rate', 0):.1f}%)")
print(f"    • Single detections: {stats.get('single_detections', 0)} frames")
print(f"    • No detections: {stats.get('no_detections', 0)} frames")

print(f"\n🔧 Strategy Metrics:")
print(f"  • Spatial Split Usage: {stats.get('spatial_split_usage_rate', 'N/A')}%")
print(f"  • Avg Detection Confidence: {stats.get('avg_detection_confidence', 'N/A')}")

print(f"\n📊 Player Tracking Quality:")
print(f"  • Player 1 Avg Keypoints: {stats.get('avg_keypoints_p1', 0):.1f} / 33")
print(f"  • Player 2 Avg Keypoints: {stats.get('avg_keypoints_p2', 0):.1f} / 33")

print(f"\n📹 Video Info:")
print(f"  • Total Frames: {data['metadata']['total_frames']}")
print(f"  • Resolution: {data['metadata']['resolution']}")
print(f"  • FPS: {data['metadata']['fps']}")

print("=" * 70)
print("\n✅ ANALYSIS:")
print(f"  → Initial detection found 2 poses in {dist.get('2_poses', 0)/stats['processed_frames']*100:.1f}% of frames")
print(f"  → Tracking algorithm kept {stats.get('dual_detections', 0)/dist.get('2_poses', 1)*100:.1f}% of dual detections")
print(f"  → This suggests tracking algorithm is filtering low-confidence poses")
print("=" * 70)
