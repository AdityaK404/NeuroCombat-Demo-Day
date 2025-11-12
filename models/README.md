# Model Directory

This directory stores trained models and weights for NeuroCombat.

## Model Files

### 1. Move Classifier (`move_classifier.pkl`)
- **Type**: Scikit-learn classifier (RandomForest/SVM/LSTM)
- **Input**: Temporal pose features (window_size × feature_dim)
- **Output**: 6 move classes (neutral, jab, cross, uppercut, front_kick, roundhouse_kick)
- **Training**: Train on Harmony4D + local MMA dataset

### 2. Pose Model (Optional - `pose_model.onnx`)
- **Type**: Custom pose estimation model
- **Default**: Uses MediaPipe (no file needed)
- **Alternative**: Can use custom ONNX models for better accuracy

## Training Instructions

To train the move classifier:

```bash
# TODO: Add training script
python train_classifier.py --data data/processed/ --output models/
```

## Model Performance Targets

| Metric | Target | Current (Mock) |
|--------|--------|----------------|
| Accuracy | >85% | ~70% (heuristic) |
| Latency | <50ms | ~30ms |
| FPS | >20 | ~15-20 |

## Placeholder

The system currently uses **mock predictions** based on simple heuristics. 
For production, train actual models using the provided datasets.
