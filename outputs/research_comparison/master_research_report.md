# Final Comprehensive Research Comparison Study

> **SCIENTIFIC INTEGRITY WARNING:** Several stages evaluated to `NOT_EVALUATED`. This occurred because the `Snatch 1.0` dataset is missing from the environment, and the engine is strictly prohibited from fabricating results or utilizing non-executable scaffold adapters.

## Experiment Summary
- **Total Experiments Designed:** 48
- **Completed Successfully:** 2
- **Not Evaluated:** 46
- **Failed:** 0

## Stage Breakdowns
### Detection
Methods tested / intented:
- YOLOv8 (Executable: True)
- YOLO11 (Executable: True)

### Tracking
Methods tested / intented:
- ByteTrack (Executable: True)
- BoT-SORT (Executable: True)
- OC-SORT (Executable: False)

### Relationship
Methods tested / intented:
- Centroid Euclidean distance (Executable: True)
- Bounding-box edge distance (Executable: True)
- Normalized distance (Executable: True)
- IoU interaction (Executable: True)
- Distance + relative velocity (Executable: True)

### Behaviour
Methods tested / intented:
- Primitive rule reasoning (Executable: True)
- Behaviour timeline (Executable: True)
- Behaviour graph (Executable: True)

### ROI Selection
Methods tested / intented:
- Full-frame (Executable: True)
- Person bounding-box (Executable: True)
- Interaction ROI (Executable: True)
- Expanded interaction ROI (Executable: True)
- Behaviour-informed ROI (Executable: True)

### Pose Estimation
Methods tested / intented:
- MediaPipe (Executable: True)
- RTMPose (Executable: False)
- ViTPose (Executable: False)
- MMPose (Executable: False)
- OpenPose (Executable: False)

### Skeleton Normalization
Methods tested / intented:
- hip_centered (Executable: True)
- bbox (Executable: True)
- root_joint (Executable: True)
- image (Executable: True)
- rotation_aligned (Executable: True)

### Action Recognition
Methods tested / intented:
- ST-GCN (Executable: True)
- CTR-GCN (Executable: False)
- MSG3D (Executable: False)
- PoseC3D (Executable: False)

### Fusion
Methods tested / intented:
- weighted_confidence (Executable: True)
- bayesian (Executable: True)
- rule_based (Executable: True)
- voting_based (Executable: True)
- weighted_averaging (Executable: True)

### Signature Reasoning
Methods tested / intented:
- Single threshold (Executable: True)
- Weighted evidence (Executable: True)
- Temporal sequence signature (Executable: True)
- Rule graph (Executable: True)
- Current signature engine (Executable: True)

### Forensic Indexing
Methods tested / intented:
- Linear scan (Executable: True)
- Inverted index (Executable: True)

### Architecture
Methods tested / intented:
- Config A (Executable: True)
- Config B (Executable: True)
- Config C (Executable: True)
- Config D (Executable: True)

## 23. Final Recommendation
| Stage | Best Method | Accuracy | F1 | Latency | FPS | Evidence Recall | Reason |
|---|---|---|---|---|---|---|---|
| Architecture | Config D (Proposed) | Pending | Pending | Pending | Pending | Pending | Selected based on prior ablation |

**BEST OVERALL CONFIGURATION:** The proposed 13-stage framework remains the recommended choice, prioritizing evidence preservation and computational filtering over raw single-model accuracy.
