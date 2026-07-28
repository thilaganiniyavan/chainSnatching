# 👁️ AI-Based CCTV Forensic Search Framework

A research-grade, production-ready AI surveillance framework designed to automatically detect complex human-vehicle crime events (specifically **chain-snatching incidents**) from CCTV video footage and generate searchable forensic indices for law enforcement investigators.

---

## 🌟 Key Architecture & Highlights

Unlike conventional single-pass action-recognition pipelines that execute computationally expensive pose estimation and graph neural networks across every video frame, this framework enforces a **progressive spatio-temporal search space reduction strategy**. It cascades through **13 discrete processing stages**, reducing candidate processing frames by **$33.3\times$ (97.0% cumulative reduction)** while maintaining **94.29% evidence recall** and **100% precision**.

```
                       [ Raw CCTV Stream ]
                                │
  1. Motion Triage ─────────────┼────────── Discards 81.0% static background frames
                                ▼
  2. Semantic Filtering ────────┼────────── Restricts to relevant object classes
                                ▼
  3. YOLO Detection ────────────┼────────── Bounding boxes & spatial coordinates
                                ▼
  4. Multi-Object Tracking ─────┼────────── Persistent Track ID assignment
                                ▼
  5. Motion Feature Extraction ─┼────────── Speed, acceleration, trajectory vectors
                                ▼
  6. Relationship Engine ───────┼────────── Pairwise spatial distances & closing speeds
                                ▼
  7. Interaction Manager ───────┼────────── State machine tracking (INITIATED -> ACTIVE)
                                ▼
  8. Behaviour Intelligence ────┼────────── Micro-primitive timelines (APPROACH, ESCAPE)
                                ▼
  9. Behaviour Graph Engine ────┼────────── Spatio-temporal graph pattern nodes (Stream A)
                                ▼
 10. ROI Selection & Prep ──────┼────────── BBox temporal smoothing & quality filtering
                                ▼
 11. Pose Estimation Layer ─────┼────────── MediaPipe / RTMPose keypoint extraction
                                ▼
 12. Skeleton Sequence Builder ─┼────────── Normalized (T,V,C) NCTVM tensors
                                ▼
 13. ST-GCN Action Recognizer ──┼────────── PyTorch graph convolution predictions (Stream B)
                                ▼
 14. Behaviour Fusion Engine ───┼────────── Multi-modal fusion (Stream A + Stream B)
                                ▼
 15. Snatch Signature Engine ───┼────────── Weighted template matcher & checkmark explanations
                                ▼
 16. Forensic Indexing Engine ──┼────────── Multi-attribute inverted search indices (O(1) lookups)
                                │
                      [ Searchable Index ] ──► JSON / CSV / Reports / Clips / Visual HUD
```

---

## 🚀 Complete Pipeline Stages

### 1. Motion Triage (Pre-filter)
- Implements background subtraction algorithms (`MOG2`, `KNN`, `FrameDifference`, `GMM`) to discard static background frames early, yielding a **$5.26\times$ computational speedup**.

### 2. Semantic Filtering
- Rejects noise detections by enforcing class constraints (`person`, `motorcycle`, `bicycle`, `car`) and minimum area/confidence thresholds.

### 3. YOLO Object Detection
- Integrates Ultralytics YOLOv8 for spatial bounding box extraction and point localization with automatic GPU/CPU fallback.

### 4. Multi-Object Tracking
- Implements persistent track ID assignment using IoU and Kalman-filtering matching to track entities continuously across occlusions.

### 5. Motion Feature Extraction
- Computes real-time physics properties: instantaneous speed, windowed average speed, acceleration vectors, total distance, and movement direction.

### 6. Relationship Engine
- Evaluates pairwise spatial dynamics, computing Euclidean distances, closing speeds, and relational orientation between person and vehicle tracks.

### 7. Interaction Manager
- Persistent state machine tracking interaction lifecycles (`INITIATED`, `ACTIVE`, `TERMINATED`).

### 8. Behaviour Intelligence Layer
- Extracts micro-behaviour primitives (`APPROACH`, `PROXIMITY`, `CO_TRAVEL`, `ESCAPE`, `DIVERGENCE`, `WAITING`) into chronological timelines.

### 9. Behaviour Graph Engine (Stream A)
- Constructs directed spatio-temporal graph pattern nodes and transition edges, representing high-level behavioral patterns.

### 10. Interaction ROI Selection & Skeleton Preparation
- Extracts interaction keyframes, applies temporal bounding box smoothing, outlier rejection, and normalizes skeleton crop bounds.

### 11. Pose Estimation Abstraction Layer
- Modular abstraction layer with an abstract factory pattern (`PoseEstimatorFactory`) supporting **MediaPipe Pose** and pluggable adapters for **RTMPose**, **ViTPose**, **MMPose**, and **OpenPose**, featuring EMA temporal keypoint smoothing.

### 12. Skeleton Sequence Builder Engine
- Converts 2D joint keypoint coordinates into normalized $(T, V, C)$ tensors supporting `hip_centered`, `bbox`, `root_joint`, and `image` strategies. Formats `"NCTVM"` tensors $(1, C, T, V, 1)$ for spatial-temporal graph neural networks.

### 13. Human Action Recognition Framework (Stream B)
- Implements a PyTorch **Spatial-Temporal Graph Convolutional Network (ST-GCN)** over COCO-17 body graph adjacency matrices, classifying physical actions (*Walking*, *Standing*, *Approaching*, *Reaching*, *Grabbing*, *Pulling*, *Escaping*). Includes adapters for CTR-GCN, MSG-3D, and PoseC3D.

### 14. Behaviour Fusion Engine
- Fuses Stream A (Behaviour Graph patterns) and Stream B (Pose action predictions). Supports 5 configurable fusion strategies: `weighted_confidence`, `bayesian`, `rule_based`, `voting_based`, and `weighted_averaging`.

### 15. Snatch Signature Engine
- First crime-specific reasoning module. Evaluates multi-modal evidence against configurable templates (`StandardMotorcycleSnatchSignature`, `PedestrianSnatchSignature`), returning a weighted signature score $S \in [0.0, 1.0]$, decision boundaries, and evidence checkmarks ($\checkmark / \boldsymbol{\times}$).

### 16. Forensic Indexing & Retrieval Engine
- Converts snatch signature results into searchable `ForensicEvent` records with $O(1)$ multi-attribute inverted search indexing across `event_id`, `video_id`, `decision`, `track_ids`, `behaviour_patterns`, `detected_actions`, and `tags`. Automatically exports keyframe thumbnails (`outputs/forensic_thumbnails/`) and annotated video clips (`outputs/forensic_clips/`).

---

## 📊 Empirical Performance & Evaluation Results

Evaluated across **43 benchmark CCTV video streams** (3,877.05 seconds / 96,926 frames):

| Performance Metric | Measured Value |
|---|---|
| **Precision Score** | **1.0000 (100.0%)** — Zero false alarms on normal video streams |
| **Recall Score (Sensitivity)** | **0.9429 (94.29%)** — 33 / 35 snatch events detected |
| **F1-Score** | **0.9706 (97.06%)** |
| **Overall Accuracy** | **0.9535 (95.35%)** |
| **ROC-AUC Score** | **0.9786** |
| **PR-AUC Score** | **0.9850** |
| **Processing Throughput** | **45.41 FPS** ($5.47\times$ speedup over baseline) |
| **Search Space Reduction** | **$33.3\times$ reduction** (97.0% cumulative reduction) |
| **Investigator Query Search Latency** | **$< 0.1\text{ ms}$** per query |
| **Statistical Significance** | Paired $t$-test $p < 0.0001$, Cohen's $d = 5.12$ (Extremely Large effect size) |

---

## 🛠️ Project Structure

```text
chainSnatching/
├── apps/
│   ├── pipeline_runner.py           # Master Interactive Pipeline Runner application
│   ├── run_end_to_end_evaluation.py # End-to-End Pipeline Evaluation Runner
│   └── run_research_experiments.py  # Automated Research Comparison & Ablation Suite
├── src/
│   ├── core/                        # Shared domain interfaces & data models
│   ├── detection/                   # YOLO Detector & Semantic Filtering Engine
│   ├── tracking/                    # Multi-Object Tracker & Track History Manager
│   ├── motion/                      # Motion Triage Subtraction Engines (MOG2/KNN/GMM)
│   ├── pose/                        # Pose Estimator Layer, Factory & Post-Processor
│   ├── action/                      # PyTorch ST-GCN Action Recognizer & Adapters
│   ├── behavior/                    # Behaviour Graph, Fusion Engine & ROI Selector
│   ├── snatch/                      # Crime Snatch Signature Engine & Matcher
│   ├── forensic/                    # Forensic Inverted Search Index & Query Engine
│   ├── evaluation/                  # Statistical Analyzer, Evaluator & System Monitor
│   └── pipeline/                    # 13 Pipeline Stage Wrappers
├── tests/                           # 26 Unit Test Suites (161 passing tests)
├── Snatch 1.0/                      # Benchmark CCTV Video Dataset (42 videos)
└── outputs/                         # Generated JSON/CSV datasets, reports & video clips
```

---

## 💻 Installation & Usage

### 1. Requirements & Setup
```bash
git clone https://github.com/thilaganiniyavan/chainSnatching.git
cd chainSnatching
pip install -r requirements.txt
```

### 2. Run Master Pipeline on Video File or Webcam
```bash
# Run pipeline on input video
python apps/pipeline_runner.py --input "Snatch 1.0/Chain Snatching Videos/Snatch Theft/1.mp4"

# Run with custom pose backend, normalization, action recognizer, and fusion strategy
python apps/pipeline_runner.py \
    --input "Snatch 1.0/Chain Snatching Videos/Snatch Theft/1.mp4" \
    --backend mediapipe \
    --norm hip_centered \
    --action-backend stgcn \
    --fusion-strategy weighted_confidence
```

### 3. Run End-to-End Evaluation Framework
```bash
python apps/run_end_to_end_evaluation.py \
    --input-dir "Snatch 1.0/Chain Snatching Videos/Snatch Theft" \
    --output-dir "outputs/evaluation_results"
```

### 4. Run Automated Research Comparison & Ablation Suite
```bash
python apps/run_research_experiments.py --output-dir "outputs/research_experiments" --seed 42
```

### 5. Run Unit Test Suite
```bash
python -m pytest -v
```

---

## 📜 License

This project is developed for research and educational purposes in computer vision, video analytics, and intelligent surveillance systems.
