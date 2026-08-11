# AI-Based CCTV Forensic Search Framework: Research Comparison Plan

## Executive Summary
This document outlines the systematic research comparison study for the 13-stage progressive spatio-temporal AI surveillance framework. The objective is to evaluate alternative techniques at each architectural level to definitively answer:

**"Why is this particular progressive multi-modal architecture preferable to alternative approaches for CCTV chain-snatching detection and forensic retrieval?"**

## Execution Strategy & Feasibility Analysis

Before executing the massive batch of experiments, we have inspected the existing research infrastructure to guarantee scientific fairness and computational feasibility.

### 1. Existing Comparisons
*   **Motion Triage (Level 0):** Already completed. The baseline, Frame Difference, MOG2, KNN, and GMM algorithms have been evaluated. These results will be incorporated into the final report rather than re-executed.

### 2. Executable Models vs. Scaffolds
*   **Detection:** YOLO11n is currently active. YOLOv8n can be compared fairly using the Ultralytics engine. 
*   **Tracking:** Ultralytics natively supports BoT-SORT and ByteTrack. We can compare them directly.
*   **Relationship Engine:** The current implementation uses centroid Euclidean distance with a 150-pixel threshold. We can evaluate distance thresholds and bounding-box edge distances.
*   **Pose Estimation:** `MediaPipe` is fully executable. `MMPose`, `RTMPose`, `OpenPose`, and `ViTPose` are implemented as adapter scaffolds (falling back to MediaPipe). **Decision:** We will strictly evaluate the executable MediaPipe and document the others as "NOT EVALUATED (Scaffold Adapter)".
*   **Action Recognition:** `ST-GCN` is fully executable. `CTR-GCN`, `MSG3D`, and `PoseC3D` are implemented as adapter scaffolds. **Decision:** We will strictly evaluate ST-GCN and document the others as "NOT EVALUATED (Scaffold Adapter)".

### 3. Datasets & Annotations
*   The `Snatch 1.0` benchmark dataset (42 videos) will be used for all live evaluations to ensure the input data, ground truth, and metrics remain strictly identical across all mechanism comparisons.

### 4. Fairness Constraint
*   Every experiment will use the identical dataset, random seed (42), and metrics.
*   We will NOT modify the production architecture. A separate, isolated experimental pipeline will be constructed or the existing `ResearchAblationEngine` will be extended to test mechanism-specific variations safely.

---

## Experiment Matrix

| Level | Component | Candidate Methods to Evaluate | Notes |
| :--- | :--- | :--- | :--- |
| **1** | Object Detection | YOLO11 (Current), YOLOv8 | Evaluate on Ultralytics backend. Focus on Person/Vehicle classes. |
| **2** | Tracking | ByteTrack, BoT-SORT | Evaluate using persistent IDs on YOLO detections. |
| **3** | Relationship | Centroid Distance, BBox Edge Distance, Normalized Distance | Evaluate threshold sensitivity (100px vs 150px vs 200px). |
| **4** | Behaviour Rep. | Rule-based, Timeline, Graph (Current) | Compare graph transitions vs independent rules. |
| **5** | ROI Selection | Full-frame, Interaction ROI, Expanded ROI | Evaluate frame reduction vs. evidence preservation. |
| **6** | Pose Estimation | MediaPipe | Others marked NOT EVALUATED (scaffolds). |
| **7** | Normalization | Hip-centered, BBox, Root-joint | Keep MediaPipe and ST-GCN fixed. |
| **8** | Action Recog. | ST-GCN | Others marked NOT EVALUATED (scaffolds). |
| **9** | Fusion | Weighted Confidence, Bayesian, Rule-based | Measure downstream snatch detection F1. |
| **10** | Snatch Signature | Weighted Evidence, Temporal Sequence, Baseline | Focus on forensic event recall & explainability. |
| **11** | Forensic Indexing | Linear Scan, Inverted Index | Scalability via simulated index expansion. |
| **12** | Architecture | Configs A, B, C, D | Final multi-modal framework comparison. |

---

## Phased Execution Plan

1.  **Phase 1: Perception Stack (Levels 1-2)**
    *   Compare Object Detectors and Trackers.
    *   *Validation:* Ensure tracking IDs persist through occlusions.
2.  **Phase 2: Spatial & Behavioural Logic (Levels 3-5)**
    *   Compare Relationship metrics, Behaviour Graph, and ROI Selection.
    *   *Validation:* Ensure no true snatch events are incorrectly filtered out.
3.  **Phase 3: Action & Modalities (Levels 6-9)**
    *   Evaluate Normalization strategies and Fusion methods.
    *   *Validation:* Document exact F1 impacts.
4.  **Phase 4: Reasoning & Indexing (Levels 10-12)**
    *   Compare Signature logic, Indexing scalability, and full Architectures.
    *   *Validation:* Generate final master research table and report.

Each phase will output its respective CSV, report, and plots in the `outputs/research_comparison/` directory before moving to the next.
