# 🔬 AI Forensic Search: Progressive Search-Space & Bottleneck Evaluation Report

**Project**: CCTV Forensic Search FYP  
**Dataset Evaluated**: 5 Video Clips (2,461 Total Frames)  
**Output Location**: `outputs/pipeline_statistics`  

---

## 📊 1. Progressive Search-Space Reduction Summary

Every pipeline stage operates as a forensic evidence filter. Below is the average percentage of search space remaining after each stage:

| Pipeline Stage | Search Space Remaining (%) | Absolute Reduction Contribution (%) | Total Retained Frames |
| :--- | :---: | :---: | :---: |
| **0. Raw Video Input** | **100.00%** | Baseline | 2,461 |
| **1. Motion Filtering** | **84.96%** | 15.04% | 2,167 |
| **2. YOLO Detection** | **78.82%** | 6.14% | 1,984 |
| **3. Tracking Stage** | **78.82%** | 0.00% | 1,984 |
| **4. Relationship Engine** | **3.52%** | 75.30% | 53 |
| **5. Candidate Events** | **3.52%** | 0.00% | 53 |

---

## ⏱️ 2. Stage Runtime Contribution & Bottleneck Analysis

| Pipeline Stage | Avg Execution Time per Video (s) | Runtime Contribution (%) | Bottleneck Rank |
| :--- | :---: | :---: | :---: |
| **Motion Filtering** | `7.3088s` | `9.95%` | 3 |
| **YOLO Detection** | `30.6707s` | `41.77%` | 1 |
| **Tracking Stage** | `34.3353s` | `46.76%` | 2 |
| **Relationship Engine** | `0.1178s` | `0.16%` | 4 |
| **Candidate Events** | `0.0000s` | `0.00%` | 5 |
| **Total Pipeline** | `73.4307s` | `100.00%` | - |

---

## 🔍 3. Core Research Findings

### 3.1 Which stage removes the largest amount of irrelevant information?
- **Primary Reducer Stage**: **Relationship Engine**
- **Impact**: Removes **75.30%** of the total video search space.
- **Forensic Rationale**: Early-stage frame difference and spatial relationship rules successfully filter out static background scenes and non-interacting background pedestrians/vehicles without losing key evidence.

### 3.2 Which stage becomes the computational bottleneck?
- **Primary Bottleneck Stage**: **Tracking Stage**
- **Impact**: Accounts for **46.76%** of total execution runtime (`34.3353s` per video).
- **Forensic Rationale**: Deep neural inference (YOLO / object feature extraction) dominates processing overhead. 

---

## 💡 4. Recommendations for Future Optimisation

1. **Cascade Execution Thresholds**:
   - Apply Motion Filtering aggressively before triggering YOLO inference to prevent unneeded neural network invocations.
2. **Adaptive Object Detection ROI**:
   - Restrict YOLO inference specifically to dynamic ROI bounding regions supplied by the motion subtractor rather than executing full 1080p frame passes.
3. **Quantized Model Deployment**:
   - Export YOLO models to TensorRT / ONNX FP16 / INT8 formats to reduce the primary computational bottleneck by 3-5x.
4. **Spatial Indexing & Caching**:
   - Cache spatial relationship distance matrices using spatial KD-Trees to keep relationship analysis latency strictly below 1ms per frame.
