# 🔬 CCTV Forensic Search: Dataset Analysis & Motion Detection Benchmark Report

**Project:** CCTV Forensic Search FYP  
**Dataset:** Snatch 1.0  
**Date:** July 13, 2026  

---

## 📹 Part 1: Dataset Exploration Summary

The **Snatch 1.0** dataset contains high-resolution and low-resolution video feeds representing standard CCTV footage (`Normal` directory) and targets containing snatch theft incidents (`Snatch Theft` directory).

### 📊 Summary Statistics
* **Total Video Files**: 42
* **Total File Size**: 1.76 GB (1801.74 MB)
* **Total Duration**: 3839.75 seconds (approx. 64 minutes)
* **Average Video Duration**: 91.42 seconds
* **Unique Resolutions**: 18 resolutions found (ranging from `198x360` up to `1920x1080`)
* **FPS Configurations**: 23.98, 24.0, 25.0, 29.97, 30.0 FPS
* **Codecs Found**: FMP4, h264
* **Integrity Status**: 100% healthy. 0 corrupted files, 0 duplicate files.

### 📂 Folder Distribution
* **Normal**: 7 videos (large background files, total ~87,500 frames)
* **Snatch Theft**: 35 videos (incident clips, total ~10,500 frames)

---

## 📈 Part 2: Motion Detection Benchmark Results

Every motion detection method was executed across all 42 videos in the dataset to measure computational speeds and forensic reduction performance.

### 📊 Overall Performance Averages
| Method | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Segment Length (Frames) | Avg Segment Count | Avg Motion Area (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.00% | 477.1 | 0.995 | 2282.2 | 1.0 | 100.000% |
| **FrameDifference** | **50.40%** | **335.6** | 0.608 | 281.2 | 21.4 | 6.588% |
| **GMM** | 34.87% | 39.3 | 0.848 | 1027.4 | 3.2 | 6.423% |
| **KNN** | 4.10% | 54.5 | 0.992 | 1866.5 | 2.2 | 28.308% |
| **MOG2** | 5.29% | 89.7 | 0.985 | 1790.6 | 3.3 | 22.328% |

---

## 🏷️ Part 3: Category-wise Analysis

### 1. Activity Class Averages (Normal vs Snatch Theft)
* **Normal Videos**: Characterized by long durations and continuous static backgrounds. FrameDifference achieved a **14.84% reduction**, while GMM showed a **5.69% reduction**.
* **Snatch Theft Videos**: Shorter durations with fast, localized motion. FrameDifference achieved **57.51% reduction**, while GMM achieved **40.70% reduction**.

### 2. Lighting Conditions Averages (Day vs Night)
* **Low Light / IR Sensor Noise**: In night scenes, high noise levels degrade background subtractor performance.
  * **FrameDifference**: Drops from **52.40%** (Day) to **24.49%** (Night) reduction.
  * **GMM**: Drops from **36.95%** (Day) to **7.81%** (Night) reduction.
  * **KNN & MOG2**: Fall to $\approx 0\%$ reduction due to continuous sensor flickering.

### 3. Camera Type Averages (Static vs Movement)
* **Camera Movement (Panning/Zooming)**: Shifted background structures cause GMM, KNN, and MOG2 to discard **0%** of frames.
* **FrameDifference**: Retains a **14.26%** reduction in moving camera feeds.

---

## 🔬 Scientific Conclusions

1. **Gating Subtractor Selection**:
   * **FrameDifference** is the most effective candidate for an upstream gating filter. It achieves a **50.40% overall frame reduction** while processing at **335.6 FPS** (far exceeding real-time requirements).
2. **Noise Gating**:
   * Environmental noise (rain, wind, low-light sensor noise) heavily affects pixel-based subtractors. Implementing a small area ratio gate (e.g. ignoring frames with $<0.5\%$ motion area ratio) will be crucial for retaining high reduction percentages under night settings.
