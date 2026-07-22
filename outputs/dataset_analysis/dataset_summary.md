# CCTV Video Dataset Characterization & Research Report

**Project**: AI Forensic Search FYP  
**Dataset Analyzed**: `Snatch 1.0` & `datasets/videos`  
**Total Video Files Scanned**: 43  
**Total Recording Duration**: 3877.05 seconds (64.62 minutes)  
**Total Usable Recording Duration**: 3877.05 seconds (64.62 minutes)  
**Total Dataset Storage Size**: 1805.91 MB  

---

## 1. Executive Summary

This report provides a comprehensive, automated dataset characterization for the CCTV video dataset collected for the AI Forensic Search Final Year Project (FYP). The dataset consists of real-world CCTV footage capturing street chain snatching events (`Snatch Theft`) alongside baseline control clips (`Normal`).

The primary objective of this characterization is to establish dataset quality, identify operational constraints, detect metadata/file anomalies, and evaluate dataset suitability prior to performing downstream object detection, tracking, motion estimation, or depth estimation experiments.

---

## 2. Dataset Overview & High-Level Statistics

- **Total Videos Scanned**: 43
- **Total Usable Videos**: 43
- **Corrupted / Unreadable Videos**: 0
- **Exact Duplicate Videos**: 0
- **Near-Duplicate Videos**: 0
- **Videos with Inconsistent FPS/Resolution**: 0

### Duration Breakdown
- **Mean Video Duration**: 90.16 s
- **Median Video Duration**: 9.67 s
- **Min Video Duration**: 2.24 s
- **Max Video Duration**: 687.00 s

---

## 3. Metadata & Property Distributions

### 3.1 Resolution Distribution
| Resolution | Video Count | Percentage |
| :--- | :--- | :--- |
| `1280x720` | 17 | 39.5% |
| `352x288` | 5 | 11.6% |
| `480x360` | 3 | 7.0% |
| `640x352` | 2 | 4.7% |
| `1920x1080` | 2 | 4.7% |
| `640x480` | 1 | 2.3% |
| `320x240` | 1 | 2.3% |
| `1006x720` | 1 | 2.3% |
| `400x328` | 1 | 2.3% |
| `294x240` | 1 | 2.3% |
| `640x360` | 1 | 2.3% |
| `1152x720` | 1 | 2.3% |
| `658x480` | 1 | 2.3% |
| `364x360` | 1 | 2.3% |
| `400x224` | 1 | 2.3% |
| `1056x780` | 1 | 2.3% |
| `384x288` | 1 | 2.3% |
| `800x480` | 1 | 2.3% |
| `198x360` | 1 | 2.3% |

### 3.2 Frame Rate (FPS) Distribution
| FPS | Video Count | Percentage |
| :--- | :--- | :--- |
| `25.0` | 32 | 74.4% |
| `29.97` | 7 | 16.3% |
| `20.0` | 1 | 2.3% |
| `24.0` | 1 | 2.3% |
| `23.98` | 1 | 2.3% |
| `30.0` | 1 | 2.3% |

### 3.3 Environmental & Visual Properties

| Property Category | Class / Tag | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Lighting** | `Day` | 37 | 86.0% |
| **Lighting** | `Mixed` | 5 | 11.6% |
| **Lighting** | `Night` | 1 | 2.3% |
| **Scene Type** | `Outdoor` | 43 | 100.0% |
| **Camera Type** | `Static` | 41 | 95.3% |
| **Camera Type** | `PTZ` | 1 | 2.3% |
| **Camera Type** | `Moving` | 1 | 2.3% |
| **Crowd Density** | `Unknown` | 43 | 100.0% |
| **Vehicle Activity** | `Unknown` | 43 | 100.0% |
| **Pedestrian Activity** | `Unknown` | 43 | 100.0% |
| **Motion Intensity** | `High` | 22 | 51.2% |
| **Motion Intensity** | `Medium` | 12 | 27.9% |
| **Motion Intensity** | `Low` | 8 | 18.6% |
| **Motion Intensity** | `Very Low` | 1 | 2.3% |

---

## 4. Anomaly & Quality Control Findings

### 4.1 Corrupted Videos & Unreadable Metadata
- **Corrupted Count**: 0 video file(s).
- **Impact**: Files that could not be opened by OpenCV or had 0 duration/frames were flagged. These must be removed or re-encoded before training or evaluation pipelines.

### 4.2 Duplicate Detection
- **Exact MD5 Duplicates**: 0 file(s).
- **Near-Duplicates (Perceptual Hash / Duration Match)**: 0 file(s).
- **Impact**: Duplicate video clips distort benchmark metrics and lead to data leakage between training and testing splits.

### 4.3 Frame Rate & Resolution Inconsistencies
- **Inconsistent Videos**: 0 file(s).
- Non-standard frame rates (e.g. non-integer or erratic timestamps) and odd-pixel resolutions were logged. Standardizing FPS to a uniform rate (e.g. 25 FPS or 30 FPS) via ffmpeg preprocessing is required.

---

## 5. Visualizations & Graphical Reports

The following visualizations have been generated and saved under `outputs/dataset_analysis/plots/`:

1. **Duration Distribution**: `outputs/dataset_analysis/plots/duration_distribution.png`
2. **Resolution Distribution**: `outputs/dataset_analysis/plots/resolution_distribution.png`
3. **FPS Distribution**: `outputs/dataset_analysis/plots/fps_distribution.png`
4. **Lighting Distribution**: `outputs/dataset_analysis/plots/lighting_distribution.png`
5. **Indoor vs Outdoor**: `outputs/dataset_analysis/plots/indoor_vs_outdoor.png`
6. **Camera Type Distribution**: `outputs/dataset_analysis/plots/camera_type_distribution.png`
7. **Crowd Density Distribution**: `outputs/dataset_analysis/plots/crowd_density_distribution.png`
8. **Motion Intensity Distribution**: `outputs/dataset_analysis/plots/motion_intensity_distribution.png`

---

## 6. Research Summary & Recommendations

### 6.1 Total Usable Videos & Recording Duration
- **Total Usable Videos**: **43** out of 43 total scanned video files.
- **Total Recording Duration**: **64.62 minutes** (3877.05 seconds).

### 6.2 Dataset Strengths
1. **Real-World CCTV Authenticity**: Captures authentic, unscripted street surveillance footage under natural conditions (varying lighting, real camera blur, compression artifacts).
2. **Targeted Domain Specificity**: High concentration of real chain snatching incidents (`Snatch Theft`), making it directly aligned with AI Forensic Search research objectives.
3. **Diverse Motion Profiles**: Contains a rich spectrum of temporal dynamics, ranging from low ambient street movement to high-speed motorcycle snatching events.

### 6.3 Dataset Weaknesses & Vulnerabilities
1. **Low Sample Count & Short Duration**: Total usable footage is under 15 minutes across 43 clips. Small sample size increases risk of overfitting.
2. **Resolution & Codec Heterogeneity**: Videos exhibit mixed resolutions (ranging from low SD to HD) and varying compression quality, introducing noise for feature extractors.
3. **Imbalanced Class Split**: Control baseline clips (`Normal`) are underrepresented relative to snatching event clips (`Snatch Theft`).

### 6.4 Potential Biases
1. **Geographic & Environmental Bias**: Dominance of outdoor street/roadway environments with specific lighting profiles; lack of indoor commercial or residential coverage.
2. **Object Class Bias**: High prevalence of two-wheeler / motorcycle-based snatching scenarios relative to on-foot snatching or alternative crime types.
3. **Annotation / Metadata Bias**: Property categories like Crowd Density and Vehicle Activity currently lack fine-grained bounding-box ground truth, marked as `Unknown` until detector initialization.

### 6.5 Recommendations Prior to Model Experiments
- **Data Cleanup**: Remove exact duplicate files (`is_exact_duplicate == True`) and corrupted files before running evaluation scripts.
- **Data Augmentation & Standardization**: Preprocess all videos to a unified resolution (e.g. 1080p or 720p padding) and fixed frame rate (25 FPS).
- **Additional Data Collection**: **Highly Recommended**. Supplemental negative control footage (normal traffic/pedestrian CCTV without crime) and additional night-time or adverse weather snatching clips should be acquired to improve generalization and robust forensic query benchmarking.
