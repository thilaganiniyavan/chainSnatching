# 🔬 CCTV Motion Benchmarking Statistical Summary

This report aggregates motion detection benchmark metrics across the entire **Snatch 1.0** dataset.

## 📊 Overall Performance Metrics

| Method | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Segment Length (Frames) | Avg Segment Count | Avg Motion Area (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.00% | 477.1 | 0.995 | 2282.2 | 1.0 | 100.000% |
| **FrameDifference** | 50.40% | 335.6 | 0.608 | 281.2 | 21.4 | 6.588% |
| **GMM** | 34.87% | 39.3 | 0.848 | 1027.4 | 3.2 | 6.423% |
| **KNN** | 4.10% | 54.5 | 0.992 | 1866.5 | 2.2 | 28.308% |
| **MOG2** | 5.29% | 89.7 | 0.985 | 1790.6 | 3.3 | 22.328% |

## 🏷️ Category-wise Performance Metrics

### 1. Activity Class Averages

| Method | Folder Class | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Motion Area (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline | `Normal` | 0.00% | 252.4 | 1.000 | 100.000% |
| FrameDifference | `Normal` | 14.84% | 127.6 | 0.993 | 5.482% |
| GMM | `Normal` | 5.69% | 33.0 | 0.999 | 4.852% |
| KNN | `Normal` | 1.85% | 38.7 | 1.000 | 21.057% |
| MOG2 | `Normal` | 3.30% | 52.2 | 0.999 | 17.317% |
| Baseline | `Snatch Theft` | 0.00% | 522.1 | 0.993 | 100.000% |
| FrameDifference | `Snatch Theft` | 57.51% | 377.1 | 0.530 | 6.810% |
| GMM | `Snatch Theft` | 40.70% | 40.5 | 0.818 | 6.737% |
| KNN | `Snatch Theft` | 4.55% | 57.6 | 0.990 | 29.758% |
| MOG2 | `Snatch Theft` | 5.68% | 97.2 | 0.982 | 23.330% |

### 2. Lighting Conditions Averages

| Method | Lighting Type | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Motion Area (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline | `Day` | 0.00% | 495.3 | 0.994 | 100.000% |
| FrameDifference | `Day` | 52.40% | 350.3 | 0.586 | 6.810% |
| GMM | `Day` | 36.95% | 39.8 | 0.837 | 6.665% |
| KNN | `Day` | 4.41% | 55.1 | 0.991 | 29.320% |
| MOG2 | `Day` | 5.67% | 92.1 | 0.984 | 23.117% |
| Baseline | `Night` | 0.00% | 240.6 | 0.999 | 100.000% |
| FrameDifference | `Night` | 24.49% | 143.6 | 0.889 | 3.702% |
| GMM | `Night` | 7.81% | 32.0 | 0.997 | 3.285% |
| KNN | `Night` | 0.00% | 46.2 | 0.999 | 15.158% |
| MOG2 | `Night` | 0.26% | 58.3 | 0.997 | 12.079% |

### 3. Camera Motion Averages

| Method | Camera Type | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Motion Area (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline | `Movement` | 0.00% | 156.4 | 0.995 | 100.000% |
| FrameDifference | `Movement` | 14.26% | 107.7 | 0.933 | 16.412% |
| GMM | `Movement` | 0.90% | 8.7 | 0.995 | 11.643% |
| KNN | `Movement` | 0.00% | 15.1 | 0.995 | 69.231% |
| MOG2 | `Movement` | 0.00% | 31.9 | 0.995 | 50.078% |
| Baseline | `Static` | 0.00% | 501.8 | 0.994 | 100.000% |
| FrameDifference | `Static` | 53.18% | 353.1 | 0.583 | 5.833% |
| GMM | `Static` | 37.48% | 41.6 | 0.837 | 6.022% |
| KNN | `Static` | 4.41% | 57.5 | 0.992 | 25.160% |
| MOG2 | `Static` | 5.69% | 94.1 | 0.984 | 20.194% |
