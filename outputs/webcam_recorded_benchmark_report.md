# 📊 Motion Detection Benchmarking Report (Same Recorded Video)

This report presents a performance and efficiency analysis of five motion detection algorithms evaluated on the **exact same 15-second recorded webcam video** (`outputs/webcam_recording.mp4`) over **438 frames**.

---

## 📈 Comparative Results Table

| Method | Total Frames | Motion Frames | Discarded Frames | Reduction % | Execution Time | Processing Speed (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏁 **Baseline (No Filtering)** | 438 | 438 | 0 | **0.0%** | **1.664s** | **263.2 FPS** |
| ⚡ **Frame Difference** | 438 | 41 | 397 | **90.6%** | **2.462s** | **177.9 FPS** |
| 🔄 **MOG2 (Background Subtractor)** | 438 | 148 | 290 | **66.2%** | **3.373s** | **129.9 FPS** |
| 👥 **KNN (Background Subtractor)** | 438 | 150 | 288 | **65.8%** | **3.794s** | **115.4 FPS** |
| 🌌 **GMM (Gaussian Mixture Model)** | 438 | 66 | 372 | **84.9%** | **4.560s** | **96.1 FPS** |

---

## 🔍 In-Depth Webcam Recording Analysis

Evaluating all five algorithms on the **exact same video file** ensures a completely fair and identical comparison, eliminating any bias from timing, motion, or lighting changes:

### 1. Performance and Workload Reduction
* **Frame Difference** achieved a phenomenal **90.6% frame reduction** (ignoring 397 of the 438 frames as static), processing at **177.9 FPS**.
* **GMM** performed exceptionally well with **84.9% reduction**, but ran at a lower speed of **96.1 FPS**.
* **MOG2 and KNN** both achieved around **66% reduction**, but were slower than Frame Difference.

### 2. Computational Speed
* **Frame Difference** remains the fastest option (**177.9 FPS**), offering high performance while introducing very little latency.
* **GMM** and **KNN** show significant execution overhead compared to the Frame Difference and Baseline methods.

---

## 💡 Conclusion

For real-time webcam deployment, the **Frame Difference Detector** remains the best overall pre-filter choice:
1. **Exceptional Workload Reduction**: Discards **90.6%** of static frames under normal webcam conditions, drastically reducing the load on the downstream YOLO and Depth estimation modules.
2. **High Speed**: Processes frames at **177.9 FPS**, leaving plenty of CPU/GPU headroom for the rest of the tracking pipeline.
