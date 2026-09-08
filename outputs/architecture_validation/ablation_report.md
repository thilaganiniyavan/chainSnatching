# 🧪 Pipeline Architecture Ablation Study Report

| Configuration | Runtime (s) | Processed Frames | Retained Candidate Frames | Total Detections | Total Tracks | Candidate Events | Cost Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A (YOLO Only)** | `13.35s` | `1500` | `750` | `5801` | `0` | `0` | `8.01` |
| **Config B (Motion + YOLO)** | `77.34s` | `1500` | `1500` | `11671` | `0` | `0` | `85.07` |
| **Config C (Motion + YOLO + Tracking)** | `185.06s` | `1500` | `1500` | `11671` | `7498` | `0` | `203.57` |
| **Config D (Full Pipeline)** | `557.58s` | `1500` | `1011` | `11671` | `7498` | `2913` | `431.57` |
