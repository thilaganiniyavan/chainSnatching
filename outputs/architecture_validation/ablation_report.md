# 🧪 Pipeline Architecture Ablation Study Report

| Configuration | Runtime (s) | Processed Frames | Retained Candidate Frames | Total Detections | Total Tracks | Candidate Events | Cost Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A (YOLO Only)** | `36.51s` | `2227` | `1021` | `2569` | `0` | `0` | `20.39` |
| **Config B (Motion + YOLO)** | `95.31s` | `2227` | `1984` | `4947` | `0` | `0` | `94.44` |
| **Config C (Motion + YOLO + Tracking)** | `196.0s` | `2227` | `1984` | `4947` | `4022` | `0` | `194.22` |
| **Config D (Full Pipeline)** | `187.4s` | `2227` | `53` | `4947` | `4022` | `67` | `23.2` |
