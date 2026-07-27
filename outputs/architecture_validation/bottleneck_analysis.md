# ⏱️ Pipeline Bottleneck & Latency Analysis Report

| Pipeline Stage | Total Runtime (s) | Runtime % | Avg Latency (ms/frame) | Peak Latency (ms/frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Motion Filtering** | `31.1461s` | `10.90%` | `13.99 ms` | `72.00 ms` |
| **YOLO Detection** | `118.7951s` | `41.57%` | `54.82 ms` | `1943.19 ms` |
| **Tracking Stage** | `135.3338s` | `47.36%` | `68.21 ms` | `152.97 ms` |
| **Relationship Engine** | `0.4648s` | `0.16%` | `0.23 ms` | `22.00 ms` |
| **Candidate Events** | `0.0000s` | `0.00%` | `0.00 ms` | `0.00 ms` |
