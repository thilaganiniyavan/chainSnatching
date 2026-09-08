# ⏱️ Pipeline Bottleneck & Latency Analysis Report

| Pipeline Stage | Total Runtime (s) | Runtime % | Avg Latency (ms/frame) | Peak Latency (ms/frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Motion Filtering** | `40.1644s` | `34.79%` | `26.78 ms` | `61.65 ms` |
| **YOLO Detection** | `28.5180s` | `24.71%` | `19.01 ms` | `2204.27 ms` |
| **Tracking Stage** | `46.3931s` | `40.19%` | `30.93 ms` | `133.54 ms` |
| **Relationship Engine** | `0.3561s` | `0.31%` | `0.24 ms` | `10.41 ms` |
| **Candidate Events** | `0.0000s` | `0.00%` | `0.00 ms` | `0.00 ms` |
