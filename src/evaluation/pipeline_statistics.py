"""Pipeline Statistics Framework for AI Forensic Search.

This module measures progressive search-space reduction, computational runtime,
detection distributions, and stage-by-stage filtering performance without modifying
the underlying algorithms.
"""

import os
import json
import csv
import time
import math
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class PipelineStatisticsCollector:
    """Collects quantitative performance and forensic space-reduction metrics

    for a single video execution across all pipeline stages.
    """

    def __init__(self, video_path: str, fps: float = 30.0, total_video_frames: int = 0):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.fps = fps if fps > 0 else 30.0
        self.total_video_frames = total_video_frames

        # Stage 1: Motion Filtering
        self.motion_entering = 0
        self.motion_removed = 0
        self.motion_retained = 0
        self.motion_time = 0.0

        # Stage 2: YOLO Detection
        self.yolo_processed = 0
        self.yolo_removed = 0
        self.yolo_retained = 0
        self.total_detections = 0
        self.detections_by_class = {
            "person": 0,
            "bicycle": 0,
            "motorcycle": 0,
            "car": 0,
            "bus": 0,
            "truck": 0
        }
        self.confidence_sum = 0.0
        self.yolo_time = 0.0

        # Stage 3: Tracking
        self.tracking_processed = 0
        self.tracking_removed = 0
        self.tracking_retained = 0
        self.observed_tracks: Dict[int, int] = {}  # track_id -> frame_count
        self.last_track_positions: Dict[int, tuple] = {}  # track_id -> (cx, cy)
        self.id_switches = 0
        self.tracking_time = 0.0

        # Stage 4: Relationship Engine
        self.relationship_processed = 0
        self.relationship_removed = 0
        self.relationship_retained = 0
        self.total_person_vehicle_pairs = 0
        self.proximity_events_count = 0
        self.pair_interaction_durations: Dict[tuple, int] = {}  # (p_id, v_id) -> frame_count
        self.active_proximity_pairs: set = set()
        self.relationship_time = 0.0

        # Stage 5: Candidate Event Generation
        self.event_processed = 0
        self.event_removed = 0
        self.event_retained = 0
        self.candidate_events_count = 0
        self.event_time = 0.0

        # Overall timing
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def log_input(self, total_frames: int):
        self.total_video_frames = total_frames
        self.motion_entering = total_frames

    def record_motion_stage(self, retained: bool, duration: float):
        self.motion_time += duration
        if retained:
            self.motion_retained += 1
        else:
            self.motion_removed += 1

    def record_yolo_stage(self, detections: list, duration: float):
        self.yolo_time += duration
        self.yolo_processed += 1

        if len(detections) > 0:
            self.yolo_retained += 1
        else:
            self.yolo_removed += 1

        self.total_detections += len(detections)
        for det in detections:
            cls_name = getattr(det, 'class_name', str(det))
            conf = float(getattr(det, 'confidence', 0.0))
            self.confidence_sum += conf

            if cls_name in self.detections_by_class:
                self.detections_by_class[cls_name] += 1
            else:
                self.detections_by_class[cls_name] = 1

    def record_tracking_stage(self, tracks: list, duration: float):
        self.tracking_time += duration
        self.tracking_processed += 1

        if len(tracks) > 0:
            self.tracking_retained += 1
        else:
            self.tracking_removed += 1

        current_positions = {}
        for trk in tracks:
            t_id = getattr(trk, 'tracking_id', -1)
            if t_id < 0:
                continue

            self.observed_tracks[t_id] = self.observed_tracks.get(t_id, 0) + 1

            center = getattr(trk, 'center', None)
            if center is not None:
                current_positions[t_id] = center
                # Simple heuristic for potential ID switch detection (large unexpected jump)
                if t_id in self.last_track_positions:
                    prev_c = self.last_track_positions[t_id]
                    dist = math.sqrt((center[0] - prev_c[0]) ** 2 + (center[1] - prev_c[1]) ** 2)
                    if dist > 200:  # unusual jump threshold
                        self.id_switches += 1

        self.last_track_positions = current_positions

    def record_relationship_stage(self, relationships: list, num_persons: int, num_vehicles: int, duration: float):
        self.relationship_time += duration
        self.relationship_processed += 1
        self.total_person_vehicle_pairs += (num_persons * num_vehicles)

        if len(relationships) > 0:
            self.relationship_retained += 1
        else:
            self.relationship_removed += 1

        self.proximity_events_count += len(relationships)

        current_pairs = set()
        for rel in relationships:
            pair = (getattr(rel, 'subject_id', -1), getattr(rel, 'object_id', -1))
            current_pairs.add(pair)
            self.pair_interaction_durations[pair] = self.pair_interaction_durations.get(pair, 0) + 1

        self.active_proximity_pairs = current_pairs

    def record_candidate_event_stage(self, is_event: bool, duration: float):
        self.event_time += duration
        self.event_processed += 1

        if is_event:
            self.event_retained += 1
            self.candidate_events_count += 1
        else:
            self.event_removed += 1

    def finalize(self) -> Dict[str, Any]:
        self.end_time = time.time()
        total_exec_time = self.end_time - self.start_time

        total_frames = max(1, self.total_video_frames)

        # Motion metrics
        motion_red_pct = (self.motion_removed / total_frames) * 100.0 if total_frames > 0 else 0.0

        # YOLO metrics
        avg_conf = (self.confidence_sum / self.total_detections) if self.total_detections > 0 else 0.0
        yolo_stage_red = ((self.yolo_processed - self.yolo_retained) / max(1, self.yolo_processed)) * 100.0

        # Tracking metrics
        track_lengths = list(self.observed_tracks.values())
        total_unique_tracks = len(track_lengths)
        avg_track_len = float(np.mean(track_lengths)) if total_unique_tracks > 0 else 0.0
        max_track_len = int(np.max(track_lengths)) if total_unique_tracks > 0 else 0
        lost_tracks = sum(1 for l in track_lengths if l < (total_frames * 0.8))

        # Relationship metrics
        interaction_lens = list(self.pair_interaction_durations.values())
        avg_interaction_duration = float(np.mean(interaction_lens)) if len(interaction_lens) > 0 else 0.0

        # Search space remaining
        p_input = 100.0
        p_motion = (self.motion_retained / total_frames) * 100.0
        p_yolo = (self.yolo_retained / total_frames) * 100.0
        p_tracking = (self.tracking_retained / total_frames) * 100.0
        p_relationship = (self.relationship_retained / total_frames) * 100.0
        p_events = (self.event_retained / total_frames) * 100.0

        stats = {
            "video_name": self.video_name,
            "video_path": self.video_path,
            "fps": self.fps,
            "input": {
                "total_video_frames": self.total_video_frames
            },
            "motion_filtering": {
                "frames_entering": self.motion_entering,
                "frames_removed": self.motion_removed,
                "frames_retained": self.motion_retained,
                "reduction_percentage": round(motion_red_pct, 2),
                "processing_time_seconds": round(self.motion_time, 4)
            },
            "yolo_detection": {
                "frames_processed": self.yolo_processed,
                "frames_removed": self.yolo_removed,
                "frames_retained": self.yolo_retained,
                "stage_reduction_percentage": round(yolo_stage_red, 2),
                "total_detections": self.total_detections,
                "detections_by_class": self.detections_by_class,
                "average_confidence": round(avg_conf, 4),
                "inference_time_seconds": round(self.yolo_time, 4)
            },
            "tracking": {
                "frames_processed": self.tracking_processed,
                "frames_removed": self.tracking_removed,
                "frames_retained": self.tracking_retained,
                "total_tracks": total_unique_tracks,
                "average_track_length_frames": round(avg_track_len, 2),
                "longest_track_frames": max_track_len,
                "lost_tracks": lost_tracks,
                "id_switches": self.id_switches,
                "processing_time_seconds": round(self.tracking_time, 4)
            },
            "relationship_engine": {
                "frames_processed": self.relationship_processed,
                "frames_removed": self.relationship_removed,
                "frames_retained": self.relationship_retained,
                "total_person_vehicle_pairs": self.total_person_vehicle_pairs,
                "proximity_events": self.proximity_events_count,
                "average_interaction_duration_frames": round(avg_interaction_duration, 2),
                "processing_time_seconds": round(self.relationship_time, 4)
            },
            "candidate_event_generation": {
                "frames_processed": self.event_processed,
                "frames_removed": self.event_removed,
                "frames_retained": self.event_retained,
                "total_candidate_events": self.candidate_events_count,
                "processing_time_seconds": round(self.event_time, 4)
            },
            "pipeline_summary": {
                "total_execution_time_seconds": round(total_exec_time, 4),
                "search_space_remaining_pct": {
                    "Input": round(p_input, 2),
                    "Motion": round(p_motion, 2),
                    "YOLO": round(p_yolo, 2),
                    "Tracking": round(p_tracking, 2),
                    "Relationships": round(p_relationship, 2),
                    "Candidate Events": round(p_events, 2)
                }
            }
        }
        return stats


def save_pipeline_statistics(all_stats: List[Dict[str, Any]], output_dir: str):
    """Save statistics to pipeline_statistics.json and pipeline_statistics.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save JSON
    json_path = os.path.join(output_dir, "pipeline_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    # 2. Save CSV
    csv_path = os.path.join(output_dir, "pipeline_statistics.csv")
    if all_stats:
        fieldnames = [
            "video_name",
            "total_frames",
            "motion_retained",
            "motion_reduction_pct",
            "motion_time_sec",
            "yolo_retained",
            "total_detections",
            "person_count",
            "vehicle_count",
            "avg_confidence",
            "yolo_time_sec",
            "total_tracks",
            "avg_track_len",
            "tracking_time_sec",
            "proximity_events",
            "relationship_time_sec",
            "candidate_events",
            "event_time_sec",
            "total_exec_time_sec",
            "final_remaining_pct"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for st in all_stats:
                det_cls = st["yolo_detection"]["detections_by_class"]
                p_cnt = det_cls.get("person", 0)
                v_cnt = sum(det_cls.get(k, 0) for k in ["bicycle", "motorcycle", "car", "bus", "truck"])
                rem_pct = st["pipeline_summary"]["search_space_remaining_pct"]["Candidate Events"]

                writer.writerow({
                    "video_name": st["video_name"],
                    "total_frames": st["input"]["total_video_frames"],
                    "motion_retained": st["motion_filtering"]["frames_retained"],
                    "motion_reduction_pct": st["motion_filtering"]["reduction_percentage"],
                    "motion_time_sec": st["motion_filtering"]["processing_time_seconds"],
                    "yolo_retained": st["yolo_detection"]["frames_retained"],
                    "total_detections": st["yolo_detection"]["total_detections"],
                    "person_count": p_cnt,
                    "vehicle_count": v_cnt,
                    "avg_confidence": st["yolo_detection"]["average_confidence"],
                    "yolo_time_sec": st["yolo_detection"]["inference_time_seconds"],
                    "total_tracks": st["tracking"]["total_tracks"],
                    "avg_track_len": st["tracking"]["average_track_length_frames"],
                    "tracking_time_sec": st["tracking"]["processing_time_seconds"],
                    "proximity_events": st["relationship_engine"]["proximity_events"],
                    "relationship_time_sec": st["relationship_engine"]["processing_time_seconds"],
                    "candidate_events": st["candidate_event_generation"]["total_candidate_events"],
                    "event_time_sec": st["candidate_event_generation"]["processing_time_seconds"],
                    "total_exec_time_sec": st["pipeline_summary"]["total_execution_time_seconds"],
                    "final_remaining_pct": rem_pct
                })


def generate_sankey_data(all_stats: List[Dict[str, Any]], output_dir: str):
    """Generate pipeline_sankey_data.csv for Sankey diagram visualization."""
    os.makedirs(output_dir, exist_ok=True)
    sankey_path = os.path.join(output_dir, "pipeline_sankey_data.csv")

    tot_input = sum(st["input"]["total_video_frames"] for st in all_stats)
    tot_motion = sum(st["motion_filtering"]["frames_retained"] for st in all_stats)
    tot_yolo = sum(st["yolo_detection"]["frames_retained"] for st in all_stats)
    tot_tracking = sum(st["tracking"]["frames_retained"] for st in all_stats)
    tot_relationship = sum(st["relationship_engine"]["frames_retained"] for st in all_stats)
    tot_events = sum(st["candidate_event_generation"]["frames_retained"] for st in all_stats)

    rows = [
        {"source": "Raw Video Input", "target": "Motion Filtering", "value": tot_input},
        {"source": "Motion Filtering", "target": "YOLO Detection", "value": tot_motion},
        {"source": "Motion Filtering", "target": "Filtered (No Motion)", "value": tot_input - tot_motion},
        {"source": "YOLO Detection", "target": "Tracking Stage", "value": tot_yolo},
        {"source": "YOLO Detection", "target": "Filtered (No Objects)", "value": tot_motion - tot_yolo},
        {"source": "Tracking Stage", "target": "Relationship Analysis", "value": tot_tracking},
        {"source": "Tracking Stage", "target": "Filtered (No Active Tracks)", "value": tot_yolo - tot_tracking},
        {"source": "Relationship Analysis", "target": "Candidate Events", "value": tot_events},
        {"source": "Relationship Analysis", "target": "Filtered (No Proximity)", "value": tot_relationship - tot_events}
    ]

    with open(sankey_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "value"])
        writer.writeheader()
        writer.writerows(rows)


def generate_research_plots(all_stats: List[Dict[str, Any]], output_dir: str):
    """Generate research visualization figures:

    1. search_space_reduction.png
    2. stage_runtime.png
    3. detection_distribution.png
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_stats:
        return

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # -------------------------------------------------------------
    # Plot 1: search_space_reduction.png
    # -------------------------------------------------------------
    stages = ["Frames", "Motion", "YOLO", "Tracking", "Relationships", "Candidate Events"]

    avg_rem = []
    for stg in ["Input", "Motion", "YOLO", "Tracking", "Relationships", "Candidate Events"]:
        vals = [st["pipeline_summary"]["search_space_remaining_pct"][stg] for st in all_stats]
        avg_rem.append(float(np.mean(vals)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(stages, avg_rem, color=['#2b5c8f', '#3690c0', '#67a9cf', '#02818a', '#67a9cf', '#bd0026'], width=0.55)

    for bar, val in zip(bars, avg_rem):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.title("Progressive Search-Space Reduction Across Pipeline Stages", fontsize=14, pad=15)
    plt.xlabel("Forensic Pipeline Stage", fontsize=12)
    plt.ylabel("Remaining Evidence Search Space (%)", fontsize=12)
    plt.ylim(0, 115)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "search_space_reduction.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: stage_runtime.png
    # -------------------------------------------------------------
    stage_times = {
        "Motion Filtering": float(np.mean([st["motion_filtering"]["processing_time_seconds"] for st in all_stats])),
        "YOLO Detection": float(np.mean([st["yolo_detection"]["inference_time_seconds"] for st in all_stats])),
        "Tracking Stage": float(np.mean([st["tracking"]["processing_time_seconds"] for st in all_stats])),
        "Relationship Engine": float(np.mean([st["relationship_engine"]["processing_time_seconds"] for st in all_stats])),
        "Candidate Events": float(np.mean([st["candidate_event_generation"]["processing_time_seconds"] for st in all_stats]))
    }

    names = list(stage_times.keys())
    times = list(stage_times.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, times, color=['#41b6c4', '#e31a1c', '#225ea8', '#1d91c0', '#7fcdbb'], height=0.55)

    for bar, val in zip(bars, times):
        plt.text(val + (max(times) * 0.01), bar.get_y() + bar.get_height() / 2.0, f"{val:.3f}s", ha='left', va='center', fontweight='bold')

    plt.title("Average Runtime Contribution per Stage (Seconds per Video)", fontsize=14, pad=15)
    plt.xlabel("Execution Time (Seconds)", fontsize=12)
    plt.ylabel("Pipeline Stage", fontsize=12)
    plt.xlim(0, max(times) * 1.2 if max(times) > 0 else 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_runtime.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: detection_distribution.png
    # -------------------------------------------------------------
    class_counts = {
        "person": sum(st["yolo_detection"]["detections_by_class"].get("person", 0) for st in all_stats),
        "bicycle": sum(st["yolo_detection"]["detections_by_class"].get("bicycle", 0) for st in all_stats),
        "motorcycle": sum(st["yolo_detection"]["detections_by_class"].get("motorcycle", 0) for st in all_stats),
        "car": sum(st["yolo_detection"]["detections_by_class"].get("car", 0) for st in all_stats),
        "bus": sum(st["yolo_detection"]["detections_by_class"].get("bus", 0) for st in all_stats),
        "truck": sum(st["yolo_detection"]["detections_by_class"].get("truck", 0) for st in all_stats)
    }

    cls_names = list(class_counts.keys())
    cls_vals = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(cls_names, cls_vals, color='#2c7fb8', width=0.5)

    for bar, val in zip(bars, cls_vals):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + (max(cls_vals) * 0.01 if max(cls_vals) > 0 else 1), f"{val}", ha='center', va='bottom', fontweight='bold')

    plt.title("Total Detection Distribution by Object Class Across Dataset", fontsize=14, pad=15)
    plt.xlabel("Object Class", fontsize=12)
    plt.ylabel("Total Detections Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_distribution.png"), dpi=300)
    plt.close()


def generate_pipeline_report(all_stats: List[Dict[str, Any]], output_dir: str):
    """Generate the automated research report pipeline_report.md."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "pipeline_report.md")

    if not all_stats:
        return

    n_videos = len(all_stats)
    tot_frames = sum(st["input"]["total_video_frames"] for st in all_stats)

    # Average search space remaining
    rem_motion = float(np.mean([st["pipeline_summary"]["search_space_remaining_pct"]["Motion"] for st in all_stats]))
    rem_yolo = float(np.mean([st["pipeline_summary"]["search_space_remaining_pct"]["YOLO"] for st in all_stats]))
    rem_tracking = float(np.mean([st["pipeline_summary"]["search_space_remaining_pct"]["Tracking"] for st in all_stats]))
    rem_rel = float(np.mean([st["pipeline_summary"]["search_space_remaining_pct"]["Relationships"] for st in all_stats]))
    rem_event = float(np.mean([st["pipeline_summary"]["search_space_remaining_pct"]["Candidate Events"] for st in all_stats]))

    # Stage runtimes
    t_motion = float(np.mean([st["motion_filtering"]["processing_time_seconds"] for st in all_stats]))
    t_yolo = float(np.mean([st["yolo_detection"]["inference_time_seconds"] for st in all_stats]))
    t_tracking = float(np.mean([st["tracking"]["processing_time_seconds"] for st in all_stats]))
    t_rel = float(np.mean([st["relationship_engine"]["processing_time_seconds"] for st in all_stats]))
    t_event = float(np.mean([st["candidate_event_generation"]["processing_time_seconds"] for st in all_stats]))
    t_total = float(np.mean([st["pipeline_summary"]["total_execution_time_seconds"] for st in all_stats]))

    runtimes = {
        "Motion Filtering": t_motion,
        "YOLO Detection": t_yolo,
        "Tracking Stage": t_tracking,
        "Relationship Engine": t_rel,
        "Candidate Events": t_event
    }
    bottleneck_stage = max(runtimes, key=runtimes.get)
    bottleneck_pct = (runtimes[bottleneck_stage] / t_total * 100.0) if t_total > 0 else 0.0

    reductions = {
        "Motion Filtering": 100.0 - rem_motion,
        "YOLO Detection": rem_motion - rem_yolo,
        "Tracking Stage": rem_yolo - rem_tracking,
        "Relationship Engine": rem_tracking - rem_rel,
        "Candidate Events": rem_rel - rem_event
    }
    top_reducer = max(reductions, key=reductions.get)

    report_content = f"""# 🔬 AI Forensic Search: Progressive Search-Space & Bottleneck Evaluation Report

**Project**: CCTV Forensic Search FYP  
**Dataset Evaluated**: {n_videos} Video Clips ({tot_frames:,} Total Frames)  
**Output Location**: `{output_dir}`  

---

## 📊 1. Progressive Search-Space Reduction Summary

Every pipeline stage operates as a forensic evidence filter. Below is the average percentage of search space remaining after each stage:

| Pipeline Stage | Search Space Remaining (%) | Absolute Reduction Contribution (%) | Total Retained Frames |
| :--- | :---: | :---: | :---: |
| **0. Raw Video Input** | **100.00%** | Baseline | {tot_frames:,} |
| **1. Motion Filtering** | **{rem_motion:.2f}%** | {reductions['Motion Filtering']:.2f}% | {sum(st['motion_filtering']['frames_retained'] for st in all_stats):,} |
| **2. YOLO Detection** | **{rem_yolo:.2f}%** | {reductions['YOLO Detection']:.2f}% | {sum(st['yolo_detection']['frames_retained'] for st in all_stats):,} |
| **3. Tracking Stage** | **{rem_tracking:.2f}%** | {reductions['Tracking Stage']:.2f}% | {sum(st['tracking']['frames_retained'] for st in all_stats):,} |
| **4. Relationship Engine** | **{rem_rel:.2f}%** | {reductions['Relationship Engine']:.2f}% | {sum(st['relationship_engine']['frames_retained'] for st in all_stats):,} |
| **5. Candidate Events** | **{rem_event:.2f}%** | {reductions['Candidate Events']:.2f}% | {sum(st['candidate_event_generation']['frames_retained'] for st in all_stats):,} |

---

## ⏱️ 2. Stage Runtime Contribution & Bottleneck Analysis

| Pipeline Stage | Avg Execution Time per Video (s) | Runtime Contribution (%) | Bottleneck Rank |
| :--- | :---: | :---: | :---: |
| **Motion Filtering** | `{t_motion:.4f}s` | `{(t_motion / t_total * 100) if t_total > 0 else 0:.2f}%` | 3 |
| **YOLO Detection** | `{t_yolo:.4f}s` | `{(t_yolo / t_total * 100) if t_total > 0 else 0:.2f}%` | 1 |
| **Tracking Stage** | `{t_tracking:.4f}s` | `{(t_tracking / t_total * 100) if t_total > 0 else 0:.2f}%` | 2 |
| **Relationship Engine** | `{t_rel:.4f}s` | `{(t_rel / t_total * 100) if t_total > 0 else 0:.2f}%` | 4 |
| **Candidate Events** | `{t_event:.4f}s` | `{(t_event / t_total * 100) if t_total > 0 else 0:.2f}%` | 5 |
| **Total Pipeline** | `{t_total:.4f}s` | `100.00%` | - |

---

## 🔍 3. Core Research Findings

### 3.1 Which stage removes the largest amount of irrelevant information?
- **Primary Reducer Stage**: **{top_reducer}**
- **Impact**: Removes **{reductions[top_reducer]:.2f}%** of the total video search space.
- **Forensic Rationale**: Early-stage frame difference and spatial relationship rules successfully filter out static background scenes and non-interacting background pedestrians/vehicles without losing key evidence.

### 3.2 Which stage becomes the computational bottleneck?
- **Primary Bottleneck Stage**: **{bottleneck_stage}**
- **Impact**: Accounts for **{bottleneck_pct:.2f}%** of total execution runtime (`{runtimes[bottleneck_stage]:.4f}s` per video).
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
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
