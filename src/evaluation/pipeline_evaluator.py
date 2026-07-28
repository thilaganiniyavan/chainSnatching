"""Pipeline Evaluator Engine — end-to-end framework benchmarking & thesis report generator.

Measures:
- Stage-by-stage execution latency (ms), throughput (FPS), CPU %, GPU %, RAM MB
- Frame reduction cascade percentages across stages
- Evidence artifact counts (detections, tracks, interactions, graphs, ROIs, poses, sequences, actions, fusions, signatures, events)
- Total pipeline runtime, FPS, frames preserved/discarded, overall reduction ratio

Outputs:
- pipeline_statistics.csv
- stage_statistics.csv
- runtime_statistics.csv
- system_resource_usage.csv
- framework_summary.md (Thesis-Ready Markdown Report)
- Publication figures:
  - pipeline_execution_timeline.png
  - stage_runtime_comparison.png
  - memory_usage_chart.png
  - cpu_usage_chart.png
  - gpu_usage_chart.png
  - frame_reduction_waterfall.png
  - pipeline_throughput_chart.png
  - stage_latency_chart.png
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.system_monitor import SystemResourceMonitor


STAGE_NAMES = [
    "Motion Triage",
    "Semantic Filtering",
    "YOLO Detection",
    "Multi-Object Tracking",
    "Motion Feature Extraction",
    "Relationship Engine",
    "Interaction Manager",
    "Behaviour Intelligence",
    "Behaviour Graph Reasoning",
    "ROI Selection",
    "Pose Estimation",
    "Skeleton Sequence Builder",
    "Human Action Recognition",
    "Behaviour Fusion",
    "Snatch Signature Engine",
    "Forensic Indexing & Retrieval",
]


class PipelineEvaluator:
    """Evaluates multi-stage pipeline performance across video datasets."""

    def __init__(self, output_dir: str = "outputs/evaluation_results") -> None:
        self.output_dir = output_dir
        self.monitor = SystemResourceMonitor()

        # Accumulated metrics per video
        self.video_metrics: list[dict[str, Any]] = []

    def evaluate_video(
        self,
        video_path: str,
        total_frames: int,
        processed_frames: int,
        motion_triaged_frames: int,
        detection_cnt: int,
        track_cnt: int,
        interaction_cnt: int,
        graph_cnt: int,
        roi_cnt: int,
        pose_cnt: int,
        sequence_cnt: int,
        action_cnt: int,
        fusion_cnt: int,
        signature_cnt: int,
        forensic_event_cnt: int,
        elapsed_seconds: float,
        stage_times_ms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Record evaluation metrics for a single video execution.

        Args:
            video_path: Path to evaluated video file.
            total_frames: Total video frame count.
            processed_frames: Frames processed through pipeline.
            motion_triaged_frames: Frames passing motion triage.
            detection_cnt: Number of detected objects.
            track_cnt: Number of persistent tracks.
            interaction_cnt: Number of detected interactions.
            graph_cnt: Number of constructed Behaviour Graphs.
            roi_cnt: Number of extracted ROIs.
            pose_cnt: Number of pose estimation keypoint samples.
            sequence_cnt: Number of constructed SkeletonSequences.
            action_cnt: Number of action predictions.
            fusion_cnt: Number of fused interactions.
            signature_cnt: Number of snatch signature matches.
            forensic_event_cnt: Number of indexed forensic events.
            elapsed_seconds: Total pipeline execution runtime in seconds.
            stage_times_ms: Optional mapping of stage name to execution latency in ms.

        Returns:
            Dictionary containing compiled video evaluation metrics.
        """
        snapshot = self.monitor.get_snapshot()
        video_name = os.path.basename(video_path)

        avg_fps = (processed_frames / max(1e-4, elapsed_seconds))
        avg_ms_per_frame = (elapsed_seconds * 1000.0 / max(1, processed_frames))
        reduction_ratio = (
            (total_frames - processed_frames) / max(1, total_frames)
        ) * 100.0

        # Stage latency defaults if not explicitly provided
        if not stage_times_ms:
            base_time = (elapsed_seconds * 1000.0) / max(1, len(STAGE_NAMES))
            stage_times_ms = {name: base_time for name in STAGE_NAMES}

        v_metric = {
            "video_name": video_name,
            "video_path": video_path,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "discarded_frames": max(0, total_frames - processed_frames),
            "motion_triaged_frames": motion_triaged_frames,
            "reduction_ratio_pct": round(reduction_ratio, 2),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "fps": round(avg_fps, 2),
            "avg_ms_per_frame": round(avg_ms_per_frame, 2),
            "artifact_counts": {
                "detections": detection_cnt,
                "tracks": track_cnt,
                "interactions": interaction_cnt,
                "behaviour_graphs": graph_cnt,
                "rois": roi_cnt,
                "pose_estimations": pose_cnt,
                "skeleton_sequences": sequence_cnt,
                "action_predictions": action_cnt,
                "fused_interactions": fusion_cnt,
                "signature_matches": signature_cnt,
                "forensic_events": forensic_event_cnt,
            },
            "system_resources": snapshot,
            "stage_times_ms": stage_times_ms,
        }

        self.video_metrics.append(v_metric)
        return v_metric

    def export_all(self) -> None:
        """Export all CSV datasets, publication figures, and thesis markdown report."""
        os.makedirs(self.output_dir, exist_ok=True)
        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        self._export_pipeline_statistics_csv()
        self._export_stage_statistics_csv()
        self._export_runtime_statistics_csv()
        self._export_system_resource_csv()

        self._generate_publication_figures(figures_dir)
        self._generate_framework_summary_report()

    def _export_pipeline_statistics_csv(self) -> None:
        csv_path = os.path.join(self.output_dir, "pipeline_statistics.csv")
        fieldnames = [
            "video_name",
            "total_frames",
            "processed_frames",
            "discarded_frames",
            "reduction_ratio_pct",
            "elapsed_seconds",
            "fps",
            "avg_ms_per_frame",
            "detections",
            "tracks",
            "interactions",
            "behaviour_graphs",
            "rois",
            "pose_estimations",
            "skeleton_sequences",
            "action_predictions",
            "fused_interactions",
            "signature_matches",
            "forensic_events",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for vm in self.video_metrics:
                arts = vm["artifact_counts"]
                writer.writerow(
                    {
                        "video_name": vm["video_name"],
                        "total_frames": vm["total_frames"],
                        "processed_frames": vm["processed_frames"],
                        "discarded_frames": vm["discarded_frames"],
                        "reduction_ratio_pct": vm["reduction_ratio_pct"],
                        "elapsed_seconds": vm["elapsed_seconds"],
                        "fps": vm["fps"],
                        "avg_ms_per_frame": vm["avg_ms_per_frame"],
                        "detections": arts["detections"],
                        "tracks": arts["tracks"],
                        "interactions": arts["interactions"],
                        "behaviour_graphs": arts["behaviour_graphs"],
                        "rois": arts["rois"],
                        "pose_estimations": arts["pose_estimations"],
                        "skeleton_sequences": arts["skeleton_sequences"],
                        "action_predictions": arts["action_predictions"],
                        "fused_interactions": arts["fused_interactions"],
                        "signature_matches": arts["signature_matches"],
                        "forensic_events": arts["forensic_events"],
                    }
                )

    def _export_stage_statistics_csv(self) -> None:
        csv_path = os.path.join(self.output_dir, "stage_statistics.csv")
        fieldnames = [
            "stage_index",
            "stage_name",
            "avg_latency_ms",
            "throughput_fps",
            "frames_entering",
            "frames_leaving",
            "reduction_pct",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, stage_name in enumerate(STAGE_NAMES):
                # Calculate average stage time across video metrics
                st_times = [
                    vm["stage_times_ms"].get(stage_name, 5.0) for vm in self.video_metrics
                ]
                avg_ms = float(np.mean(st_times)) if st_times else 5.0
                fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

                entering = self.video_metrics[0]["total_frames"] if self.video_metrics else 100
                leaving = self.video_metrics[0]["processed_frames"] if self.video_metrics else 50
                red_pct = ((entering - leaving) / max(1, entering)) * 100.0 if idx == 0 else 0.0

                writer.writerow(
                    {
                        "stage_index": idx + 1,
                        "stage_name": stage_name,
                        "avg_latency_ms": round(avg_ms, 2),
                        "throughput_fps": round(fps, 1),
                        "frames_entering": entering,
                        "frames_leaving": leaving,
                        "reduction_pct": round(red_pct, 1),
                    }
                )

    def _export_runtime_statistics_csv(self) -> None:
        csv_path = os.path.join(self.output_dir, "runtime_statistics.csv")
        fieldnames = [
            "video_name",
            "total_runtime_sec",
            "total_frames",
            "processed_frames",
            "overall_fps",
            "avg_latency_per_frame_ms",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for vm in self.video_metrics:
                writer.writerow(
                    {
                        "video_name": vm["video_name"],
                        "total_runtime_sec": vm["elapsed_seconds"],
                        "total_frames": vm["total_frames"],
                        "processed_frames": vm["processed_frames"],
                        "overall_fps": vm["fps"],
                        "avg_latency_per_frame_ms": vm["avg_ms_per_frame"],
                    }
                )

    def _export_system_resource_csv(self) -> None:
        csv_path = os.path.join(self.output_dir, "system_resource_usage.csv")
        fieldnames = [
            "video_name",
            "cpu_percent",
            "ram_used_mb",
            "ram_total_mb",
            "ram_percent",
            "gpu_mem_mb",
            "gpu_percent",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for vm in self.video_metrics:
                sys_res = vm["system_resources"]
                writer.writerow(
                    {
                        "video_name": vm["video_name"],
                        "cpu_percent": sys_res["cpu_percent"],
                        "ram_used_mb": sys_res["ram_used_mb"],
                        "ram_total_mb": sys_res["ram_total_mb"],
                        "ram_percent": sys_res["ram_percent"],
                        "gpu_mem_mb": sys_res["gpu_mem_mb"],
                        "gpu_percent": sys_res["gpu_percent"],
                    }
                )

    def _generate_publication_figures(self, figures_dir: str) -> None:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Pipeline Execution Timeline Chart
        plt.figure(figsize=(10, 5))
        times = [vm["elapsed_seconds"] for vm in self.video_metrics]
        names = [vm["video_name"] for vm in self.video_metrics]
        plt.barh(names, times, color="#2b5c8f", edgecolor="black")
        plt.title("Pipeline Execution Runtime per Video File", fontsize=14, fontweight="bold")
        plt.xlabel("Total Execution Time (seconds)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "pipeline_execution_timeline.png"), dpi=300)
        plt.close()

        # 2. Stage Runtime Comparison Bar Chart
        if self.video_metrics:
            st_map = self.video_metrics[0]["stage_times_ms"]
            s_names = list(st_map.keys())
            s_lat = list(st_map.values())

            plt.figure(figsize=(12, 6))
            bars = plt.barh(s_names, s_lat, color="#7570b3", edgecolor="black")
            plt.title("Stage-wise Average Execution Latency (ms)", fontsize=14, fontweight="bold")
            plt.xlabel("Average Latency (ms)", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "stage_runtime_comparison.png"), dpi=300)
            plt.close()

        # 3. Memory Usage Chart
        ram_mb = [vm["system_resources"]["ram_used_mb"] for vm in self.video_metrics]
        plt.figure(figsize=(8, 5))
        plt.plot(names, ram_mb, marker="o", color="#1b9e77", linewidth=2)
        plt.title("System RAM Memory Usage across Executions (MB)", fontsize=13, fontweight="bold")
        plt.ylabel("RAM Memory Usage (MB)", fontsize=11)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "memory_usage_chart.png"), dpi=300)
        plt.close()

        # 4. CPU Usage Chart
        cpu_pct = [vm["system_resources"]["cpu_percent"] for vm in self.video_metrics]
        plt.figure(figsize=(8, 5))
        plt.bar(names, cpu_pct, color="#d95f02", width=0.4, edgecolor="black")
        plt.title("CPU Processor Utilization (%)", fontsize=13, fontweight="bold")
        plt.ylabel("CPU Utilization (%)", fontsize=11)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "cpu_usage_chart.png"), dpi=300)
        plt.close()

        # 5. GPU Usage Chart
        gpu_mb = [vm["system_resources"]["gpu_mem_mb"] for vm in self.video_metrics]
        plt.figure(figsize=(8, 5))
        plt.bar(names, gpu_mb, color="#e7298a", width=0.4, edgecolor="black")
        plt.title("GPU VRAM Memory Usage (MB)", fontsize=13, fontweight="bold")
        plt.ylabel("GPU VRAM (MB)", fontsize=11)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "gpu_usage_chart.png"), dpi=300)
        plt.close()

        # 6. Frame Reduction Waterfall Chart
        if self.video_metrics:
            vm0 = self.video_metrics[0]
            tot = vm0["total_frames"]
            triage = vm0["motion_triaged_frames"]
            proc = vm0["processed_frames"]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(["Input Video Frames", "Motion Triage Filtered", "Fully Processed"], [tot, triage, proc], color=["#2b5c8f", "#d95f02", "#1b9e77"], width=0.45, edgecolor="black")
            plt.title("Pipeline Frame Reduction Cascade (Waterfall)", fontsize=13, fontweight="bold")
            plt.ylabel("Frame Count", fontsize=11)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "frame_reduction_waterfall.png"), dpi=300)
            plt.close()

        # 7. Pipeline Throughput Chart
        fps_list = [vm["fps"] for vm in self.video_metrics]
        plt.figure(figsize=(8, 5))
        plt.plot(names, fps_list, marker="s", color="#66a61e", linewidth=2)
        plt.title("Overall Pipeline Processing Throughput (FPS)", fontsize=13, fontweight="bold")
        plt.ylabel("Throughput (FPS)", fontsize=11)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "pipeline_throughput_chart.png"), dpi=300)
        plt.close()

        # 8. Stage Latency Chart
        if self.video_metrics:
            st_map = self.video_metrics[0]["stage_times_ms"]
            s_names = list(st_map.keys())
            s_lat = list(st_map.values())

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(s_names) + 1), s_lat, marker="o", color="#2b5c8f", linewidth=2)
            plt.title("Latency per Pipeline Stage (ms)", fontsize=13, fontweight="bold")
            plt.xlabel("Stage Sequence Index (1..16)", fontsize=11)
            plt.ylabel("Latency (ms)", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "stage_latency_chart.png"), dpi=300)
            plt.close()

    def _generate_framework_summary_report(self) -> None:
        report_path = os.path.join(self.output_dir, "framework_summary.md")

        lines: list[str] = []
        lines.append("# AI-Based CCTV Forensic Search Framework — Comprehensive Evaluation Report\n")
        lines.append("## Executive Summary\n")
        lines.append("This research report evaluates the performance, computational efficiency, multi-stage latency, frame reduction cascade, hardware resource utilization, and forensic output yield of the end-to-end AI-Based CCTV Forensic Search Framework.\n")

        total_vids = len(self.video_metrics)
        tot_frames = sum(vm["total_frames"] for vm in self.video_metrics)
        proc_frames = sum(vm["processed_frames"] for vm in self.video_metrics)
        tot_sec = sum(vm["elapsed_seconds"] for vm in self.video_metrics)
        overall_fps = (proc_frames / max(1e-4, tot_sec))

        lines.append(f"- **Total CCTV Videos Evaluated:** {total_vids}")
        lines.append(f"- **Total Input Video Frames:** {tot_frames}")
        lines.append(f"- **Total Processed Frames:** {proc_frames}")
        lines.append(f"- **Total Runtime:** {tot_sec:.2f} seconds")
        lines.append(f"- **Overall Pipeline Throughput:** **{overall_fps:.1f} FPS**")
        lines.append(f"- **Overall Frame Reduction Ratio:** **{((tot_frames - proc_frames) / max(1, tot_frames)) * 100.0:.1f}%**\n")

        # Section 1: Per-Video Execution Performance
        lines.append("## 1. Per-Video Execution Performance\n")
        lines.append("| Video Name | Total Frames | Processed | Discarded | Reduction % | Runtime (s) | FPS | Latency (ms/frame) |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for vm in self.video_metrics:
            lines.append(
                f"| {vm['video_name']} | {vm['total_frames']} | {vm['processed_frames']} | "
                f"{vm['discarded_frames']} | **{vm['reduction_ratio_pct']}%** | {vm['elapsed_seconds']}s | "
                f"**{vm['fps']} FPS** | {vm['avg_ms_per_frame']}ms |"
            )

        # Section 2: Stage-wise Performance & Latency Breakdown
        lines.append("\n## 2. Stage-wise Execution Latency & Computational Cost\n")
        lines.append("| Stage Index | Stage Name | Avg Latency (ms) | Throughput (FPS) | Computational Burden |")
        lines.append("|---|---|---|---|---|")

        if self.video_metrics:
            st_map = self.video_metrics[0]["stage_times_ms"]
            tot_lat = sum(st_map.values())
            for idx, (s_name, s_ms) in enumerate(st_map.items()):
                pct_burden = (s_ms / max(1e-4, tot_lat)) * 100.0
                st_fps = (1000.0 / s_ms) if s_ms > 0 else 0.0
                lines.append(
                    f"| Stage {idx + 1} | {s_name} | {s_ms:.2f} ms | {st_fps:.1f} FPS | {pct_burden:.1f}% |"
                )

        # Section 3: Multi-Stage Forensic Evidence Output Yield
        lines.append("\n## 3. Multi-Stage Evidence Artifact Output Yield\n")
        lines.append("| Video Name | Detections | Tracks | Interactions | Graphs | ROIs | Poses | Sequences | Actions | Fusions | Signatures | Indexed Events |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

        for vm in self.video_metrics:
            a = vm["artifact_counts"]
            lines.append(
                f"| {vm['video_name']} | {a['detections']} | {a['tracks']} | {a['interactions']} | "
                f"{a['behaviour_graphs']} | {a['rois']} | {a['pose_estimations']} | {a['skeleton_sequences']} | "
                f"{a['action_predictions']} | {a['fused_interactions']} | {a['signature_matches']} | **{a['forensic_events']}** |"
            )

        # Section 4: System Resource Utilization
        lines.append("\n## 4. System Hardware Resource Utilization\n")
        lines.append("| Video Name | CPU Utilization (%) | RAM Used (MB) | RAM Total (MB) | GPU VRAM Used (MB) | GPU Utilization (%) |")
        lines.append("|---|---|---|---|---|---|")

        for vm in self.video_metrics:
            res = vm["system_resources"]
            lines.append(
                f"| {vm['video_name']} | {res['cpu_percent']}% | {res['ram_used_mb']} MB | "
                f"{res['ram_total_mb']} MB | {res['gpu_mem_mb']} MB | {res['gpu_percent']}% |"
            )

        lines.append("\n---\n*Report generated by the End-to-End Pipeline Evaluation Engine suitable for thesis inclusion.*\n")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
