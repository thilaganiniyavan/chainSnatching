"""Live Snatch 1.0 evaluation for research configurations A–D and ablations.

Runs the real pipeline on each dataset video (no synthetic F1 scores).
Video-level ground truth is taken from Snatch 1.0 folder labels:
- ``Snatch Theft`` → incident (positive)
- ``Normal`` → control (negative)
"""

from __future__ import annotations

import gc
import glob
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from configs.detection_config import ALLOWED_CLASSES
from src.core.models.frame_context import FrameContext
from src.detection.detector import Detector
from src.evaluation.system_monitor import SystemResourceMonitor
from src.pipeline.action_recognition_stage import ActionRecognitionStage
from src.pipeline.behaviour_fusion_stage import BehaviourFusionStage
from src.pipeline.behaviour_stage import BehaviourStage
from src.pipeline.forensic_indexing_stage import ForensicIndexingStage
from src.pipeline.graph_reasoning_stage import GraphReasoningStage
from src.pipeline.interaction_stage import InteractionStage
from src.pipeline.pipeline import Pipeline
from src.pipeline.pose_estimation_stage import PoseEstimationStage
from src.pipeline.reasoning_stage import ReasoningStage
from src.pipeline.relationship_stage import RelationshipStage
from src.pipeline.roi_selection_stage import ROISelectionStage
from src.pipeline.skeleton_sequence_stage import SkeletonSequenceStage
from src.pipeline.snatch_signature_stage import SnatchSignatureStage
from src.pipeline.tracking_stage import TrackingStage


VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm", "*.3gp")
DEFAULT_SNATCH_DATASET_DIRS = [
    os.path.join("Snatch 1.0", "Chain Snatching Videos", "Snatch Theft"),
    os.path.join("Snatch 1.0", "Chain Snatching Videos", "Normal"),
    "Snatch 1.0",
]
VEHICLE_CLASSES = {"motorcycle", "bicycle", "car", "bus", "truck"}
SNATCH_PATTERNS = {
    "APPROACH_PATTERN",
    "FOLLOW_PATTERN",
    "INTERACTION_PATTERN",
    "PROXIMITY_PATTERN",
    "ESCAPE_PATTERN",
    "SEPARATION_PATTERN",
}
POSITIVE_THRESHOLD = 0.75
PROXIMITY_PX = 150.0
MOTION_PIXEL_THRESHOLD = 5000


@dataclass
class PipelineSpec:
    """Which pipeline stages are enabled for a configuration or ablation."""

    name: str
    use_motion: bool = True
    use_semantic_filter: bool = True
    use_tracking: bool = True
    use_relationship: bool = False
    use_interaction: bool = False
    use_behaviour: bool = False
    use_reasoning: bool = False
    use_graph: bool = False
    use_roi: bool = False
    use_pose: bool = False
    use_sequence: bool = False
    use_action: bool = False
    use_fusion: bool = False
    use_signature: bool = False
    use_indexing: bool = False


def baseline_spec(name: str, use_motion: bool) -> PipelineSpec:
    """YOLO + tracking + spatial proximity (Config A / B)."""
    return PipelineSpec(
        name=name,
        use_motion=use_motion,
        use_semantic_filter=True,
        use_tracking=True,
        use_relationship=True,
    )


def graph_spec(name: str) -> PipelineSpec:
    """Motion + tracking + behaviour graph (Config C)."""
    return PipelineSpec(
        name=name,
        use_motion=True,
        use_semantic_filter=True,
        use_tracking=True,
        use_relationship=True,
        use_interaction=True,
        use_behaviour=True,
        use_reasoning=True,
        use_graph=True,
    )


def full_spec(name: str, **overrides: bool) -> PipelineSpec:
    """Full 13-stage proposed framework (Config D), with optional ablation overrides."""
    spec = PipelineSpec(
        name=name,
        use_motion=True,
        use_semantic_filter=True,
        use_tracking=True,
        use_relationship=True,
        use_interaction=True,
        use_behaviour=True,
        use_reasoning=True,
        use_graph=True,
        use_roi=True,
        use_pose=True,
        use_sequence=True,
        use_action=True,
        use_fusion=True,
        use_signature=True,
        use_indexing=True,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


CONFIG_SPECS: list[PipelineSpec] = [
    baseline_spec("Config A (Baseline)", use_motion=False),
    baseline_spec("Config B (+ Motion Triage)", use_motion=True),
    graph_spec("Config C (+ Behaviour Graph)"),
    full_spec("Config D (Proposed Framework)"),
]

ABLATION_SPECS: list[PipelineSpec] = [
    full_spec("Ablation: Motion Triage Removed", use_motion=False),
    full_spec("Ablation: Semantic Filtering Removed", use_semantic_filter=False),
    full_spec("Ablation: Behaviour Graph Removed", use_graph=False, use_fusion=False),
    full_spec("Ablation: ROI Selection Removed", use_roi=False, use_pose=False, use_sequence=False, use_action=False),
    full_spec("Ablation: Pose Estimation Removed", use_pose=False, use_sequence=False, use_action=False),
    full_spec("Ablation: Behaviour Fusion Removed", use_fusion=False, use_signature=False),
    full_spec("Ablation: Action Recognition Removed", use_action=False),
    full_spec("Ablation: Relationship Engine Removed", use_relationship=False, use_interaction=False, use_behaviour=False, use_reasoning=False, use_graph=False, use_roi=False, use_fusion=False),
    full_spec("Ablation: Interaction Manager Removed", use_interaction=False, use_behaviour=False, use_reasoning=False, use_graph=False, use_roi=False, use_fusion=False),
    full_spec("Ablation: Forensic Indexing Removed", use_indexing=False),
]


def discover_snatch_videos(dataset_paths: Iterable[str] | None = None) -> list[str]:
    """Recursively discover Snatch 1.0 (and override) video files, de-duplicated."""
    paths = list(dataset_paths) if dataset_paths else list(DEFAULT_SNATCH_DATASET_DIRS)
    found: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        if os.path.isfile(path):
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                found.append(os.path.abspath(path))
            continue
        for ext in VIDEO_EXTENSIONS:
            matches = glob.glob(os.path.join(path, "**", ext), recursive=True)
            matches.extend(glob.glob(os.path.join(path, "**", ext.upper()), recursive=True))
            for match in matches:
                key = os.path.normcase(os.path.abspath(match))
                if key not in seen:
                    seen.add(key)
                    found.append(os.path.abspath(match))
    return sorted(found)


def is_incident_video(video_path: str) -> bool:
    """Return True if the video is labelled as a snatch incident in Snatch 1.0."""
    normalized = video_path.replace("\\", "/")
    name = os.path.basename(video_path).lower()
    if "/Normal/" in normalized or normalized.rstrip("/").endswith("/Normal"):
        return False
    if "Snatch Theft" in video_path or "snatch" in name:
        return True
    return "Normal" not in video_path


def person_vehicle_proximity_score(detections: list[Any]) -> float:
    """Score [0, 1] from closest person–vehicle pair in a frame."""
    persons = [d for d in detections if getattr(d, "class_name", "") == "person"]
    vehicles = [d for d in detections if getattr(d, "class_name", "") in VEHICLE_CLASSES]
    if not persons or not vehicles:
        return 0.0
    min_dist = min(
        math.hypot(p.center_x - v.center_x, p.center_y - v.center_y)
        for p in persons
        for v in vehicles
    )
    if min_dist <= PROXIMITY_PX:
        return max(0.35, min(1.0, 1.0 - (min_dist / (PROXIMITY_PX * 2.0))))
    return max(0.0, 1.0 - (min_dist / 400.0)) * 0.5


def binary_roc_auc(y_true: list[int], y_score: list[float]) -> float:
    """Trapezoidal ROC-AUC. Returns 0.5 if only one class is present."""
    if not y_true or len(set(y_true)) < 2:
        return 0.5
    order = np.argsort(-np.asarray(y_score, dtype=float))
    y = np.asarray(y_true, dtype=float)[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    tpr = tps / max(tps[-1], 1e-9)
    fpr = fps / max(fps[-1], 1e-9)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr, fpr))


def classification_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, float]:
    """Dataset-level precision, recall, F1, accuracy, rates, and ROC-AUC."""
    n = max(1, len(y_true))
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall / max(1e-9, precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n
    tpr = recall
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(binary_roc_auc(y_true, y_score), 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "tpr": round(tpr, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_videos": len(y_true),
        "n_positives": sum(y_true),
        "n_negatives": len(y_true) - sum(y_true),
    }


def _finalize_stage(stage: Any) -> None:
    if hasattr(stage, "finalize"):
        try:
            stage.finalize()
        except Exception:
            pass
    # Explicitly release MediaPipe C++ session memory if present
    if hasattr(stage, "estimator"):
        est = stage.estimator
        if hasattr(est, "_pose_solution") and est._pose_solution is not None:
            try:
                est._pose_solution.close()
                est._pose_solution = None
            except Exception:
                pass


class LiveResearchEvaluator:
    """Executes Config A–D and optional ablations on real dataset videos."""

    def __init__(
        self,
        detector: Detector | None = None,
        pose_backend: str = "mediapipe",
        action_backend: str = "stgcn",
        fusion_strategy: str = "weighted_confidence",
        decision_threshold: float = POSITIVE_THRESHOLD,
        max_frames: int | None = None,
    ) -> None:
        self.detector = detector
        self.pose_backend = pose_backend
        self.action_backend = action_backend
        self.fusion_strategy = fusion_strategy
        self.decision_threshold = decision_threshold
        self.max_frames = max_frames
        self.monitor = SystemResourceMonitor()

    def _ensure_detector(self) -> Detector:
        if self.detector is None:
            self.detector = Detector()
        return self.detector

    def _build_stages(self, spec: PipelineSpec, fps: float, video_name: str) -> tuple[list[Any], dict[str, Any]]:
        handles: dict[str, Any] = {}
        stages: list[Any] = []
        if spec.use_tracking:
            handles["tracking"] = TrackingStage()
            stages.append(handles["tracking"])
        if spec.use_relationship:
            handles["relationship"] = RelationshipStage(distance_threshold=PROXIMITY_PX)
            stages.append(handles["relationship"])
        if spec.use_interaction:
            handles["interaction"] = InteractionStage(distance_threshold=PROXIMITY_PX)
            stages.append(handles["interaction"])
        if spec.use_behaviour:
            handles["behaviour"] = BehaviourStage(fps=fps)
            stages.append(handles["behaviour"])
        if spec.use_reasoning:
            handles["reasoning"] = ReasoningStage(fps=fps)
            stages.append(handles["reasoning"])
        if spec.use_graph:
            handles["graph"] = GraphReasoningStage(fps=fps)
            stages.append(handles["graph"])
        if spec.use_roi:
            handles["roi"] = ROISelectionStage(fps=fps)
            stages.append(handles["roi"])
        if spec.use_pose:
            handles["pose"] = PoseEstimationStage(backend_name=self.pose_backend, fps=fps)
            stages.append(handles["pose"])
        if spec.use_sequence:
            handles["sequence"] = SkeletonSequenceStage(fps=fps)
            stages.append(handles["sequence"])
        if spec.use_action:
            handles["action"] = ActionRecognitionStage(backend_name=self.action_backend, fps=fps)
            stages.append(handles["action"])
        if spec.use_fusion:
            handles["fusion"] = BehaviourFusionStage(fusion_strategy=self.fusion_strategy, fps=fps)
            stages.append(handles["fusion"])
        if spec.use_signature:
            handles["signature"] = SnatchSignatureStage()
            stages.append(handles["signature"])
        if spec.use_indexing:
            handles["indexing"] = ForensicIndexingStage(video_id=video_name, location="Snatch 1.0")
            stages.append(handles["indexing"])
        return stages, handles

    def _decision_score(self, spec: PipelineSpec, handles: dict[str, Any], proximity_score: float) -> tuple[float, float]:
        """Return (decision_score, evidence_completeness) from enabled stages."""
        signature_stage = handles.get("signature")
        if spec.use_signature and signature_stage is not None:
            results = signature_stage.engine.get_all_results()
            if results:
                best = max(float(r.signature_score) for r in results)
                n_matched = max((len(r.matched_evidence) for r in results), default=0)
                n_total = max((len(r.matched_evidence) + len(r.missing_evidence) for r in results), default=1)
                return best, n_matched / max(1, n_total)

        fusion_stage = handles.get("fusion")
        if spec.use_fusion and fusion_stage is not None:
            fusions = fusion_stage.engine.get_completed_fusions()
            if fusions:
                best = max(float(f.fusion_confidence) for f in fusions)
                return best, best

        graph_stage = handles.get("graph")
        if spec.use_graph and graph_stage is not None:
            graphs = graph_stage.engine.get_all_graphs()
            patterns = {node.pattern_type for g in graphs for node in g.nodes}
            hits = patterns & SNATCH_PATTERNS
            if hits:
                score = min(1.0, 0.35 + 0.12 * len(hits))
                return score, len(hits) / max(1, len(SNATCH_PATTERNS))

        return proximity_score, min(1.0, proximity_score)

    def evaluate_video(self, spec: PipelineSpec, video_path: str, max_frames: int | None = None) -> dict[str, Any]:
        """Run one configuration on one video and return live metrics."""
        detector = self._ensure_detector()
        video_name = os.path.basename(video_path)
        gt_positive = is_incident_video(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 20.0

        limit_frames = max_frames if max_frames is not None else self.max_frames
        target_frames = min(total_frames, limit_frames) if (limit_frames and total_frames > 0) else total_frames

        stages, handles = self._build_stages(spec, fps=fps, video_name=video_name)
        pipeline = Pipeline(stages=stages) if stages else None
        mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True) if spec.use_motion else None

        frame_number = 0
        processed_cnt = 0
        max_proximity = 0.0
        t0 = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_number += 1
                if limit_frames is not None and frame_number > limit_frames:
                    break

                if mog2 is not None:
                    mask = mog2.apply(frame)
                    if cv2.countNonZero(mask) <= MOTION_PIXEL_THRESHOLD:
                        continue

                detections = detector.detect(frame)
                max_proximity = max(max_proximity, person_vehicle_proximity_score(detections))

                if spec.use_semantic_filter and not detections:
                    continue

                processed_cnt += 1
                if pipeline is None:
                    continue

                context = FrameContext(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=frame_number / fps,
                    detections=detections,
                )
                pipeline.run(context)
        finally:
            cap.release()
            for stage in stages:
                _finalize_stage(stage)

        elapsed = max(1e-6, time.perf_counter() - t0)
        score, evidence = self._decision_score(spec, handles, max_proximity)
        predicted = int(score >= self.decision_threshold)
        gt = int(gt_positive)
        correct = int(predicted == gt)

        # Release video frame buffers and model memory immediately
        del pipeline, stages, handles
        gc.collect()

        snapshot = self.monitor.get_snapshot()
        eval_total = max(target_frames, frame_number) if target_frames > 0 else max(total_frames, frame_number)
        reduction = ((eval_total - processed_cnt) / max(1, eval_total)) * 100.0
        latency_ms = (elapsed * 1000.0) / max(1, processed_cnt)
        proc_fps = processed_cnt / elapsed

        return {
            "video_name": video_name,
            "video_path": video_path,
            "ground_truth_positive": gt,
            "predicted_positive": predicted,
            "decision_score": round(float(score), 4),
            "correct": correct,
            "precision": float(correct),
            "recall": float(correct),
            "f1": float(correct),
            "accuracy": float(correct),
            "latency_ms": round(latency_ms, 2),
            "fps": round(proc_fps, 2),
            "frame_reduction_pct": round(reduction, 1),
            "ram_mb": float(snapshot.get("ram_used_mb", 0.0)),
            "cpu_pct": float(snapshot.get("cpu_percent", 0.0)),
            "gpu_mb": float(snapshot.get("gpu_mem_mb", 0.0)),
            "signature_score": round(float(score), 4),
            "evidence_completeness": round(float(evidence), 4),
            "processed_frames": processed_cnt,
            "total_frames": eval_total,
            "elapsed_seconds": round(elapsed, 2),
        }

    def evaluate_spec_on_videos(
        self,
        spec: PipelineSpec,
        video_paths: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
        max_frames: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Evaluate one spec on every video; return per-video rows and dataset metrics."""
        rows: list[dict[str, Any]] = []
        for idx, video_path in enumerate(video_paths, start=1):
            if progress_callback:
                progress_callback(idx, len(video_paths), video_path)
            try:
                row = self.evaluate_video(spec, video_path, max_frames=max_frames)
            except Exception as exc:
                print(f"  ! Failed {os.path.basename(video_path)}: {exc}")
                row = {
                    "video_name": os.path.basename(video_path),
                    "video_path": video_path,
                    "ground_truth_positive": int(is_incident_video(video_path)),
                    "predicted_positive": 0,
                    "decision_score": 0.0,
                    "correct": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                    "latency_ms": 0.0,
                    "fps": 0.0,
                    "frame_reduction_pct": 0.0,
                    "ram_mb": 0.0,
                    "cpu_pct": 0.0,
                    "gpu_mb": 0.0,
                    "signature_score": 0.0,
                    "evidence_completeness": 0.0,
                    "processed_frames": 0,
                    "total_frames": 0,
                    "elapsed_seconds": 0.0,
                }
            rows.append(row)
            gc.collect()

        y_true = [int(r["ground_truth_positive"]) for r in rows]
        y_pred = [int(r["predicted_positive"]) for r in rows]
        y_score = [float(r["decision_score"]) for r in rows]
        metrics = classification_metrics(y_true, y_pred, y_score)
        metrics["config_name"] = spec.name
        return rows, metrics
