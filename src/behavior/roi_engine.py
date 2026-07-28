"""ROI Engine — Interaction ROI Selection & Skeleton Sample preparation manager.

Consumes completed and active Behaviour Graphs and Track histories to select,
process, evaluate, and extract Interaction ROIs for downstream pose analysis.

Exposes clean API suite:
- create_roi()
- update_roi()
- finalize_roi()
- get_active_rois()
- get_completed_rois()
- get_accepted_rois()
- prepare_skeleton_samples()
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.models.behaviour_graph import BehaviourGraph
from src.core.models.interaction_roi import InteractionROI, PreparedSkeletonSample
from src.core.models.track import Track
from src.behavior.roi_bbox_processor import ROIBBoxProcessor
from src.behavior.roi_quality_evaluator import ROIQualityEvaluator
from src.behavior.skeleton_preparer import SkeletonPreparer


class ROIEngine:
    """Manages creation, frame tracking, quality evaluation, finalization,
    and skeleton sample preparation for Interaction ROIs.

    Args:
        bbox_processor: Custom ROIBBoxProcessor instance. If None, default parameters are used.
        quality_evaluator: Custom ROIQualityEvaluator instance. If None, default thresholds are used.
        fps: Video frame rate.
    """

    def __init__(
        self,
        bbox_processor: ROIBBoxProcessor | None = None,
        quality_evaluator: ROIQualityEvaluator | None = None,
        fps: float = 30.0,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.bbox_processor = bbox_processor if bbox_processor is not None else ROIBBoxProcessor()
        self.quality_evaluator = quality_evaluator if quality_evaluator is not None else ROIQualityEvaluator()
        self.preparer = SkeletonPreparer()

        # Storage: interaction_id -> InteractionROI
        self._rois: dict[str, InteractionROI] = {}
        # Raw box history per interaction: interaction_id -> list[(frame, bbox_or_None)]
        self._raw_box_history: dict[str, list[tuple[int, float, Optional[tuple[int, int, int, int]]]]] = {}

    def create_roi(
        self,
        graph: BehaviourGraph,
        frame_number: int,
    ) -> InteractionROI:
        """Initialize a new InteractionROI from a BehaviourGraph."""
        iid = graph.interaction_id
        roi_id = f"ROI-{iid}"

        roi = InteractionROI(
            roi_id=roi_id,
            interaction_id=iid,
            video_id="video_001",
            start_frame=graph.start_frame,
            person_track_id=graph.person_track_id,
            vehicle_track_id=graph.vehicle_track_id,
            graph_reference_id=graph.graph_id,
            interaction_confidence=max((n.confidence for n in graph.nodes), default=0.5),
            pattern_sequence=[n.pattern_type for n in graph.nodes],
            is_accepted=False,
        )

        self._rois[iid] = roi
        self._raw_box_history[iid] = []
        return roi

    def update_roi(
        self,
        graph: BehaviourGraph,
        tracks: list[Track],
        frame_number: int,
    ) -> InteractionROI:
        """Per-frame update collecting track bounding box history for an interaction ROI."""
        iid = graph.interaction_id

        if iid not in self._rois:
            roi = self.create_roi(graph, frame_number)
        else:
            roi = self._rois[iid]

        # Locate the person track
        track_map = {t.tracking_id: t for t in tracks}
        person_track = track_map.get(roi.person_track_id)

        raw_box: Optional[tuple[int, int, int, int]] = None
        if person_track and "bbox" in person_track.metadata:
            raw_box = tuple(person_track.metadata["bbox"])

        timestamp = round(frame_number / self.fps, 3)
        self._raw_box_history[iid].append((frame_number, timestamp, raw_box))

        # Update ROI frame span
        roi.end_frame = frame_number
        roi.frame_count = len(self._raw_box_history[iid])
        roi.duration_seconds = round(roi.frame_count / self.fps, 3)
        roi.pattern_sequence = [n.pattern_type for n in graph.nodes]
        if graph.nodes:
            roi.interaction_confidence = max(n.confidence for n in graph.nodes)

        # Process bounding boxes
        frames = [h[0] for h in self._raw_box_history[iid]]
        ts_list = [h[1] for h in self._raw_box_history[iid]]
        boxes_or_none = [h[2] for h in self._raw_box_history[iid]]

        smoothed, expanded, aligned_frames = self.bbox_processor.process_sequence(
            frames, boxes_or_none
        )

        roi.bounding_box_sequence = smoothed
        roi.expanded_bounding_boxes = expanded
        roi.frame_index_mapping = aligned_frames
        roi.timestamps = ts_list

        # Evaluate quality
        raw_mask = [b is not None for b in boxes_or_none]
        self.quality_evaluator.evaluate(roi, raw_box_mask=raw_mask)

        return roi

    def finalize_roi(
        self,
        interaction_id: str,
        frame_number: int,
    ) -> InteractionROI:
        """Finalize an InteractionROI when its interaction finishes."""
        roi = self._rois.get(interaction_id)
        if roi is not None:
            roi.end_frame = frame_number
            raw_boxes = [h[2] for h in self._raw_box_history.get(interaction_id, [])]
            raw_mask = [b is not None for b in raw_boxes]
            self.quality_evaluator.evaluate(roi, raw_box_mask=raw_mask)
        return roi if roi is not None else self._rois.get(interaction_id)

    def get_active_rois(self) -> list[InteractionROI]:
        """Return ROIs for currently active interactions."""
        return list(self._rois.values())

    def get_completed_rois(self) -> list[InteractionROI]:
        """Return all finalized ROIs."""
        return list(self._rois.values())

    def get_accepted_rois(self) -> list[InteractionROI]:
        """Return all ROIs that passed quality threshold acceptance checks."""
        return [roi for roi in self._rois.values() if roi.is_accepted]

    def prepare_skeleton_samples(
        self,
        roi: InteractionROI,
    ) -> list[PreparedSkeletonSample]:
        """Delegate skeleton sample generation to SkeletonPreparer."""
        return self.preparer.prepare_samples(roi)

    def clear(self) -> None:
        """Reset internal registries."""
        self._rois.clear()
        self._raw_box_history.clear()
