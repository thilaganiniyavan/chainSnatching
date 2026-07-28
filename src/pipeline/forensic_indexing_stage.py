"""Pipeline Stage for the Forensic Indexing & Retrieval Engine.

Consumes SnatchSignatureResults from upstream SnatchSignatureStage, converts them into
searchable :class:`ForensicEvent` records, builds multi-attribute search indices via
:class:`ForensicQueryEngine`, exports thumbnails and video clips, annotates video frames via
:class:`ForensicOverlayVisualizer`, and logs datasets via :class:`ForensicLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.forensic.forensic_query_engine import ForensicQueryEngine
from src.forensic.forensic_visualizer import (
    ForensicOverlayVisualizer,
    ForensicThumbnailExporter,
    ForensicClipExporter,
)
from src.forensic.forensic_logger import ForensicLogger


class ForensicIndexingStage(Stage):
    """Pipeline stage executing forensic event indexing and evidence traceability linking.

    Args:
        video_id: Identifier or file name of source video.
        location: Camera or CCTV location identifier string.
        output_json_path: Path for forensic_events.json export.
        output_csv_path: Path for forensic_index.csv export.
        output_report_path: Path for forensic_index_report.md export.
        export_thumbnails_dir: Directory path to export event keyframe thumbnails.
        export_clips_dir: Directory path to export annotated event video clips.
    """

    def __init__(
        self,
        video_id: str = "default_video",
        location: str = "Camera 1",
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_thumbnails_dir: str | None = None,
        export_clips_dir: str | None = None,
    ) -> None:
        self.video_id = video_id
        self.location = location

        self.query_engine = ForensicQueryEngine()
        self.visualizer = ForensicOverlayVisualizer()
        self.logger = ForensicLogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_thumbnails_dir = export_thumbnails_dir
        self.export_clips_dir = export_clips_dir

        self._indexed_signature_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, index active SnatchSignatureResults."""

        snatch_signatures = context.metadata.get("snatch_signatures", [])
        current_frame_events = []

        for sig_result in snatch_signatures:
            if sig_result.signature_id not in self._indexed_signature_ids:
                event = self.query_engine.create_event_from_signature(
                    sig_result=sig_result,
                    video_id=self.video_id,
                    location=self.location,
                )

                # Export thumbnail image if thumbnail directory is set
                if self.export_thumbnails_dir:
                    thumb_path = f"{self.export_thumbnails_dir}/thumb_{event.event_id}.jpg"
                    ForensicThumbnailExporter.export_thumbnail(event, context.frame, thumb_path)

                self.logger.log_event(event)
                self._indexed_signature_ids.add(sig_result.signature_id)
                current_frame_events.append(event)
            else:
                event_id = f"EVT-{sig_result.signature_id}"
                event = self.query_engine.get_event(event_id)
                if event:
                    current_frame_events.append(event)

        all_events = self.query_engine.get_all_events()

        context.forensic_events = all_events
        context.metadata["forensic_events"] = all_events
        context.metadata["query_engine"] = self.query_engine

        # Render forensic event HUD overlay on video frame
        base_frame = context.metadata.get(
            "signature_frame",
            context.metadata.get(
                "fusion_frame",
                context.metadata.get(
                    "action_frame",
                    context.metadata.get(
                        "sequence_frame",
                        context.metadata.get(
                            "pose_frame",
                            context.metadata.get(
                                "roi_frame",
                                context.metadata.get(
                                    "graph_frame",
                                    context.metadata.get(
                                        "reasoning_frame",
                                        context.metadata.get(
                                            "behaviour_frame",
                                            context.metadata.get(
                                                "relationship_frame",
                                                context.metadata.get("trajectory_frame", context.frame),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        forensic_frame = self.visualizer.draw(base_frame, all_events)
        context.metadata["forensic_frame"] = forensic_frame

        return context

    def finalize(self) -> None:
        """Export accumulated ForensicEvents to JSON, CSV, and report files."""
        for event in self.query_engine.get_all_events():
            if event.event_id not in [e.event_id for e in self.logger.get_events()]:
                self.logger.log_event(event)

        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)
