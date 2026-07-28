"""Pipeline Stage for the Snatch Signature Engine.

Consumes FusedInteraction objects from upstream BehaviourFusionStage, executes
crime-specific forensic signature matching via :class:`SnatchSignatureEngine`, annotates
video frames via :class:`SignatureOverlayVisualizer`, and logs datasets via :class:`SignatureLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.snatch.signature_config import SignatureTemplate
from src.snatch.snatch_signature_engine import SnatchSignatureEngine
from src.snatch.signature_visualizer import SignatureOverlayVisualizer, SignaturePreviewExporter
from src.snatch.signature_logger import SignatureLogger


class SnatchSignatureStage(Stage):
    """Pipeline stage executing forensic snatch signature evaluation over FusedInteractions.

    Args:
        template: Optional custom SignatureTemplate (defaults to StandardMotorcycleSnatchSignature).
        output_json_path: Path for snatch_signature_results.json export.
        output_csv_path: Path for signature_statistics.csv export.
        output_report_path: Path for signature_report.md export.
        export_previews_dir: Optional directory path to export preview videos.
    """

    def __init__(
        self,
        template: SignatureTemplate | None = None,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_previews_dir: str | None = None,
    ) -> None:
        self.engine = SnatchSignatureEngine(template=template)
        self.visualizer = SignatureOverlayVisualizer()
        self.logger = SignatureLogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_previews_dir = export_previews_dir

        self._evaluated_signature_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, evaluate active FusedInteractions."""

        fused_interactions = context.metadata.get("fused_interactions", [])
        current_frame_signatures = []

        for fusion in fused_interactions:
            sig_result = self.engine.evaluate_interaction(fusion)
            current_frame_signatures.append(sig_result)

            if sig_result.signature_id not in self._evaluated_signature_ids:
                self.logger.log_result(sig_result)
                self._evaluated_signature_ids.add(sig_result.signature_id)

        all_signatures = self.engine.get_all_results()

        context.snatch_signatures = all_signatures
        context.metadata["snatch_signatures"] = all_signatures
        context.metadata["snatch_engine"] = self.engine

        # Render forensic signature HUD overlay on video frame
        base_frame = context.metadata.get(
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
        )

        signature_frame = self.visualizer.draw(base_frame, all_signatures)
        context.metadata["signature_frame"] = signature_frame

        return context

    def finalize(self) -> None:
        """Export accumulated SnatchSignatureResults to JSON, CSV, and report files."""
        for sig in self.engine.get_all_results():
            if sig.signature_id not in self._evaluated_signature_ids:
                self.logger.log_result(sig)
                self._evaluated_signature_ids.add(sig.signature_id)

        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)
