"""Unit tests for the SystemResourceMonitor and PipelineEvaluator modules.

Tests cover:
- Hardware resource metric profiling (CPU, RAM MB, GPU MB)
- Single and multi-video pipeline evaluation metric recording
- Generation of CSV datasets, markdown report, and publication figures
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.evaluation.system_monitor import SystemResourceMonitor
from src.evaluation.pipeline_evaluator import PipelineEvaluator, STAGE_NAMES


# ======================================================================
# SystemResourceMonitor Tests
# ======================================================================

class TestSystemResourceMonitor:

    def test_get_snapshot(self):
        monitor = SystemResourceMonitor()
        snapshot = monitor.get_snapshot()

        assert "cpu_percent" in snapshot
        assert "ram_used_mb" in snapshot
        assert "ram_total_mb" in snapshot
        assert "gpu_mem_mb" in snapshot
        assert snapshot["ram_used_mb"] > 0.0


# ======================================================================
# PipelineEvaluator Tests
# ======================================================================

class TestPipelineEvaluator:

    def test_evaluate_video_and_export_all(self, tmp_path):
        eval_dir = str(tmp_path / "eval_results")
        evaluator = PipelineEvaluator(output_dir=eval_dir)

        v_metric = evaluator.evaluate_video(
            video_path="test_video_01.mp4",
            total_frames=300,
            processed_frames=150,
            motion_triaged_frames=200,
            detection_cnt=450,
            track_cnt=10,
            interaction_cnt=3,
            graph_cnt=2,
            roi_cnt=2,
            pose_cnt=60,
            sequence_cnt=2,
            action_cnt=2,
            fusion_cnt=2,
            signature_cnt=1,
            forensic_event_cnt=1,
            elapsed_seconds=5.0,
        )

        assert v_metric["video_name"] == "test_video_01.mp4"
        assert v_metric["reduction_ratio_pct"] == 50.0
        assert v_metric["fps"] == 30.0

        # Export all datasets and figures
        evaluator.export_all()

        assert os.path.exists(os.path.join(eval_dir, "pipeline_statistics.csv"))
        assert os.path.exists(os.path.join(eval_dir, "stage_statistics.csv"))
        assert os.path.exists(os.path.join(eval_dir, "runtime_statistics.csv"))
        assert os.path.exists(os.path.join(eval_dir, "system_resource_usage.csv"))
        assert os.path.exists(os.path.join(eval_dir, "framework_summary.md"))
        assert os.path.exists(os.path.join(eval_dir, "figures", "pipeline_execution_timeline.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
