"""Unit tests for Action Recognizer Interface, STGCNRecognizer, Scaffold Adapters, and Factory.

Tests cover:
- ActionRecognizerFactory backend instantiation
- STGCNRecognizer single-sequence and batch classification
- Scaffold adapters (CTRGCN, MSG-3D, PoseC3D)
- ActionResult field structure and class probability distributions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.core.models.action_result import ActionResult
from src.core.models.skeleton_sequence import SkeletonSequence
from src.action.factory import ActionRecognizerFactory
from src.action.stgcn_recognizer import STGCNRecognizer
from src.action.adapters.ctrgcn_adapter import CTRGCNRecognizer
from src.action.adapters.msg3d_adapter import MSG3DRecognizer
from src.action.adapters.posec3d_adapter import PoseC3DRecognizer


# ======================================================================
# Helpers
# ======================================================================

def _make_dummy_sequence(seq_id: str = "SEQ-001", T: int = 10) -> SkeletonSequence:
    tensor = np.zeros((T, 17, 4), dtype=float)
    for t in range(T):
        for v in range(17):
            tensor[t, v] = (100.0 + v, 100.0 + v, 0.8, 0.9)

    return SkeletonSequence(
        sequence_id=seq_id,
        interaction_id="INT-001",
        person_track_id=1,
        start_frame=1,
        end_frame=T,
        frame_count=T,
        duration_seconds=round(T / 30.0, 3),
        topology="COCO_17",
        num_joints=17,
        skeleton_tensor=tensor,
        joint_confidence_matrix=np.ones((T, 17), dtype=float) * 0.8,
        quality_score=0.8,
        is_accepted=True,
    )


# ======================================================================
# Action Recognizer Tests
# ======================================================================

class TestActionRecognizerFactory:

    def test_supported_backends(self):
        backends = ActionRecognizerFactory.get_supported_backends()
        assert "stgcn" in backends
        assert "ctrgcn" in backends
        assert "msg3d" in backends
        assert "posec3d" in backends

    def test_create_stgcn_backend(self):
        recognizer = ActionRecognizerFactory.create("stgcn")
        assert isinstance(recognizer, STGCNRecognizer)
        assert recognizer.backend_name == "ST-GCN"

    def test_create_scaffold_adapters(self):
        ctr = ActionRecognizerFactory.create("ctrgcn")
        assert isinstance(ctr, CTRGCNRecognizer)
        assert ctr.backend_name == "CTR-GCN"

        msg = ActionRecognizerFactory.create("msg3d")
        assert isinstance(msg, MSG3DRecognizer)
        assert msg.backend_name == "MSG-3D"

        posec = ActionRecognizerFactory.create("posec3d")
        assert isinstance(posec, PoseC3DRecognizer)
        assert posec.backend_name == "PoseC3D"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            ActionRecognizerFactory.create("invalid_action_backend")


class TestSTGCNRecognizer:

    def test_predict_action_single_sequence(self):
        recognizer = STGCNRecognizer()
        seq = _make_dummy_sequence("SEQ-001", T=10)

        res = recognizer.predict_action(seq)
        assert isinstance(res, ActionResult)
        assert res.sequence_id == "SEQ-001"
        assert res.model_name == "ST-GCN"
        assert res.predicted_action in recognizer.action_taxonomy
        assert res.action_confidence >= 0.0
        assert len(res.class_probabilities) == len(recognizer.action_taxonomy)
        assert res.inference_time_ms >= 0.0

    def test_predict_batch(self):
        recognizer = STGCNRecognizer()
        seqs = [_make_dummy_sequence("SEQ-001", T=10), _make_dummy_sequence("SEQ-002", T=10)]

        results = recognizer.predict_batch(seqs)
        assert len(results) == 2
        assert results[0].sequence_id == "SEQ-001"
        assert results[1].sequence_id == "SEQ-002"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
