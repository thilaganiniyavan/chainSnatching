"""Human Action Recognition Framework for the forensic surveillance framework.

Provides model-agnostic Action Recognizer interface, ST-GCN implementation,
scaffolded adapters for CTR-GCN, MSG-3D, PoseC3D, ActionRecognizerFactory,
post-processing, visualization, and structured dataset export.
"""

from src.action.base_recognizer import AbstractActionRecognizer, DEFAULT_ACTION_TAXONOMY
from src.action.stgcn_recognizer import STGCNRecognizer
from src.action.adapters.ctrgcn_adapter import CTRGCNRecognizer
from src.action.adapters.msg3d_adapter import MSG3DRecognizer
from src.action.adapters.posec3d_adapter import PoseC3DRecognizer
from src.action.factory import ActionRecognizerFactory
from src.action.action_post_processor import ActionPostProcessor
from src.action.action_visualizer import ActionOverlayVisualizer, ActionPreviewExporter
from src.action.action_logger import ActionLogger
