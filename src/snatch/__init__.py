"""Snatch Signature Engine for the forensic surveillance framework.

Provides crime-specific forensic signature matching, configurable evidence templates,
weighted signature match scoring, human-readable explainable evidence checkmarks,
forensic HUD visualizer, clip exporter, and dataset loggers.
"""

from src.snatch.signature_config import (
    SignatureTemplate,
    StandardMotorcycleSnatchSignature,
    PedestrianSnatchSignature,
)
from src.snatch.signature_matcher import SignatureMatcher
from src.snatch.signature_explainer import SignatureExplainer
from src.snatch.snatch_signature_engine import SnatchSignatureEngine
from src.snatch.signature_visualizer import SignatureOverlayVisualizer, SignaturePreviewExporter
from src.snatch.signature_logger import SignatureLogger
