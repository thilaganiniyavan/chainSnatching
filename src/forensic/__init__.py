"""Forensic Indexing & Retrieval Engine for the surveillance framework.

Provides searchable forensic event dataclass, multi-attribute inverted search index engine,
query engine APIs (search_events, filter_events, export_events), forensic HUD visualizer,
thumbnail and clip exporters, and dataset loggers.
"""

from src.forensic.forensic_index_engine import ForensicIndexEngine
from src.forensic.forensic_query_engine import ForensicQueryEngine
from src.forensic.forensic_visualizer import (
    ForensicOverlayVisualizer,
    ForensicThumbnailExporter,
    ForensicClipExporter,
)
from src.forensic.forensic_logger import ForensicLogger
