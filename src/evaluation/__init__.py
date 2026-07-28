"""Research evaluation modules for the forensic surveillance framework.

Provides:
- Interaction & relationship metrics
- Behaviour reasoning evaluator
- Pose quality & sequence statistics
- Action recognition statistics
- Behaviour fusion statistics
- Snatch signature statistics
- Forensic indexing & retrieval statistics
- System resource monitor & end-to-end Pipeline Evaluator Engine
"""

from src.evaluation.system_monitor import SystemResourceMonitor
from src.evaluation.pipeline_evaluator import PipelineEvaluator, STAGE_NAMES
from src.evaluation.statistical_analyzer import StatisticalAnalyzer
from src.evaluation.research_ablation_engine import (
    ResearchAblationEngine,
    CONFIG_NAMES,
    ABLATION_VARIANTS,
)
