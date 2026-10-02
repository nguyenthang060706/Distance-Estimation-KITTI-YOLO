"""
src/evaluation/: Evaluation metrics and diagnostic reporting.
"""

from src.evaluation.metrics import (
    DepthMetrics,
    compute_depth_metrics,
    compute_metrics_by_range,
    format_metrics_table,
    DEPTH_BINS,
)

__all__ = [
    "DepthMetrics",
    "compute_depth_metrics",
    "compute_metrics_by_range",
    "format_metrics_table",
    "DEPTH_BINS",
]
