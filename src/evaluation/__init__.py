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
from src.evaluation.eval import (
    evaluate_report,
    evaluate_groups,
    depth_metrics,
    cluster_bootstrap,
    paired_cluster_bootstrap,
    BootstrapResult,
)

__all__ = [
    "DepthMetrics",
    "compute_depth_metrics",
    "compute_metrics_by_range",
    "format_metrics_table",
    "DEPTH_BINS",
    "evaluate_report",
    "evaluate_groups",
    "depth_metrics",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
    "BootstrapResult",
]

