"""
[LEGACY / DEPRECATED] src/evaluation/metrics.py: Early Day 4 evaluation helper.

NOTE: This module is retained for historical backward compatibility with Day 4 scripts only.
It silently drops NaNs and counts only valid rows (n = n_valid), without tracking `valid_frac`,
and uses non-standard band labels ("0-10m" instead of "0-10").

For all Week 2 code, research questions, and official benchmarks, use `src/evaluation/eval.py`,
which provides:
  - Strict tracking of valid_frac, n_valid, and total n.
  - Frozen band edges and labels [0,10), [10,20), [20,30), [30,50), [50,inf) and ">30" (D3).
  - Paired cluster bootstrap over whole drives.
"""

import warnings
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


DEPTH_BINS = [
    ("0-10m", 0.0, 10.0),
    ("10-20m", 10.0, 20.0),
    ("20-30m", 20.0, 30.0),
    ("30-50m", 30.0, 50.0),
    (">50m", 50.0, float("inf")),
    (">30m", 30.0, float("inf")),
]


@dataclass
class DepthMetrics:
    n: int
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta_1: float
    delta_2: float
    delta_3: float
    mae: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "AbsRel": round(self.abs_rel, 4),
            "SqRel": round(self.sq_rel, 4),
            "RMSE": round(self.rmse, 3),
            "RMSElog": round(self.rmse_log, 4),
            "delta1": round(self.delta_1, 4),
            "delta2": round(self.delta_2, 4),
            "delta3": round(self.delta_3, 4),
            "MAE": round(self.mae, 3),
        }


def compute_depth_metrics(z_pred: np.ndarray, z_gt: np.ndarray) -> DepthMetrics:
    """
    [LEGACY] Compute standard depth estimation metrics on valid pairs.
    Use src.evaluation.eval (evaluate_report, depth_metrics) for official evaluations.
    """
    warnings.warn(
        "src.evaluation.metrics.compute_depth_metrics is legacy. "
        "Use src.evaluation.eval (evaluate_report, depth_metrics) for official evaluations.",
        DeprecationWarning,
        stacklevel=2,
    )
    valid = (~np.isnan(z_pred)) & (~np.isnan(z_gt)) & (z_pred > 0) & (z_gt > 0)
    pred = z_pred[valid]
    gt = z_gt[valid]
    n = len(pred)

    if n == 0:
        return DepthMetrics(
            n=0,
            abs_rel=np.nan,
            sq_rel=np.nan,
            rmse=np.nan,
            rmse_log=np.nan,
            delta_1=np.nan,
            delta_2=np.nan,
            delta_3=np.nan,
            mae=np.nan,
        )

    abs_diff = np.abs(pred - gt)
    sq_diff = (pred - gt) ** 2

    abs_rel = float(np.mean(abs_diff / gt))
    sq_rel = float(np.mean(sq_diff / gt))
    rmse = float(np.sqrt(np.mean(sq_diff)))
    rmse_log = float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))

    # Threshold accuracy
    ratio = np.maximum(pred / gt, gt / pred)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < (1.25 ** 2)))
    delta_3 = float(np.mean(ratio < (1.25 ** 3)))

    mae = float(np.mean(abs_diff))

    return DepthMetrics(
        n=n,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        rmse=rmse,
        rmse_log=rmse_log,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=delta_3,
        mae=mae,
    )


def compute_metrics_by_range(
    z_pred: np.ndarray,
    z_gt: np.ndarray,
    bins: Optional[List[tuple]] = None,
) -> Dict[str, DepthMetrics]:
    """
    Compute metrics stratified across depth ranges (§6, D3).
    """
    if bins is None:
        bins = DEPTH_BINS

    results = {}
    # Overall metrics
    results["Overall"] = compute_depth_metrics(z_pred, z_gt)

    for name, z_min, z_max in bins:
        in_bin = (z_gt >= z_min) & (z_gt < z_max)
        results[name] = compute_depth_metrics(z_pred[in_bin], z_gt[in_bin])

    return results


def format_metrics_table(results_by_cue: Dict[str, Dict[str, DepthMetrics]]) -> str:
    """
    Format a comparison markdown table across cues and depth ranges.
    """
    lines = []
    lines.append("| Cue / Method | Range | N | AbsRel | SqRel | RMSE (m) | RMSElog | δ < 1.25 | MAE (m) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for cue_name, range_dict in results_by_cue.items():
        for r_name, m in range_dict.items():
            flag = "*" if m.n < 100 and m.n > 0 else ""
            n_str = f"{m.n}{flag}"
            if m.n == 0:
                lines.append(f"| {cue_name} | {r_name} | 0 | - | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {cue_name} | {r_name} | {n_str} | {m.abs_rel:.4f} | {m.sq_rel:.4f} | "
                    f"{m.rmse:.2f} | {m.rmse_log:.4f} | {m.delta_1:.3f} | {m.mae:.2f} |"
                )
    return "\n".join(lines)
