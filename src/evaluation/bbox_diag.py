"""
src/evaluation/bbox_diag.py: Diagnostic calculations for bounding box distribution shift (Task T03, v4 §0.1, §5.3).

Computes:
- TP matching join between detector predictions, matches, and ground truth
- Error metrics: IoU, bottom-edge shift (px and relative), relative width and height shift
- Distribution statistics: median, IQR (25th and 75th percentiles)
- Two-sample Kolmogorov-Smirnov test (A vs B, B vs C)
- Recall comparison across splits
- Strict guard: Split T is forbidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.pipeline.build_dataset import load_artifacts


def extract_tp_bbox_data(
    model_key: str,
    split: str,
    predictions_dir: str | Path = "results/predictions",
) -> tuple[pd.DataFrame, int]:
    """
    Extract True Positive detections at pass_thr joined with their corresponding Ground Truth.

    Args:
        model_key: Model key (e.g. 'yolov8s_640', 'yolo11s_640', 'yolov5su_640').
        split: Split name ('A', 'B', 'C'). Split 'T' is strictly forbidden.
        predictions_dir: Path to prediction parquet artifacts.

    Returns:
        (df_tp_matched, total_gt_hard):
            - df_tp_matched: DataFrame with pred and GT bounding boxes and derived error metrics.
            - total_gt_hard: Count of total GT Hard objects in the split.

    Raises:
        PermissionError: If split is 'T' (AGENT_RULES §1.1).
    """
    split_upper = split.upper()
    if split_upper == "T":
        raise PermissionError(
            "Access to Split T is strictly forbidden for diagnostic analysis (AGENT_RULES §1.1)!"
        )

    df_dets, df_matches, df_gt = load_artifacts(model_key, split_upper, predictions_dir=predictions_dir)

    total_gt_hard = len(df_gt)

    # Ensure integer indices for join keys
    dets = df_dets.copy()
    matches = df_matches.copy()
    gt = df_gt.copy()

    dets["pred_idx"] = dets["pred_idx"].astype(int)
    matches["pred_idx"] = matches["pred_idx"].astype(int)
    gt["gt_idx"] = gt["gt_idx"].astype(int)

    # 1. Join detections with matches
    det_matches = dets.merge(
        matches[["frame_id", "pred_idx", "status", "matched_gt_idx", "matched_iou"]],
        on=["frame_id", "pred_idx"],
        how="inner",
    )

    # 2. Filter TP at pass_thr
    tp_mask = (det_matches["pass_thr"] == True) & (det_matches["status"] == "TP")
    df_tp = det_matches[tp_mask].copy()
    df_tp["matched_gt_idx"] = df_tp["matched_gt_idx"].astype(int)
    df_tp = df_tp[df_tp["matched_gt_idx"] >= 0]

    # 3. Join with GT
    merged = df_tp.merge(
        gt,
        left_on=["frame_id", "matched_gt_idx"],
        right_on=["frame_id", "gt_idx"],
        how="inner",
        suffixes=("_pred", "_gt"),
    )

    # Compute bounding box widths and heights
    w_pred = np.maximum(merged["x2_pred"] - merged["x1_pred"], 1.0)
    h_pred = np.maximum(merged["y2_pred"] - merged["y1_pred"], 1.0)
    w_gt = np.maximum(merged["x2_gt"] - merged["x1_gt"], 1.0)
    h_gt = np.maximum(merged["y2_gt"] - merged["y1_gt"], 1.0)

    merged["w_pred"] = w_pred
    merged["h_pred"] = h_pred
    merged["w_gt"] = w_gt
    merged["h_gt"] = h_gt

    # Metric 1: matched IoU
    merged["iou"] = merged["matched_iou"].astype(float)

    # Metric 2: bottom-edge shift y2_pred - y2_gt (pixels)
    merged["delta_y2_px"] = (merged["y2_pred"] - merged["y2_gt"]).astype(float)

    # Metric 3: relative bottom-edge shift (y2_pred - y2_gt) / h_gt
    merged["delta_y2_rel"] = (merged["delta_y2_px"] / h_gt).astype(float)

    # Metric 4: relative width shift (w_pred - w_gt) / w_gt
    merged["delta_w_rel"] = ((w_pred - w_gt) / w_gt).astype(float)

    # Metric 5: relative height shift (h_pred - h_gt) / h_gt
    merged["delta_h_rel"] = ((h_pred - h_gt) / h_gt).astype(float)

    return merged, total_gt_hard


def compute_distribution_stats(series: pd.Series | np.ndarray) -> dict[str, float]:
    """
    Compute median, IQR (q75 - q25), q25, q75, mean, and std for a metric distribution.
    """
    arr = np.asarray(series, dtype=float)
    valid_arr = arr[np.isfinite(arr)]
    if len(valid_arr) == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }

    q25 = float(np.percentile(valid_arr, 25))
    q75 = float(np.percentile(valid_arr, 75))
    median = float(np.median(valid_arr))
    iqr = q75 - q25

    return {
        "n": len(valid_arr),
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "mean": float(np.mean(valid_arr)),
        "std": float(np.std(valid_arr, ddof=1)) if len(valid_arr) > 1 else 0.0,
    }


def compute_ks_test(
    sample_1: pd.Series | np.ndarray,
    sample_2: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Perform two-sample Kolmogorov-Smirnov test between two distributions.
    """
    s1 = np.asarray(sample_1, dtype=float)
    s2 = np.asarray(sample_2, dtype=float)
    s1 = s1[np.isfinite(s1)]
    s2 = s2[np.isfinite(s2)]

    if len(s1) == 0 or len(s2) == 0:
        return {"statistic": float("nan"), "pvalue": float("nan")}

    res = ks_2samp(s1, s2)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
    }
