"""
tests/test_diag_bbox_shift.py: Unit tests for bounding box shift diagnostics (Task T03).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.evaluation.bbox_diag import (
    compute_distribution_stats,
    compute_ks_test,
    extract_tp_bbox_data,
)


def test_compute_distribution_stats_known_values():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_distribution_stats(data)
    assert stats["n"] == 5
    assert stats["median"] == 30.0
    assert stats["q25"] == 20.0
    assert stats["q75"] == 40.0
    assert stats["iqr"] == 20.0
    assert stats["mean"] == 30.0


def test_compute_distribution_stats_empty():
    stats = compute_distribution_stats([])
    assert stats["n"] == 0
    assert np.isnan(stats["median"])
    assert np.isnan(stats["iqr"])


def test_compute_ks_test_identical_and_different():
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res_same = compute_ks_test(s1, s2)
    assert res_same["statistic"] == 0.0
    assert res_same["pvalue"] == 1.0

    s3 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    res_diff = compute_ks_test(s1, s3)
    assert res_diff["statistic"] == 1.0
    assert res_diff["pvalue"] < 0.05


def test_extract_tp_bbox_data_guard_split_t():
    with pytest.raises(PermissionError, match="Split T is strictly forbidden"):
        extract_tp_bbox_data("yolov8s_640", "T")
    with pytest.raises(PermissionError, match="Split T is strictly forbidden"):
        extract_tp_bbox_data("yolov8s_640", "t")


def test_extract_tp_bbox_data_mock():
    # Mock prediction data
    df_dets = pd.DataFrame({
        "frame_id": ["000001", "000001", "000002"],
        "pred_idx": [0, 1, 0],
        "x1": [10.0, 100.0, 50.0],
        "y1": [20.0, 150.0, 60.0],
        "x2": [60.0, 150.0, 150.0],
        "y2": [70.0, 200.0, 160.0],
        "confidence": [0.85, 0.90, 0.50],
        "pass_thr": [True, True, False],
    })

    df_matches = pd.DataFrame({
        "frame_id": ["000001", "000001", "000002"],
        "pred_idx": [0, 1, 0],
        "status": ["TP", "FP", "TP"],
        "matched_gt_idx": [0, -1, 0],
        "matched_iou": [0.8, 0.0, 0.75],
    })

    df_gt = pd.DataFrame({
        "frame_id": ["000001", "000002"],
        "gt_idx": [0, 0],
        "x1": [12.0, 45.0],
        "y1": [18.0, 55.0],
        "x2": [62.0, 145.0],
        "y2": [68.0, 155.0],
        "z_gt": [15.0, 25.0],
        "difficulty": ["Easy", "Moderate"],
    })

    with patch("src.evaluation.bbox_diag.load_artifacts", return_value=(df_dets, df_matches, df_gt)):
        merged, total_gt = extract_tp_bbox_data("yolov8s_640", "A")

    assert total_gt == 2
    # Only frame 000001, pred 0 is pass_thr=True and status=TP
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["frame_id"] == "000001"
    assert row["iou"] == 0.8
    # y2_pred = 70.0, y2_gt = 68.0 -> delta_y2 = 2.0 px
    assert row["delta_y2_px"] == 2.0
    # h_gt = 68 - 18 = 50.0 -> delta_y2_rel = 2.0 / 50.0 = 0.04
    assert row["delta_y2_rel"] == pytest.approx(0.04)
    # w_pred = 50.0, w_gt = 50.0 -> delta_w_rel = 0.0
    assert row["delta_w_rel"] == pytest.approx(0.0)
    # h_pred = 50.0, h_gt = 50.0 -> delta_h_rel = 0.0
    assert row["delta_h_rel"] == pytest.approx(0.0)
