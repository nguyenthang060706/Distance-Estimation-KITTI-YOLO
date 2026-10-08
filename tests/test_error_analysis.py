"""
tests/test_error_analysis.py: Unit tests for Task T14 error analysis module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    assign_distance_bin,
    compute_point_metrics,
    compute_subgroup_metrics,
    compute_binned_distance_breakdown,
    compute_viewing_angle_breakdown,
    compute_physical_bias_analysis,
    extract_top_failures_and_successes,
)


def test_assign_distance_bin():
    assert assign_distance_bin(5.0) == "0-10"
    assert assign_distance_bin(10.0) == "10-20"
    assert assign_distance_bin(25.0) == "20-30"
    assert assign_distance_bin(35.0) == "30-50"
    assert assign_distance_bin(55.0) == ">50"

    arr = np.array([2.0, 12.0, 22.0, 45.0, 70.0])
    res = assign_distance_bin(arr)
    np.testing.assert_array_equal(res, ["0-10", "10-20", "20-30", "30-50", ">50"])


def test_compute_point_metrics_basic():
    # Empty / invalid
    res_empty = compute_point_metrics(np.array([]), np.array([]))
    assert res_empty["n"] == 0
    assert np.isnan(res_empty["absrel"])

    # Perfect prediction
    g = np.array([10.0, 20.0, 30.0])
    p = np.array([10.0, 20.0, 30.0])
    res = compute_point_metrics(g, p)
    assert res["absrel"] == 0.0
    assert res["mae"] == 0.0
    assert res["delta1"] == 1.0


def test_subgroup_metrics_clusters_and_low_n():
    df = pd.DataFrame({
        "difficulty": ["Easy", "Easy", "Hard", "Hard"],
        "drive": ["drive_1", "drive_1", "drive_2", "drive_3"],
        "z_gt": [10.0, 20.0, 30.0, 40.0],
        "z_hat_f": [11.0, 19.0, 33.0, 44.0],
    })
    res = compute_subgroup_metrics(df, "difficulty")
    assert len(res) == 2
    easy_res = [r for r in res if r["group_name"] == "Easy"][0]
    hard_res = [r for r in res if r["group_name"] == "Hard"][0]

    assert easy_res["n"] == 2
    assert easy_res["k_clusters"] == 1
    assert easy_res["low_n"] is True

    assert hard_res["n"] == 2
    assert hard_res["k_clusters"] == 2
    assert hard_res["low_n"] is True


def test_binned_distance_breakdown_with_recall():
    pred_df = pd.DataFrame({
        "z_gt": [5.0, 15.0, 25.0, 35.0, 45.0, 60.0],
        "z_hat_f": [5.5, 15.0, 24.0, 36.0, 44.0, 58.0],
        "drive": ["d1", "d1", "d2", "d2", "d3", "d3"],
    })
    fn_df = pd.DataFrame({
        "z_gt": [5.0, 35.0],
        "drive": ["d1", "d2"],
    })

    res = compute_binned_distance_breakdown(pred_df, fn_df)
    res_dict = {r["distance_bin"]: r for r in res}

    # 0-10m: 1 TP, 1 FN => Recall = 1 / 2 = 0.5
    assert res_dict["0-10"]["n_tp"] == 1
    assert res_dict["0-10"]["n_fn"] == 1
    assert res_dict["0-10"]["recall"] == 0.5

    # 10-20m: 1 TP, 0 FN => Recall = 1.0
    assert res_dict["10-20"]["n_tp"] == 1
    assert res_dict["10-20"]["n_fn"] == 0
    assert res_dict["10-20"]["recall"] == 1.0

    # >30m grouped row: 35m, 45m, 60m => 3 TP; fn has 35m => 1 FN. Total 3 / 4 = 0.75
    assert res_dict[">30"]["n_tp"] == 3
    assert res_dict[">30"]["n_fn"] == 1
    assert res_dict[">30"]["recall"] == 0.75


def test_viewing_angle_breakdown_symmetry():
    # alpha = 0 => theta = 0 (Side)
    # alpha = pi => theta = 0 (Side)
    # alpha = pi/4 => theta = pi/4 = 45 deg (Diagonal)
    # alpha = pi/2 => theta = pi/2 = 90 deg (Front/Rear)
    df = pd.DataFrame({
        "alpha": [0.0, np.pi, np.pi / 4, np.pi / 2],
        "drive": ["d1", "d1", "d2", "d2"],
        "z_gt": [10.0, 15.0, 20.0, 25.0],
        "z_w": [10.0, 15.0, 20.0, 25.0],
        "z_h": [10.0, 15.0, 20.0, 25.0],
        "z_g": [10.0, 15.0, 20.0, 25.0],
        "z_d": [10.0, 15.0, 20.0, 25.0],
        "z_hat_f0": [10.0, 15.0, 20.0, 25.0],
        "z_hat_f": [10.0, 15.0, 20.0, 25.0],
    })
    res = compute_viewing_angle_breakdown(df)

    assert res["side"]["n"] == 2
    assert res["diagonal"]["n"] == 1
    assert res["front_rear"]["n"] == 1


def test_physical_bias_pattern_111():
    # Ground truth = 8.0m, Pred = 7.0m => Bias = -1.0m (negative)
    df = pd.DataFrame({
        "z_gt": [8.0, 8.0, 20.0],
        "z_hat_f": [7.0, 6.0, 20.0],
        "valid_w": [1, 0, 1],
        "valid_h": [1, 1, 1],
        "valid_g": [1, 1, 1],
    })
    res = compute_physical_bias_analysis(df)

    near_p111 = res["pattern_111_verification"]["near_0_10m_pattern_111_no_border_cut"]
    assert near_p111["n"] == 1
    assert near_p111["bias_rel_mean"] == -0.125  # (7 - 8) / 8 = -0.125


def test_extract_top_failures_and_successes():
    df = pd.DataFrame({
        "frame_id": ["001", "002", "003"],
        "drive": ["d1", "d1", "d2"],
        "pred_idx": [0, 1, 0],
        "gt_idx": [0, 1, 0],
        "z_gt": [10.0, 20.0, 30.0],
        "z_hat_f": [20.0, 20.1, 30.0],  # 001 AbsRel = 1.0 (fail), 002 AbsRel = 0.005 (success)
        "z_d": [18.0, 20.0, 30.0],
        "bbox_x1": [10, 20, 30],
        "bbox_y1": [10, 20, 30],
        "bbox_x2": [100, 200, 300],
        "bbox_y2": [100, 200, 300],
        "matched_iou": [0.8, 0.9, 0.95],
        "alpha": [0.1, 1.2, 0.5],
        "occluded": [0, 1, 0],
        "truncated": [0.0, 0.1, 0.0],
        "difficulty": ["Easy", "Moderate", "Hard"],
        "fallback_flag": [False, False, False],
        "valid_w": [1, 1, 1],
        "valid_h": [1, 1, 1],
        "valid_g": [1, 1, 1],
    })

    res = extract_top_failures_and_successes(df, detector_name="yolo11s_640", top_k_fail=2, n_success=2)
    assert res["n_top_failures"] == 2
    assert res["top_failures"][0]["frame_id"] == "001"
    assert res["top_failures"][0]["absrel"] == 1.0
    assert len(res["representative_successes"]) > 0


def test_kitti_difficulty_nested_definition():
    """Definition check: Verify nested hierarchy Easy is a subset of Moderate is a subset of Hard."""
    # Definition KITTI:
    # Easy: h >= 40, occ == 0, trunc <= 0.15
    # Moderate: h >= 25, occ <= 1, trunc <= 0.30
    # Hard: h >= 25, occ <= 2, trunc <= 0.50
    # Every Easy object MUST satisfy Moderate and Hard conditions!
    h_candidates = [45, 50, 100]
    occ_candidates = [0]
    trunc_candidates = [0.0, 0.10]

    for h in h_candidates:
        for occ in occ_candidates:
            for tr in trunc_candidates:
                is_easy = (h >= 40) and (occ == 0) and (tr <= 0.15)
                is_mod = (h >= 25) and (occ <= 1) and (tr <= 0.30)
                is_hard = (h >= 25) and (occ <= 2) and (tr <= 0.50)
                if is_easy:
                    assert is_mod and is_hard, f"Hierarchy violation for h={h}, occ={occ}, trunc={tr}"
