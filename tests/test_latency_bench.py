"""
tests/test_latency_bench.py: Unit tests for Tier 1 latency benchmarking and Parity check (D40, D44, D94, D95).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.bench_latency import (
    compute_box_iou,
    compute_latency_stats,
    extract_10_features_e_vectorized,
    extract_17_features_vectorized,
    letterbox_image,
    postprocess_onnx_boxes,
)
from src.geometry.geometric_cues import CameraIntrinsics, CueResult
from src.residual.models import FEATURE_COLS_F
from src.uncertainty.cqr import predict_interval


def test_compute_latency_stats_exact():
    """Verify statistical functions (median, mean, std, iqr, p95) on known distribution."""
    # Data: 1, 2, 3, 4, 5, 6, 7, 8, 9
    arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    stats = compute_latency_stats(arr)
    assert stats["median_ms"] == 5.0
    assert stats["mean_ms"] == 5.0
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 9.0
    assert stats["iqr_ms"] == 4.0  # 7.0 - 3.0
    assert stats["p95_ms"] == 8.6


def test_end_to_end_vs_sum_of_medians_mathematical_difference():
    """
    Demonstrate Decision D44: Median(A + B) != Median(A) + Median(B) for correlated/skewed distributions.
    This mathematical property confirms why summing stage medians is invalid and per-image totals are mandatory.
    """
    stage_a = [1.0, 2.0, 3.0, 100.0, 100.0]  # median = 3.0
    stage_b = [100.0, 100.0, 3.0, 2.0, 1.0]  # median = 3.0
    total = [a + b for a, b in zip(stage_a, stage_b)]  # [101.0, 102.0, 6.0, 102.0, 101.0] -> median = 101.0

    stat_a = compute_latency_stats(stage_a)
    stat_b = compute_latency_stats(stage_b)
    stat_total = compute_latency_stats(total)

    sum_of_medians = stat_a["median_ms"] + stat_b["median_ms"]  # 3.0 + 3.0 = 6.0
    assert sum_of_medians == 6.0
    assert stat_total["median_ms"] == 101.0
    assert stat_total["median_ms"] != sum_of_medians  # 101.0 != 6.0


def test_compute_box_iou():
    """Verify IoU calculation on known boxes."""
    b1 = np.array([0.0, 0.0, 10.0, 10.0])
    b2 = np.array([0.0, 0.0, 10.0, 10.0])
    assert compute_box_iou(b1, b2) == pytest.approx(1.0)

    b3 = np.array([5.0, 0.0, 15.0, 10.0])
    # Inter: 5x10=50, Area1=100, Area3=100, Union=150 -> IoU = 50/150 = 1/3
    assert compute_box_iou(b1, b3) == pytest.approx(1.0 / 3.0)

    b4 = np.array([20.0, 20.0, 30.0, 30.0])
    assert compute_box_iou(b1, b4) == 0.0


def test_letterbox_image_dimensions():
    """Verify letterbox output shapes and normalization."""
    mock_img = np.zeros((375, 1242, 3), dtype=np.uint8)
    blob, scale, left, top = letterbox_image(mock_img, target_size=640)

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() <= blob.max() <= 1.0
    assert scale == pytest.approx(640.0 / 1242.0)
    assert left >= 0
    assert top >= 0


def test_postprocess_onnx_boxes():
    """Verify box decoding and NMS on mock tensor output."""
    # Mock raw output: (1, 7, 8400)
    # rows: xc, yc, w, h, score_car, score_van, score_truck
    out_raw = np.zeros((1, 7, 8400), dtype=np.float32)

    # Box 1: high score car
    out_raw[0, 0, 10] = 320.0  # xc
    out_raw[0, 1, 10] = 320.0  # yc
    out_raw[0, 2, 10] = 100.0  # bw
    out_raw[0, 3, 10] = 80.0   # bh
    out_raw[0, 4, 10] = 0.85   # score car

    # Box 2: overlapping duplicate with lower score
    out_raw[0, 0, 11] = 321.0
    out_raw[0, 1, 11] = 321.0
    out_raw[0, 2, 11] = 100.0
    out_raw[0, 3, 11] = 80.0
    out_raw[0, 4, 11] = 0.70

    # Box 3: non-overlapping car
    out_raw[0, 0, 20] = 100.0
    out_raw[0, 1, 20] = 100.0
    out_raw[0, 2, 20] = 50.0
    out_raw[0, 3, 20] = 40.0
    out_raw[0, 4, 20] = 0.90

    boxes, scores = postprocess_onnx_boxes(
        out_raw,
        scale=1.0,
        left=0.0,
        top=0.0,
        orig_w=640,
        orig_h=640,
        conf_min=0.50,
        iou_nms=0.70,
    )

    # Box 2 should be suppressed by Box 1 -> exactly 2 boxes left
    assert len(boxes) == 2
    assert len(scores) == 2
    assert scores[0] == pytest.approx(0.90)
    assert scores[1] == pytest.approx(0.85)


def test_extract_17_features_vectorized_matches_whitelist():
    """Verify that vectorized 17-feature extraction produces exactly FEATURE_COLS_F without leak."""
    boxes = np.array([
        [100.0, 150.0, 250.0, 220.0],
        [0.0, 180.0, 120.0, 260.0],
    ])
    scores = np.array([0.88, 0.75])
    cues_res = CueResult(
        Z_w=np.array([25.0, 15.0]),
        Z_h=np.array([24.5, 14.8]),
        Z_g=np.array([26.0, 16.0]),
        valid_w=np.array([True, False]),
        valid_h=np.array([True, True]),
        valid_g=np.array([True, False]),
    )
    z_base = np.array([25.2, 14.8])
    intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8)

    X_mat = extract_17_features_vectorized(
        active_boxes=boxes,
        active_scores=scores,
        cues_res=cues_res,
        z_base=z_base,
        intrinsics=intrinsics,
        orig_w=1242,
        orig_h=375,
    )

    assert X_mat.shape == (2, 17)
    assert len(FEATURE_COLS_F) == 17
    assert np.all(np.isfinite(X_mat))
    # Confidence column (idx 9) must match input scores
    assert np.allclose(X_mat[:, 9], scores)
    # touch_left column (idx 5) for second box touching x1=0 must be 1.0
    assert X_mat[1, 5] == 1.0


def test_cqr_stage_timing_mock():
    """Verify CQR interval calculation in log-space."""
    z_base = np.array([15.0, 25.0, 35.0])
    q05 = np.array([-0.05, -0.04, -0.06])
    q95 = np.array([0.05, 0.06, 0.04])
    q_hat = 0.051

    z_lo, z_hi, r_lo, r_hi, n_cross = predict_interval(z_base, q05, q95, q_hat)
    assert len(z_lo) == 3
    assert len(z_hi) == 3
    assert np.all(z_lo < z_base)
    assert np.all(z_hi > z_base)
    assert n_cross == 0


def test_extract_17_features_vectorized_matches_pipeline_exact():
    """Verify that vectorized 17-feature extraction exactly matches build_feature_matrices from models.py."""
    import pandas as pd
    from src.residual.feature_extractor import extract_inference_features
    from src.residual.models import build_feature_matrices

    boxes = np.array([
        [100.0, 150.0, 250.0, 220.0],
        [0.0, 180.0, 120.0, 260.0],
    ])
    scores = np.array([0.88, 0.75])
    cues_res = CueResult(
        Z_w=np.array([25.0, 15.0]),
        Z_h=np.array([24.5, 14.8]),
        Z_g=np.array([26.0, 16.0]),
        valid_w=np.array([True, False]),
        valid_h=np.array([True, True]),
        valid_g=np.array([True, False]),
    )
    z_base = np.array([25.2, 14.8])
    intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8)
    orig_w, orig_h = 1242, 375

    X_vec = extract_17_features_vectorized(
        active_boxes=boxes,
        active_scores=scores,
        cues_res=cues_res,
        z_base=z_base,
        intrinsics=intrinsics,
        orig_w=orig_w,
        orig_h=orig_h,
    )

    # Build reference features using canonical pipeline
    raw_df = pd.DataFrame({
        "x1": boxes[:, 0],
        "y1": boxes[:, 1],
        "x2": boxes[:, 2],
        "y2": boxes[:, 3],
        "confidence": scores,
        "img_w": orig_w,
        "img_h": orig_h,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "z_w": cues_res.Z_w,
        "z_h": cues_res.Z_h,
        "z_g": cues_res.Z_g,
        "valid_w": cues_res.valid_w,
        "valid_h": cues_res.valid_h,
        "valid_g": cues_res.valid_g,
    })
    base_feats = extract_inference_features(raw_df)
    feat_mats = build_feature_matrices(base_feats, z_base=z_base)
    X_ref = feat_mats["f"].to_numpy(dtype=float)

    assert X_vec.shape == X_ref.shape
    np.testing.assert_allclose(X_vec, X_ref, rtol=1e-5, atol=1e-5)


def test_extract_10_features_vectorized_matches_pipeline_exact():
    """Verify that vectorized 10-feature extraction for Model (e) exactly matches build_feature_matrices from models.py."""
    import pandas as pd
    from src.residual.feature_extractor import extract_inference_features
    from src.residual.models import build_feature_matrices

    boxes = np.array([
        [100.0, 150.0, 250.0, 220.0],
        [0.0, 180.0, 120.0, 260.0],
    ])
    scores = np.array([0.88, 0.75])
    intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8)
    orig_w, orig_h = 1242, 375

    X_vec_e = extract_10_features_e_vectorized(
        active_boxes=boxes,
        active_scores=scores,
        intrinsics=intrinsics,
        orig_w=orig_w,
        orig_h=orig_h,
    )

    # Build reference features using canonical pipeline
    raw_df = pd.DataFrame({
        "x1": boxes[:, 0],
        "y1": boxes[:, 1],
        "x2": boxes[:, 2],
        "y2": boxes[:, 3],
        "confidence": scores,
        "img_w": orig_w,
        "img_h": orig_h,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "z_w": [25.0, 15.0],
        "z_h": [24.5, 14.8],
        "z_g": [26.0, 16.0],
        "valid_w": [True, False],
        "valid_h": [True, True],
        "valid_g": [True, False],
    })
    base_feats = extract_inference_features(raw_df)
    feat_mats = build_feature_matrices(base_feats, z_base=np.array([25.2, 14.8]))
    X_ref_e = feat_mats["e"].to_numpy(dtype=float)

    assert X_vec_e.shape == (2, 10)
    assert X_vec_e.shape == X_ref_e.shape
    # Specifically check cx_offset_norm column (index 4) matches (divided by fx=721.5, not orig_w=1242)
    np.testing.assert_allclose(X_vec_e, X_ref_e, rtol=1e-5, atol=1e-5)

