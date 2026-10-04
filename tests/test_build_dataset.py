"""
tests/test_build_dataset.py: Unit tests for dataset building pipeline (T01, Decisions D11, D22, D23).

Tests using synthetic data:
1. Correct join order invariance (shuffled row order).
2. Elimination of FP, IGNORED_NONHARD, IGNORED_DONTCARE.
3. 1-to-1 bijection between TP and matched GT (no duplicate GT).
4. Features DataFrame does not contain any forbidden substrings.
5. Features DataFrame is fully compatible with extract_inference_features() and passes whitelist.
6. Superset relationship: floor ⊇ pass_thr.
7. Split T permission guard.
"""

import pytest
import numpy as np
import pandas as pd

from src.pipeline.build_dataset import (
    build_population_from_dfs,
    load_artifacts,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    compute_population_summary,
)
from src.residual.feature_extractor import extract_inference_features


@pytest.fixture
def synthetic_data():
    """Generate controlled synthetic detections, matches, and gt DataFrames."""
    # 2 frames: 000001 (drive 0001), 000002 (drive 0002)
    # Frame 1: 3 GT objects (idx 0, 1, 2)
    # Frame 2: 2 GT objects (idx 0, 1)
    # Total GT = 5
    gt_df = pd.DataFrame([
        # Frame 1
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "gt_idx": 0, "z_gt": 15.5, "x1": 100.0, "y1": 150.0, "x2": 200.0, "y2": 250.0, "difficulty": 0, "truncated": 0.0, "occluded": 0, "alpha": -0.5},
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "gt_idx": 1, "z_gt": 25.0, "x1": 300.0, "y1": 160.0, "x2": 380.0, "y2": 240.0, "difficulty": 1, "truncated": 0.1, "occluded": 1, "alpha": 0.2},
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "gt_idx": 2, "z_gt": 45.0, "x1": 500.0, "y1": 170.0, "x2": 550.0, "y2": 210.0, "difficulty": 2, "truncated": 0.0, "occluded": 2, "alpha": 1.5},
        # Frame 2
        {"frame_id": "000002", "drive": "2011_09_26_drive_0002", "gt_idx": 0, "z_gt": 10.2, "x1": 50.0, "y1": 140.0, "x2": 180.0, "y2": 260.0, "difficulty": 0, "truncated": 0.0, "occluded": 0, "alpha": -1.2},
        {"frame_id": "000002", "drive": "2011_09_26_drive_0002", "gt_idx": 1, "z_gt": 32.0, "x1": 400.0, "y1": 165.0, "x2": 470.0, "y2": 225.0, "difficulty": 1, "truncated": 0.3, "occluded": 1, "alpha": 0.0},
    ])

    # Predictions:
    # Frame 1:
    #  pred 0: pass_thr=True, matched to GT 0 (TP)
    #  pred 1: pass_thr=False, matched to GT 1 (TP at floor only)
    #  pred 2: pass_thr=True, FP (matched_gt_idx = -1)
    #  pred 3: pass_thr=True, IGNORED_NONHARD
    #  pred 4: pass_thr=True, IGNORED_DONTCARE
    # Frame 2:
    #  pred 0: pass_thr=True, matched to GT 0 (TP)
    #  pred 1: pass_thr=True, matched to GT 1 (TP)
    dets_df = pd.DataFrame([
        # Frame 1
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "pred_idx": 0, "x1": 102.0, "y1": 151.0, "x2": 198.0, "y2": 249.0, "confidence": 0.92, "class_id": 0, "pass_thr": True, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "pred_idx": 1, "x1": 298.0, "y1": 159.0, "x2": 382.0, "y2": 241.0, "confidence": 0.35, "class_id": 0, "pass_thr": False, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "pred_idx": 2, "x1": 600.0, "y1": 150.0, "x2": 700.0, "y2": 220.0, "confidence": 0.88, "class_id": 0, "pass_thr": True, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "pred_idx": 3, "x1": 750.0, "y1": 150.0, "x2": 820.0, "y2": 210.0, "confidence": 0.80, "class_id": 0, "pass_thr": True, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
        {"frame_id": "000001", "drive": "2011_09_26_drive_0001", "pred_idx": 4, "x1": 850.0, "y1": 150.0, "x2": 900.0, "y2": 200.0, "confidence": 0.82, "class_id": 0, "pass_thr": True, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
        # Frame 2
        {"frame_id": "000002", "drive": "2011_09_26_drive_0002", "pred_idx": 0, "x1": 52.0, "y1": 141.0, "x2": 179.0, "y2": 258.0, "confidence": 0.95, "class_id": 0, "pass_thr": True, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
        {"frame_id": "000002", "drive": "2011_09_26_drive_0002", "pred_idx": 1, "x1": 399.0, "y1": 166.0, "x2": 469.0, "y2": 224.0, "confidence": 0.85, "class_id": 0, "pass_thr": True, "fx": 721.5, "fy": 721.5, "cx": 609.5, "cy": 172.8, "img_w": 1242, "img_h": 375},
    ])

    matches_df = pd.DataFrame([
        # Frame 1
        {"frame_id": "000001", "pred_idx": 0, "status": "TP", "matched_gt_idx": 0, "matched_iou": 0.91},
        {"frame_id": "000001", "pred_idx": 1, "status": "TP", "matched_gt_idx": 1, "matched_iou": 0.88},
        {"frame_id": "000001", "pred_idx": 2, "status": "FP", "matched_gt_idx": -1, "matched_iou": 0.0},
        {"frame_id": "000001", "pred_idx": 3, "status": "IGNORED_NONHARD", "matched_gt_idx": -1, "matched_iou": 0.65},
        {"frame_id": "000001", "pred_idx": 4, "status": "IGNORED_DONTCARE", "matched_gt_idx": -1, "matched_iou": 0.70},
        # Frame 2
        {"frame_id": "000002", "pred_idx": 0, "status": "TP", "matched_gt_idx": 0, "matched_iou": 0.93},
        {"frame_id": "000002", "pred_idx": 1, "status": "TP", "matched_gt_idx": 1, "matched_iou": 0.89},
    ])

    return dets_df, matches_df, gt_df


def test_join_row_order_invariance(synthetic_data):
    """Shuffling rows in input tables should produce identical merged results."""
    dets_df, matches_df, gt_df = synthetic_data

    # Standard run
    feats_1, eval_1, fn_1 = build_population_from_dfs(dets_df, matches_df, gt_df, population="pass_thr")

    # Shuffled run
    np.random.seed(42)
    dets_shuffled = dets_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    matches_shuffled = matches_df.sample(frac=1.0, random_state=43).reset_index(drop=True)
    gt_shuffled = gt_df.sample(frac=1.0, random_state=44).reset_index(drop=True)

    feats_2, eval_2, fn_2 = build_population_from_dfs(dets_shuffled, matches_shuffled, gt_shuffled, population="pass_thr")

    # Sort by frame_id, pred_idx for comparison
    f1_sorted = feats_1.sort_values(["frame_id", "pred_idx"]).reset_index(drop=True)
    f2_sorted = feats_2.sort_values(["frame_id", "pred_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(f1_sorted, f2_sorted)

    e1_sorted = eval_1.sort_values(["frame_id", "pred_idx"]).reset_index(drop=True)
    e2_sorted = eval_2.sort_values(["frame_id", "pred_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(e1_sorted, e2_sorted)

    fn1_sorted = fn_1.sort_values(["frame_id", "gt_idx"]).reset_index(drop=True)
    fn2_sorted = fn_2.sort_values(["frame_id", "gt_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(fn1_sorted, fn2_sorted)


def test_fp_and_ignored_excluded_from_population(synthetic_data):
    """FP, IGNORED_NONHARD, IGNORED_DONTCARE must be excluded from features and eval."""
    dets_df, matches_df, gt_df = synthetic_data
    feats, eval_df, fn_df = build_population_from_dfs(dets_df, matches_df, gt_df, population="pass_thr")

    # Frame 1 preds 2 (FP), 3 (IGNORED_NONHARD), 4 (IGNORED_DONTCARE) must NOT be present
    frame1_preds = eval_df[eval_df["frame_id"] == "000001"]["pred_idx"].tolist()
    assert 2 not in frame1_preds
    assert 3 not in frame1_preds
    assert 4 not in frame1_preds

    # At pass_thr, frame 1 pred 1 (confidence below thr) is also excluded
    assert 1 not in frame1_preds
    # Only pred 0 is in TP for frame 1
    assert frame1_preds == [0]


def test_tp_gt_bijection_and_no_duplicate_gt(synthetic_data):
    """Every TP corresponds to exactly one GT object, and no GT is matched twice."""
    dets_df, matches_df, gt_df = synthetic_data
    feats, eval_df, fn_df = build_population_from_dfs(dets_df, matches_df, gt_df, population="pass_thr")

    assert len(feats) == len(eval_df)
    # Check uniqueness of (frame_id, gt_idx)
    assert not eval_df.duplicated(subset=["frame_id", "gt_idx"]).any()
    # Check uniqueness of (frame_id, pred_idx)
    assert not eval_df.duplicated(subset=["frame_id", "pred_idx"]).any()

    # Check conservation: len(eval) + len(fn) == total_gt
    assert len(eval_df) + len(fn_df) == len(gt_df)


def test_features_contain_no_forbidden_substrings(synthetic_data):
    """Features columns must never contain forbidden substrings (gt, depth, alpha, etc.)."""
    dets_df, matches_df, gt_df = synthetic_data
    feats, _, _ = build_population_from_dfs(dets_df, matches_df, gt_df, population="pass_thr")

    for col in feats.columns:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert forbidden not in col_lower, f"Forbidden substring '{forbidden}' in column '{col}'"


def test_features_compatible_with_extractor_and_whitelist(synthetic_data):
    """extract_inference_features(features) must succeed and pass whitelist (Decision D11)."""
    dets_df, matches_df, gt_df = synthetic_data
    feats, _, _ = build_population_from_dfs(dets_df, matches_df, gt_df, population="pass_thr")

    inf_feats = extract_inference_features(feats)
    assert isinstance(inf_feats, pd.DataFrame)
    assert len(inf_feats) == len(feats)
    # Ensure all columns in inf_feats are safe
    for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
        for col in inf_feats.columns:
            assert forbidden not in col.lower()


def test_floor_superset_of_pass_thr(synthetic_data):
    """Population 'floor' must be a superset of 'pass_thr' (floor ⊇ pass_thr)."""
    dets_df, matches_df, gt_df = synthetic_data
    feats_thr, eval_thr, fn_thr = build_population_from_dfs(dets_df, matches_df, gt_df, population="pass_thr")
    feats_flr, eval_flr, fn_flr = build_population_from_dfs(dets_df, matches_df, gt_df, population="floor")

    # Pass_thr TP keys must be a subset of floor TP keys
    thr_keys = set(zip(eval_thr["frame_id"], eval_thr["pred_idx"]))
    flr_keys = set(zip(eval_flr["frame_id"], eval_flr["pred_idx"]))

    assert thr_keys.issubset(flr_keys)
    assert len(flr_keys) > len(thr_keys)  # In synthetic data, pred 1 is added in floor

    # Correspondingly, fn_flr must be a subset of fn_thr
    fn_thr_keys = set(zip(fn_thr["frame_id"], fn_thr["gt_idx"]))
    fn_flr_keys = set(zip(fn_flr["frame_id"], fn_flr["gt_idx"]))
    assert fn_flr_keys.issubset(fn_thr_keys)


def test_split_t_permission_guard():
    """Attempting to load split T must raise PermissionError (Decision D4, AGENT_RULES.md §1.1)."""
    with pytest.raises(PermissionError, match="Split T is strictly forbidden"):
        load_artifacts("yolov8s_640", "T")


def test_compute_population_summary(synthetic_data):
    """Summary counts must accurately partition all detections and ground truth."""
    dets_df, matches_df, gt_df = synthetic_data
    summary = compute_population_summary(
        dets_df, matches_df, gt_df,
        model_key="test_model", split="B", population="pass_thr",
    )

    assert summary["n_gt"] == 5
    assert summary["n_TP"] == 3  # (frame 1 pred 0, frame 2 pred 0, frame 2 pred 1)
    assert summary["n_FN"] == 2  # 5 - 3 = 2
    assert summary["n_FP"] == 1  # frame 1 pred 2
    assert summary["n_ignored_nonhard"] == 1
    assert summary["n_ignored_dontcare"] == 1
    assert summary["n_ignored"] == 2
    assert summary["total_preds_considered"] == 6  # 5 pass_thr in frame 1 (pred 0,2,3,4) + 2 in frame 2 = 6
