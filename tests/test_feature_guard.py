"""
tests/test_feature_guard.py: Unit tests for Decision D11.
Verifies that feature extractor strictly prevents data leakage by rejecting
ground truth columns (alpha, truncated, occluded, depth, location_z, gt_*).
"""

import pytest
import pandas as pd
import numpy as np

from src.residual.feature_extractor import (
    check_no_gt_leakage,
    extract_inference_features,
    validate_feature_columns_against_whitelist,
    DEFAULT_FEATURE_WHITELIST,
)



def test_clean_features_pass():
    """Valid feature column names must pass without error."""
    safe_cols = [
        "w", "h", "w_h_ratio", "y_bottom_minus_cy", "cx_offset_norm",
        "touch_left", "touch_right", "touch_top", "touch_bottom",
        "confidence", "class_id",
        "ln_z_w", "ln_z_h", "ln_z_g",
        "valid_w", "valid_h", "valid_g",
    ]
    check_no_gt_leakage(safe_cols)


@pytest.mark.parametrize("forbidden_col", [
    "alpha", "truncated", "occluded", "depth", "location_z",
    "location_x", "location_y", "z_gt", "rotation_y",
    "gt_depth", "gt_alpha", "gt_truncated", "gt_occluded", "gt_box",
])
def test_forbidden_gt_columns_raise_error(forbidden_col):
    """Any forbidden ground truth column must raise ValueError."""
    cols = ["w", "h", "confidence", forbidden_col]
    with pytest.raises(ValueError, match="Data leakage violation"):
        check_no_gt_leakage(cols)


def test_extract_inference_features_clean_output():
    """Extractor produces a DataFrame containing only safe features, even if input has GT."""
    raw_df = pd.DataFrame({
        "bbox": [[100.0, 50.0, 200.0, 150.0], [50.0, 20.0, 120.0, 80.0]],
        "img_w": [1242, 1224],
        "img_h": [375, 370],
        "cy": [172.8, 168.0],
        "confidence": [0.85, 0.92],
        "class_id": [0, 0],
        "z_w": [25.0, np.nan],
        "z_h": [24.5, 30.0],
        "z_g": [26.0, 29.5],
        "valid_w": [True, False],
        "valid_h": [True, True],
        "valid_g": [True, True],
        # Input contains GT labels
        "z_gt": [25.2, 30.1],
        "alpha": [-0.5, 1.2],
        "truncated": [0.0, 0.2],
        "occluded": [0, 1],
    })

    feats = extract_inference_features(raw_df)

    # Output MUST NOT contain any of the GT columns
    for forbidden in ["z_gt", "alpha", "truncated", "occluded", "depth"]:
        assert forbidden not in feats.columns

    # Verify check_no_gt_leakage passes on output columns
    check_no_gt_leakage(feats.columns)

    # Verify expected feature columns match whitelist exactly
    assert set(feats.columns) == DEFAULT_FEATURE_WHITELIST


@pytest.mark.parametrize("unapproved_col", [
    "dist", "y3d", "z_true", "pseudo_depth", "car_speed", "weather",
])
def test_unapproved_columns_rejected_by_whitelist(unapproved_col):
    """Any unapproved column not in the whitelist must raise ValueError."""
    cols = ["w", "h", "confidence", unapproved_col]
    with pytest.raises(ValueError, match="Feature whitelist violation"):
        validate_feature_columns_against_whitelist(cols)

