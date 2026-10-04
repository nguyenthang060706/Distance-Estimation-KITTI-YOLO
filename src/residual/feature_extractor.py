"""
src/residual/feature_extractor.py: Feature extractor and strict ground-truth separation guard (§5.3, Decision D11).

Inference features must ONLY use information available at test time:
- Bounding box geometry: w, h, w_h_ratio, y_bottom_minus_cy, cx_offset_norm
- Detection attributes: confidence, class_id
- Boundary contact flags: touch_left, touch_right, touch_top, touch_bottom
- Geometric depth cues: ln_Z_w, ln_Z_h, ln_Z_g
- Cue validity flags: valid_w, valid_h, valid_g

FORBIDDEN: Any ground truth labels (alpha, truncated, occluded, depth, location_z, or gt_*).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Set
from src.geometry.geometric_cues import BORDER_EPS

# Strictly forbidden ground truth columns that must never leak into features (Decision D11)
FORBIDDEN_GT_EXACT = {
    "alpha",
    "truncated",
    "occluded",
    "depth",
    "location_x",
    "location_y",
    "location_z",
    "z_gt",
    "rotation_y",
}

FORBIDDEN_GT_PREFIX = "gt_"

# Approved inference feature whitelist (derived from configs/residual/residual_config.yaml)
DEFAULT_FEATURE_WHITELIST = {
    "w",
    "h",
    "w_h_ratio",
    "y_bottom_minus_cy",
    "cx_offset_norm",
    "touch_left",
    "touch_right",
    "touch_top",
    "touch_bottom",
    "confidence",
    "class_id",
    "valid_w",
    "valid_h",
    "valid_g",
    "ln_z_w",
    "ln_z_h",
    "ln_z_g",
}


def check_no_gt_leakage(columns: List[str] | Set[str]) -> None:
    """
    Validates that a list of feature column names does not contain any ground-truth fields.

    Raises:
        ValueError: If any forbidden column or gt_* prefix is present.
    """
    for col in columns:
        col_lower = col.strip().lower()
        if col_lower in FORBIDDEN_GT_EXACT:
            raise ValueError(
                f"Data leakage violation (Decision D11): Forbidden ground truth column '{col}' "
                f"detected in feature matrix."
            )
        if col_lower.startswith(FORBIDDEN_GT_PREFIX):
            raise ValueError(
                f"Data leakage violation (Decision D11): Column '{col}' with forbidden prefix "
                f"'{FORBIDDEN_GT_PREFIX}' detected in feature matrix."
            )


def validate_feature_columns_against_whitelist(
    columns: List[str] | Set[str],
    whitelist: Set[str] = DEFAULT_FEATURE_WHITELIST,
) -> None:
    """
    Validates that every feature column is strictly in the allowed whitelist (Decision D11).
    Raises ValueError if any unknown column (e.g. dist, y3d, etc.) is found.
    """
    check_no_gt_leakage(columns)
    for col in columns:
        if col not in whitelist:
            raise ValueError(
                f"Feature whitelist violation (Decision D11): Column '{col}' is not in the "
                f"approved inference feature whitelist."
            )


def extract_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features for residual prediction, ensuring zero leakage of ground truth.

    Expected input df columns (at least):
        - bbox: [x1, y1, x2, y2] or separate columns (x1, y1, x2, y2)
        - img_w, img_h, cy
        - confidence (score)
        - class_id
        - z_w, z_h, z_g
        - valid_w, valid_h, valid_g

    Returns:
        pd.DataFrame containing only safe, validated inference features.
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Bbox geometry
    if "bbox" in df.columns:
        bboxes = np.vstack(df["bbox"].values)
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    else:
        x1, y1, x2, y2 = df["x1"], df["y1"], df["x2"], df["y2"]

    w = np.maximum(x2 - x1, 1.0)
    h = np.maximum(y2 - y1, 1.0)
    feats["w"] = w
    feats["h"] = h
    feats["w_h_ratio"] = w / h

    # Normalised center offset from optical center (§5.3: (cx - cx0) / fx or img_w)
    cx = df["cx"] if "cx" in df.columns else df["img_w"] / 2.0
    cy = df["cy"] if "cy" in df.columns else df["img_h"] / 2.0
    box_cx = (x1 + x2) / 2.0
    if "fx" in df.columns:
        feats["cx_offset_norm"] = (box_cx - cx) / df["fx"]
    else:
        feats["cx_offset_norm"] = (box_cx - cx) / df["img_w"]
    feats["y_bottom_minus_cy"] = y2 - cy

    # 2. Boundary contact flags (using canonical BORDER_EPS = 2.0 px)
    eps = BORDER_EPS
    feats["touch_left"] = (x1 <= eps).astype(int)
    feats["touch_right"] = (x2 >= (df["img_w"] - 1 - eps)).astype(int)
    feats["touch_top"] = (y1 <= eps).astype(int)
    feats["touch_bottom"] = (y2 >= (df["img_h"] - 1 - eps)).astype(int)

    # 3. Detection attributes
    if "confidence" in df.columns:
        feats["confidence"] = df["confidence"].astype(float)
    if "class_id" in df.columns:
        # Note: In single-class Car detection, class_id is constant (0)
        feats["class_id"] = df["class_id"].astype(int)

    # 4. Geometric depth cues in log-space (with safe NaN replacement + validity flags)
    for cue_col, valid_col in [("z_w", "valid_w"), ("z_h", "valid_h"), ("z_g", "valid_g")]:
        if cue_col in df.columns:
            valid = df[valid_col].astype(bool) if valid_col in df.columns else (df[cue_col] > 0) & df[cue_col].notna()
            feats[valid_col] = valid.astype(int)
            # Safe log: use log(val) where valid, 0.0 where invalid
            log_val = np.zeros(len(df), dtype=float)
            mask = valid & (df[cue_col] > 0)
            log_val[mask] = np.log(df.loc[mask, cue_col].astype(float))
            feats[f"ln_{cue_col}"] = log_val

    # Strict check before returning (blacklist + whitelist, Decision D11)
    validate_feature_columns_against_whitelist(feats.columns)
    return feats

