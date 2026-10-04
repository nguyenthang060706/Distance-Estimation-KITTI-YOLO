"""
src/pipeline/geometry_stage.py: Geometric cues computation, LODO fusion, and evaluation stage (Decisions D10, D14, D17, D29).

Key components:
- load_geometry_priors: Loads calibrated priors from configs/geometry_params.yaml (never reads deprecated geometry_priors.yaml).
- add_cues: Vectorised batch computation of width (Z_w), height (Z_h), and ground (Z_g) cues with per-image camera intrinsics and 2px border tolerance.
- fit_fusion_lodo: Leave-One-Drive-Out (LODO) log-space fusion on Split B to prevent in-sample lookahead leakage for residual targets (D29), plus full Split B refit for Split C.
- load_frozen_gt_fusion_weights: Loads theoretical lower-bound weights from geometry-v2 for baseline comparison (D17).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.geometry.geometric_cues import (
    BORDER_EPS,
    CameraIntrinsics,
    CueResult,
    GeometricPriors,
    compute_cues_batch,
    load_geometry_v2,
)
from src.geometry.fusion import (
    CUE_NAMES,
    FusionWeights,
    fit_fusion_weights,
    fuse_depths_vectorised,
)


def load_geometry_priors(
    yaml_path: str | Path = "configs/geometry_params.yaml",
) -> tuple[GeometricPriors, dict]:
    """
    Load calibrated geometry-v2 GeometricPriors and full config dictionary.

    Strictly reads configs/geometry_params.yaml (Decision D10/D14).
    DO NOT read configs/geometry_priors.yaml (deprecated, uncalibrated).
    """
    p = Path(yaml_path)
    if not p.is_file():
        raise FileNotFoundError(f"Geometry config file not found: {p.resolve()}")

    return load_geometry_v2(str(p))


def load_frozen_gt_fusion_weights(
    yaml_path: str | Path = "configs/geometry_params.yaml",
) -> FusionWeights:
    """
    Load frozen GT-bbox fusion weights and covariance from configs/geometry_params.yaml (D10, D17).

    Returns a FusionWeights object representing the theoretical lower bound on GT bboxes.
    """
    _, cfg = load_geometry_priors(yaml_path)
    fusion_cfg = cfg["fusion_split_B"]

    weights = np.array(fusion_cfg["weights"], dtype=float)
    cov_shrunk = np.array(fusion_cfg["covariance_shrunk"], dtype=float)
    alpha = float(fusion_cfg.get("shrinkage_alpha", 0.0008))
    cue_names = list(fusion_cfg.get("cue_order", CUE_NAMES))

    return FusionWeights(
        weights=weights,
        cov_matrix=cov_shrunk,
        cov_shrunk=cov_shrunk,
        shrinkage_alpha=alpha,
        cue_names=cue_names,
        constrained=bool(np.any(weights < 0)),
        n_samples=int(cfg["metadata"].get("split_B_frames", 4776)),
        n_drives=12,
    )


def add_cues(
    features: pd.DataFrame,
    priors: Optional[GeometricPriors] = None,
    eps: float = BORDER_EPS,
) -> pd.DataFrame:
    """
    Compute geometric depth cues (Z_w, Z_h, Z_g) and validity flags for detections.

    Computation is batched per frame_id using exact per-frame camera intrinsics (fx, fy, cx, cy)
    and exact image dimensions (img_w, img_h).

    Args:
        features: DataFrame containing canonical feature columns (x1, y1, x2, y2, fx, fy, cx, cy, img_w, img_h).
        priors: Optional pre-loaded GeometricPriors (defaults to configs/geometry_params.yaml).
        eps: Border tolerance in pixels (default: 2.0 px).

    Returns:
        pd.DataFrame: A copy of features with new columns:
            z_w, z_h, z_g (float, NaN if masked/invalid)
            valid_w, valid_h, valid_g (bool, True if valid)
    """
    if priors is None:
        priors, _ = load_geometry_priors()

    df = features.copy()
    n = len(df)

    z_w = np.full(n, np.nan, dtype=float)
    z_h = np.full(n, np.nan, dtype=float)
    z_g = np.full(n, np.nan, dtype=float)
    valid_w = np.zeros(n, dtype=bool)
    valid_h = np.zeros(n, dtype=bool)
    valid_g = np.zeros(n, dtype=bool)

    if n == 0:
        df["z_w"] = z_w
        df["z_h"] = z_h
        df["z_g"] = z_g
        df["valid_w"] = valid_w
        df["valid_h"] = valid_h
        df["valid_g"] = valid_g
        return df

    # Group by frame_id to use per-frame intrinsics and image dimensions
    for frame_id, frame_df in df.groupby("frame_id", sort=False):
        row0 = frame_df.iloc[0]
        intrinsics = CameraIntrinsics(
            fx=float(row0["fx"]),
            fy=float(row0["fy"]),
            cx=float(row0["cx"]),
            cy=float(row0["cy"]),
        )
        img_w = int(row0["img_w"])
        img_h = int(row0["img_h"])

        bboxes = frame_df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        indices = frame_df.index.to_numpy()

        # Vectorised cues for this frame
        res: CueResult = compute_cues_batch(
            bboxes=bboxes,
            intrinsics=intrinsics,
            priors=priors,
            img_width=img_w,
            img_height=img_h,
            eps=eps,
        )

        pos = df.index.get_indexer(indices)
        z_w[pos] = res.Z_w
        z_h[pos] = res.Z_h
        z_g[pos] = res.Z_g
        valid_w[pos] = res.valid_w
        valid_h[pos] = res.valid_h
        valid_g[pos] = res.valid_g

    df["z_w"] = z_w
    df["z_h"] = z_h
    df["z_g"] = z_g
    df["valid_w"] = valid_w
    df["valid_h"] = valid_h
    df["valid_g"] = valid_g

    return df


def fit_fusion_lodo(
    cues_df: pd.DataFrame,
    z_gt: np.ndarray | pd.Series,
    min_train_drives: int = 3,
) -> tuple[np.ndarray, FusionWeights, dict[str, FusionWeights]]:
    """
    Leave-One-Drive-Out (LODO) fusion on Split B (Decisions D16c, D29).

    For each drive d:
        - Weights and covariance are fit strictly on all other drives (out-of-fold).
        - Prevents in-sample lookahead bias when constructing residual targets r = ln(Z_gt) - ln(Z_d).

    In addition:
        - Fits global weights on all drives of Split B to be used when evaluating Split C and Split T.

    Args:
        cues_df: DataFrame with columns z_w, z_h, z_g, valid_w, valid_h, valid_g, drive.
        z_gt: Ground truth distance array matching rows of cues_df.
        min_train_drives: Minimum required training drives in each fold.

    Returns:
        (z_d_oof, full_b_weights, lodo_fits_dict):
            z_d_oof: (N,) array of out-of-fold fused depth for Split B.
            full_b_weights: FusionWeights fit on all Split B samples.
            lodo_fits_dict: Mapping drive -> FusionWeights for each fold.
    """
    n = len(cues_df)
    z_gt_arr = np.asarray(z_gt, dtype=float)
    if len(z_gt_arr) != n:
        raise ValueError(f"z_gt length {len(z_gt_arr)} != cues_df length {n}")

    z_cues = cues_df[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = cues_df[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)
    drive_ids = cues_df["drive"].to_numpy(dtype=str)

    unique_drives = np.unique(drive_ids)
    z_d_oof = np.full(n, np.nan, dtype=float)
    lodo_fits: dict[str, FusionWeights] = {}

    for d in unique_drives:
        held = (drive_ids == d)
        train_mask = ~held
        train_drives = np.unique(drive_ids[train_mask])

        if len(train_drives) < min_train_drives:
            raise ValueError(
                f"Too few training drives for LODO fusion on fold '{d}': "
                f"{len(train_drives)} < {min_train_drives}"
            )

        # Fit weights strictly without fold d
        fw = fit_fusion_weights(
            Z_cues=z_cues[train_mask],
            Z_gt=z_gt_arr[train_mask],
            valid_mask=valid_mask[train_mask],
            drive_ids=drive_ids[train_mask],
        )

        # Predict held-out fold d using vectorised fusion
        z_d_oof[held] = fuse_depths_vectorised(
            Z_cues=z_cues[held],
            valid_mask=valid_mask[held],
            fusion_weights=fw,
        )
        lodo_fits[str(d)] = fw

    # Fit full model across all Split B samples (to be applied to Split C and Split T)
    full_b_weights = fit_fusion_weights(
        Z_cues=z_cues,
        Z_gt=z_gt_arr,
        valid_mask=valid_mask,
        drive_ids=drive_ids,
    )

    return z_d_oof, full_b_weights, lodo_fits


def fuse_with_weights(
    cues_df: pd.DataFrame,
    fusion_weights: FusionWeights,
) -> np.ndarray:
    """
    Fuse geometric cues using a fixed pre-fitted FusionWeights object.

    Args:
        cues_df: DataFrame with columns z_w, z_h, z_g, valid_w, valid_h, valid_g.
        fusion_weights: Pre-fitted FusionWeights (e.g. from Split B or frozen GT).

    Returns:
        (N,) array of fused depth estimates Z_d.
    """
    z_cues = cues_df[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = cues_df[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)
    return fuse_depths_vectorised(z_cues, valid_mask, fusion_weights)
