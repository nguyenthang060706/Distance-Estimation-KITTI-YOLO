"""
tests/test_geometry_stage.py: Unit tests for src/pipeline/geometry_stage.py.

Verifies:
- LODO fusion has zero leakage across folds (train mask strictly excludes held drive).
- NaN in one cue does not contaminate other cues or block fusion when other cues are valid.
- load_geometry_priors correctly reads calibrated parameters from configs/geometry_params.yaml.
- add_cues properly computes cues using per-frame intrinsics and boundary rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.geometry.geometric_cues import GeometricPriors
from src.pipeline.geometry_stage import (
    add_cues,
    fit_fusion_lodo,
    fuse_with_weights,
    load_frozen_gt_fusion_weights,
    load_geometry_priors,
)


def test_load_geometry_priors() -> None:
    priors, cfg = load_geometry_priors()
    assert isinstance(priors, GeometricPriors)
    assert priors.W_eff == pytest.approx(2.6184, rel=1e-3)
    assert priors.H_obj == pytest.approx(1.6797, rel=1e-3)
    assert priors.H_cam == pytest.approx(2.0422, rel=1e-3)
    assert priors.delta_horizon == pytest.approx(-4.6782, rel=1e-3)
    assert "priors_split_A" in cfg
    assert "fusion_split_B" in cfg


def test_load_frozen_gt_fusion_weights() -> None:
    fw = load_frozen_gt_fusion_weights()
    assert len(fw.weights) == 3
    assert np.sum(fw.weights) == pytest.approx(1.0, abs=1e-3)
    assert fw.weights[0] == pytest.approx(0.0807, abs=1e-3)
    assert fw.weights[1] == pytest.approx(0.6634, abs=1e-3)
    assert fw.weights[2] == pytest.approx(0.2560, abs=1e-3)


def test_add_cues_border_masking_and_nan_isolation() -> None:
    # Synthetic frame with 3 detections:
    # 1. Normal box well inside image
    # 2. Box touching left border (x1=0 <= eps) -> Z_w must be NaN, Z_h & Z_g must be valid
    # 3. Box touching bottom border (y2=374 >= img_h - 1 - eps) -> Z_h & Z_g must be NaN, Z_w must be valid
    features = pd.DataFrame({
        "frame_id": ["000001", "000001", "000001"],
        "drive": ["d001", "d001", "d001"],
        "pred_idx": [0, 1, 2],
        "x1": [100.0, 0.0, 200.0],
        "y1": [180.0, 180.0, 250.0],
        "x2": [150.0, 50.0, 260.0],
        "y2": [220.0, 220.0, 374.0],
        "confidence": [0.9, 0.85, 0.88],
        "class_id": [0, 0, 0],
        "fx": [721.53, 721.53, 721.53],
        "fy": [721.53, 721.53, 721.53],
        "cx": [609.55, 609.55, 609.55],
        "cy": [172.85, 172.85, 172.85],
        "img_w": [1242, 1242, 1242],
        "img_h": [375, 375, 375],
    })

    priors, _ = load_geometry_priors()
    df_cues = add_cues(features, priors=priors, eps=2.0)

    # Box 1: All cues valid
    assert df_cues.loc[0, "valid_w"] and np.isfinite(df_cues.loc[0, "z_w"])
    assert df_cues.loc[0, "valid_h"] and np.isfinite(df_cues.loc[0, "z_h"])
    assert df_cues.loc[0, "valid_g"] and np.isfinite(df_cues.loc[0, "z_g"])

    # Box 2: Touches left border -> Z_w is NaN/invalid, but Z_h and Z_g remain valid!
    assert not df_cues.loc[1, "valid_w"]
    assert np.isnan(df_cues.loc[1, "z_w"])
    assert df_cues.loc[1, "valid_h"] and np.isfinite(df_cues.loc[1, "z_h"])
    assert df_cues.loc[1, "valid_g"] and np.isfinite(df_cues.loc[1, "z_g"])

    # Box 3: Touches bottom border -> Z_h and Z_g are NaN/invalid, but Z_w remains valid!
    assert df_cues.loc[2, "valid_w"] and np.isfinite(df_cues.loc[2, "z_w"])
    assert not df_cues.loc[2, "valid_h"]
    assert np.isnan(df_cues.loc[2, "z_h"])
    assert not df_cues.loc[2, "valid_g"]
    assert np.isnan(df_cues.loc[2, "z_g"])

    # Fusion on these boxes: NaN does not cause Z_d to become NaN as long as >= 1 cue is valid
    fw = load_frozen_gt_fusion_weights()
    z_d = fuse_with_weights(df_cues, fw)
    assert np.all(np.isfinite(z_d))
    # Box 2 (w missing) should fuse h and g
    assert z_d[1] > 0
    # Box 3 (h and g missing) should fall back directly to Z_w
    assert z_d[2] == pytest.approx(df_cues.loc[2, "z_w"])


def test_fit_fusion_lodo_no_leakage() -> None:
    # Create synthetic dataset with 4 drives, 25 rows per drive
    rng = np.random.default_rng(42)
    n_drives = 4
    rows_per_drive = 25
    n = n_drives * rows_per_drive

    drives = np.repeat([f"drive_{i:02d}" for i in range(n_drives)], rows_per_drive)
    z_gt = rng.uniform(10.0, 40.0, size=n)

    # Synthetic cues with realistic noise
    z_w = z_gt * np.exp(rng.normal(0, 0.08, size=n))
    z_h = z_gt * np.exp(rng.normal(0, 0.05, size=n))
    z_g = z_gt * np.exp(rng.normal(0, 0.10, size=n))

    # Mask a few values
    valid_w = rng.uniform(size=n) > 0.1
    valid_h = rng.uniform(size=n) > 0.05
    valid_g = rng.uniform(size=n) > 0.1
    z_w[~valid_w] = np.nan
    z_h[~valid_h] = np.nan
    z_g[~valid_g] = np.nan

    cues_df = pd.DataFrame({
        "drive": drives,
        "z_w": z_w,
        "z_h": z_h,
        "z_g": z_g,
        "valid_w": valid_w,
        "valid_h": valid_h,
        "valid_g": valid_g,
    })

    z_d_oof, full_b_weights, lodo_fits = fit_fusion_lodo(cues_df, z_gt, min_train_drives=2)

    # 1. Verify LODO predictions are finite where at least 1 cue is valid
    any_valid = valid_w | valid_h | valid_g
    assert np.all(np.isfinite(z_d_oof[any_valid]))

    # 2. Strict no-leakage test:
    # For a specific drive, fit directly on dataset WITHOUT that drive, and assert identical weights
    test_drive = "drive_01"
    held_mask = drives == test_drive
    train_mask = ~held_mask

    # Manual fit on train_mask
    from src.geometry.fusion import fit_fusion_weights
    z_cues = cues_df[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = cues_df[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)

    fw_manual = fit_fusion_weights(
        Z_cues=z_cues[train_mask],
        Z_gt=z_gt[train_mask],
        valid_mask=valid_mask[train_mask],
        drive_ids=drives[train_mask],
    )

    fw_lodo = lodo_fits[test_drive]
    np.testing.assert_allclose(fw_manual.weights, fw_lodo.weights, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fw_manual.cov_shrunk, fw_lodo.cov_shrunk, rtol=1e-5, atol=1e-5)
    assert fw_lodo.n_drives == 3  # exactly n_drives - 1
