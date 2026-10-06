"""
src/pipeline/apply_frozen.py: Shared inference pipeline for out-of-sample splits (C and T) (Decisions D13, D29, D49).

Ensures exact code path and mathematical exchangeability between Split C (calibration) and Split T (testing):
1. Reads detector features, cues, and evaluation metadata for the target split.
2. Applies frozen geometric fusion weights (full_fw.json fit on full Split B) to compute Z_d.
3. Asserts exact numerical match against stored z_d in {model}_{split}_cues.parquet (D49).
4. Handles fallback pattern 000 using Model (e) fit on full Split B to produce seamless Z_base.
5. Builds feature matrices with derived ln_z_base for residual/quantile model inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.geometry.fusion import load_fusion_weights
from src.pipeline.geometry_stage import fuse_with_weights
from src.residual.feature_extractor import extract_inference_features
from src.residual.models import (
    build_feature_matrices,
    load_model_e,
    predict_e,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def apply_frozen_pipeline(
    model_key: str,
    split: str = "C",
    data_dir: Path | str | None = None,
    runs_dir: Path | str | None = None,
    verify_stored_zd: bool = True,
) -> dict[str, Any]:
    """
    Execute frozen pipeline inference for Split C or Split T (Decision D49).

    Args:
        model_key: Detector model identifier (e.g. 'yolo11s_640').
        split: Target split name ('C' for calibration, 'T' for final test).
        data_dir: Directory containing parquet dataset artifacts (defaults to results/datasets).
        runs_dir: Directory containing trained model artifacts (defaults to runs/residual).
        verify_stored_zd: If True and cues parquet has 'z_d', asserts numerical equality.

    Returns:
        dict containing:
            - 'model_key': str
            - 'split': str
            - 'features_df': pd.DataFrame
            - 'cues_df': pd.DataFrame
            - 'eval_df': pd.DataFrame | None
            - 'z_d': np.ndarray (fused depth from geometry)
            - 'z_base': np.ndarray (fused depth with fallback handling)
            - 'fallback_flag': np.ndarray (bool mask for pattern 000)
            - 'base_feats': pd.DataFrame (17 canonical inference features)
            - 'feat_mats': dict[str, pd.DataFrame] (matrices for f0, f, e)
            - 'r_actual': np.ndarray | None (log-residual ln(Z_gt) - ln(Z_base) if eval exists)
            - 'n_samples': int
    """
    d_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "results" / "datasets"
    r_dir = Path(runs_dir) if runs_dir is not None else PROJECT_ROOT / "runs" / "residual"

    feat_path = d_dir / f"{model_key}_{split}_features.parquet"
    cues_path = d_dir / f"{model_key}_{split}_cues.parquet"
    eval_path = d_dir / f"{model_key}_{split}_eval.parquet"

    fw_path = r_dir / model_key / "full_fw.json"
    model_e_path = r_dir / model_key / "model_e.json"

    if not feat_path.is_file():
        raise FileNotFoundError(f"Missing features parquet: {feat_path}")
    if not cues_path.is_file():
        raise FileNotFoundError(f"Missing cues parquet: {cues_path}")
    if not fw_path.is_file():
        raise FileNotFoundError(f"Missing full fusion weights: {fw_path}")

    features_df = pd.read_parquet(feat_path)
    cues_df = pd.read_parquet(cues_path)
    eval_df = pd.read_parquet(eval_path) if eval_path.is_file() else None

    n_samples = len(features_df)
    if len(cues_df) != n_samples:
        raise ValueError(f"Length mismatch: features ({n_samples}) != cues ({len(cues_df)})")
    if eval_df is not None and len(eval_df) != n_samples:
        raise ValueError(f"Length mismatch: features ({n_samples}) != eval ({len(eval_df)})")

    # Step 1: Compute Z_d using frozen full Split B fusion weights (Decision D29)
    full_fw = load_fusion_weights(fw_path)
    z_d = fuse_with_weights(cues_df, full_fw)

    # Step 2: Assert match against stored z_d in cues parquet (Decision D49)
    if verify_stored_zd and "z_d" in cues_df.columns:
        stored_zd = cues_df["z_d"].to_numpy(dtype=float)
        valid_mask = ~np.isnan(stored_zd)
        if np.any(valid_mask):
            diff = np.abs(z_d[valid_mask] - stored_zd[valid_mask])
            max_diff = float(np.max(diff))
            if not np.allclose(z_d[valid_mask], stored_zd[valid_mask], atol=1e-5):
                raise AssertionError(
                    f"Recalculated Z_d does not match stored cues parquet for {model_key} {split}! "
                    f"Max absolute diff: {max_diff:.6e}"
                )

    # Step 3: Identify pattern 000 (all cues invalid)
    valid_w = cues_df["valid_w"].to_numpy(dtype=bool)
    valid_h = cues_df["valid_h"].to_numpy(dtype=bool)
    valid_g = cues_df["valid_g"].to_numpy(dtype=bool)
    pattern_000 = (~valid_w) & (~valid_h) & (~valid_g)

    # Step 4: Extract base inference features
    combined = features_df.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined[col] = cues_df[col]
    base_feats = extract_inference_features(combined)

    # Step 5: Fallback to Model (e) for pattern 000 (Decision D13, D34)
    if np.any(pattern_000):
        if not model_e_path.is_file():
            raise FileNotFoundError(f"Missing Model (e) for fallback: {model_e_path}")
        model_e = load_model_e(model_e_path)
        z_hat_e, _ = predict_e(model_e, base_feats)
        z_base = np.where(~pattern_000, z_d, z_hat_e)
    else:
        z_base = z_d.copy()

    # Step 6: Verify Z_base invariants
    if np.any(np.isnan(z_base)):
        raise ValueError(f"z_base contains {np.sum(np.isnan(z_base))} NaNs")
    if np.any(z_base <= 0):
        raise ValueError(f"z_base contains {np.sum(z_base <= 0)} non-positive values")

    # Step 7: Build full feature matrices with derived ln_z_base (Decision D30)
    feat_mats = build_feature_matrices(base_feats, z_base=z_base)

    # Step 8: Log-residual target if ground truth is present
    r_actual = None
    if eval_df is not None and "z_gt" in eval_df.columns:
        z_gt = eval_df["z_gt"].to_numpy(dtype=float)
        r_actual = np.log(z_gt) - np.log(z_base)

    return {
        "model_key": model_key,
        "split": split,
        "features_df": features_df,
        "cues_df": cues_df,
        "eval_df": eval_df,
        "z_d": z_d,
        "z_base": z_base,
        "fallback_flag": pattern_000,
        "base_feats": base_feats,
        "feat_mats": feat_mats,
        "r_actual": r_actual,
        "n_samples": n_samples,
    }
