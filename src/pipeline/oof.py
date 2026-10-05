"""
src/pipeline/oof.py: Nested Leave-One-Drive-Out (LODO) pipeline on Split B (Decisions D13, D16b, D24, D25, D29, D30, D33, D34).

Key components:
- Outer LODO: 12 folds over Split B drives.
- For each outer fold d:
  - Z_d: evaluated using fusion weights w_-d fit strictly on the remaining 11 drives.
  - Z_e for train (pattern 000): inner-LODO within the 11 train drives to prevent target lookahead.
  - Z_e for test fold d: evaluated using model (e) fit on all 11 train drives.
  - Z_base: Z_d if >= 1 valid cue (pattern != 000), fallback Z_e if pattern 000.
  - Target r = ln(Z_gt) - ln(Z_base).
  - Model (f0): Ridge + StandardScaler, alpha selected via inner GroupKFold CV.
  - Model (f): XGBoost residual, hyperparams selected via inner GroupKFold CV across 12 pre-registered configs.
  - Predictions on test fold d: z_hat_f0, z_hat_f, z_hat_e.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

from src.geometry.fusion import (
    FusionWeights,
    fit_fusion_weights,
    fuse_depths_vectorised,
)
from src.residual.feature_extractor import (
    extract_inference_features,
    check_no_gt_leakage,
)
from src.residual.models import (
    FEATURE_COLS_E,
    FEATURE_COLS_F,
    FEATURE_COLS_F0,
    build_feature_matrices,
    fit_e,
    fit_f,
    fit_f0,
    predict_e,
    predict_f,
    predict_f0,
)


def run_nested_lodo_b(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cues_df: pd.DataFrame,
    min_train_drives: int = 3,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: bool = True,
    scramble_train_targets: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Execute nested Leave-One-Drive-Out (LODO) cross-validation on Split B.

    Args:
        features_df: Canonical features parquet (frame_id, drive, pred_idx, x1..y2, confidence, etc.).
        eval_df: Evaluation parquet containing z_gt and metadata.
        cues_df: Geometric cues parquet (z_w, z_h, z_g, valid_w, valid_h, valid_g, z_d).
        min_train_drives: Minimum required training drives in each fold.
        random_state: Deterministic seed.
        n_jobs: Thread count for XGBoost (1 for strict reproducibility).
        verbose: If True, prints progress for each outer fold.

    Returns:
        (oof_df, fold_records):
            oof_df: DataFrame with columns:
                [frame_id, drive, pred_idx, z_d, z_base, z_hat_f0, z_hat_f, z_hat_e,
                 r_hat_f0, r_hat_f, fallback_flag]
            fold_records: List of dictionaries recording parameters per fold.
    """
    n_samples = len(features_df)
    if len(eval_df) != n_samples or len(cues_df) != n_samples:
        raise ValueError(
            f"Input row counts mismatch: features={len(features_df)}, "
            f"eval={len(eval_df)}, cues={len(cues_df)}"
        )

    # 1. Base inference features extraction (strictly no GT leakage, Decision D11)
    combined = features_df.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined[col] = cues_df[col]

    base_feats = extract_inference_features(combined)

    # Extract target and cues
    z_gt = eval_df["z_gt"].to_numpy(dtype=float)
    check_no_gt_leakage(base_feats.columns)
    if np.any(np.isnan(z_gt)) or np.any(z_gt <= 0):
        raise ValueError("z_gt must be strictly positive and finite")
    ln_z_gt = np.log(z_gt)

    z_cues = cues_df[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = cues_df[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)
    drive_ids = cues_df["drive"].to_numpy(dtype=str)

    pattern_000 = (~valid_mask[:, 0] & ~valid_mask[:, 1] & ~valid_mask[:, 2])
    fallback_flag = pattern_000.copy()

    unique_drives = np.unique(drive_ids)
    n_drives = len(unique_drives)
    if n_drives < min_train_drives + 1:
        raise ValueError(f"Too few drives for LODO: {n_drives} < {min_train_drives + 1}")

    # Output arrays
    z_d_test_arr = np.full(n_samples, np.nan, dtype=float)
    z_base_arr = np.full(n_samples, np.nan, dtype=float)
    z_hat_f0_arr = np.full(n_samples, np.nan, dtype=float)
    z_hat_f_arr = np.full(n_samples, np.nan, dtype=float)
    z_hat_e_arr = np.full(n_samples, np.nan, dtype=float)
    r_hat_f0_arr = np.full(n_samples, np.nan, dtype=float)
    r_hat_f_arr = np.full(n_samples, np.nan, dtype=float)

    fold_records: list[dict[str, Any]] = []

    for fold_idx, held_drive in enumerate(unique_drives):
        test_mask = (drive_ids == held_drive)
        train_mask = ~test_mask
        train_drives = drive_ids[train_mask]
        n_test = int(test_mask.sum())
        n_train = int(train_mask.sum())

        if verbose:
            print(f"[{fold_idx + 1:02d}/{n_drives:02d}] Outer LODO fold: drive '{held_drive}' "
                  f"(test={n_test}, train={n_train})...")

        # -------------------------------------------------------------
        # Step A: Fusion weights w_-d fit on train drives (D29, D34)
        # -------------------------------------------------------------
        fw_minus_d = fit_fusion_weights(
            Z_cues=z_cues[train_mask],
            Z_gt=z_gt[train_mask],
            valid_mask=valid_mask[train_mask],
            drive_ids=train_drives,
        )

        z_d_train = fuse_depths_vectorised(
            Z_cues=z_cues[train_mask],
            valid_mask=valid_mask[train_mask],
            fusion_weights=fw_minus_d,
        )
        z_d_test = fuse_depths_vectorised(
            Z_cues=z_cues[test_mask],
            valid_mask=valid_mask[test_mask],
            fusion_weights=fw_minus_d,
        )
        z_d_test_arr[test_mask] = z_d_test

        # -------------------------------------------------------------
        # Step B: Inner-LODO for Z_e on train drives (pattern 000, D34)
        # -------------------------------------------------------------
        train_indices = np.where(train_mask)[0]
        inner_unique_drives = np.unique(train_drives)
        z_e_train_oof = np.full(n_train, np.nan, dtype=float)

        for inner_d in inner_unique_drives:
            inner_test_local = (train_drives == inner_d)
            inner_train_local = ~inner_test_local

            inner_train_global = train_indices[inner_train_local]
            inner_test_global = train_indices[inner_test_local]

            # Fit Model (e) on the 10 inner drives
            m_e_inner = fit_e(
                X=base_feats.iloc[inner_train_global],
                y_ln_gt=ln_z_gt[inner_train_global],
                random_state=random_state,
                n_jobs=n_jobs,
            )
            z_e_pred, _ = predict_e(m_e_inner, base_feats.iloc[inner_test_global])
            z_e_train_oof[inner_test_local] = z_e_pred

        # -------------------------------------------------------------
        # Step C: Model (e) on all 11 train drives -> predict test fold d
        # -------------------------------------------------------------
        m_e_outer = fit_e(
            X=base_feats.iloc[train_mask],
            y_ln_gt=ln_z_gt[train_mask],
            random_state=random_state,
            n_jobs=n_jobs,
        )
        z_e_test, _ = predict_e(m_e_outer, base_feats.iloc[test_mask])
        z_hat_e_arr[test_mask] = z_e_test

        # -------------------------------------------------------------
        # Step D: Construct Z_base for train and test (D13, D34)
        # -------------------------------------------------------------
        z_base_train = np.where(~pattern_000[train_mask], z_d_train, z_e_train_oof)
        z_base_test = np.where(~pattern_000[test_mask], z_d_test, z_e_test)
        z_base_arr[test_mask] = z_base_test

        assert not np.any(np.isnan(z_base_train)), "z_base_train must not contain NaN"
        assert not np.any(np.isnan(z_base_test)), "z_base_test must not contain NaN"
        assert np.all(z_base_train > 0), "z_base_train must be strictly positive"
        assert np.all(z_base_test > 0), "z_base_test must be strictly positive"

        # Log-residual target for train
        r_train = ln_z_gt[train_mask] - np.log(z_base_train)
        if scramble_train_targets:
            rng_canary = np.random.RandomState(random_state + fold_idx * 100)
            r_train = rng_canary.permutation(r_train)

        # -------------------------------------------------------------
        # Step E: Construct feature matrices with derived ln_z_base (D30)
        # -------------------------------------------------------------
        feat_mats_train = build_feature_matrices(
            base_features_df=base_feats.iloc[train_mask],
            z_base=z_base_train,
        )
        feat_mats_test = build_feature_matrices(
            base_features_df=base_feats.iloc[test_mask],
            z_base=z_base_test,
        )

        # -------------------------------------------------------------
        # Step F: Fit Model (f0) - Ridge baseline
        # -------------------------------------------------------------
        m_f0, best_alpha_f0, f0_cv = fit_f0(
            X=feat_mats_train["f0"],
            y=r_train,
            groups=train_drives,
            random_state=random_state,
        )
        z_hat_f0, r_hat_f0 = predict_f0(
            model=m_f0,
            X=feat_mats_test["f0"],
            z_base=z_base_test,
        )
        z_hat_f0_arr[test_mask] = z_hat_f0
        r_hat_f0_arr[test_mask] = r_hat_f0

        # -------------------------------------------------------------
        # Step G: Fit Model (f) - XGBoost residual with grid search
        # -------------------------------------------------------------
        m_f, best_params_f, f_cv = fit_f(
            X=feat_mats_train["f"],
            y=r_train,
            groups=train_drives,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        z_hat_f, r_hat_f = predict_f(
            model=m_f,
            X=feat_mats_test["f"],
            z_base=z_base_test,
        )
        z_hat_f_arr[test_mask] = z_hat_f
        r_hat_f_arr[test_mask] = r_hat_f

        fold_records.append({
            "fold_idx": fold_idx,
            "held_drive": str(held_drive),
            "n_test": n_test,
            "n_train": n_train,
            "best_alpha_f0": best_alpha_f0,
            "f0_rmse_inner": f0_cv["best_rmse"],
            "best_params_f": best_params_f,
            "f_rmse_inner": f_cv["best_rmse"],
        })

    # Assemble output dataframe
    oof_df = pd.DataFrame({
        "frame_id": features_df["frame_id"].values,
        "drive": features_df["drive"].values,
        "pred_idx": features_df["pred_idx"].values,
        "z_d": z_d_test_arr,
        "z_base": z_base_arr,
        "z_hat_f0": z_hat_f0_arr,
        "z_hat_f": z_hat_f_arr,
        "z_hat_e": z_hat_e_arr,
        "r_hat_f0": r_hat_f0_arr,
        "r_hat_f": r_hat_f_arr,
        "fallback_flag": fallback_flag,
    })

    return oof_df, fold_records


def fit_full_b_models(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cues_df: pd.DataFrame,
    best_params_f: dict[str, Any],
    best_alpha_f0: float,
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[Pipeline, xgb.XGBRegressor, xgb.XGBRegressor, FusionWeights]:
    """
    Fit full models across all 12 drives of Split B to be serialized into runs/residual/{model}/.

    Args:
        features_df: Features DataFrame for Split B.
        eval_df: Evaluation DataFrame for Split B (contains z_gt).
        cues_df: Geometric cues DataFrame for Split B.
        best_params_f: Optimal hyperparameters chosen from CV.
        best_alpha_f0: Optimal Ridge alpha chosen from CV.
        random_state: Seed.
        n_jobs: Threads.

    Returns:
        (model_f0, model_f, model_e, full_b_weights)
    """
    combined = features_df.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined[col] = cues_df[col]
    base_feats = extract_inference_features(combined)

    z_gt = eval_df["z_gt"].to_numpy(dtype=float)
    ln_z_gt = np.log(z_gt)
    z_cues = cues_df[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = cues_df[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)
    drive_ids = cues_df["drive"].to_numpy(dtype=str)
    pattern_000 = (~valid_mask[:, 0] & ~valid_mask[:, 1] & ~valid_mask[:, 2])

    # 1. Full geometric fusion weights
    full_fw = fit_fusion_weights(
        Z_cues=z_cues,
        Z_gt=z_gt,
        valid_mask=valid_mask,
        drive_ids=drive_ids,
    )
    z_d_full = fuse_depths_vectorised(z_cues, valid_mask, full_fw)

    # 2. Full Model (e)
    model_e = fit_e(
        X=base_feats,
        y_ln_gt=ln_z_gt,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    z_e_full, _ = predict_e(model_e, base_feats)

    # 3. Construct Z_base for full B
    z_base_full = np.where(~pattern_000, z_d_full, z_e_full)
    r_full = ln_z_gt - np.log(z_base_full)

    feat_mats_full = build_feature_matrices(base_feats, z_base=z_base_full)

    # 4. Full Model (f0)
    scaler = StandardScaler()
    X_f0_scaled = scaler.fit_transform(feat_mats_full["f0"].to_numpy(dtype=float))
    ridge = Ridge(alpha=best_alpha_f0, random_state=random_state)
    ridge.fit(X_f0_scaled, r_full)
    model_f0 = Pipeline([("scaler", scaler), ("ridge", ridge)])

    # 5. Full Model (f)
    f_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    f_params.update(best_params_f)
    model_f = xgb.XGBRegressor(**f_params)
    model_f.fit(feat_mats_full["f"].to_numpy(dtype=float), r_full)

    return model_f0, model_f, model_e, full_fw
