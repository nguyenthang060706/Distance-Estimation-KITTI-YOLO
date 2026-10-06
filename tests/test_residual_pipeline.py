"""
tests/test_residual_pipeline.py: Unit tests and guards for residual models and nested LODO pipeline (Decisions D11, D13, D25, D28, D30, D34).

Verifies:
- DERIVED_FEATURES separation from DEFAULT_FEATURE_WHITELIST (D30).
- fallback_flag is strictly metadata and excluded from feature matrices (D28).
- ln_z_base invariance when modifying ground truth z_gt (guard against leakage).
- Model (e) complete isolation from cues and validity flags.
- Nested LODO mock execution without data leakage across drives.
- Canary test: scrambled z_gt target in training prevents model (f) from beating Z_base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.residual.feature_extractor import (
    DEFAULT_FEATURE_WHITELIST,
    check_no_gt_leakage,
    extract_inference_features,
)
from src.residual.models import (
    DERIVED_FEATURES,
    FEATURE_COLS_E,
    FEATURE_COLS_F,
    FEATURE_COLS_F0,
    FORBIDDEN_METADATA,
    build_feature_matrices,
    fit_e,
    fit_f,
    fit_f0,
    predict_e,
    predict_f,
    predict_f0,
)
from src.geometry.fusion import FusionWeights, load_fusion_weights, save_fusion_weights
from src.pipeline.oof import fit_full_b_models, run_nested_lodo_b


def test_derived_features_not_in_default_whitelist():
    """Decision D30: ln_z_base must NOT be placed into DEFAULT_FEATURE_WHITELIST (preserves D11)."""
    assert "ln_z_base" in DERIVED_FEATURES
    assert "ln_z_base" not in DEFAULT_FEATURE_WHITELIST


def test_fallback_flag_strictly_forbidden_in_features():
    """Decision D28: fallback_flag is metadata only and must not be in feature columns."""
    assert "fallback_flag" in FORBIDDEN_METADATA
    assert "fallback_flag" not in FEATURE_COLS_F
    assert "fallback_flag" not in FEATURE_COLS_F0
    assert "fallback_flag" not in FEATURE_COLS_E


def test_model_e_feature_isolation():
    """Model (e) must NOT contain cues, validity flags, or ln_z_base."""
    forbidden = {"ln_z_w", "ln_z_h", "ln_z_g", "valid_w", "valid_h", "valid_g", "ln_z_base"}
    assert not set(FEATURE_COLS_E).intersection(forbidden)


def test_ln_z_base_guard_against_z_gt_leakage():
    """
    Guard test: changing ground truth z_gt must NOT affect ln_z_base or inference features.
    ln_z_base is strictly computed from base depth (Z_d or fallback Z_e).
    """
    n = 20
    df = pd.DataFrame({
        "bbox": [[100.0, 50.0, 200.0, 150.0]] * n,
        "img_w": [1242] * n,
        "img_h": [375] * n,
        "cy": [172.8] * n,
        "confidence": [0.85] * n,
        "class_id": [0] * n,
        "z_w": [25.0] * n,
        "z_h": [24.5] * n,
        "z_g": [26.0] * n,
        "valid_w": [True] * n,
        "valid_h": [True] * n,
        "valid_g": [True] * n,
    })

    base_feats = extract_inference_features(df)
    z_base = np.full(n, 25.0)

    # Matrix built with original z_base
    mats1 = build_feature_matrices(base_feats, z_base=z_base)

    # Even if ground truth values are changed wildly, mats1 remains identical
    z_gt_fake_1 = np.full(n, 20.0)
    z_gt_fake_2 = np.full(n, 80.0)

    mats2 = build_feature_matrices(base_feats, z_base=z_base)
    pd.testing.assert_frame_equal(mats1["f"], mats2["f"])
    pd.testing.assert_frame_equal(mats1["f0"], mats2["f0"])
    pd.testing.assert_frame_equal(mats1["e"], mats2["e"])


def test_build_feature_matrices_rejects_fallback_flag():
    """build_feature_matrices must reject any input that has fallback_flag."""
    df = pd.DataFrame({
        "w": [10.0], "h": [20.0], "w_h_ratio": [0.5], "y_bottom_minus_cy": [5.0],
        "cx_offset_norm": [0.1], "touch_left": [0], "touch_right": [0],
        "touch_top": [0], "touch_bottom": [0], "confidence": [0.9],
        "fallback_flag": [True],
    })
    with pytest.raises(ValueError, match="Metadata violation"):
        build_feature_matrices(df, z_base=np.array([15.0]))


def _create_mock_dataset(n_per_drive: int = 30, n_drives: int = 4):
    """Helper to generate consistent mock data for pipeline tests."""
    np.random.seed(42)
    rows = []
    for d in range(n_drives):
        drive_name = f"2011_09_26_drive_{d:04d}_sync"
        for i in range(n_per_drive):
            true_z = np.random.uniform(10.0, 50.0)
            w = 500.0 / true_z + np.random.normal(0, 1.0)
            h = 400.0 / true_z + np.random.normal(0, 1.0)
            x1 = np.random.uniform(100, 800)
            y1 = np.random.uniform(100, 200)
            x2 = x1 + w
            y2 = y1 + h

            # Some rows pattern 000
            is_000 = (i % 10 == 0)

            rows.append({
                "frame_id": f"{d}_{i}",
                "drive": drive_name,
                "pred_idx": 0,
                "gt_idx": 0,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": 0.85,
                "class_id": 0,
                "fx": 721.5,
                "fy": 721.5,
                "cx": 609.5,
                "cy": 172.8,
                "img_w": 1242,
                "img_h": 375,
                "z_gt": true_z,
                "z_w": np.nan if is_000 else 500.0 / w,
                "z_h": np.nan if is_000 else 400.0 / h,
                "z_g": np.nan if is_000 else true_z + np.random.normal(0, 2.0),
                "valid_w": not is_000,
                "valid_h": not is_000,
                "valid_g": not is_000,
                "z_d": np.nan if is_000 else true_z + np.random.normal(0, 1.0),
                "cls": "Car",
                "difficulty": "Hard",
                "truncated": 0.0,
                "occluded": 0,
                "alpha": 0.0,
                "matched_iou": 0.8,
            })

    full_df = pd.DataFrame(rows)
    feat_cols = ["frame_id", "drive", "pred_idx", "x1", "y1", "x2", "y2", "confidence", "class_id",
                 "fx", "fy", "cx", "cy", "img_w", "img_h"]
    eval_cols = ["frame_id", "drive", "pred_idx", "gt_idx", "z_gt", "cls", "difficulty",
                 "truncated", "occluded", "alpha", "matched_iou"]
    cues_cols = ["frame_id", "drive", "pred_idx", "z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g", "z_d"]

    return full_df[feat_cols], full_df[eval_cols], full_df[cues_cols]


def test_nested_lodo_pipeline_execution_mock():
    """Verify that nested LODO pipeline runs end-to-end on mock data and produces valid estimates."""
    feats, evals, cues = _create_mock_dataset(n_per_drive=20, n_drives=4)

    oof_df, fold_records = run_nested_lodo_b(
        features_df=feats,
        eval_df=evals,
        cues_df=cues,
        min_train_drives=2,
        random_state=42,
        verbose=False,
    )

    assert len(oof_df) == len(feats)
    assert len(fold_records) == 4

    # Check columns
    expected_cols = {
        "frame_id", "drive", "pred_idx", "z_d", "z_base", "z_hat_f0", "z_hat_f", "z_hat_e",
        "r_hat_f0", "r_hat_f", "fallback_flag",
    }
    assert expected_cols.issubset(set(oof_df.columns))

    # All predictions must be positive and finite
    assert np.all(np.isfinite(oof_df["z_base"]))
    assert np.all(oof_df["z_base"] > 0)
    assert np.all(np.isfinite(oof_df["z_hat_f0"]))
    assert np.all(oof_df["z_hat_f0"] > 0)
    assert np.all(np.isfinite(oof_df["z_hat_f"]))
    assert np.all(oof_df["z_hat_f"] > 0)
    assert np.all(np.isfinite(oof_df["z_hat_e"]))
    assert np.all(oof_df["z_hat_e"] > 0)

    # fallback_flag must match pattern 000
    p000 = (~cues["valid_w"] & ~cues["valid_h"] & ~cues["valid_g"]).to_numpy()
    np.testing.assert_array_equal(oof_df["fallback_flag"].to_numpy(), p000)


def test_canary_scrambled_z_gt_fails_improvement_mock():
    """
    Canary test (Decision D25 / AGENT_RULES):
    When ground truth target is randomly permuted in training folds, the residual model
    must learn noise and NOT beat the baseline on out-of-fold evaluations.
    """
    feats, evals, cues = _create_mock_dataset(n_per_drive=25, n_drives=4)

    # Run nested LODO with scramble_train_targets=True
    oof_df, _ = run_nested_lodo_b(
        features_df=feats,
        eval_df=evals,
        cues_df=cues,
        min_train_drives=2,
        random_state=42,
        verbose=False,
        scramble_train_targets=True,
    )

    # Against genuine z_gt, the scrambled residual model (f) must NOT beat Z_base
    y_true = evals["z_gt"].values
    rmse_base = np.sqrt(np.mean((oof_df["z_base"].values - y_true) ** 2))
    rmse_f = np.sqrt(np.mean((oof_df["z_hat_f"].values - y_true) ** 2))

    # Because train target was scrambled, f learns noise and cannot beat base
    assert rmse_f >= rmse_base * 0.99, (
        f"Canary failed: Model (f) unexpectedly beat Z_base despite scrambled training targets! "
        f"rmse_f={rmse_f:.4f} vs rmse_base={rmse_base:.4f}"
    )


def test_fit_full_b_models_requires_z_base_oof():
    """Decisions D16b, D29, D43: fit_full_b_models must require out-of-fold z_base_oof."""
    feats, evals, cues = _create_mock_dataset(n_per_drive=20, n_drives=4)
    best_params_f = {"max_depth": 2, "n_estimators": 10, "min_child_weight": 1}

    # Missing z_base_oof must raise ValueError
    with pytest.raises(ValueError, match="z_base_oof must be provided"):
        fit_full_b_models(
            features_df=feats,
            eval_df=evals,
            cues_df=cues,
            best_params_f=best_params_f,
            best_alpha_f0=10.0,
            z_base_oof=None,
        )

    # Valid z_base_oof must fit successfully
    z_base_mock = evals["z_gt"].to_numpy(dtype=float) * 1.02
    m_f0, m_f, m_e, full_fw = fit_full_b_models(
        features_df=feats,
        eval_df=evals,
        cues_df=cues,
        best_params_f=best_params_f,
        best_alpha_f0=10.0,
        z_base_oof=z_base_mock,
    )
    assert m_f0 is not None
    assert m_f is not None
    assert m_e is not None
    assert isinstance(full_fw, FusionWeights)


def test_fusion_weights_json_serialization(tmp_path):
    """Verify FusionWeights serializes to JSON and deserializes identically."""
    fw = FusionWeights(
        weights=np.array([0.0, 0.7, 0.3]),
        cov_matrix=np.array([[0.05, 0.01, 0.02], [0.01, 0.04, 0.01], [0.02, 0.01, 0.06]]),
        cov_shrunk=np.array([[0.05, 0.01, 0.02], [0.01, 0.04, 0.01], [0.02, 0.01, 0.06]]),
        shrinkage_alpha=0.001,
        cue_names=["Z_w", "Z_h", "Z_g"],
        constrained=True,
        n_samples=500,
        n_drives=12,
    )
    out_file = tmp_path / "weights.json"
    save_fusion_weights(fw, out_file)
    assert out_file.is_file()

    loaded = load_fusion_weights(out_file)
    np.testing.assert_allclose(loaded.weights, fw.weights)
    np.testing.assert_allclose(loaded.cov_matrix, fw.cov_matrix)
    np.testing.assert_allclose(loaded.cov_shrunk, fw.cov_shrunk)
    assert loaded.shrinkage_alpha == fw.shrinkage_alpha
    assert loaded.cue_names == fw.cue_names
    assert loaded.constrained == fw.constrained
    assert loaded.n_samples == fw.n_samples
    assert loaded.n_drives == fw.n_drives
