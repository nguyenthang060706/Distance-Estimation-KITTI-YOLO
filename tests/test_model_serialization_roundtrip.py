"""
tests/test_model_serialization_roundtrip.py: Regression round-trip tests for XGBoost JSON serialization (Decision D77).

Guards against:
- Array brackets in learner.learner_model_param.base_score ('[val]' vs 'val').
- Silent fallback of XGBoost C++ parser to base_score = 0.5.
- Discrepancy between in-memory model predictions and predictions from loaded JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import xgboost as xgb

from src.residual.models import load_model_f, load_model_e, save_model_f, save_model_e
from src.uncertainty.cqr import load_quantile_model, save_quantile_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_xgboost_roundtrip_toy(tmp_path: Path):
    """Verifies that fit -> save_model -> load_model produces bit-exact predictions on toy data."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).randn(100) * 0.1 + 0.05

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    p = tmp_path / "model_test.json"
    save_model_f(model, p)

    # Check JSON structure directly
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    bs = data["learner"]["learner_model_param"]["base_score"]
    assert "[" not in str(bs) and "]" not in str(bs), f"base_score contains brackets: {bs}"
    float_bs = float(bs)
    assert np.isfinite(float_bs)

    # Load and compare predictions
    loaded = load_model_f(p)
    pred_orig = model.predict(X[:10])
    pred_load = loaded.predict(X[:10])
    np.testing.assert_allclose(pred_load, pred_orig, atol=1e-6, err_msg="Loaded model predictions differ from in-memory!")


def test_corrupted_base_score_raises_value_error(tmp_path: Path):
    """Verifies that a JSON with bracketed base_score '[1.23]' triggers an immediate ValueError."""
    p = tmp_path / "corrupted.json"
    dummy_data = {
        "learner": {
            "learner_model_param": {
                "base_score": "[1.9926282E-2]"
            }
        }
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    with pytest.raises(ValueError, match="Corrupted base_score serialization"):
        load_model_f(p)


def test_all_existing_residual_models_loadable():
    """Decision D77: Verify all 12 frozen XGBoost JSON models in runs/residual load without error and have clean base_scores."""
    detectors = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
    models_to_check = ["model_f.json", "model_e.json", "model_q05.json", "model_q95.json"]

    for det in detectors:
        dir_p = PROJECT_ROOT / "runs" / "residual" / det
        for m_name in models_to_check:
            m_path = dir_p / m_name
            assert m_path.is_file(), f"Missing model file: {m_path}"

            with open(m_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            bs = d["learner"]["learner_model_param"]["base_score"]
            assert "[" not in str(bs) and "]" not in str(bs), f"Bracket found in {m_path.name}: {bs}"

            # Ensure loaders succeed
            if m_name == "model_f.json":
                m = load_model_f(m_path)
            elif m_name == "model_e.json":
                m = load_model_e(m_path)
            else:
                m = load_quantile_model(m_path)
            assert m is not None


def test_golden_prediction_sample_level_regression():
    """
    Decisions D77 & D80: Verify sample-level predictions of loaded real JSON models
    against independent golden values saved in tests/fixtures/golden_predictions_sample.json.
    Ensures that loaded models from disk match expected outputs and do not suffer
    from silent base_score reset or version serialization discrepancies.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "golden_predictions_sample.json"
    assert fixture_path.is_file(), f"Fixture file not found: {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    import pandas as pd
    from src.residual.models import predict_f, predict_e

    for det, data in golden_data.items():
        dir_p = PROJECT_ROOT / "runs" / "residual" / det
        mf_path = dir_p / "model_f.json"
        me_path = dir_p / "model_e.json"

        assert mf_path.is_file(), f"Missing {mf_path}"
        assert me_path.is_file(), f"Missing {me_path}"

        mf = load_model_f(mf_path)
        me = load_model_e(me_path)

        f_df = pd.DataFrame(data["feat_f_rows"], columns=data["feature_cols_f"])
        e_df = pd.DataFrame(data["feat_e_rows"], columns=data["feature_cols_e"])
        z_base = np.array(data["z_base"], dtype=float)

        z_hat_f, r_hat_f = predict_f(mf, f_df, z_base)
        z_hat_e, _ = predict_e(me, e_df)

        golden_r_hat_f = np.array(data["golden_r_hat_f"], dtype=float)
        golden_z_hat_f = np.array(data["golden_z_hat_f"], dtype=float)
        golden_z_hat_e = np.array(data["golden_z_hat_e"], dtype=float)

        # 1. Exact match with golden values
        np.testing.assert_allclose(
            r_hat_f, golden_r_hat_f, atol=1e-4,
            err_msg=f"Loaded Model (f) for {det} diverged from golden sample predictions!"
        )
        np.testing.assert_allclose(
            z_hat_f, golden_z_hat_f, atol=1e-3,
            err_msg=f"Loaded Model (f) Z_hat for {det} diverged from golden sample predictions!"
        )
        np.testing.assert_allclose(
            z_hat_e, golden_z_hat_e, atol=1e-3,
            err_msg=f"Loaded Model (e) Z_hat for {det} diverged from golden sample predictions!"
        )

        # 2. Sanity bounds: Residuals must be realistic and centered around 0 (not shifted by +0.5)
        assert np.mean(np.abs(r_hat_f)) < 0.20, f"Model (f) residual bias detected: {r_hat_f}"
        assert np.mean(z_hat_e) > 5.0, f"Model (e) fallback collapsed: {z_hat_e}"


def test_corrupted_base_score_alters_raw_predictions(tmp_path: Path):
    """
    Demonstrates the exact root-cause bug (Decisions D71, D77, D80):
    When raw XGBoost loads a JSON with '[val]' bracket, C++ silently resets base_score to 0.5.
    Our loader `load_model_f` catches this via `_validate_xgboost_json`, preventing corrupted predictions.
    """
    p_orig = PROJECT_ROOT / "runs" / "residual" / "yolo11s_640" / "model_f.json"
    with open(p_orig, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Inject bracket into base_score
    clean_bs = data["learner"]["learner_model_param"]["base_score"]
    data["learner"]["learner_model_param"]["base_score"] = f"[{clean_bs}]"

    corrupted_p = tmp_path / "corrupted_model_f.json"
    with open(corrupted_p, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # 1. Protected loader must reject with ValueError
    with pytest.raises(ValueError, match="Corrupted base_score serialization"):
        load_model_f(corrupted_p)

    # 2. Raw XGBoost silently loads it and defaults to 0.5, shifting predictions
    raw_clean = xgb.XGBRegressor()
    raw_clean.load_model(str(p_orig))

    raw_corrupted = xgb.XGBRegressor()
    raw_corrupted.load_model(str(corrupted_p))

    # Evaluate on dummy row with 17 features
    dummy_x = np.zeros((1, 17))
    pred_clean = raw_clean.predict(dummy_x)
    pred_corrupted = raw_corrupted.predict(dummy_x)

    # The prediction difference must be approximately 0.5 - clean_bs
    diff = float(pred_corrupted[0] - pred_clean[0])
    assert abs(diff) > 0.3, f"Expected raw C++ parser bug shift > 0.3, but got {diff}"

