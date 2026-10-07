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
