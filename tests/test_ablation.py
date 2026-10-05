"""Unit tests for T05 ablation components (Decisions D37, D38, D39).

Verifies:
- MLPRegressor pipeline with StandardScaler (fit_mlp, predict_mlp)
- Model (f) fitting and prediction with custom feature_cols
- Drop single cue feature filtering and fallback detection logic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.residual.models import (
    FEATURE_COLS_F,
    fit_f,
    fit_mlp,
    predict_f,
    predict_mlp,
)


@pytest.fixture
def mock_ablation_data():
    """Generates synthetic data matching Model (f) feature contract."""
    np.random.seed(42)
    n = 60
    drives = np.array([f"drive_{i % 6:02d}" for i in range(n)])

    data = {
        "w": np.random.uniform(50, 150, n),
        "h": np.random.uniform(40, 120, n),
        "w_h_ratio": np.random.uniform(0.8, 1.8, n),
        "y_bottom_minus_cy": np.random.uniform(10, 80, n),
        "cx_offset_norm": np.random.uniform(-0.3, 0.3, n),
        "touch_left": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "touch_right": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "touch_top": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "touch_bottom": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "confidence": np.random.uniform(0.5, 0.95, n),
        "ln_z_w": np.random.uniform(2.0, 3.5, n),
        "ln_z_h": np.random.uniform(2.0, 3.5, n),
        "ln_z_g": np.random.uniform(2.0, 3.5, n),
        "valid_w": np.random.choice([1, 0], n, p=[0.8, 0.2]),
        "valid_h": np.random.choice([1, 0], n, p=[0.85, 0.15]),
        "valid_g": np.random.choice([1, 0], n, p=[0.8, 0.2]),
        "ln_z_base": np.random.uniform(2.0, 3.5, n),
    }
    df = pd.DataFrame(data)
    z_base = np.exp(df["ln_z_base"].to_numpy())
    r = np.random.normal(0, 0.05, n)
    return df, r, z_base, drives


def test_mlp_fit_and_predict(mock_ablation_data):
    """fit_mlp and predict_mlp execute cleanly with scaling and positive outputs."""
    df, r, z_base, _ = mock_ablation_data

    model = fit_mlp(
        X=df,
        y=r,
        hidden_layer_sizes=(32, 16),  # small for fast test
        max_iter=50,
        random_state=42,
    )

    z_hat, r_hat = predict_mlp(model, df, z_base)

    assert len(z_hat) == len(df)
    assert len(r_hat) == len(df)
    assert not np.any(np.isnan(z_hat))
    assert not np.any(np.isnan(r_hat))
    assert np.all(z_hat > 0)


def test_fit_f_with_reduced_feature_cols(mock_ablation_data):
    """fit_f and predict_f support custom feature subsets for ablation (A1-A6)."""
    df, r, z_base, drives = mock_ablation_data

    # A1: Drop bbox_geometry
    dropped = ["w", "h", "w_h_ratio", "y_bottom_minus_cy", "cx_offset_norm"]
    reduced_cols = [c for c in FEATURE_COLS_F if c not in dropped]
    assert len(reduced_cols) == 12

    grid = {
        "n_estimators": [10],
        "max_depth": [2],
        "min_child_weight": [1],
    }

    model, best_params, cv_results = fit_f(
        X=df,
        y=r,
        groups=drives,
        grid=grid,
        n_splits=3,
        feature_cols=reduced_cols,
    )

    z_hat, r_hat = predict_f(model, df, z_base, feature_cols=reduced_cols)

    assert len(z_hat) == len(df)
    assert not np.any(np.isnan(z_hat))
    assert np.all(z_hat > 0)
    assert best_params["max_depth"] == 2


def test_drop_single_cue_feature_filtering():
    """Decision D38: Dropping cue k must remove ln_z_k AND valid_k."""
    # When dropping Z_w:
    drop_w_features = {"ln_z_w", "valid_w"}
    remaining_f = [c for c in FEATURE_COLS_F if c not in drop_w_features]
    assert len(remaining_f) == 15
    assert "ln_z_w" not in remaining_f
    assert "valid_w" not in remaining_f
    assert "ln_z_h" in remaining_f
    assert "valid_h" in remaining_f

    # Pattern 000 re-evaluation logic:
    # If valid_w was True and valid_h, valid_g were False:
    # Originally pattern != 000; after dropping Z_w, it becomes pattern 000!
    valid_mask = np.array([
        [True, False, False],  # only w valid -> becomes 000 when w dropped
        [True, True, False],   # w and h valid -> h remains valid
        [False, False, False], # already 000
    ])

    # Dropping cue index 0 (w):
    remaining_mask = valid_mask[:, [1, 2]]
    new_pattern_000 = ~np.any(remaining_mask, axis=1)

    assert new_pattern_000[0] == True   # Was valid via w, now 000!
    assert new_pattern_000[1] == False  # h is still valid
    assert new_pattern_000[2] == True   # Still 000
