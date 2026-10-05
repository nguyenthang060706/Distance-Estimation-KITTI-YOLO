"""
src/residual/models.py: Residual and direct depth models (Decisions D25, D28, D30, D33, D34).

Key specifications:
- Model (f): XGBoost log-residual model with pre-registered grid search (12 configs, depth <= 4).
- Model (f0): Ridge regression baseline with StandardScaler and inner-CV alpha selection.
- Model (e): XGBoost direct depth regression with fixed single configuration (no grid).
- DERIVED_FEATURES = {"ln_z_base"} (D30): strictly managed outside DEFAULT_FEATURE_WHITELIST.
- FORBIDDEN_METADATA = {"fallback_flag"} (D28): metadata only, never in feature matrices.
- class_id is excluded from all models (constant 0 for single-class Car).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Compatibility fix: scikit-learn >= 1.6 removed _estimator_type from RegressorMixin,
# but XGBoost 2.0.x save_model() relies on it.
if not hasattr(xgb.XGBRegressor, "_estimator_type"):
    xgb.XGBRegressor._estimator_type = "regressor"

from src.residual.feature_extractor import check_no_gt_leakage

# Decision D30: ln_z_base is a derived feature at inference time, kept distinct from D11 whitelist
DERIVED_FEATURES: frozenset[str] = frozenset({"ln_z_base"})

# Decision D28: fallback_flag is strictly metadata and must never appear in any feature matrix
FORBIDDEN_METADATA: frozenset[str] = frozenset({"fallback_flag"})

# Feature set for Model (f): 17 features
FEATURE_COLS_F: list[str] = [
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
    "ln_z_w",
    "ln_z_h",
    "ln_z_g",
    "valid_w",
    "valid_h",
    "valid_g",
    "ln_z_base",
]

# Feature set for Model (f0): 5 features
FEATURE_COLS_F0: list[str] = [
    "ln_z_base",
    "w",
    "h",
    "w_h_ratio",
    "confidence",
]

# Feature set for Model (e): 10 features (isolated from cues, validity flags, and ln_z_base)
FEATURE_COLS_E: list[str] = [
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
]

DEFAULT_F0_ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]

DEFAULT_F_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 4],
    "min_child_weight": [10, 20],
}

FIXED_E_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "min_child_weight": 20,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
}


def build_feature_matrices(
    base_features_df: pd.DataFrame,
    z_base: np.ndarray | pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Construct safe, isolated feature matrices for models f, f0, and e.

    Args:
        base_features_df: DataFrame containing the 16 whitelist features (extracted by
            extract_inference_features). May contain class_id, which will be filtered out.
        z_base: Array of base depth estimates (Z_d or fallback Z_e). Required for f and f0.

    Returns:
        dict with keys "f", "f0", "e", each mapping to the corresponding pd.DataFrame.
    """
    # Guard against ground truth leakage
    check_no_gt_leakage(base_features_df.columns)

    for forbidden in FORBIDDEN_METADATA:
        if forbidden in base_features_df.columns:
            raise ValueError(
                f"Metadata violation (Decision D28): '{forbidden}' must not be present in feature matrix."
            )

    df_base = base_features_df.copy()

    # Model (e) does not use z_base
    X_e = df_base[FEATURE_COLS_E].copy()
    check_no_gt_leakage(X_e.columns)

    if z_base is None:
        return {"e": X_e}

    z_base_arr = np.asarray(z_base, dtype=float)
    if len(z_base_arr) != len(df_base):
        raise ValueError(
            f"z_base length ({len(z_base_arr)}) != feature dataframe length ({len(df_base)})"
        )
    if np.any(np.isnan(z_base_arr)) or np.any(z_base_arr <= 0):
        raise ValueError("z_base must be strictly positive and finite to compute ln_z_base")

    ln_z_base = np.log(z_base_arr)
    df_base["ln_z_base"] = ln_z_base

    X_f = df_base[FEATURE_COLS_F].copy()
    X_f0 = df_base[FEATURE_COLS_F0].copy()

    check_no_gt_leakage(X_f.columns)
    check_no_gt_leakage(X_f0.columns)

    assert "class_id" not in X_f.columns, "class_id must be excluded from model (f)"
    assert "class_id" not in X_f0.columns, "class_id must be excluded from model (f0)"
    assert "class_id" not in X_e.columns, "class_id must be excluded from model (e)"

    return {
        "f": X_f,
        "f0": X_f0,
        "e": X_e,
    }


def fit_f0(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    alpha_grid: list[float] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, float, dict[str, Any]]:
    """
    Fit baseline linear Model (f0) using Ridge regression and StandardScaler.

    Hyperparameter alpha is selected via inner GroupKFold CV to minimize log-residual RMSE.

    Args:
        X: Feature DataFrame (must contain FEATURE_COLS_F0).
        y: Target log-residual array r = ln(Z_gt) - ln(Z_base).
        groups: Cluster/drive IDs for GroupKFold.
        alpha_grid: List of alpha values to evaluate.
        n_splits: Maximum number of inner CV folds (default 5).
        random_state: Random state for deterministic fitting.

    Returns:
        (best_pipeline, best_alpha, cv_results)
    """
    check_no_gt_leakage(X.columns)
    X_mat = X[FEATURE_COLS_F0].to_numpy(dtype=float)
    y_arr = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups, dtype=str)

    if alpha_grid is None:
        alpha_grid = DEFAULT_F0_ALPHA_GRID

    unique_groups = np.unique(groups_arr)
    actual_splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=actual_splits)

    folds = list(gkf.split(X_mat, y_arr, groups=groups_arr))
    best_alpha = alpha_grid[0]
    best_rmse = float("inf")
    cv_scores: dict[float, float] = {}

    for alpha in alpha_grid:
        oof_preds = np.zeros_like(y_arr)
        for train_idx, val_idx in folds:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=random_state)),
            ])
            pipe.fit(X_mat[train_idx], y_arr[train_idx])
            oof_preds[val_idx] = pipe.predict(X_mat[val_idx])

        rmse = float(np.sqrt(np.mean((oof_preds - y_arr) ** 2)))
        cv_scores[float(alpha)] = rmse

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    # Refit best pipeline on all data
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha, random_state=random_state)),
    ])
    final_pipe.fit(X_mat, y_arr)

    cv_results = {
        "alpha_scores": cv_scores,
        "best_alpha": float(best_alpha),
        "best_rmse": float(best_rmse),
        "n_splits": actual_splits,
    }
    return final_pipe, float(best_alpha), cv_results


def fit_e(
    X: pd.DataFrame,
    y_ln_gt: np.ndarray,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> xgb.XGBRegressor:
    """
    Fit Model (e): XGBoost direct depth regression on ln(Z_gt).

    Uses a single pre-registered configuration (no grid, no early stopping).
    Features must NOT contain cues, validity flags, or ln_z_base.

    Args:
        X: Feature DataFrame (must contain FEATURE_COLS_E).
        y_ln_gt: Target array ln(Z_gt).
        params: Optional override parameters (defaults to FIXED_E_PARAMS).
        random_state: Seed for reproducibility.
        n_jobs: Number of threads (default 1 for deterministic execution).

    Returns:
        Fitted xgb.XGBRegressor instance.
    """
    check_no_gt_leakage(X.columns)
    X_mat = X[FEATURE_COLS_E].to_numpy(dtype=float)
    y_arr = np.asarray(y_ln_gt, dtype=float)

    cfg = dict(FIXED_E_PARAMS)
    if params:
        cfg.update(params)

    cfg["random_state"] = random_state
    cfg["n_jobs"] = n_jobs

    model = xgb.XGBRegressor(**cfg)
    model.fit(X_mat, y_arr)
    return model


def fit_f(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    feature_cols: list[str] | None = None,
) -> tuple[xgb.XGBRegressor, dict[str, Any], dict[str, Any]]:
    """
    Fit Model (f): XGBoost residual model using pre-registered inner CV grid search (D25).

    Selection criterion: min rmse_log on inner GroupKFold OOF.
    Tie-breaking: smaller max_depth -> smaller n_estimators.

    Args:
        X: Feature DataFrame (must contain feature_cols or FEATURE_COLS_F).
        y: Target log-residual array r = ln(Z_gt) - ln(Z_base).
        groups: Cluster/drive IDs for GroupKFold.
        grid: Hyperparameter grid (defaults to DEFAULT_F_GRID).
        n_splits: Number of inner CV folds (default 5).
        random_state: Seed for reproducibility.
        n_jobs: Number of worker threads (default 1).
        feature_cols: Optional subset of feature column names for ablation (defaults to FEATURE_COLS_F).

    Returns:
        (best_model, best_params, cv_results)
    """
    check_no_gt_leakage(X.columns)
    cols = feature_cols if feature_cols is not None else FEATURE_COLS_F
    X_mat = X[cols].to_numpy(dtype=float)
    y_arr = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups, dtype=str)

    if grid is None:
        grid = DEFAULT_F_GRID

    param_names = list(grid.keys())
    param_values = [grid[k] for k in param_names]
    candidate_combos = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    assert len(candidate_combos) <= 24, f"Grid size {len(candidate_combos)} exceeds limit 24 (D25)"

    unique_groups = np.unique(groups_arr)
    actual_splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=actual_splits)
    folds = list(gkf.split(X_mat, y_arr, groups=groups_arr))

    best_params: dict[str, Any] | None = None
    best_rmse = float("inf")
    combo_results: list[dict[str, Any]] = []

    for combo in candidate_combos:
        current_params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        current_params.update(combo)

        oof_preds = np.zeros_like(y_arr)
        for train_idx, val_idx in folds:
            m = xgb.XGBRegressor(**current_params)
            m.fit(X_mat[train_idx], y_arr[train_idx])
            oof_preds[val_idx] = m.predict(X_mat[val_idx])

        rmse = float(np.sqrt(np.mean((oof_preds - y_arr) ** 2)))
        combo_results.append({
            "params": combo,
            "rmse": rmse,
        })

        # Tie-break logic: smaller RMSE -> smaller max_depth -> smaller n_estimators
        is_better = False
        if rmse < best_rmse - 1e-7:
            is_better = True
        elif abs(rmse - best_rmse) <= 1e-7 and best_params is not None:
            # Check max_depth
            if combo["max_depth"] < best_params["max_depth"]:
                is_better = True
            elif combo["max_depth"] == best_params["max_depth"]:
                if combo["n_estimators"] < best_params["n_estimators"]:
                    is_better = True

        if is_better or best_params is None:
            best_rmse = rmse
            best_params = combo

    assert best_params is not None

    final_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    final_params.update(best_params)

    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_mat, y_arr)

    cv_results = {
        "best_params": best_params,
        "best_rmse": float(best_rmse),
        "all_combos": combo_results,
        "n_splits": actual_splits,
    }
    return final_model, best_params, cv_results


def predict_f0(
    model: Pipeline,
    X: pd.DataFrame,
    z_base: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict depth using fitted Model (f0): Z_hat = Z_base * exp(r_hat).

    Returns:
        (z_hat_f0, r_hat_f0)
    """
    check_no_gt_leakage(X.columns)
    X_mat = X[FEATURE_COLS_F0].to_numpy(dtype=float)
    z_base_arr = np.asarray(z_base, dtype=float)

    r_hat = model.predict(X_mat)
    z_hat = z_base_arr * np.exp(r_hat)
    return z_hat, r_hat


def predict_f(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    z_base: np.ndarray | pd.Series,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict depth using fitted Model (f): Z_hat = Z_base * exp(r_hat).

    Returns:
        (z_hat_f, r_hat_f)
    """
    check_no_gt_leakage(X.columns)
    cols = feature_cols if feature_cols is not None else FEATURE_COLS_F
    X_mat = X[cols].to_numpy(dtype=float)
    z_base_arr = np.asarray(z_base, dtype=float)

    r_hat = model.predict(X_mat)
    z_hat = z_base_arr * np.exp(r_hat)
    return z_hat, r_hat


def predict_e(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict depth using fitted Model (e): Z_hat = exp(ln_z_hat).

    Returns:
        (z_hat_e, ln_z_hat_e)
    """
    check_no_gt_leakage(X.columns)
    X_mat = X[FEATURE_COLS_E].to_numpy(dtype=float)

    ln_z_hat = model.predict(X_mat)
    z_hat = np.exp(ln_z_hat)
    return z_hat, ln_z_hat


def save_model_f(model: xgb.XGBRegressor, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))


def load_model_f(path: str | Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def save_model_e(model: xgb.XGBRegressor, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))


def load_model_e(path: str | Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def save_model_f0(model: Pipeline, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(p))


def load_model_f0(path: str | Path) -> Pipeline:
    return joblib.load(str(path))


def fit_mlp(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str] | None = None,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    activation: str = "relu",
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    """
    Fit alternative Model (MLP) for ablation M1 (residual_prereg_v1.yaml).
    Uses StandardScaler within a Pipeline to prevent data leakage and handle scaling.

    Args:
        X: Feature DataFrame containing feature_cols (defaults to FEATURE_COLS_F).
        y: Target log-residual array r = ln(Z_gt) - ln(Z_base).
        feature_cols: Optional list of features to extract from X.
        hidden_layer_sizes: MLP layer topology (default 128, 64).
        activation: Activation function (default relu).
        max_iter: Maximum optimization iterations (default 500).
        random_state: Deterministic seed (default 42).

    Returns:
        Fitted sklearn Pipeline(StandardScaler, MLPRegressor).
    """
    check_no_gt_leakage(X.columns)
    cols = feature_cols if feature_cols is not None else FEATURE_COLS_F
    X_mat = X[cols].to_numpy(dtype=float)
    y_arr = np.asarray(y, dtype=float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])
    pipe.fit(X_mat, y_arr)
    return pipe


def predict_mlp(
    model: Pipeline,
    X: pd.DataFrame,
    z_base: np.ndarray | pd.Series,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict depth using fitted MLP Pipeline: Z_hat = Z_base * exp(r_hat).

    Returns:
        (z_hat_mlp, r_hat_mlp)
    """
    check_no_gt_leakage(X.columns)
    cols = feature_cols if feature_cols is not None else FEATURE_COLS_F
    X_mat = X[cols].to_numpy(dtype=float)
    z_base_arr = np.asarray(z_base, dtype=float)

    r_hat = model.predict(X_mat)
    z_hat = z_base_arr * np.exp(r_hat)
    return z_hat, r_hat
