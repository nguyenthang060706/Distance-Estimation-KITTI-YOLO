"""
src/uncertainty/cqr.py: Conformal Quantile Regression (CQR) core module (Decisions D46, D47, D48, D49).

Theoretical Foundation:
- Romano, Sesia, Candès (2019): "Conformalized Quantile Regression", NeurIPS 2019.
- Finite-sample marginal coverage guarantee 1 - alpha via exact order statistics (D46).
- Winkler score (Gneiting & Raftery 2007) in log-space (r-space) for interval sharpness & penalty (D47).
- Quantile sorting consistency and crossing counter without silent clipping (D47).
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb


def sort_quantiles(
    q_lo: np.ndarray | pd.Series,
    q_hi: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure quantile predictions satisfy q_lo <= q_hi element-wise (Decision D47).

    Args:
        q_lo: Estimated lower quantile array (e.g. q=0.05).
        q_hi: Estimated upper quantile array (e.g. q=0.95).

    Returns:
        (q_lo_sorted, q_hi_sorted) where q_lo_sorted <= q_hi_sorted everywhere.
    """
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    if len(lo) != len(hi):
        raise ValueError(f"Length mismatch: len(q_lo)={len(lo)} != len(q_hi)={len(hi)}")
    return np.minimum(lo, hi), np.maximum(lo, hi)


def compute_nonconformity_scores(
    q_lo: np.ndarray | pd.Series,
    q_hi: np.ndarray | pd.Series,
    r_actual: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Compute symmetric CQR nonconformity score E_i = max(q_lo - r, r - q_hi) (Romano et al. 2019).

    Uses sort_quantiles beforehand to maintain mathematical consistency with prediction (D47).

    Args:
        q_lo: Lower quantile predictions on calibration set.
        q_hi: Upper quantile predictions on calibration set.
        r_actual: Actual ground-truth log-residuals r = ln(Z_gt) - ln(Z_base).

    Returns:
        (N,) array of nonconformity scores.
    """
    lo_sorted, hi_sorted = sort_quantiles(q_lo, q_hi)
    r_arr = np.asarray(r_actual, dtype=float)
    if len(lo_sorted) != len(r_arr):
        raise ValueError(f"Length mismatch: len(quantiles)={len(lo_sorted)} != len(r_actual)={len(r_arr)}")
    return np.maximum(lo_sorted - r_arr, r_arr - hi_sorted)


def conformalize(scores: np.ndarray | pd.Series | list[float], alpha: float = 0.1) -> float:
    """
    Calculate conformal correction threshold Q_hat using exact finite-sample order statistic (Decision D46).

    Theoretical Formula:
        k = ceil((n + 1) * (1 - alpha))
        If k > n: return +inf
        Else: return s_{(k)} (k-th smallest value in 1-based index, i.e. sorted[k - 1])

    Fixes off-by-one error present in np.quantile(scores, q, method='higher') which evaluates
    virtual index q*(n - 1) and returns s_{(k + 1)}.

    Args:
        scores: 1D array of calibration nonconformity scores.
        alpha: Nominal miscoverage level (default 0.1 for 90% coverage).

    Returns:
        float: Q_hat value (or float('inf') if sample size n is too small for level alpha).
    """
    scores_arr = np.asarray(scores, dtype=float)
    n = len(scores_arr)
    if n == 0:
        raise ValueError("scores array cannot be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = int(np.ceil((n + 1) * (1.0 - alpha) - 1e-12))
    if k > n:
        return float("inf")

    sorted_scores = np.sort(scores_arr)
    return float(sorted_scores[k - 1])


def predict_interval(
    z_base: np.ndarray | pd.Series,
    q_lo: np.ndarray | pd.Series,
    q_hi: np.ndarray | pd.Series,
    Q_hat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Construct conformal prediction interval in log-space r and convert to depth Z (Decisions D46, D47).

    Formulas:
        q_lo_sorted, q_hi_sorted = sort_quantiles(q_lo, q_hi)
        r_lo = q_lo_sorted - Q_hat
        r_hi = q_hi_sorted + Q_hat
        Z_lo = Z_base * exp(r_lo)
        Z_hi = Z_base * exp(r_hi)

    Crossing check:
        If Q_hat is negative with |Q_hat| > (q_hi - q_lo)/2, r_lo > r_hi can occur.
        Decision D47 dictates: count and report crossings, NEVER silently clip!

    Args:
        z_base: Baseline distance estimate Z_base.
        q_lo: Predicted lower quantile on target log-residual r.
        q_hi: Predicted upper quantile on target log-residual r.
        Q_hat: Conformal correction parameter from calibration.

    Returns:
        (Z_lo, Z_hi, r_lo, r_hi, n_crossings):
            Z_lo: Lower depth bound in meters.
            Z_hi: Upper depth bound in meters.
            r_lo: Lower residual bound in log-space.
            r_hi: Upper residual bound in log-space.
            n_crossings: Number of instances where r_lo > r_hi (crossing).
    """
    zb = np.asarray(z_base, dtype=float)
    if np.any(np.isnan(zb)) or np.any(zb <= 0):
        raise ValueError("z_base must be strictly positive and finite")

    lo_sorted, hi_sorted = sort_quantiles(q_lo, q_hi)
    if len(zb) != len(lo_sorted):
        raise ValueError(f"Length mismatch: len(z_base)={len(zb)} != len(quantiles)={len(lo_sorted)}")

    if np.isneginf(Q_hat):
        raise ValueError("Q_hat cannot be -inf")

    if np.isposinf(Q_hat):
        r_lo = np.full_like(lo_sorted, -np.inf)
        r_hi = np.full_like(hi_sorted, np.inf)
        z_lo = np.zeros_like(zb)
        z_hi = np.full_like(zb, np.inf)
        return z_lo, z_hi, r_lo, r_hi, 0

    r_lo = lo_sorted - Q_hat
    r_hi = hi_sorted + Q_hat

    # Decision D47: Count crossings without silent clipping
    crossings = r_lo > r_hi
    n_crossings = int(np.sum(crossings))

    z_lo = zb * np.exp(r_lo)
    z_hi = zb * np.exp(r_hi)

    return z_lo, z_hi, r_lo, r_hi, n_crossings


def winkler_score(
    r_lo: np.ndarray | pd.Series,
    r_hi: np.ndarray | pd.Series,
    r_actual: np.ndarray | pd.Series,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Compute Winkler interval score (Gneiting & Raftery 2007) in log-space r (Decision D47).

    Formula for (1 - alpha) coverage interval [L, U] and target Y:
        Score = (U - L) + (2 / alpha) * (L - Y) * I(Y < L) + (2 / alpha) * (Y - U) * I(Y > U)

    Lower score is strictly better (rewards sharp intervals and penalises miscoverage).

    Args:
        r_lo: Lower interval bound in log-space.
        r_hi: Upper interval bound in log-space.
        r_actual: Ground truth log-residual r.
        alpha: Miscoverage rate (default 0.1).

    Returns:
        (N,) array of Winkler scores.
    """
    lo = np.asarray(r_lo, dtype=float)
    hi = np.asarray(r_hi, dtype=float)
    act = np.asarray(r_actual, dtype=float)

    if len(lo) != len(hi) or len(lo) != len(act):
        raise ValueError("Length mismatch between r_lo, r_hi, and r_actual")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    width = hi - lo
    penalty_under = (2.0 / alpha) * np.maximum(0.0, lo - act)
    penalty_over = (2.0 / alpha) * np.maximum(0.0, act - hi)
    return width + penalty_under + penalty_over


def fit_quantile_xgb(
    X: pd.DataFrame | np.ndarray,
    r: np.ndarray | pd.Series,
    q: float,
    best_params_f: dict[str, Any],
    random_state: int = 42,
    n_jobs: int = 1,
) -> xgb.XGBRegressor:
    """
    Fit XGBoost quantile regression model using fixed hyperparams from manifest.json (Decision D24).

    Strict constraints:
    - NO re-gridding or hyperparameter tuning.
    - Uses exact best_params_f (n_estimators, max_depth, min_child_weight) established in T04.
    - Objective: reg:quantileerror with quantile_alpha=q.

    Args:
        X: Feature matrix for Model (f).
        r: Target log-residual array r = ln(Z_gt) - ln(Z_base_oof).
        q: Quantile level in (0, 1) (e.g. 0.05 or 0.95).
        best_params_f: Fixed optimal hyperparameters from CV manifest.
        random_state: Seed.
        n_jobs: Worker threads.

    Returns:
        Fitted xgb.XGBRegressor model.
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"Quantile level q must be in (0, 1), got {q}")

    X_mat = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    r_arr = np.asarray(r, dtype=float)

    if len(X_mat) != len(r_arr):
        raise ValueError(f"X rows ({len(X_mat)}) != r length ({len(r_arr)})")

    params: dict[str, Any] = {
        "objective": "reg:quantileerror",
        "quantile_alpha": float(q),
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    params.update(best_params_f)

    model = xgb.XGBRegressor(**params)
    model.fit(X_mat, r_arr)
    return model


def assert_disjoint_drives(
    fit_drives: list[str] | np.ndarray | pd.Series,
    calib_drives: list[str] | np.ndarray | pd.Series,
) -> None:
    """
    Guard: Verify that fit and calibration sets have zero overlapping drives (D13, D27).

    Raises:
        ValueError if any drive is present in both sets.
    """
    set_fit = set(fit_drives)
    set_calib = set(calib_drives)
    common = set_fit.intersection(set_calib)
    if common:
        raise ValueError(
            f"Data leakage detected! Drives present in both fit and calib sets: {sorted(common)}"
        )


def _validate_xgboost_json(path: str | Path) -> None:
    """Decision D77: Verify that serialized XGBoost JSON contains valid scalar base_score without brackets."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    bs = data.get("learner", {}).get("learner_model_param", {}).get("base_score")
    if bs is not None:
        bs_str = str(bs)
        if "[" in bs_str or "]" in bs_str:
            raise ValueError(
                f"Corrupted base_score serialization in {p.name}: {bs_str}. "
                "Contains array brackets which causes XGBoost C++ parser fallback to 0.5 (Decision D71/D77)."
            )
        try:
            float(bs_str)
        except ValueError as err:
            raise ValueError(f"Invalid base_score float format in {p.name}: {bs_str}") from err


def save_quantile_model(model: xgb.XGBRegressor, path: str | Path) -> None:
    """Save XGBoost quantile model in JSON format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))


def load_quantile_model(path: str | Path) -> xgb.XGBRegressor:
    """Load XGBoost quantile model from JSON format."""
    _validate_xgboost_json(path)
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


# ==============================================================================
# Split Conformal (Absolute Residual) — Decision D26
# ==============================================================================

def compute_split_conformal_scores(
    r_predicted: np.ndarray | pd.Series,
    r_actual: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Compute standard split conformal nonconformity scores E_i = |r_actual - r_predicted|.

    Args:
        r_predicted: Predicted log-residual (point prediction r_hat from Model f).
        r_actual: Ground truth log-residual r = ln(Z_gt) - ln(Z_base).

    Returns:
        (N,) array of absolute error scores.
    """
    r_pred = np.asarray(r_predicted, dtype=float)
    r_act = np.asarray(r_actual, dtype=float)
    if len(r_pred) != len(r_act):
        raise ValueError(f"Length mismatch: len(r_pred)={len(r_pred)} != len(r_act)={len(r_act)}")
    return np.abs(r_act - r_pred)


def predict_split_interval(
    z_base: np.ndarray | pd.Series,
    r_predicted: np.ndarray | pd.Series,
    Q_hat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Construct symmetric split conformal prediction interval [r_hat - Q_hat, r_hat + Q_hat]
    and convert to depth Z (Decision D26).

    Args:
        z_base: Baseline distance estimate Z_base.
        r_predicted: Point prediction r_hat in log-space.
        Q_hat: Conformal correction threshold (>= 0).

    Returns:
        (Z_lo, Z_hi, r_lo, r_hi, n_crossings=0)
    """
    zb = np.asarray(z_base, dtype=float)
    r_pred = np.asarray(r_predicted, dtype=float)
    if np.any(np.isnan(zb)) or np.any(zb <= 0):
        raise ValueError("z_base must be strictly positive and finite")
    if len(zb) != len(r_pred):
        raise ValueError(f"Length mismatch: len(z_base)={len(zb)} != len(r_predicted)={len(r_pred)}")

    if np.isposinf(Q_hat):
        r_lo = np.full_like(r_pred, -np.inf)
        r_hi = np.full_like(r_pred, np.inf)
        z_lo = np.zeros_like(zb)
        z_hi = np.full_like(zb, np.inf)
        return z_lo, z_hi, r_lo, r_hi, 0

    r_lo = r_pred - Q_hat
    r_hi = r_pred + Q_hat

    z_lo = zb * np.exp(r_lo)
    z_hi = zb * np.exp(r_hi)

    # In split conformal, Q_hat >= 0 so r_lo <= r_hi always holds
    crossings = r_lo > r_hi
    n_crossings = int(np.sum(crossings))

    return z_lo, z_hi, r_lo, r_hi, n_crossings


# ==============================================================================
# Mondrian Conformal Quantile Regression (Mondrian CQR) — Decision D26
# ==============================================================================

class MondrianBinning:
    """
    Mondrian distance binning on estimated depth Ẑ with pre-registered merge rule (Decision D26).

    Bins are partitioned by prospective Ẑ (midpoint estimate Z_base * exp(r_hat)).
    Merge rule: Merges farthest bin inward if calibration sample size < min_samples.
    """

    def __init__(
        self,
        calib_z_hat: np.ndarray | pd.Series,
        base_edges: list[float] | None = None,
        min_samples: int = 50,
    ) -> None:
        """
        Fit Mondrian bin edges on calibration Ẑ.

        Args:
            calib_z_hat: Ẑ on calibration set.
            base_edges: Pre-registered edges, default [0.0, 10.0, 20.0, 30.0].
            min_samples: Minimum calibration samples per bin (default 50).
        """
        if base_edges is None:
            base_edges = [0.0, 10.0, 20.0, 30.0]

        self.min_samples = min_samples
        self.base_edges = sorted(base_edges)
        calib_z = np.asarray(calib_z_hat, dtype=float)

        # Start with base bins
        # e.g. edges [0, 10, 20, 30] -> bins: [0, 10), [10, 20), [20, 30), [30, inf)
        edges = list(self.base_edges)

        # Merge from the far end inward if sample count < min_samples
        while len(edges) > 2:
            # Check count in the last bin [edges[-1], inf)
            last_bin_count = np.sum(calib_z >= edges[-1])
            if last_bin_count < self.min_samples:
                # Merge into the preceding bin by dropping the last edge
                edges.pop()
            else:
                break

        self.edges = edges
        self.labels: list[str] = []
        for i in range(len(self.edges)):
            if i == len(self.edges) - 1:
                self.labels.append(f">={int(self.edges[i])}m")
            else:
                self.labels.append(f"{int(self.edges[i])}-{int(self.edges[i+1])}m")

        self.n_bins = len(self.labels)

    def assign_bins(self, z_hat: np.ndarray | pd.Series) -> np.ndarray:
        """
        Assign prospective Ẑ samples to integer bin indices [0 .. n_bins - 1].

        Args:
            z_hat: Estimated depth array.

        Returns:
            (N,) integer array of bin indices.
        """
        z_arr = np.asarray(z_hat, dtype=float)
        # Bins:
        # 0: [edges[0], edges[1])
        # 1: [edges[1], edges[2])
        # ...
        # n_bins - 1: [edges[-1], inf)
        bin_idx = np.zeros(len(z_arr), dtype=int)
        for i in range(1, len(self.edges)):
            bin_idx[z_arr >= self.edges[i]] = i
        return bin_idx


def conformalize_mondrian(
    scores: np.ndarray | pd.Series,
    bin_indices: np.ndarray,
    n_bins: int,
    alpha: float = 0.1,
) -> dict[int, float]:
    """
    Compute conformal correction threshold Q_hat separately for each Mondrian bin (Decision D26).

    Args:
        scores: Nonconformity scores on calibration set.
        bin_indices: Integer bin indices for calibration set.
        n_bins: Total number of bins.
        alpha: Miscoverage level (default 0.1).

    Returns:
        Dict mapping bin_index -> Q_hat value.
    """
    scores_arr = np.asarray(scores, dtype=float)
    bin_arr = np.asarray(bin_indices, dtype=int)
    if len(scores_arr) != len(bin_arr):
        raise ValueError(f"Length mismatch: len(scores)={len(scores_arr)} != len(bin_indices)={len(bin_arr)}")

    q_hat_dict: dict[int, float] = {}
    for b in range(n_bins):
        mask = bin_arr == b
        if np.sum(mask) == 0:
            q_hat_dict[b] = float("inf")
        else:
            q_hat_dict[b] = conformalize(scores_arr[mask], alpha=alpha)

    return q_hat_dict


def predict_interval_mondrian(
    z_base: np.ndarray | pd.Series,
    q_lo: np.ndarray | pd.Series,
    q_hi: np.ndarray | pd.Series,
    q_hat_per_bin: dict[int, float],
    bin_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Construct Mondrian CQR prediction intervals using bin-specific Q_hat thresholds (Decision D26).

    Args:
        z_base: Baseline distance estimate Z_base.
        q_lo: Predicted lower quantile on target log-residual r.
        q_hi: Predicted upper quantile on target log-residual r.
        q_hat_per_bin: Dict of Q_hat thresholds per bin.
        bin_indices: Integer bin indices for the test samples.

    Returns:
        (Z_lo, Z_hi, r_lo, r_hi, n_crossings)
    """
    zb = np.asarray(z_base, dtype=float)
    if np.any(np.isnan(zb)) or np.any(zb <= 0):
        raise ValueError("z_base must be strictly positive and finite")

    lo_sorted, hi_sorted = sort_quantiles(q_lo, q_hi)
    bins = np.asarray(bin_indices, dtype=int)
    if len(zb) != len(lo_sorted) or len(zb) != len(bins):
        raise ValueError("Length mismatch between z_base, quantiles, and bin_indices")

    # Map Q_hat for each sample based on its bin
    q_hat_arr = np.array([q_hat_per_bin.get(b, float("inf")) for b in bins], dtype=float)

    r_lo = np.where(np.isposinf(q_hat_arr), -np.inf, lo_sorted - q_hat_arr)
    r_hi = np.where(np.isposinf(q_hat_arr), np.inf, hi_sorted + q_hat_arr)

    crossings = r_lo > r_hi
    n_crossings = int(np.sum(crossings))

    z_lo = np.where(np.isneginf(r_lo), 0.0, zb * np.exp(r_lo))
    z_hi = np.where(np.isposinf(r_hi), np.inf, zb * np.exp(r_hi))

    return z_lo, z_hi, r_lo, r_hi, n_crossings

