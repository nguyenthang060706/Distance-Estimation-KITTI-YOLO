"""
src/geometry/fusion.py: Log-space optimal fusion of geometric depth cues (§5.2 of KE_HOACH_V4).

Given K valid cues Z_1, …, Z_K per object, the fused depth is:
    ln Z_d = Σ w_k · ln Z_k
    Z_d = exp(ln Z_d)

where weights w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1) minimise the variance of ln Z_d,
and Σ is the K×K covariance matrix of log-depth errors e_k = ln Z_k − ln Z_gt
estimated from Split B using grouped cross-validation by drive.

Design decisions (see NHAT_KY_QUYET_DINH.md):
    - Σ estimated via grouped CV by drive, not in-sample.
    - For objects with missing cues, use the sub-matrix Σ_S.
    - Apply Oracle Approximating Shrinkage (OAS, Chen et al. 2010) to Σ.
    - If any weight < 0, fall back to constrained w ≥ 0 (NNLS).
    - NaN in one cue must not contaminate others.
    - Decision D16c: LODO OOF fusion (fuse_depths_lodo) on Split B for target residual r.
"""

import json
from pathlib import Path
import warnings
from dataclasses import dataclass
import numpy as np


CUE_NAMES = ["Z_w", "Z_h", "Z_g"]
N_CUES = len(CUE_NAMES)


@dataclass
class FusionWeights:
    """Optimal fusion weights and diagnostics."""
    weights: np.ndarray         # (K,) optimal weights, sum to 1
    cov_matrix: np.ndarray      # (K, K) covariance matrix Σ
    cov_shrunk: np.ndarray      # (K, K) shrunk covariance matrix
    shrinkage_alpha: float      # shrinkage intensity [0, 1]
    cue_names: list             # names of cues in order
    constrained: bool           # True if NNLS was needed (negative weight)
    n_samples: int              # number of samples used to estimate Σ
    n_drives: int               # number of drives in grouped CV


def _oas_shrinkage(S: np.ndarray, n: int) -> tuple:
    """
    Oracle Approximating Shrinkage (OAS, Chen et al. 2010) linear shrinkage towards diagonal target.

    Note: This computes linear shrinkage towards a scaled identity target F = (Tr(S)/K)*I.
    For large N (e.g. N > 4000), alpha is very small (alpha < 0.001) and essentially identical
    to standard Ledoit-Wolf (2004).

    Args:
        S: (K, K) sample covariance matrix
        n: number of samples

    Returns:
        (S_shrunk, alpha) where alpha ∈ [0, 1] is the shrinkage intensity
    """
    K = S.shape[0]
    mu = np.trace(S) / K
    F = mu * np.eye(K)

    # OAS shrinkage intensity (Chen et al. 2010)
    rho_num = (1 - 2.0 / K) * np.sum(S ** 2) + np.trace(S) ** 2
    rho_denom = (n + 1 - 2.0 / K) * (np.sum(S ** 2) - np.trace(S) ** 2 / K)

    if rho_denom == 0:
        alpha = 1.0
    else:
        alpha = np.clip(rho_num / rho_denom, 0.0, 1.0)

    S_shrunk = (1 - alpha) * S + alpha * F
    return S_shrunk, float(alpha)


# Alias for backward compatibility
_ledoit_wolf_shrinkage = _oas_shrinkage


def _optimal_weights(sigma: np.ndarray) -> np.ndarray:
    """
    Compute optimal weights w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1).

    Args:
        sigma: (K, K) covariance matrix (positive definite)

    Returns:
        (K,) weights summing to 1
    """
    K = sigma.shape[0]
    ones = np.ones(K)
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        warnings.warn("Covariance matrix is singular; cannot invert. Falling back to equal weights.", UserWarning, stacklevel=2)
        return np.ones(K) / K

    w = sigma_inv @ ones
    denom = ones @ sigma_inv @ ones

    if denom <= 0 or not np.isfinite(denom):
        warnings.warn(f"Optimal weights denominator {denom} <= 0 or non-finite. Falling back to equal weights.", UserWarning, stacklevel=2)
        return np.ones(K) / K

    w = w / denom
    return w


def _nnls_weights(sigma: np.ndarray) -> np.ndarray:
    """
    Constrained optimal weights with w ≥ 0 using simple projection.

    Iteratively sets negative weights to zero and re-solves on the
    remaining subset until all weights are non-negative.

    Args:
        sigma: (K, K) covariance matrix

    Returns:
        (K,) non-negative weights summing to 1
    """
    K = sigma.shape[0]
    active = np.ones(K, dtype=bool)

    for _iter in range(K):
        idx = np.where(active)[0]
        if len(idx) == 0:
            return np.ones(K) / K

        sub_sigma = sigma[np.ix_(idx, idx)]
        sub_w = _optimal_weights(sub_sigma)

        if np.all(sub_w >= 0):
            w_full = np.zeros(K)
            w_full[idx] = sub_w
            return w_full

        # Remove the most negative weight
        worst = idx[np.argmin(sub_w)]
        active[worst] = False

    # Fallback: equal weights on remaining
    idx = np.where(active)[0]
    if len(idx) == 0:
        return np.ones(K) / K
    w_full = np.zeros(K)
    w_full[idx] = 1.0 / len(idx)
    return w_full


def estimate_covariance_grouped_cv(
    log_errors: np.ndarray,
    drive_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple:
    """
    Estimate the covariance matrix of log-depth errors using leave-one-drive-out CV.

    For each drive d, compute mean error on all OTHER drives, then compute
    the residual for drive d relative to that mean. The covariance is computed
    on these out-of-drive residuals.

    Args:
        log_errors: (N, K) array of e_k = ln Z_k - ln Z_gt
        drive_ids: (N,) array of drive identifiers (strings)
        valid_mask: (N, K) boolean array, True if cue k is valid for sample n

    Returns:
        (cov_matrix, n_complete, n_drives) where cov_matrix is (K, K)
    """
    N, K = log_errors.shape
    unique_drives = np.unique(drive_ids)
    n_drives = len(unique_drives)

    # Use only complete cases (all K cues valid) for covariance estimation
    all_valid = np.all(valid_mask, axis=1)
    complete_errors = log_errors[all_valid]
    complete_drives = drive_ids[all_valid]
    n_complete = complete_errors.shape[0]

    if n_complete < K + 1:
        raise ValueError(
            f"Insufficient complete cases for covariance estimation: n_complete={n_complete} < K+1={K+1}."
        )

    # Leave-one-drive-out residuals
    residuals = np.zeros_like(complete_errors)
    for drive in unique_drives:
        in_drive = complete_drives == drive
        out_drive = ~in_drive
        if np.sum(out_drive) == 0:
            continue
        if np.sum(in_drive) == 0:
            continue
        # Mean on out-of-drive samples
        mean_ood = np.mean(complete_errors[out_drive], axis=0)
        # Residual for this drive
        residuals[in_drive] = complete_errors[in_drive] - mean_ood

    # Compute covariance of residuals
    cov = np.cov(residuals, rowvar=False, ddof=1)

    # Ensure cov is 2D
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    return cov, n_complete, n_drives


def fit_fusion_weights(
    Z_cues: np.ndarray,
    Z_gt: np.ndarray,
    valid_mask: np.ndarray,
    drive_ids: np.ndarray,
    cue_names: list[str] | None = None,
) -> FusionWeights:
    """
    Estimate optimal fusion weights from data (Split B).

    Args:
        Z_cues: (N, K) array of depth estimates from each cue
        Z_gt: (N,) array of ground truth depths
        valid_mask: (N, K) boolean, True if cue k is valid
        drive_ids: (N,) array of drive identifiers
        cue_names: Optional list of K cue names (defaults to CUE_NAMES[:K]).

    Returns:
        FusionWeights dataclass
    """
    N, K = Z_cues.shape
    assert Z_gt.shape == (N,), f"Z_gt shape {Z_gt.shape} != (N,)={N}"
    assert valid_mask.shape == (N, K), f"valid_mask shape mismatch"

    if cue_names is None:
        resolved_cue_names = CUE_NAMES[:K]
    else:
        assert len(cue_names) == K, f"len(cue_names)={len(cue_names)} != K={K}"
        resolved_cue_names = list(cue_names)

    # Compute log errors for valid cues
    log_errors = np.full((N, K), np.nan)
    for k in range(K):
        mask_k = valid_mask[:, k] & (Z_cues[:, k] > 0) & (Z_gt > 0)
        log_errors[mask_k, k] = np.log(Z_cues[mask_k, k]) - np.log(Z_gt[mask_k])

    # Update valid_mask to exclude NaN errors
    valid_mask_clean = ~np.isnan(log_errors)

    # Estimate covariance with grouped CV
    cov_raw, n_complete, n_drives = estimate_covariance_grouped_cv(
        log_errors, drive_ids, valid_mask_clean
    )

    # Apply OAS shrinkage (Chen et al. 2010)
    cov_shrunk, alpha = _oas_shrinkage(cov_raw, n_complete)

    # Compute optimal weights
    weights = _optimal_weights(cov_shrunk)

    # Check for negative weights
    constrained = False
    if np.any(weights < 0):
        weights = _nnls_weights(cov_shrunk)
        constrained = True

    return FusionWeights(
        weights=weights,
        cov_matrix=cov_raw,
        cov_shrunk=cov_shrunk,
        shrinkage_alpha=alpha,
        cue_names=resolved_cue_names,
        constrained=constrained,
        n_samples=n_complete,
        n_drives=n_drives,
    )


def fuse_depths(
    Z_cues: np.ndarray,
    valid_mask: np.ndarray,
    fusion_weights: FusionWeights,
) -> np.ndarray:
    """
    Fuse depth estimates using pre-computed weights.

    For each object:
    - If all cues valid: use full weight vector
    - If subset S valid: use sub-matrix Σ_S to compute weights for S
    - If no cues valid: return NaN

    Args:
        Z_cues: (N, K) array of depth estimates
        valid_mask: (N, K) boolean
        fusion_weights: pre-fitted FusionWeights

    Returns:
        (N,) array of fused depth estimates Z_d
    """
    N, K = Z_cues.shape
    Z_d = np.full(N, np.nan)

    sigma = fusion_weights.cov_shrunk

    # Pre-compute full weights (already done)
    w_full = fusion_weights.weights

    for i in range(N):
        valid = valid_mask[i] & (Z_cues[i] > 0)
        idx = np.where(valid)[0]

        if len(idx) == 0:
            continue

        log_Z = np.log(Z_cues[i, idx])

        if len(idx) == K:
            # All cues valid: use pre-computed weights
            ln_Zd = np.dot(w_full, log_Z)
        elif len(idx) == 1:
            # Single cue: just use it
            ln_Zd = log_Z[0]
        else:
            # Subset: compute weights from sub-matrix
            sub_sigma = sigma[np.ix_(idx, idx)]
            sub_w = _optimal_weights(sub_sigma)
            if np.any(sub_w < 0):
                sub_w = _nnls_weights(sub_sigma)
            ln_Zd = np.dot(sub_w, log_Z)

        Z_d[i] = np.exp(ln_Zd)

    return Z_d


def fuse_depths_vectorised(
    Z_cues: np.ndarray,
    valid_mask: np.ndarray,
    fusion_weights: FusionWeights,
) -> np.ndarray:
    """
    Faster fusion for the common case where most objects have all cues valid.

    Falls back to per-object loop for partial cases.
    """
    N, K = Z_cues.shape
    Z_d = np.full(N, np.nan)
    sigma = fusion_weights.cov_shrunk
    w_full = fusion_weights.weights

    # Identify complete and partial cases
    positive = Z_cues > 0
    effective_valid = valid_mask & positive
    all_valid = np.all(effective_valid, axis=1)
    any_valid = np.any(effective_valid, axis=1)
    partial = any_valid & ~all_valid

    # Vectorised path for complete cases
    if np.any(all_valid):
        log_Z_complete = np.log(Z_cues[all_valid])
        Z_d[all_valid] = np.exp(log_Z_complete @ w_full)

    # Per-object path for partial cases
    partial_idx = np.where(partial)[0]
    for i in partial_idx:
        idx = np.where(effective_valid[i])[0]
        log_Z = np.log(Z_cues[i, idx])

        if len(idx) == 1:
            Z_d[i] = np.exp(log_Z[0])
        else:
            sub_sigma = sigma[np.ix_(idx, idx)]
            sub_w = _optimal_weights(sub_sigma)
            if np.any(sub_w < 0):
                sub_w = _nnls_weights(sub_sigma)
            Z_d[i] = np.exp(np.dot(sub_w, log_Z))

    return Z_d


def fuse_depths_lodo(
    Z_cues: np.ndarray,
    Z_gt: np.ndarray,
    valid_mask: np.ndarray,
    drive_ids: np.ndarray,
    min_train_drives: int = 3,
) -> tuple[np.ndarray, dict[str, FusionWeights]]:
    """
    Z_d out-of-fold: weights for drive d are fit on all OTHER drives (Decision D16c).

    For each drive d, optimal fusion weights are fitted strictly on all OTHER drives (out-of-fold).
    This ensures the fused depth Z_d on Split B contains zero in-sample lookahead bias
    when constructing target residual r = ln Z_gt - ln Z_d.

    Args:
        Z_cues: (N, K) array of depth estimates from each cue
        Z_gt: (N,) array of ground truth depths
        valid_mask: (N, K) boolean, True if cue k is valid
        drive_ids: (N,) array of drive identifiers
        min_train_drives: minimum required training drives in each fold (default: 3)

    Returns:
        (Z_d, fits_dict): (N,) array of fused depths and dict mapping drive -> FusionWeights
    """
    N = len(Z_gt)
    Z_d = np.full(N, np.nan)
    fits: dict[str, FusionWeights] = {}
    unique_drives = np.unique(drive_ids)

    for d in unique_drives:
        held = (drive_ids == d)
        train_mask = ~held
        train_drives = np.unique(drive_ids[train_mask])
        if len(train_drives) < min_train_drives:
            raise ValueError(
                f"Too few training drives for LODO fusion on fold '{d}': "
                f"{len(train_drives)} < min_train_drives={min_train_drives}"
            )
        fw = fit_fusion_weights(
            Z_cues=Z_cues[train_mask],
            Z_gt=Z_gt[train_mask],
            valid_mask=valid_mask[train_mask],
            drive_ids=drive_ids[train_mask],
        )
        Z_d[held] = fuse_depths(Z_cues[held], valid_mask[held], fw)
        fits[str(d)] = fw

    return Z_d, fits


def save_fusion_weights(fw: FusionWeights, path: Path | str) -> None:
    """
    Serialize FusionWeights object to a JSON file.

    Args:
        fw: FusionWeights dataclass instance.
        path: Output path for JSON serialization.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": fw.weights.tolist(),
        "cov_matrix": fw.cov_matrix.tolist(),
        "cov_shrunk": fw.cov_shrunk.tolist(),
        "shrinkage_alpha": float(fw.shrinkage_alpha),
        "cue_names": list(fw.cue_names),
        "constrained": bool(fw.constrained),
        "n_samples": int(fw.n_samples),
        "n_drives": int(fw.n_drives),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_fusion_weights(path: Path | str) -> FusionWeights:
    """
    Load serialized FusionWeights object from a JSON file.

    Args:
        path: Path to serialized JSON file.

    Returns:
        Reconstructed FusionWeights dataclass instance.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return FusionWeights(
        weights=np.array(payload["weights"], dtype=float),
        cov_matrix=np.array(payload["cov_matrix"], dtype=float),
        cov_shrunk=np.array(payload["cov_shrunk"], dtype=float),
        shrinkage_alpha=float(payload["shrinkage_alpha"]),
        cue_names=list(payload["cue_names"]),
        constrained=bool(payload["constrained"]),
        n_samples=int(payload["n_samples"]),
        n_drives=int(payload["n_drives"]),
    )

