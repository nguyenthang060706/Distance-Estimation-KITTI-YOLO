"""
src/geometry/geometric_cues.py: Three pinhole depth cues (§5.2 of KE_HOACH_V4).

Cues:
    (a) Z_w = fx · W_eff / w            — width cue
    (b) Z_h = fy · H_obj / h            — height cue
    (c) Z_g = fy · H_cam / (y_bot - y_h) — ground-plane cue

Boundary masking (ε = 2 px):
    - bbox touches left or right border  → Z_w = NaN
    - bbox touches top or bottom border  → Z_h = NaN, Z_g = NaN
    - y_bottom ≤ y_horizon + ε           → Z_g = NaN

Each function returns (depth, valid) where valid is a boolean flag.
Vectorised versions accept arrays and return (depths, valids) arrays.

Design decisions:
    - Uses median priors (W_eff_median, H_obj_median, H_cam_median) from Split A.
    - Uses per-image P2 intrinsics (fx, fy, cx, cy).
    - ε = 2 px for all boundary checks (configurable).
    - Image size (W, H) needed for border checks; loaded lazily if not provided.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Default border tolerance (pixels)
BORDER_EPS = 2


@dataclass
class GeometricPriors:
    """Prior values estimated from Split A for a single class."""
    W_eff: float   # Effective width (median), metres
    H_obj: float   # Object height (median), metres
    H_cam: float   # Camera height above ground (median), metres
    y_horizon: float = 174.8  # Fallback horizon line y-coordinate, pixels
    delta_horizon: Optional[float] = None  # Offset from per-image cy: y_h = cy_i + delta_horizon


@dataclass
class CameraIntrinsics:
    """Per-image camera intrinsics from P2."""
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CueResult:
    """Result of computing geometric cues for one or more objects."""
    Z_w: np.ndarray         # Width-based depth, NaN if invalid
    Z_h: np.ndarray         # Height-based depth, NaN if invalid
    Z_g: np.ndarray         # Ground-plane depth, NaN if invalid
    valid_w: np.ndarray     # Boolean, True if Z_w is valid
    valid_h: np.ndarray     # Boolean, True if Z_h is valid
    valid_g: np.ndarray     # Boolean, True if Z_g is valid

    @property
    def n(self) -> int:
        return len(self.Z_w)


# ---------------------------------------------------------------------------
# Single-cue functions
# ---------------------------------------------------------------------------

def compute_Z_w(bbox: np.ndarray, fx: float, W_eff: float,
                img_width: int, eps: float = BORDER_EPS) -> tuple:
    """
    Width cue: Z_w = fx * W_eff / w.

    Args:
        bbox: [x1, y1, x2, y2]
        fx: focal length (horizontal)
        W_eff: effective object width (median prior)
        img_width: image width in pixels
        eps: border tolerance

    Returns:
        (Z_w, valid): float depth and bool validity flag
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1

    # Border check: left/right
    touches_left = x1 <= eps
    touches_right = x2 >= (img_width - 1 - eps)
    valid = not (touches_left or touches_right) and w > 0

    if valid:
        Z = fx * W_eff / w
        if Z <= 0:
            return np.nan, False
        return float(Z), True
    else:
        return np.nan, False


def compute_Z_h(bbox: np.ndarray, fy: float, H_obj: float,
                img_height: int, eps: float = BORDER_EPS) -> tuple:
    """
    Height cue: Z_h = fy * H_obj / h.

    Args:
        bbox: [x1, y1, x2, y2]
        fy: focal length (vertical)
        H_obj: object height (median prior)
        img_height: image height in pixels
        eps: border tolerance

    Returns:
        (Z_h, valid): float depth and bool validity flag
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1

    # Border check: top/bottom
    touches_top = y1 <= eps
    touches_bottom = y2 >= (img_height - 1 - eps)
    valid = not (touches_top or touches_bottom) and h > 0

    if valid:
        Z = fy * H_obj / h
        if Z <= 0:
            return np.nan, False
        return float(Z), True
    else:
        return np.nan, False


def compute_Z_g(bbox: np.ndarray, fy: float, H_cam: float,
                y_horizon: float, img_height: int,
                eps: float = BORDER_EPS) -> tuple:
    """
    Ground-plane cue: Z_g = fy * H_cam / (y_bottom - y_horizon).

    Args:
        bbox: [x1, y1, x2, y2]
        fy: focal length (vertical)
        H_cam: camera height above ground (median prior)
        y_horizon: horizon line y-coordinate
        img_height: image height in pixels
        eps: border tolerance

    Returns:
        (Z_g, valid): float depth and bool validity flag
    """
    x1, y1, x2, y2 = bbox
    y_bottom = y2

    # Border checks:
    # 1. Top/bottom border (same as Z_h: if bbox is cut vertically, y_bottom is unreliable)
    touches_top = y1 <= eps
    touches_bottom = y2 >= (img_height - 1 - eps)

    # 2. y_bottom must be strictly below horizon + eps
    below_horizon = y_bottom > (y_horizon + eps)

    valid = not (touches_top or touches_bottom) and below_horizon

    if valid:
        denom = y_bottom - y_horizon
        Z = fy * H_cam / denom
        if Z <= 0:
            return np.nan, False
        return float(Z), True
    else:
        return np.nan, False


# ---------------------------------------------------------------------------
# Vectorised computation for arrays of bboxes
# ---------------------------------------------------------------------------

def compute_cues_batch(
    bboxes: np.ndarray,
    intrinsics: CameraIntrinsics,
    priors: GeometricPriors,
    img_width: int,
    img_height: int,
    eps: float = BORDER_EPS,
) -> CueResult:
    """
    Compute all three geometric cues for a batch of bounding boxes.

    Args:
        bboxes: (N, 4) array of [x1, y1, x2, y2]
        intrinsics: per-image camera intrinsics
        priors: class-specific geometric priors
        img_width: image width in pixels
        img_height: image height in pixels
        eps: border tolerance

    Returns:
        CueResult with arrays of shape (N,)
    """
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, 4)

    N = bboxes.shape[0]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    w = x2 - x1
    h = y2 - y1
    y_bottom = y2

    fx = intrinsics.fx
    fy = intrinsics.fy
    if priors.delta_horizon is not None:
        y_horizon = intrinsics.cy + priors.delta_horizon
    else:
        y_horizon = priors.y_horizon

    # --- Border masks ---
    touches_left = x1 <= eps
    touches_right = x2 >= (img_width - 1 - eps)
    touches_top = y1 <= eps
    touches_bottom = y2 >= (img_height - 1 - eps)

    # --- Z_w ---
    valid_w = ~(touches_left | touches_right) & (w > 0)
    Z_w = np.full(N, np.nan)
    mask_w = valid_w
    Z_w[mask_w] = fx * priors.W_eff / w[mask_w]
    # Invalidate non-positive depths
    bad_w = mask_w & (Z_w <= 0)
    Z_w[bad_w] = np.nan
    valid_w[bad_w] = False

    # --- Z_h ---
    valid_h = ~(touches_top | touches_bottom) & (h > 0)
    Z_h = np.full(N, np.nan)
    mask_h = valid_h
    Z_h[mask_h] = fy * priors.H_obj / h[mask_h]
    bad_h = mask_h & (Z_h <= 0)
    Z_h[bad_h] = np.nan
    valid_h[bad_h] = False

    # --- Z_g ---
    below_horizon = y_bottom > (y_horizon + eps)
    valid_g = ~(touches_top | touches_bottom) & below_horizon
    Z_g = np.full(N, np.nan)
    mask_g = valid_g
    denom = y_bottom[mask_g] - y_horizon
    Z_g[mask_g] = fy * priors.H_cam / denom
    bad_g = mask_g & (Z_g <= 0)
    Z_g[bad_g] = np.nan
    valid_g[bad_g] = False

    return CueResult(
        Z_w=Z_w, Z_h=Z_h, Z_g=Z_g,
        valid_w=valid_w, valid_h=valid_h, valid_g=valid_g,
    )


def load_priors_from_yaml(yaml_path: str, obj_class: str = "Car") -> tuple:
    """
    Load GeometricPriors from configs/geometry_priors.yaml.

    Returns:
        (GeometricPriors, metadata_dict)
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    meta = data["metadata"]
    cls_data = data["classes"][obj_class]

    priors = GeometricPriors(
        W_eff=cls_data["W_eff_median"],
        H_obj=cls_data["H_obj_median"],
        H_cam=meta["H_cam_median"],
        y_horizon=meta["y_horizon_cy_mean"],
    )

    return priors, meta
