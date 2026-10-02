"""
src/geometry/: Geometric depth estimation cues and fusion (§5.2).
"""

from src.geometry.geometric_cues import (
    GeometricPriors,
    CameraIntrinsics,
    CueResult,
    compute_Z_w,
    compute_Z_h,
    compute_Z_g,
    compute_cues_batch,
    load_priors_from_yaml,
    BORDER_EPS,
)

from src.geometry.fusion import (
    FusionWeights,
    fit_fusion_weights,
    fuse_depths,
    fuse_depths_vectorised,
    CUE_NAMES,
)
