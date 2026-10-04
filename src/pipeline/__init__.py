"""
src/pipeline: Data preparation, geometric cues, residual modeling, and evaluation pipeline.
"""

from src.pipeline.build_dataset import build_population, load_artifacts
from src.pipeline.geometry_stage import (
    add_cues,
    fit_fusion_lodo,
    fuse_with_weights,
    load_frozen_gt_fusion_weights,
    load_geometry_priors,
)

__all__ = [
    "load_artifacts",
    "build_population",
    "load_geometry_priors",
    "load_frozen_gt_fusion_weights",
    "add_cues",
    "fit_fusion_lodo",
    "fuse_with_weights",
]
