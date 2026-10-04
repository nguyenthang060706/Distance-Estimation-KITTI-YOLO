"""
src/pipeline: Data preparation, geometric cues, residual modeling, and evaluation pipeline.
"""

from src.pipeline.build_dataset import build_population, load_artifacts

__all__ = ["load_artifacts", "build_population"]
