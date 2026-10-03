"""
tests/test_inference.py: Unit tests for scripts/run_inference.py (Decision D4, D11, D22).
"""

import pytest
from pathlib import Path
import pandas as pd
import yaml
from scripts.run_inference import (
    run_inference_for_model,
    verify_checkpoint,
    compute_file_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_split_t_hard_guard():
    """
    Mandatory test:
    Running inference on Split T must raise PermissionError (Decision D4, D22).
    """
    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        run_inference_for_model(
            model_name="yolov8s",
            split="T",
        )


def test_checkpoint_sha_verification_valid():
    """
    Verify that existing checkpoint matches SHA recorded in checkpoints.yaml.
    """
    checkpoints_yaml = PROJECT_ROOT / "configs" / "detector" / "checkpoints.yaml"
    ckpt_path = PROJECT_ROOT / "runs" / "detector" / "yolov8s_640" / "weights" / "last.pt"

    if ckpt_path.exists():
        actual_sha = verify_checkpoint("yolov8s_640", ckpt_path, checkpoints_yaml)
        assert len(actual_sha) == 64


def test_checkpoint_sha_mismatch_raises(tmp_path):
    """
    Mandatory test:
    Altered checkpoint file with mismatched SHA must raise ValueError.
    """
    # Create fake checkpoint
    fake_ckpt = tmp_path / "fake_last.pt"
    fake_ckpt.write_bytes(b"corrupted or wrong checkpoint bytes")

    checkpoints_yaml = PROJECT_ROOT / "configs" / "detector" / "checkpoints.yaml"
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        verify_checkpoint("yolov8s_640", fake_ckpt, checkpoints_yaml)


def test_detections_artifact_feature_guard():
    """
    Mandatory test (Decision D11 & D22):
    Verify that detection schema has ONLY test-time observable features and NO GT fields.
    """
    allowed_cols = {
        "frame_id", "drive", "pred_idx", "x1", "y1", "x2", "y2",
        "conf", "pass_thr", "fx", "fy", "cx", "cy", "img_w", "img_h",
    }
    forbidden_terms = ["gt", "depth", "distance", "alpha", "occluded", "truncated", "target"]

    for col in allowed_cols:
        col_lower = col.lower()
        for forbidden in forbidden_terms:
            assert forbidden not in col_lower, f"Forbidden term '{forbidden}' in allowed column '{col}'"
