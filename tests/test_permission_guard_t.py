"""
Tests for Split T Permission Guard across all pipeline entry points (Decisions D4, D22, D27).
Ensures Split T is strictly locked by default and can only be accessed with allow_test=True.
"""

import pytest
from pathlib import Path
from scripts.run_inference import run_inference_for_model, run_inference_core
from src.pipeline.build_dataset import load_artifacts
from src.pipeline.apply_frozen import apply_frozen_pipeline


def test_run_inference_for_model_hard_blocks_t():
    """run_inference_for_model must ALWAYS raise PermissionError for Split T."""
    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        run_inference_for_model("yolo11s", "T")

    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        run_inference_for_model("yolov8s", "t")


def test_run_inference_core_blocks_t_by_default():
    """run_inference_core must raise PermissionError for Split T when allow_test=False."""
    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        run_inference_core("yolo11s", "T", allow_test=False)

    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        run_inference_core("yolov5su", "t", allow_test=False)


def test_load_artifacts_blocks_t_by_default():
    """load_artifacts must raise PermissionError for Split T when allow_test=False."""
    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        load_artifacts("yolo11s_640", "T", allow_test=False)

    with pytest.raises(PermissionError, match="Access to Split T is strictly forbidden"):
        load_artifacts("yolov8s_640", "t", allow_test=False)


def test_apply_frozen_pipeline_blocks_t_by_default():
    """apply_frozen_pipeline must raise PermissionError for Split T when allow_test=False."""
    with pytest.raises(PermissionError, match="Access to Split T pipeline execution is strictly forbidden"):
        apply_frozen_pipeline("yolo11s_640", "T", allow_test=False)

    with pytest.raises(PermissionError, match="Access to Split T pipeline execution is strictly forbidden"):
        apply_frozen_pipeline("yolov5su_640", "t", allow_test=False)


def test_allow_test_bypasses_permission_guard(tmp_path: Path):
    """When allow_test=True, permission guard does not raise PermissionError (raises FileNotFoundError for missing data instead)."""
    # 1. load_artifacts with allow_test=True passes permission check
    with pytest.raises(FileNotFoundError):
        load_artifacts("yolo11s_640", "T", predictions_dir=tmp_path, allow_test=True)

    # 2. apply_frozen_pipeline with allow_test=True passes permission check
    with pytest.raises(FileNotFoundError):
        apply_frozen_pipeline("yolo11s_640", "T", data_dir=tmp_path, allow_test=True)
