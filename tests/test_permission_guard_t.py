"""
Tests for Split T Permission Guard across all pipeline entry points (Decisions D4, D22, D27).
Ensures Split T is strictly locked by default and can only be accessed with allow_test=True.
"""

import pytest
from pathlib import Path
from typing import Any
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


def test_run_inference_core_allow_test_passes_load_split(monkeypatch):
    """Confirm run_inference_core(split='T', allow_test=True) passes allow_test to load_split without touching T data (Decision D66)."""
    import scripts.run_inference as ri

    class Sentinel(Exception):
        pass

    seen: dict[str, Any] = {}

    def fake_load_split(splits_dir, split, allow_test=False):
        seen["allow_test"] = allow_test
        seen["split"] = split
        raise Sentinel("Aborting test execution before loading any Split T frames!")

    monkeypatch.setattr(ri, "verify_checkpoint", lambda *args, **kwargs: "dummy_sha")
    monkeypatch.setattr(ri, "load_split", fake_load_split)

    with pytest.raises(Sentinel):
        ri.run_inference_core("yolo11s", "T", allow_test=True)

    assert seen["allow_test"] is True
    assert seen["split"] == "T"


def test_guard_3_hashes_verification_passes_and_detects_tampering(tmp_path: Path):
    """Test check_guard_3_hashes against real configs and verify tampering detection."""
    import yaml
    from scripts.run_final_T import check_guard_3_hashes, PROJECT_ROOT

    real_cfg_path = PROJECT_ROOT / "configs" / "pipeline_frozen_v1.yaml"
    assert real_cfg_path.is_file()

    # Real config must pass 100%
    check_guard_3_hashes(real_cfg_path)

    # Tampered config (e.g. altered model SHA) must be caught
    with open(real_cfg_path, "r", encoding="utf-8") as f:
        tampered_cfg = yaml.safe_load(f)

    tampered_cfg["detectors"]["yolo11s_640"]["models"]["model_f"]["sha256"] = "00000000000000000000000000000000"
    tampered_path = tmp_path / "tampered_pipeline.yaml"
    with open(tampered_path, "w", encoding="utf-8") as f:
        yaml.dump(tampered_cfg, f)

    with pytest.raises(ValueError, match="Guard 3 FAILED: Model model_f SHA mismatch"):
        check_guard_3_hashes(tampered_path)


def test_guard_4_lock_logic(tmp_path: Path, monkeypatch):
    """Test Guard 4 lockfile requires explicit confirmation and prevents re-execution."""
    from scripts.run_final_T import check_and_create_guard_4_lock
    import scripts.run_final_T as rft

    # 1. Dry-run mode bypasses lock creation
    check_and_create_guard_4_lock(dry_run=True, confirm_flag=None)

    # 2. Split T without confirm flag raises ValueError
    with pytest.raises(ValueError, match="Guard 4 FAILED: Running on Split T requires explicit confirmation flag"):
        check_and_create_guard_4_lock(dry_run=False, confirm_flag=None)

    with pytest.raises(ValueError, match="Guard 4 FAILED: Running on Split T requires explicit confirmation flag"):
        check_and_create_guard_4_lock(dry_run=False, confirm_flag="INVALID_FLAG")

    # 3. Split T with confirm flag creates lockfile
    test_runs_dir = tmp_path / "runs"
    test_runs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rft, "PROJECT_ROOT", tmp_path)

    check_and_create_guard_4_lock(dry_run=False, confirm_flag="FINAL_T_RUN")
    assert (test_runs_dir / "final_T.lock").is_file()

    # 4. Running again when lockfile exists raises FileExistsError (Decision D27)
    with pytest.raises(FileExistsError, match="Guard 4 FAILED: Lockfile .* already exists"):
        check_and_create_guard_4_lock(dry_run=False, confirm_flag="FINAL_T_RUN")
