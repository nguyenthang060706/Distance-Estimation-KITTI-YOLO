"""
tests/test_splits.py: Unit tests and regression tests for KITTI splits and drive mapping.
"""

import json
import pytest
from pathlib import Path
import numpy as np

from src.utils.kitti_loader import KITTILoader, read_drive_mapping
from src.utils.split_builder import load_splits, compute_split_hash


def test_mapping_regression_h0_vs_h1(tmp_path):
    """
    Regression test for mapping bug (train_rand.txt permutation).

    Ensures that read_drive_mapping uses rand[i]-1 permutation (H1)
    and DOES NOT simply read line i (H0).
    """
    mapping_file = tmp_path / "train_mapping.txt"
    rand_file = tmp_path / "train_rand.txt"

    # 3 lines in mapping
    mapping_file.write_text(
        "2011_09_26 2011_09_26_drive_0001_sync\n"
        "2011_09_26 2011_09_26_drive_0002_sync\n"
        "2011_09_26 2011_09_26_drive_0003_sync\n"
    )

    # Permutation: frame 0 -> line 3 (drive_0003), frame 1 -> line 1 (drive_0001), frame 2 -> line 2 (drive_0002)
    rand_file.write_text("3, 1, 2\n")

    mapping = read_drive_mapping(str(mapping_file), str(rand_file))

    # Assert H1 behavior (correct permutation)
    assert mapping["000000"] == "2011_09_26_drive_0003_sync"
    assert mapping["000001"] == "2011_09_26_drive_0001_sync"
    assert mapping["000002"] == "2011_09_26_drive_0002_sync"

    # Verify that this is strictly different from buggy H0 (direct row indexing)
    h0_mapping_frame_0 = "2011_09_26_drive_0001_sync"
    assert mapping["000000"] != h0_mapping_frame_0, "Regression: mapping returned H0 unpermuted row!"


def test_splits_hash_matches_metadata():
    """Verify that split files on disk match split_metadata.json hashes exactly."""
    meta_path = Path("splits/split_metadata.json")
    if not meta_path.exists():
        pytest.skip("splits/split_metadata.json not found")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = load_splits("splits")
    for name, expected_info in meta["splits"].items():
        actual_hash = compute_split_hash(splits[name])
        assert actual_hash == expected_info["hash"], (
            f"Split {name} hash mismatch! File altered on disk."
        )


def test_splits_disjoint_by_drive():
    """Verify that all splits (A, V, B, C, T) share zero drives."""
    data_root = Path("data/kitti")
    if not (data_root / "devkit").exists():
        pytest.skip("KITTI devkit not found in data/kitti")

    loader = KITTILoader(str(data_root))
    splits = load_splits("splits")

    split_drives = {}
    for name, fids in splits.items():
        drives = {loader.get_drive(fid) for fid in fids}
        split_drives[name] = drives

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = split_drives[s1].intersection(split_drives[s2])
            assert len(overlap) == 0, f"Drive leakage between {s1} and {s2}: {overlap}"


def test_splits_frame_counts():
    """Verify exact frame counts for frozen splits-v1."""
    splits = load_splits("splits")
    expected_counts = {
        "A": 3740,
        "V": 374,
        "B": 1496,
        "C": 749,
        "T": 1122,
    }
    for name, expected in expected_counts.items():
        assert len(splits[name]) == expected, f"Split {name} has {len(splits[name])} frames, expected {expected}"
    assert sum(len(fids) for fids in splits.values()) == 7481


def test_load_split_guard():
    """Verify that loading Split T is strictly blocked unless allow_test=True (§4.2, D4)."""
    from src.utils.split_builder import load_split
    import os

    # Normal splits load without error
    frames_a = load_split("splits", "A")
    assert len(frames_a) == 3740

    # Ensure environment variable is not active
    old_env = os.environ.pop("ALLOW_TEST_SPLIT", None)
    try:
        # Split T without allow_test must raise PermissionError
        with pytest.raises(PermissionError) as exc_info:
            load_split("splits", "T")
        assert "Split T is FROZEN" in str(exc_info.value)

        # Split T with allow_test=True must succeed
        frames_t = load_split("splits", "T", allow_test=True)
        assert len(frames_t) == 1122

        # Split T with ALLOW_TEST_SPLIT=1 must succeed
        os.environ["ALLOW_TEST_SPLIT"] = "1"
        frames_t_env = load_split("splits", "T")
        assert len(frames_t_env) == 1122
    finally:
        if old_env is not None:
            os.environ["ALLOW_TEST_SPLIT"] = old_env
        else:
            os.environ.pop("ALLOW_TEST_SPLIT", None)

