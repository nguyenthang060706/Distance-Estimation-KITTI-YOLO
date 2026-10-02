"""
scripts/verify_data.py: Comprehensive dataset and split verification script.

Runs before every pipeline stage to guarantee data integrity:
1. Validates KITTI frame counts (7481 training frames).
2. Verifies drive mapping via train_rand.txt.
3. Asserts P2 intrinsic consistency = 1.0000 across all 141 drives.
4. Validates split metadata hashes and ensures strict drive disjointness.
"""

import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")
from src.utils.kitti_loader import KITTILoader, parse_calib, read_drive_mapping
from src.utils.split_builder import load_splits, compute_split_hash


def verify_kitti_structure(data_root: Path = Path("data/kitti")):
    print("=== 1. Verifying KITTI Directory Structure ===")
    required_dirs = [
        data_root / "image_2",
        data_root / "label_2",
        data_root / "calib",
        data_root / "devkit" / "mapping",
    ]
    for d in required_dirs:
        assert d.exists(), f"Missing directory: {d}"
        print(f"  [OK] Found {d}")

    # Check 7481 frames
    img_count = len(list((data_root / "image_2").glob("*.png")))
    lbl_count = len(list((data_root / "label_2").glob("*.txt")))
    cal_count = len(list((data_root / "calib").glob("*.txt")))
    print(f"  Images: {img_count}, Labels: {lbl_count}, Calibs: {cal_count}")
    assert img_count == 7481, f"Expected 7481 images, got {img_count}"
    assert lbl_count == 7481, f"Expected 7481 labels, got {lbl_count}"
    assert cal_count == 7481, f"Expected 7481 calibs, got {cal_count}"
    print("  [OK] Frame counts match exactly (7,481 frames).")


def verify_p2_consistency(data_root: Path = Path("data/kitti")):
    print("\n=== 2. Verifying P2 Intrinsic Consistency (rand mapping) ===")
    mapping_path = data_root / "devkit" / "mapping" / "train_mapping.txt"
    rand_path = data_root / "devkit" / "mapping" / "train_rand.txt"
    mapping = read_drive_mapping(str(mapping_path), str(rand_path))

    drive_p2 = defaultdict(list)
    for fid, drive in mapping.items():
        calib = parse_calib(str(data_root / "calib" / f"{fid}.txt"))
        drive_p2[drive].append(calib.P2)

    consistent_drives = 0
    total_drives = len(drive_p2)
    for drive, p2_list in drive_p2.items():
        first = p2_list[0]
        if all(np.allclose(first, p, atol=1e-5) for p in p2_list):
            consistent_drives += 1

    ratio = consistent_drives / total_drives
    print(f"  Total drives: {total_drives}")
    print(f"  Consistent drives: {consistent_drives}/{total_drives} ({ratio:.4f})")
    assert ratio == 1.0, f"P2 consistency check failed! Expected 1.0, got {ratio:.4f}"
    print("  [OK] P2 intrinsic consistency = 1.0000 across all drives.")


def verify_splits(splits_dir: Path = Path("splits"), data_root: Path = Path("data/kitti")):
    print("\n=== 3. Verifying Splits Integrity & Disjointness ===")
    meta_path = splits_dir / "split_metadata.json"
    assert meta_path.exists(), f"Metadata not found: {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = load_splits(str(splits_dir))
    expected_splits = ["A", "V", "B", "C", "T"]
    assert set(splits.keys()) == set(expected_splits), f"Split names mismatch: {splits.keys()}"

    total_frames = sum(len(ids) for ids in splits.values())
    assert total_frames == 7481, f"Expected 7481 total frames across splits, got {total_frames}"

    loader = KITTILoader(str(data_root))
    all_drives_assigned = {}
    split_drives = {}

    for name in expected_splits:
        ids = splits[name]
        calc_hash = compute_split_hash(ids)
        expected_hash = meta["splits"][name]["hash"]
        assert calc_hash == expected_hash, (
            f"Hash mismatch for split {name}!\n"
            f"  Calculated: {calc_hash}\n"
            f"  Expected:   {expected_hash}"
        )

        drives_in_split = set()
        for fid in ids:
            d = loader.get_drive(fid)
            drives_in_split.add(d)
        split_drives[name] = drives_in_split
        print(f"  Split {name}: {len(ids):>4d} frames, {len(drives_in_split):>2d} drives, hash {calc_hash[:8]}... [OK]")

    # Check pairwise drive disjointness
    for i in range(len(expected_splits)):
        for j in range(i + 1, len(expected_splits)):
            s1 = expected_splits[i]
            s2 = expected_splits[j]
            overlap = split_drives[s1].intersection(split_drives[s2])
            assert len(overlap) == 0, f"Drive leakage detected between {s1} and {s2}: {overlap}"

    print("  [OK] All splits strictly disjoint by raw drive (zero drive leakage).")
    print("  [OK] All split hashes match metadata perfectly.")


def main():
    try:
        verify_kitti_structure()
        verify_p2_consistency()
        verify_splits()
        print("\n>>> ALL DATA VERIFICATIONS PASSED SUCCESSFULLY (READY FOR TRAINING) <<<\n")
    except AssertionError as e:
        print(f"\n[FAIL] Verification error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
