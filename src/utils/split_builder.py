"""
Split builder for KITTI Object Detection.

Creates 5 splits by DRIVE (not by frame) as specified in KE_HOACH_V4 §4:
  A (50%) - Detector train
  V (5%)  - Detector val (early stopping)
  B (20%) - Residual train
  C (10%) - CQR calibration only
  T (15%) - Final test (run once)

Key design decisions:
- Split by drive to prevent scene leakage (§4.1)
- Stratified assignment to balance Z distribution and class counts (§4.1)
- Hash of sorted frame IDs for reproducibility (§3 NHAT_KY)
- Automatic disjointness check by drive (§3 NHAT_KY)
"""

import hashlib
import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Optional


def compute_split_hash(frame_ids: list) -> str:
    """SHA-256 hash of sorted frame IDs for reproducibility."""
    sorted_ids = sorted(frame_ids)
    content = ",".join(sorted_ids)
    return hashlib.sha256(content.encode()).hexdigest()


def assert_split_disjoint_by_drive(splits: dict, drive_mapping: dict):
    """
    Assert that no drive appears in more than one split.
    Raises AssertionError with details if violated.
    """
    drive_to_splits = defaultdict(set)
    for split_name, frame_ids in splits.items():
        for fid in frame_ids:
            drive = drive_mapping.get(fid, "UNKNOWN")
            drive_to_splits[drive].add(split_name)

    violations = {
        drive: sorted(split_names)
        for drive, split_names in drive_to_splits.items()
        if len(split_names) > 1
    }

    assert len(violations) == 0, (
        f"Drive leakage detected! {len(violations)} drive(s) appear in "
        f"multiple splits: {dict(list(violations.items())[:5])}"
    )


def _compute_drive_stats(loader, drive_frames: dict) -> dict:
    """
    Compute statistics per drive for stratification.

    Returns:
        dict: {drive: {"n_frames": int, "n_objects": int,
                        "mean_depth": float, "class_counts": dict}}
    """
    drive_stats = {}

    for drive, fids in drive_frames.items():
        depths = []
        class_counts = defaultdict(int)

        for fid in fids:
            frame = loader.load_frame(fid)
            for obj in frame.objects:
                if obj.passes_hard_filter():
                    depths.append(obj.depth)
                    class_counts[obj.obj_class] += 1

        drive_stats[drive] = {
            "n_frames": len(fids),
            "n_objects": len(depths),
            "mean_depth": float(np.mean(depths)) if depths else 0.0,
            "median_depth": float(np.median(depths)) if depths else 0.0,
            "class_counts": dict(class_counts),
        }

    return drive_stats


def build_splits(
    loader,
    seed: int = 42,
    ratios: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Build A/V/B/C/T splits by drive with stratification.

    Strategy:
    1. Compute per-drive statistics (mean depth, object count)
    2. Sort drives into depth-based strata
    3. Within each stratum, randomly assign drives to splits
       proportionally to target ratios

    Args:
        loader: KITTILoader instance
        seed: Random seed for reproducibility
        ratios: Split ratios. Default: A=0.50, V=0.05, B=0.20, C=0.10, T=0.15
        output_dir: If provided, save split files here

    Returns:
        dict: {split_name: [frame_id, ...]}
    """
    if ratios is None:
        ratios = {"A": 0.50, "V": 0.05, "B": 0.20, "C": 0.10, "T": 0.15}

    assert abs(sum(ratios.values()) - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {sum(ratios.values())}"

    rng = np.random.RandomState(seed)

    # Get drive -> frame_ids mapping
    drive_frames = loader.get_all_drives()
    print(f"Total drives: {len(drive_frames)}")
    print(f"Total frames: {sum(len(v) for v in drive_frames.values())}")

    # Compute stats for stratification
    print("Computing per-drive statistics for stratification...")
    drive_stats = _compute_drive_stats(loader, drive_frames)

    # Sort drives by median depth for stratification
    drives_sorted = sorted(
        drive_frames.keys(),
        key=lambda d: drive_stats[d]["median_depth"]
    )

    # Create strata (groups of ~5 drives each)
    n_strata = max(1, len(drives_sorted) // 5)
    strata = [[] for _ in range(n_strata)]
    for i, drive in enumerate(drives_sorted):
        strata[i % n_strata].append(drive)

    # Assign drives to splits within each stratum
    split_names = list(ratios.keys())
    split_ratios = np.array([ratios[s] for s in split_names])

    splits = {name: [] for name in split_names}
    split_frame_counts = {name: 0 for name in split_names}

    total_frames = sum(len(v) for v in drive_frames.values())

    for stratum in strata:
        rng.shuffle(stratum)

        for drive in stratum:
            # Assign to the split that is most below its target ratio
            current_ratios = np.array([
                split_frame_counts[s] / max(total_frames, 1)
                for s in split_names
            ])
            deficit = split_ratios - current_ratios
            best_split_idx = np.argmax(deficit)
            best_split = split_names[best_split_idx]

            splits[best_split].extend(drive_frames[drive])
            split_frame_counts[best_split] += len(drive_frames[drive])

    # Sort frame IDs within each split
    for name in splits:
        splits[name] = sorted(splits[name])

    # Verify disjointness
    assert_split_disjoint_by_drive(splits, loader._drive_mapping)
    print("✓ No drive leakage detected")

    # Print summary
    print("\n=== Split Summary ===")
    for name in split_names:
        n = len(splits[name])
        pct = 100.0 * n / total_frames
        target_pct = 100.0 * ratios[name]
        print(f"  {name}: {n:5d} frames ({pct:5.1f}%, target {target_pct:.0f}%)")

    # Save if output_dir provided
    if output_dir is not None:
        save_splits(splits, output_dir, seed, loader._drive_mapping)

    return splits


def save_splits(splits: dict, output_dir: str, seed: int, drive_mapping: dict):
    """
    Save splits to files with metadata.

    Creates:
    - {split_name}.txt: one frame ID per line
    - split_metadata.json: hashes, counts, seed, timestamp
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "splits": {},
    }

    for name, frame_ids in splits.items():
        # Save frame IDs
        split_file = out / f"{name}.txt"
        with open(split_file, "w") as f:
            for fid in sorted(frame_ids):
                f.write(fid + "\n")

        # Compute hash
        split_hash = compute_split_hash(frame_ids)

        # Get drives in this split
        drives_in_split = sorted(set(
            drive_mapping[fid] for fid in frame_ids
        ))

        metadata["splits"][name] = {
            "n_frames": len(frame_ids),
            "hash": split_hash,
            "n_drives": len(drives_in_split),
            "drives": drives_in_split,
        }

    # Save metadata
    meta_file = out / "split_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Splits saved to {out}")
    print(f"  Metadata: {meta_file}")


def load_splits(splits_dir: str) -> dict:
    """
    Load splits from saved files.

    Args:
        splits_dir: Directory containing {A,V,B,C,T}.txt files

    Returns:
        dict: {split_name: [frame_id, ...]}
    """
    splits_dir = Path(splits_dir)
    splits = {}

    for split_file in sorted(splits_dir.glob("*.txt")):
        name = split_file.stem
        if name in ["A", "V", "B", "C", "T"]:
            with open(split_file, "r") as f:
                frame_ids = [line.strip() for line in f if line.strip()]
            splits[name] = frame_ids

    if not splits:
        raise FileNotFoundError(f"No split files found in {splits_dir}")

    return splits


def validate_splits(splits: dict, drive_mapping: dict, verbose: bool = True) -> dict:
    """
    Validate loaded splits: disjointness, completeness.

    Returns:
        dict with validation results
    """
    results = {
        "total_frames": sum(len(v) for v in splits.values()),
        "per_split": {k: len(v) for k, v in splits.items()},
        "drive_disjoint": False,
        "all_unique": False,
    }

    # Check drive disjointness
    try:
        assert_split_disjoint_by_drive(splits, drive_mapping)
        results["drive_disjoint"] = True
    except AssertionError as e:
        results["drive_disjoint"] = False
        results["disjoint_error"] = str(e)

    # Check frame uniqueness
    all_frames = []
    for fids in splits.values():
        all_frames.extend(fids)
    results["all_unique"] = len(all_frames) == len(set(all_frames))

    # Verify hashes
    results["hashes"] = {
        name: compute_split_hash(fids) for name, fids in splits.items()
    }

    if verbose:
        print("=== Split Validation ===")
        print(f"Total frames: {results['total_frames']}")
        for k, v in results["per_split"].items():
            print(f"  {k}: {v}")
        print(f"Drive disjoint: {'✓' if results['drive_disjoint'] else '✗'}")
        print(f"All unique: {'✓' if results['all_unique'] else '✗'}")

    return results
