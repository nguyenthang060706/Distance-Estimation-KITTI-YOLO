"""
Script to create and validate A/V/B/C/T splits from KITTI data.

Usage:
    python scripts/create_splits.py [--seed 42] [--data-root data/kitti] [--output-dir splits]

This script:
1. Loads all KITTI frames via KITTILoader
2. Builds splits by drive with stratification
3. Saves split files + metadata to splits/
4. Validates and prints comprehensive statistics
5. Generates a distribution comparison report
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# UTF-8 stdout/stderr for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.kitti_loader import KITTILoader, VEHICLE_CLASSES
from src.utils.split_builder import (
    build_splits,
    load_splits,
    validate_splits,
    compute_split_hash,
)


def print_depth_distribution(loader, splits: dict):
    """Print depth distribution per split for comparison."""
    bins = [0, 10, 20, 30, 50, float("inf")]
    labels = ["0-10m", "10-20m", "20-30m", "30-50m", ">50m"]

    print("\n=== Depth Distribution by Split (Hard-filtered vehicles) ===")
    header = f"{'Split':>6s} {'Total':>6s}"
    for lbl in labels:
        header += f" {lbl:>8s}"
    print(header)
    print("-" * len(header))

    for split_name in ["A", "V", "B", "C", "T"]:
        if split_name not in splits:
            continue
        fids = splits[split_name]
        depths = []
        for fid in fids:
            frame = loader.load_frame(fid)
            for obj in frame.objects:
                if obj.passes_hard_filter():
                    depths.append(obj.depth)

        depths = np.array(depths)
        row = f"{split_name:>6s} {len(depths):>6d}"
        for i in range(len(bins) - 1):
            count = int(((depths >= bins[i]) & (depths < bins[i + 1])).sum())
            row += f" {count:>8d}"
        print(row)


def print_class_distribution(loader, splits: dict):
    """Print class distribution per split."""
    print("\n=== Class Distribution by Split (Hard-filtered) ===")
    all_classes = sorted(VEHICLE_CLASSES)
    header = f"{'Split':>6s} {'Total':>6s}"
    for cls in all_classes:
        header += f" {cls:>8s}"
    print(header)
    print("-" * len(header))

    for split_name in ["A", "V", "B", "C", "T"]:
        if split_name not in splits:
            continue
        fids = splits[split_name]
        class_counts = defaultdict(int)
        total = 0
        for fid in fids:
            frame = loader.load_frame(fid)
            for obj in frame.objects:
                if obj.passes_hard_filter():
                    class_counts[obj.obj_class] += 1
                    total += 1

        row = f"{split_name:>6s} {total:>6d}"
        for cls in all_classes:
            row += f" {class_counts.get(cls, 0):>8d}"
        print(row)


def print_drive_summary(splits: dict, drive_mapping: dict):
    """Print number of drives per split."""
    print("\n=== Drives per Split ===")
    for split_name in ["A", "V", "B", "C", "T"]:
        if split_name not in splits:
            continue
        drives = set(drive_mapping[fid] for fid in splits[split_name])
        print(f"  {split_name}: {len(drives):3d} drives, {len(splits[split_name]):5d} frames")


def main():
    parser = argparse.ArgumentParser(description="Create KITTI data splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-root", type=str, default="data/kitti",
                        help="Path to KITTI data root")
    parser.add_argument("--output-dir", type=str, default="splits",
                        help="Output directory for split files")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing splits, don't create new ones")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing splits even if version is v2 (breaks D14 protocol)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Seed: {args.seed}")
    print()

    # Load KITTI
    print("Loading KITTI dataset...")
    loader = KITTILoader(str(data_root))
    print(f"  {loader.num_frames} frames loaded")
    print(f"  {len(loader.get_all_drives())} drives found")

    if args.validate_only:
        # Validate existing splits
        print("\n--- Validating existing splits ---")
        splits = load_splits(str(output_dir))
        results = validate_splits(splits, loader._drive_mapping)
        if results["drive_disjoint"] and results["all_unique"]:
            print("\n✓ All validations passed!")
        else:
            print("\n✗ Validation failed!")
            sys.exit(1)
    else:
        # Guard against accidental overwriting of frozen splits-v2 (Decision D14)
        meta_path = output_dir / "split_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("version") == "v2" and not args.force:
                    print("\n[ERROR] Refusing to overwrite frozen splits-v2 (Decision D14).")
                    print("Splits-v2 are frozen and tied to git tag 'splits-v2'.")
                    print("If you really intend to overwrite, pass --force.")
                    sys.exit(1)
            except Exception:
                pass

        # Build new splits
        print("\n--- Building splits ---")
        splits = build_splits(
            loader,
            seed=args.seed,
            output_dir=str(output_dir),
        )

    # Print comprehensive stats
    print_drive_summary(splits, loader._drive_mapping)
    print("\nComputing detailed statistics (this may take a minute)...")
    print_depth_distribution(loader, splits)
    print_class_distribution(loader, splits)

    # Check minimum sample sizes for C and T (§4.1 warning)
    print("\n=== Sample Size Warnings (§4.1) ===")
    bins = [0, 10, 20, 30, 50, float("inf")]
    labels = ["0-10m", "10-20m", "20-30m", "30-50m", ">50m"]
    for split_name in ["C", "T"]:
        fids = splits[split_name]
        depths = []
        for fid in fids:
            frame = loader.load_frame(fid)
            for obj in frame.objects:
                if obj.passes_hard_filter():
                    depths.append(obj.depth)
        depths = np.array(depths)
        for i in range(len(bins) - 1):
            count = int(((depths >= bins[i]) & (depths < bins[i + 1])).sum())
            if count < 100:
                print(f"  ⚠️ {split_name} {labels[i]}: only {count} samples (< 100)")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
