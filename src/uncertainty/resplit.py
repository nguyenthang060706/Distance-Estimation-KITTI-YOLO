"""
src/uncertainty/resplit.py: B∪C drive-level resplit generator and validator (Decisions D24, D26).

Implements the pre-registered protocol from configs/residual/coverage_prereg_v1.yaml:
- Shuffles 22 drives with Car Hard from B∪C for seeds 0–19.
- Allocates drives to three disjoint partitions:
    fit   ≈ 50% cars
    calib ≈ 25% cars
    eval  ≈ 25% cars
- Enforces strict constraints:
    - len(drives) >= 4 for each partition.
    - top1_share <= 0.50 for each partition.
    - If violated, redraws with next sequence in same RNG stream, recording n_redraws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DRIVES_22 = [
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0011_sync",
    "2011_09_26_drive_0014_sync",
    "2011_09_26_drive_0027_sync",
    "2011_09_26_drive_0028_sync",
    "2011_09_26_drive_0035_sync",
    "2011_09_26_drive_0056_sync",
    "2011_09_26_drive_0057_sync",
    "2011_09_26_drive_0059_sync",
    "2011_09_26_drive_0060_sync",
    "2011_09_26_drive_0070_sync",
    "2011_09_26_drive_0079_sync",
    "2011_09_26_drive_0084_sync",
    "2011_09_26_drive_0086_sync",
    "2011_09_26_drive_0087_sync",
    "2011_09_26_drive_0096_sync",
    "2011_09_26_drive_0104_sync",
    "2011_09_26_drive_0106_sync",
    "2011_09_26_drive_0113_sync",
    "2011_09_28_drive_0034_sync",
    "2011_09_28_drive_0047_sync",
    "2011_09_29_drive_0004_sync",
]


def get_bc_car_hard_counts(predictions_dir: Path | str | None = None) -> dict[str, int]:
    """
    Load Car Hard ground-truth object counts per drive from Split B and C annotations.

    Uses GT parquets (detector-neutral) to count Hard instances for Car.

    Args:
        predictions_dir: Directory containing *_gt.parquet files (default results/predictions).

    Returns:
        Dict mapping drive_id to ground-truth Car Hard count.
    """
    p_dir = Path(predictions_dir) if predictions_dir is not None else PROJECT_ROOT / "results" / "predictions"
    gt_b_path = p_dir / "yolo11s_640_B_gt.parquet"
    gt_c_path = p_dir / "yolo11s_640_C_gt.parquet"

    if not gt_b_path.is_file() or not gt_c_path.is_file():
        raise FileNotFoundError(f"Missing GT parquet files in {p_dir}")

    df_b = pd.read_parquet(gt_b_path)
    df_c = pd.read_parquet(gt_c_path)
    df_bc = pd.concat([df_b, df_c], ignore_index=True)

    # Filter Car Hard
    mask = df_bc["difficulty"].isin(["Easy", "Moderate", "Hard"])
    car_df = df_bc[mask]
    counts = car_df.groupby("drive").size().to_dict()

    # Verify all 22 drives exist
    for d in DRIVES_22:
        if d not in counts:
            counts[d] = 0

    return {d: counts[d] for d in sorted(DRIVES_22)}


def generate_resplit(
    seed: int,
    car_counts: dict[str, int] | None = None,
    target_fractions: tuple[float, float, float] = (0.50, 0.25, 0.25),
    min_drives_per_part: int = 4,
    max_top1_share: float = 0.50,
    max_attempts: int = 1000,
) -> dict[str, Any]:
    """
    Generate a 3-way partition (fit, calib, eval) for a given RNG seed according to D26.

    Greedy assignment distributes drives based on Car Hard count to match target fractions.
    Enforces min_drives_per_part and max_top1_share constraints.

    Args:
        seed: Random seed (0–19).
        car_counts: Drive to car count dict. If None, loaded via get_bc_car_hard_counts().
        target_fractions: Target ratio for (fit, calib, eval), summing to 1.0.
        min_drives_per_part: Minimum drives per partition (default 4).
        max_top1_share: Maximum fraction of total cars in partition from a single drive (default 0.50).
        max_attempts: Maximum redraw iterations.

    Returns:
        Dict containing partition details, drive lists, shares, and redraw count.
    """
    if car_counts is None:
        car_counts = get_bc_car_hard_counts()

    drives = sorted(car_counts.keys())
    total_cars = sum(car_counts[d] for d in drives)
    if total_cars <= 0:
        raise ValueError("Total car count must be positive")

    targets = [total_cars * f for f in target_fractions]
    rng = np.random.default_rng(seed)
    n_redraws = 0

    for _ in range(max_attempts):
        shuffled = list(drives)
        rng.shuffle(shuffled)

        parts: list[list[str]] = [[], [], []]
        part_counts = [0, 0, 0]

        for d in shuffled:
            cnt = car_counts[d]
            # Greedy assign to partition whose current fraction is furthest below target
            best_idx = int(np.argmin([part_counts[i] / targets[i] for i in range(3)]))
            parts[best_idx].append(d)
            part_counts[best_idx] += cnt

        # Constraint check
        valid = True
        top1_shares = []
        for i in range(3):
            if len(parts[i]) < min_drives_per_part:
                valid = False
                break
            if part_counts[i] == 0:
                valid = False
                break
            top1 = max(car_counts[d] for d in parts[i]) / part_counts[i]
            top1_shares.append(top1)
            if top1 > max_top1_share:
                valid = False
                break

        if valid:
            fit_d = sorted(parts[0])
            calib_d = sorted(parts[1])
            eval_d = sorted(parts[2])

            # Disjointness check
            s_fit = set(fit_d)
            s_calib = set(calib_d)
            s_eval = set(eval_d)
            if len(s_fit & s_calib) > 0 or len(s_fit & s_eval) > 0 or len(s_calib & s_eval) > 0:
                raise RuntimeError("Generated overlapping partitions!")
            if len(s_fit | s_calib | s_eval) != len(drives):
                raise RuntimeError("Generated partitions do not cover all 22 drives!")

            return {
                "seed": seed,
                "fit_drives": fit_d,
                "calib_drives": calib_d,
                "eval_drives": eval_d,
                "n_redraws": n_redraws,
                "part_counts": {
                    "fit": part_counts[0],
                    "calib": part_counts[1],
                    "eval": part_counts[2],
                },
                "part_shares": {
                    "fit": part_counts[0] / total_cars,
                    "calib": part_counts[1] / total_cars,
                    "eval": part_counts[2] / total_cars,
                },
                "top1_shares": {
                    "fit": top1_shares[0],
                    "calib": top1_shares[1],
                    "eval": top1_shares[2],
                },
                "n_drives": {
                    "fit": len(fit_d),
                    "calib": len(calib_d),
                    "eval": len(eval_d),
                },
            }

        n_redraws += 1

    raise RuntimeError(f"Could not find valid partition for seed {seed} after {max_attempts} attempts")


def validate_resplit(
    resplit_dict: dict[str, Any],
    min_drives: int = 4,
    max_top1: float = 0.50,
) -> bool:
    """
    Validate partition invariants according to Decision D26 constraints.
    """
    fit = set(resplit_dict["fit_drives"])
    calib = set(resplit_dict["calib_drives"])
    ev = set(resplit_dict["eval_drives"])

    # Disjoint check
    if fit & calib or fit & ev or calib & ev:
        return False
    if len(fit | calib | ev) != 22:
        return False
    if len(fit) < min_drives or len(calib) < min_drives or len(ev) < min_drives:
        return False

    top1 = resplit_dict["top1_shares"]
    if top1["fit"] > max_top1 or top1["calib"] > max_top1 or top1["eval"] > max_top1:
        return False

    return True
