"""
src/pipeline/build_dataset.py: Dataset builder for ranging pipeline (§5 Data Contract, Decisions D11, D22, D23).

Constructs ranging populations from raw detector parquet artifacts:
- Separates features (strictly observed at test time) from evaluation ground-truth.
- Hard guard against Split T.
- Implements both 'pass_thr' and 'floor' (conf >= 0.05) populations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.residual.feature_extractor import check_no_gt_leakage

# Substrings strictly forbidden in any feature column name (testing & leakage guard)
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "gt",
    "depth",
    "alpha",
    "occluded",
    "truncated",
    "iou",
    "status",
    "target",
)

# Canonical feature columns according to §5 Data Contract
CANONICAL_FEATURE_COLUMNS = [
    "frame_id",
    "drive",
    "pred_idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class_id",
    "fx",
    "fy",
    "cx",
    "cy",
    "img_w",
    "img_h",
]

# Canonical evaluation columns according to §5 Data Contract
CANONICAL_EVAL_COLUMNS = [
    "frame_id",
    "drive",
    "pred_idx",
    "gt_idx",
    "z_gt",
    "cls",
    "difficulty",
    "truncated",
    "occluded",
    "alpha",
    "matched_iou",
]

# Canonical false negative columns
CANONICAL_FN_COLUMNS = [
    "frame_id",
    "drive",
    "gt_idx",
    "z_gt",
    "cls",
    "x1",
    "y1",
    "x2",
    "y2",
    "difficulty",
    "truncated",
    "occluded",
    "alpha",
]


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_artifacts(
    model_key: str,
    split: str,
    predictions_dir: str | Path = "results/predictions",
    allow_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw prediction parquet files for a model and split.

    Args:
        model_key: Identifier of detector model, e.g. 'yolov8s_640'.
        split: Dataset split ('A', 'B', 'C'). Split 'T' is strictly forbidden.
        predictions_dir: Path to directory containing predictions parquet artifacts.
        allow_test: If True, permits loading Split T artifacts (for final evaluation). Defaults to False.

    Returns:
        (df_detections, df_matches, df_gt)

    Raises:
        PermissionError: If split is 'T' and allow_test is False (Decisions D4, D27).
        FileNotFoundError: If any parquet artifact is missing.
    """
    split_upper = split.upper()
    if split_upper == "T" and not allow_test:
        raise PermissionError(
            "Access to Split T is strictly forbidden during Week 2 development "
            "(Decision D4, D22, AGENT_RULES.md §1.1)!"
        )

    base_dir = Path(predictions_dir)
    det_path = base_dir / f"{model_key}_{split_upper}_detections.parquet"
    match_path = base_dir / f"{model_key}_{split_upper}_matches.parquet"
    gt_path = base_dir / f"{model_key}_{split_upper}_gt.parquet"

    for p in (det_path, match_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(f"Required prediction artifact not found: {p}")

    df_dets = pd.read_parquet(det_path)
    df_matches = pd.read_parquet(match_path)
    df_gt = pd.read_parquet(gt_path)

    return df_dets, df_matches, df_gt


def build_population(
    model_key: str,
    split: str,
    population: Literal["pass_thr", "floor"] = "pass_thr",
    predictions_dir: str | Path = "results/predictions",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build ranging population, strictly separating features from evaluation data (§5, D11, D22, D23).

    Args:
        model_key: Identifier of detector model (e.g., 'yolov8s_640').
        split: Dataset split ('B', 'C'). Split 'T' is strictly guarded.
        population: 'pass_thr' (pass_thr == True and status == 'TP') or
                    'floor' (status == 'TP' across all floor conf >= 0.05).
        predictions_dir: Path to directory containing prediction parquet files.

    Returns:
        (features_df, eval_df, fn_df):
            - features_df: DataFrame with test-observable attributes only.
            - eval_df: DataFrame with ground truth labels and matching metadata.
            - fn_df: DataFrame of Hard GT objects that were missed.
    """
    df_dets, df_matches, df_gt = load_artifacts(model_key, split, predictions_dir=predictions_dir)

    return build_population_from_dfs(
        df_dets=df_dets,
        df_matches=df_matches,
        df_gt=df_gt,
        population=population,
    )


def build_population_from_dfs(
    df_dets: pd.DataFrame,
    df_matches: pd.DataFrame,
    df_gt: pd.DataFrame,
    population: Literal["pass_thr", "floor"] = "pass_thr",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Core relational join and population extraction logic from DataFrames.

    Relational join keys:
    - detections ⋈ matches on (frame_id, pred_idx)
    - matches ⋈ gt on (frame_id, matched_gt_idx == gt_idx)
    """
    if population not in ("pass_thr", "floor"):
        raise ValueError(f"Unknown population type: {population}. Expected 'pass_thr' or 'floor'.")

    # 1. Join detections with matches on (frame_id, pred_idx)
    # Ensure types match for join keys
    dets_keyed = df_dets.copy()
    matches_keyed = df_matches.copy()
    dets_keyed["pred_idx"] = dets_keyed["pred_idx"].astype(int)
    matches_keyed["pred_idx"] = matches_keyed["pred_idx"].astype(int)

    joined = dets_keyed.merge(
        matches_keyed[["frame_id", "pred_idx", "status", "matched_gt_idx", "matched_iou"]],
        on=["frame_id", "pred_idx"],
        how="inner",
    )

    # 2. Filter population
    if population == "pass_thr":
        tp_mask = (joined["pass_thr"] == True) & (joined["status"] == "TP")
    else:  # 'floor'
        tp_mask = joined["status"] == "TP"

    df_tp = joined[tp_mask].copy()

    # 3. Join with ground truth on (frame_id, matched_gt_idx == gt_idx)
    gt_keyed = df_gt.copy()
    gt_keyed["gt_idx"] = gt_keyed["gt_idx"].astype(int)
    df_tp["matched_gt_idx"] = df_tp["matched_gt_idx"].astype(int)

    # Filter out invalid matched_gt_idx if any
    df_tp = df_tp[df_tp["matched_gt_idx"] >= 0]

    merged_tp = df_tp.merge(
        gt_keyed,
        left_on=["frame_id", "matched_gt_idx"],
        right_on=["frame_id", "gt_idx"],
        how="inner",
        suffixes=("", "_gt"),
    )

    # Verify 1-to-1 match (in Greedy matching, each TP maps to exactly 1 GT, and no GT is matched twice)
    if len(merged_tp) != len(df_tp):
        raise ValueError(
            f"Relational merge mismatch: {len(df_tp)} TP detections joined to {len(merged_tp)} GT records."
        )

    if merged_tp.duplicated(subset=["frame_id", "gt_idx"]).any():
        dup_count = merged_tp.duplicated(subset=["frame_id", "gt_idx"]).sum()
        raise ValueError(f"Integrity violation: {dup_count} duplicate GT matches detected in TP population.")

    # 4. Construct features DataFrame (strictly observable at test time)
    # Use CANONICAL_FEATURE_COLUMNS
    features = pd.DataFrame(index=merged_tp.index)
    for col in CANONICAL_FEATURE_COLUMNS:
        if col not in merged_tp.columns:
            raise KeyError(f"Expected canonical feature column '{col}' missing from merged detections.")
        features[col] = merged_tp[col]

    # Verify Decision D11: zero ground truth leakage
    check_no_gt_leakage(features.columns)
    for col in features.columns:
        col_lower = col.strip().lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            if forbidden in col_lower:
                raise ValueError(
                    f"Forbidden substring '{forbidden}' detected in feature column name '{col}'!"
                )

    # 5. Construct eval DataFrame
    eval_df = pd.DataFrame(index=merged_tp.index)
    eval_df["frame_id"] = merged_tp["frame_id"]
    eval_df["drive"] = merged_tp["drive"]
    eval_df["pred_idx"] = merged_tp["pred_idx"]
    eval_df["gt_idx"] = merged_tp["gt_idx"]
    eval_df["z_gt"] = merged_tp["z_gt"]
    eval_df["cls"] = "Car"
    eval_df["difficulty"] = merged_tp["difficulty"]
    eval_df["truncated"] = merged_tp["truncated"]
    eval_df["occluded"] = merged_tp["occluded"]
    eval_df["alpha"] = merged_tp["alpha"]
    eval_df["matched_iou"] = merged_tp["matched_iou"]

    # 6. Construct FN (False Negatives) DataFrame
    # All GT Hard objects not matched by any TP in this population
    matched_gt_set = set(zip(merged_tp["frame_id"], merged_tp["gt_idx"]))
    is_matched_gt = [
        (row.frame_id, row.gt_idx) in matched_gt_set
        for row in gt_keyed.itertuples(index=False)
    ]
    fn_gt = gt_keyed[~np.array(is_matched_gt)].copy()

    fn_df = pd.DataFrame(index=fn_gt.index)
    fn_df["frame_id"] = fn_gt["frame_id"]
    fn_df["drive"] = fn_gt["drive"]
    fn_df["gt_idx"] = fn_gt["gt_idx"]
    fn_df["z_gt"] = fn_gt["z_gt"]
    fn_df["cls"] = "Car"
    fn_df["x1"] = fn_gt["x1"]
    fn_df["y1"] = fn_gt["y1"]
    fn_df["x2"] = fn_gt["x2"]
    fn_df["y2"] = fn_gt["y2"]
    fn_df["difficulty"] = fn_gt["difficulty"]
    fn_df["truncated"] = fn_gt["truncated"]
    fn_df["occluded"] = fn_gt["occluded"]
    fn_df["alpha"] = fn_gt["alpha"]

    # Reset indices
    features.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    fn_df.reset_index(drop=True, inplace=True)

    return features, eval_df, fn_df


def compute_population_summary(
    df_dets: pd.DataFrame,
    df_matches: pd.DataFrame,
    df_gt: pd.DataFrame,
    model_key: str,
    split: str,
    population: str = "pass_thr",
    split_hash: str = "",
    source_files_sha: dict[str, str] | None = None,
) -> dict:
    """
    Compute detailed summary counts matching detector evaluation specification.
    """
    dets_keyed = df_dets.copy()
    matches_keyed = df_matches.copy()
    dets_keyed["pred_idx"] = dets_keyed["pred_idx"].astype(int)
    matches_keyed["pred_idx"] = matches_keyed["pred_idx"].astype(int)

    joined = dets_keyed.merge(
        matches_keyed[["frame_id", "pred_idx", "status", "matched_gt_idx", "matched_iou"]],
        on=["frame_id", "pred_idx"],
        how="inner",
    )

    if population == "pass_thr":
        considered = joined[joined["pass_thr"] == True]
    else:
        considered = joined

    n_gt = len(df_gt)
    n_tp = int((considered["status"] == "TP").sum())
    n_fp = int((considered["status"] == "FP").sum())
    n_ign_nonhard = int((considered["status"] == "IGNORED_NONHARD").sum())
    n_ign_dontcare = int((considered["status"] == "IGNORED_DONTCARE").sum())
    n_ignored = n_ign_nonhard + n_ign_dontcare
    n_fn = n_gt - n_tp

    return {
        "model_key": model_key,
        "split": split,
        "population": population,
        "n_gt": n_gt,
        "n_TP": n_tp,
        "n_FN": n_fn,
        "n_FP": n_fp,
        "n_ignored_nonhard": n_ign_nonhard,
        "n_ignored_dontcare": n_ign_dontcare,
        "n_ignored": n_ignored,
        "total_preds_considered": len(considered),
        "split_hash": split_hash,
        "source_files_sha": source_files_sha or {},
    }
