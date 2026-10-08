"""
src/evaluation/error_analysis.py: Comprehensive Error Analysis module for Task T14 (Decisions D19, D21, D32, D54, D70, D84).

Theoretical & Protocol Foundation:
- Operates on pre-computed static evaluation artifacts from Split T (Decision D70).
- Transparent survivorship bias reporting: always outputs n_TP, n_FN, and Recall alongside error metrics (Decision D32).
- Distance bins with sample-size awareness: 5 canonical bins + '>30m' grouped row, flagged with '*' when n < 100, cluster count k (Decisions D3, D54).
- Viewing angle stratification theta = min(|alpha|, pi - |alpha|) across Side (<30 deg), Diagonal (30-60 deg), Front/Rear (>60 deg) (Decision D19).
- Physical ground-truth center vs visual surface bias analysis in near range (0-10m), isolated on Pattern 111 without border cut (valid_w & valid_h & valid_g) (Decisions D21, D84).
- Mining top failure cases and verified successful cases for qualitative rendering in T16.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import pandas as pd


DISTANCE_BINS = [
    ("0-10", 0.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-50", 30.0, 50.0),
    (">50", 50.0, float("inf")),
]


def assign_distance_bin(z: float | np.ndarray) -> str | np.ndarray:
    """Assign distance values to canonical KITTI evaluation bins."""
    arr = np.asarray(z, dtype=float)
    bins = np.full(arr.shape, "", dtype=object)
    for name, lo, hi in DISTANCE_BINS:
        mask = (arr >= lo) & (arr < hi if np.isfinite(hi) else np.ones_like(arr, dtype=bool))
        bins[mask] = name
    return bins.item() if np.ndim(z) == 0 else bins


def compute_point_metrics(z_gt: np.ndarray, z_pred: np.ndarray) -> dict[str, float]:
    """Compute standard depth estimation point error metrics with finite masking."""
    g = np.asarray(z_gt, dtype=float)
    p = np.asarray(z_pred, dtype=float)
    valid = np.isfinite(g) & (g > 0) & np.isfinite(p) & (p > 0)

    n_total = int(len(g))
    n_valid = int(valid.sum())
    valid_frac = float(n_valid / n_total) if n_total > 0 else 0.0

    if n_valid == 0:
        return {
            "n": n_total,
            "n_valid": 0,
            "valid_frac": 0.0,
            "absrel": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "delta1": float("nan"),
        }

    gv = g[valid]
    pv = p[valid]
    diff = np.abs(pv - gv)
    ratio = np.maximum(pv / gv, gv / pv)

    return {
        "n": n_total,
        "n_valid": n_valid,
        "valid_frac": round(valid_frac, 4),
        "absrel": float(np.mean(diff / gv)),
        "mae": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def compute_subgroup_metrics(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "z_gt",
    pred_col: str = "z_hat_f",
    drive_col: str = "drive",
) -> list[dict[str, Any]]:
    """
    Compute depth metrics grouped by a specific metadata column, with cluster counts and low-n flags.
    """
    results: list[dict[str, Any]] = []
    if group_col not in df.columns:
        return results

    groups = df[group_col].dropna().unique()
    try:
        groups = sorted(groups)
    except TypeError:
        pass

    for grp in groups:
        sub = df[df[group_col] == grp]
        n_samples = len(sub)
        n_clusters = int(sub[drive_col].nunique()) if drive_col in sub.columns else 0

        g_arr = sub[target_col].to_numpy(dtype=float)
        p_arr = sub[pred_col].to_numpy(dtype=float)
        m = compute_point_metrics(g_arr, p_arr)

        results.append({
            "group_name": str(grp),
            "n": n_samples,
            "n_valid": m["n_valid"],
            "valid_frac": m["valid_frac"],
            "k_clusters": n_clusters,
            "low_n": bool(n_samples < 100),
            "absrel": round(m["absrel"], 4) if np.isfinite(m["absrel"]) else None,
            "mae": round(m["mae"], 3) if np.isfinite(m["mae"]) else None,
            "rmse": round(m["rmse"], 3) if np.isfinite(m["rmse"]) else None,
            "delta1": round(m["delta1"], 4) if np.isfinite(m["delta1"]) else None,
        })

    return results


def compute_binned_distance_breakdown(
    pred_df: pd.DataFrame,
    fn_df: pd.DataFrame | None = None,
    distance_col: str = "z_gt",
    pred_col: str = "z_hat_f",
    drive_col: str = "drive",
) -> list[dict[str, Any]]:
    """
    Compute breakdown across canonical distance bins and '>30m' grouped row.
    Transparently pairs True Positives with False Negatives to report Recall (Decision D32).
    """
    rows: list[dict[str, Any]] = []

    # Evaluate canonical bins + >30m row
    bin_specs = list(DISTANCE_BINS) + [(">30", 30.0, float("inf"))]

    for bin_name, lo, hi in bin_specs:
        tp_mask = (pred_df[distance_col] >= lo) & (
            pred_df[distance_col] < hi if np.isfinite(hi) else np.ones(len(pred_df), dtype=bool)
        )
        tp_sub = pred_df[tp_mask]
        n_tp = len(tp_sub)

        n_fn = 0
        if fn_df is not None and not fn_df.empty and distance_col in fn_df.columns:
            fn_mask = (fn_df[distance_col] >= lo) & (
                fn_df[distance_col] < hi if np.isfinite(hi) else np.ones(len(fn_df), dtype=bool)
            )
            n_fn = int(fn_mask.sum())

        n_gt = n_tp + n_fn
        recall = float(n_tp / n_gt) if n_gt > 0 else 0.0
        n_clusters = int(tp_sub[drive_col].nunique()) if drive_col in tp_sub.columns else 0

        m = compute_point_metrics(
            tp_sub[distance_col].to_numpy(dtype=float),
            tp_sub[pred_col].to_numpy(dtype=float),
        )

        rows.append({
            "distance_bin": bin_name,
            "n_tp": n_tp,
            "n_fn": n_fn,
            "n_gt": n_gt,
            "recall": round(recall, 4),
            "k_clusters": n_clusters,
            "low_n": bool(n_tp < 100),
            "absrel": round(m["absrel"], 4) if np.isfinite(m["absrel"]) else None,
            "mae": round(m["mae"], 3) if np.isfinite(m["mae"]) else None,
            "rmse": round(m["rmse"], 3) if np.isfinite(m["rmse"]) else None,
            "delta1": round(m["delta1"], 4) if np.isfinite(m["delta1"]) else None,
        })

    return rows


def compute_viewing_angle_breakdown(
    df: pd.DataFrame,
    alpha_col: str = "alpha",
    target_col: str = "z_gt",
    drive_col: str = "drive",
) -> dict[str, Any]:
    """
    Stratify evaluation by viewing angle theta = min(|alpha|, pi - |alpha|) (Decision D19).

    3 canonical bins:
    - Side (Ngang): theta < 30 deg (0.5236 rad)
    - Diagonal (Chéo): 30 <= theta <= 60 deg (0.5236 - 1.0472 rad)
    - Front/Rear (Đầu/Đuôi): theta > 60 deg (1.0472 rad)

    Compares individual visual cues (z_w, z_h, z_g), geometric fusion (z_d),
    linear baseline (z_hat_f0), and full residual model (z_hat_f).
    """
    if alpha_col not in df.columns:
        raise KeyError(f"Missing '{alpha_col}' column in DataFrame.")

    alpha_arr = df[alpha_col].to_numpy(dtype=float)
    abs_alpha = np.abs(alpha_arr)
    theta_arr = np.minimum(abs_alpha, np.pi - abs_alpha)
    theta_deg = np.degrees(theta_arr)

    subgroups = {
        "side": theta_deg < 30.0,
        "diagonal": (theta_deg >= 30.0) & (theta_deg <= 60.0),
        "front_rear": theta_deg > 60.0,
    }

    models_to_evaluate = [
        ("z_w", "Width cue"),
        ("z_h", "Height cue"),
        ("z_g", "Ground cue"),
        ("z_d", "Geometric Fused (d)"),
        ("z_hat_f0", "Linear Baseline (f0)"),
        ("z_hat_f", "Residual Model (f)"),
    ]

    breakdown_results: dict[str, Any] = {}

    for grp_key, mask in subgroups.items():
        sub = df[mask]
        n_samples = len(sub)
        n_clusters = int(sub[drive_col].nunique()) if drive_col in sub.columns else 0
        z_gt_sub = sub[target_col].to_numpy(dtype=float)

        model_metrics: dict[str, Any] = {}
        for m_col, m_label in models_to_evaluate:
            if m_col in sub.columns:
                p_arr = sub[m_col].to_numpy(dtype=float)
                m = compute_point_metrics(z_gt_sub, p_arr)
                model_metrics[m_col] = {
                    "label": m_label,
                    "valid_frac": m["valid_frac"],
                    "absrel": round(m["absrel"], 4) if np.isfinite(m["absrel"]) else None,
                    "mae": round(m["mae"], 3) if np.isfinite(m["mae"]) else None,
                    "delta1": round(m["delta1"], 4) if np.isfinite(m["delta1"]) else None,
                }

        breakdown_results[grp_key] = {
            "n": n_samples,
            "k_clusters": n_clusters,
            "low_n": bool(n_samples < 100),
            "models": model_metrics,
        }

    return breakdown_results


def compute_physical_bias_analysis(
    df: pd.DataFrame,
    target_col: str = "z_gt",
    pred_col: str = "z_hat_f",
    distance_col: str = "z_gt",
) -> dict[str, Any]:
    """
    Empirical verification of Decision D21 (Center-vs-surface distance bias).

    Calculates signed residual bias Delta Z = Z_pred - Z_gt and relative signed bias
    (Z_pred - Z_gt) / Z_gt across distance bins.
    Specifically isolates 'Pattern 111 without border cut' (valid_w & valid_h & valid_g)
    in the 0-10m range to verify if negative near bias persists physically (Decision D84).
    """
    g = df[target_col].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(g) & (g > 0) & np.isfinite(p) & (p > 0)

    clean_df = df[valid].copy()
    diff_signed = p[valid] - g[valid]
    rel_signed = diff_signed / g[valid]
    clean_df["bias_signed"] = diff_signed
    clean_df["bias_rel"] = rel_signed

    bin_bias_summary: list[dict[str, Any]] = []
    bin_specs = list(DISTANCE_BINS) + [(">30", 30.0, float("inf"))]

    for bin_name, lo, hi in bin_specs:
        mask = (clean_df[distance_col] >= lo) & (
            clean_df[distance_col] < hi if np.isfinite(hi) else np.ones(len(clean_df), dtype=bool)
        )
        b_sub = clean_df[mask]
        n_b = len(b_sub)
        if n_b == 0:
            continue

        bs = b_sub["bias_signed"].to_numpy(dtype=float)
        br = b_sub["bias_rel"].to_numpy(dtype=float)

        bin_bias_summary.append({
            "distance_bin": bin_name,
            "n": n_b,
            "low_n": bool(n_b < 100),
            "bias_mean_m": round(float(np.mean(bs)), 3),
            "bias_median_m": round(float(np.median(bs)), 3),
            "bias_rel_mean": round(float(np.mean(br)), 4),
            "bias_rel_median": round(float(np.median(br)), 4),
            "bias_rel_iqr": round(float(np.percentile(br, 75) - np.percentile(br, 25)), 4),
        })

    # Isolate Pattern 111 (all three cues valid = no border cut per eps mask, Decision D84) in 0-10m
    p111_mask = (
        (clean_df["valid_w"] == 1) &
        (clean_df["valid_h"] == 1) &
        (clean_df["valid_g"] == 1)
    ) if {"valid_w", "valid_h", "valid_g"}.issubset(clean_df.columns) else np.zeros(len(clean_df), dtype=bool)

    near_mask = clean_df[distance_col] < 10.0
    near_all = clean_df[near_mask]
    near_p111 = clean_df[near_mask & p111_mask]

    pattern_111_verification = {
        "near_0_10m_all": {
            "n": len(near_all),
            "bias_rel_mean": round(float(np.mean(near_all["bias_rel"])), 4) if len(near_all) > 0 else None,
            "bias_rel_median": round(float(np.median(near_all["bias_rel"])), 4) if len(near_all) > 0 else None,
        },
        "near_0_10m_pattern_111_no_border_cut": {
            "n": len(near_p111),
            "bias_rel_mean": round(float(np.mean(near_p111["bias_rel"])), 4) if len(near_p111) > 0 else None,
            "bias_rel_median": round(float(np.median(near_p111["bias_rel"])), 4) if len(near_p111) > 0 else None,
        },
    }

    return {
        "by_distance_bin": bin_bias_summary,
        "pattern_111_verification": pattern_111_verification,
    }


def extract_top_failures_and_successes(
    df: pd.DataFrame,
    detector_name: str,
    top_k_fail: int = 50,
    n_success: int = 10,
    seed: int = 42,
    target_col: str = "z_gt",
    pred_col: str = "z_hat_f",
) -> dict[str, Any]:
    """
    Mine top failure cases ranked by AbsRel and representative accurate cases with fixed seed.
    Outputs structured metadata for qualitative visualization in Task T16.
    """
    g = df[target_col].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(g) & (g > 0) & np.isfinite(p) & (p > 0)

    work_df = df[valid].copy()
    work_df["absrel_val"] = np.abs(p[valid] - g[valid]) / g[valid]

    # Calculate theta
    if "alpha" in work_df.columns:
        a_arr = work_df["alpha"].to_numpy(dtype=float)
        th_arr = np.minimum(np.abs(a_arr), np.pi - np.abs(a_arr))
        work_df["theta_deg"] = np.degrees(th_arr)
    else:
        work_df["theta_deg"] = float("nan")

    # 1. Top failures sorted by AbsRel descending
    fail_df = work_df.sort_values(by="absrel_val", ascending=False).head(top_k_fail)

    # Analyze subgroup distribution among top failures
    dist_bins = assign_distance_bin(fail_df[target_col].values)
    bin_counts = pd.Series(dist_bins).value_counts().to_dict()
    fb_count = int(fail_df["fallback_flag"].sum()) if "fallback_flag" in fail_df.columns else 0

    failure_items: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(fail_df.iterrows(), start=1):
        failure_items.append({
            "rank": rank,
            "detector": detector_name,
            "frame_id": str(row.get("frame_id", "")),
            "drive": str(row.get("drive", "")),
            "pred_idx": int(row.get("pred_idx", -1)),
            "gt_idx": int(row.get("gt_idx", -1)),
            "z_gt": round(float(row[target_col]), 2),
            "z_hat_f": round(float(row[pred_col]), 2),
            "z_d": round(float(row.get("z_d", float("nan"))), 2) if np.isfinite(row.get("z_d", float("nan"))) else None,
            "absrel": round(float(row["absrel_val"]), 4),
            "bbox": [
                round(float(row.get("bbox_x1", 0)), 1),
                round(float(row.get("bbox_y1", 0)), 1),
                round(float(row.get("bbox_x2", 0)), 1),
                round(float(row.get("bbox_y2", 0)), 1),
            ],
            "matched_iou": round(float(row.get("matched_iou", float("nan"))), 3),
            "alpha": round(float(row.get("alpha", float("nan"))), 3),
            "theta_deg": round(float(row["theta_deg"]), 1),
            "occluded": int(row.get("occluded", -1)),
            "truncated": round(float(row.get("truncated", 0.0)), 2),
            "difficulty": str(row.get("difficulty", "")),
            "fallback_flag": bool(row.get("fallback_flag", False)),
            "cues_valid": {
                "w": bool(row.get("valid_w", 0)),
                "h": bool(row.get("valid_h", 0)),
                "g": bool(row.get("valid_g", 0)),
            },
        })

    # 2. Representative accurate cases across distance ranges (AbsRel < 0.05)
    accurate_candidates = work_df[work_df["absrel_val"] < 0.05].copy()
    if len(accurate_candidates) > n_success:
        accurate_sample = accurate_candidates.sample(n=n_success, random_state=seed)
    else:
        accurate_sample = accurate_candidates

    success_items: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(accurate_sample.iterrows(), start=1):
        success_items.append({
            "rank": rank,
            "detector": detector_name,
            "frame_id": str(row.get("frame_id", "")),
            "drive": str(row.get("drive", "")),
            "z_gt": round(float(row[target_col]), 2),
            "z_hat_f": round(float(row[pred_col]), 2),
            "absrel": round(float(row["absrel_val"]), 4),
            "theta_deg": round(float(row["theta_deg"]), 1),
            "difficulty": str(row.get("difficulty", "")),
            "bbox": [
                round(float(row.get("bbox_x1", 0)), 1),
                round(float(row.get("bbox_y1", 0)), 1),
                round(float(row.get("bbox_x2", 0)), 1),
                round(float(row.get("bbox_y2", 0)), 1),
            ],
        })

    return {
        "detector": detector_name,
        "n_top_failures": len(failure_items),
        "failures_bin_distribution": bin_counts,
        "failures_fallback_count": fb_count,
        "top_failures": failure_items,
        "representative_successes": success_items,
    }
