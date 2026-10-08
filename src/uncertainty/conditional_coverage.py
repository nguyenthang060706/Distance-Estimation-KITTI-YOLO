"""
src/uncertainty/conditional_coverage.py: Comprehensive Conditional Coverage Analysis & Exchangeability Diagnostics on Split T (Decisions D19, D47, D50, D54, D55, D70, D74, D75, D79).

Features:
- Subgroup conditional coverage & interval sharpness across 7 categories:
  1. Prospective distance bins (z_hat_f)
  2. Retrospective distance bins (z_gt - diagnostic)
  3. Truncation & Touch-edge flags (D84)
  4. Occlusion levels (0, 1, 2)
  5. Viewing angle theta bins (D19)
  6. KITTI difficulty (nested & disjoint)
  7. Fallback pattern 000 (D74)
- Drive-level heterogeneity breakdown across all 10 drives, pooled, macro, and macro n>=30 (D50, D75).
- Exchangeability diagnostics (2-sample KS-test) between Split C and Split T (D79).
- Cluster bootstrap confidence intervals (coarse 10-cluster bootstrap) for pooled coverage and width.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from src.uncertainty.cqr import winkler_score

METHODS = ["cqr", "sc", "mondrian"]

CANONICAL_DISTANCE_BINS = [
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
    for name, lo, hi in CANONICAL_DISTANCE_BINS:
        mask = (arr >= lo) & (arr < hi if np.isfinite(hi) else np.ones_like(arr, dtype=bool))
        bins[mask] = name
    return bins.item() if np.ndim(z) == 0 else bins


def compute_viewing_angle_theta(alpha: float | np.ndarray) -> float | np.ndarray:
    """
    Compute viewing angle theta = min(|alpha|, pi - |alpha|) (Decision D19).
    theta < 30 deg (pi/6): Side view (vehicle length)
    30 <= theta <= 60 deg: Diagonal view
    theta > 60 deg (pi/3): Front/Rear view (vehicle width)
    """
    a = np.abs(np.asarray(alpha, dtype=float))
    theta = np.minimum(a, np.pi - a)
    return theta.item() if np.ndim(alpha) == 0 else theta


def assign_theta_bin(alpha: float | np.ndarray) -> str | np.ndarray:
    """Assign observation angles to 3 canonical viewing angle bins (Decision D19)."""
    theta = compute_viewing_angle_theta(alpha)
    th_deg = np.rad2deg(theta)
    arr = np.asarray(th_deg, dtype=float)
    bins = np.full(arr.shape, "", dtype=object)
    bins[arr < 30.0] = "Side (<30°)"
    bins[(arr >= 30.0) & (arr <= 60.0)] = "Diagonal (30–60°)"
    bins[arr > 60.0] = "Front/Rear (>60°)"
    return bins.item() if np.ndim(alpha) == 0 else bins


def compute_interval_subgroup_metrics(
    sub_df: pd.DataFrame,
    n_fn: int = 0,
    alpha: float = 0.10,
    drive_col: str = "drive",
) -> dict[str, Any]:
    """
    Compute comprehensive interval metrics (coverage, width, Winkler, crossings)
    and survivorship metrics (n_TP, n_FN, Recall) for a subgroup subset.
    """
    n_tp = len(sub_df)
    n_gt = n_tp + n_fn
    recall = float(n_tp / n_gt) if n_gt > 0 else 0.0
    n_clusters = int(sub_df[drive_col].nunique()) if drive_col in sub_df.columns and n_tp > 0 else 0
    low_n = bool(n_tp < 100)

    res: dict[str, Any] = {
        "n_tp": n_tp,
        "n_fn": n_fn,
        "n_gt": n_gt,
        "recall": round(recall, 4),
        "k_clusters": n_clusters,
        "low_n": low_n,
    }

    if n_tp == 0:
        for m in METHODS:
            res[m] = {
                "coverage": None,
                "mean_width": None,
                "median_width": None,
                "mean_winkler": None,
                "crossings": 0,
            }
        return res

    z_gt = sub_df["z_gt"].to_numpy(dtype=float)
    r_act = sub_df["r_actual"].to_numpy(dtype=float)

    for m in METHODS:
        col_lo = f"z_lo_{m}"
        col_hi = f"z_hi_{m}"
        col_r_lo = f"r_lo_{m}"
        col_r_hi = f"r_hi_{m}"

        if col_lo not in sub_df.columns or col_hi not in sub_df.columns:
            res[m] = {
                "coverage": None,
                "mean_width": None,
                "median_width": None,
                "mean_winkler": None,
                "crossings": 0,
            }
            continue

        z_lo = sub_df[col_lo].to_numpy(dtype=float)
        z_hi = sub_df[col_hi].to_numpy(dtype=float)
        r_lo = sub_df[col_r_lo].to_numpy(dtype=float)
        r_hi = sub_df[col_r_hi].to_numpy(dtype=float)

        # 1. Coverage
        covered = (z_lo <= z_gt) & (z_gt <= z_hi)
        cov = float(np.mean(covered))

        # 2. Width ratio
        valid_width = (z_lo > 0) & np.isfinite(z_lo) & np.isfinite(z_hi)
        if np.any(valid_width):
            w_ratios = z_hi[valid_width] / z_lo[valid_width]
            mean_w = float(np.mean(w_ratios))
            med_w = float(np.median(w_ratios))
        else:
            mean_w = float("nan")
            med_w = float("nan")

        # 3. Crossings (z_lo > z_hi) without silent clip (Decision D47)
        crossings = int(np.sum(z_lo > z_hi))

        # 4. Winkler score in log-space (Decision D47)
        w_scores = winkler_score(r_lo, r_hi, r_act, alpha=alpha)
        mean_wink = float(np.mean(w_scores)) if len(w_scores) > 0 else float("nan")

        res[m] = {
            "coverage": round(cov, 4) if np.isfinite(cov) else None,
            "mean_width": round(mean_w, 3) if np.isfinite(mean_w) else None,
            "median_width": round(med_w, 3) if np.isfinite(med_w) else None,
            "mean_winkler": round(mean_wink, 4) if np.isfinite(mean_wink) else None,
            "crossings": crossings,
        }

    return res


def compute_conditional_coverage_breakdown(
    pred_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    alpha: float = 0.10,
    drive_col: str = "drive",
) -> dict[str, list[dict[str, Any]]]:
    """
    Compute full conditional coverage breakdown across 7 canonical categories.
    """
    categories: dict[str, list[dict[str, Any]]] = {}

    # --------------------------------------------------------------------------
    # 1. Prospective distance bins based on Ẑ (z_hat_f)
    # --------------------------------------------------------------------------
    z_hat_bins_rows = []
    z_hat = pred_df["z_hat_f"].to_numpy(dtype=float)
    bin_names = assign_distance_bin(z_hat)
    pred_df_copy = pred_df.copy()
    pred_df_copy["z_hat_bin"] = bin_names

    # Primary 5 bins
    for b_name, _, _ in CANONICAL_DISTANCE_BINS:
        mask = pred_df_copy["z_hat_bin"] == b_name
        sub_p = pred_df_copy[mask]
        fn_count = int(np.sum(assign_distance_bin(fn_df["z_gt"].to_numpy(dtype=float)) == b_name)) if not fn_df.empty else 0
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=fn_count, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = b_name
        z_hat_bins_rows.append(m_dict)

    # Grouped row >30m
    mask_gt30 = pred_df_copy["z_hat_f"] > 30.0
    sub_gt30 = pred_df_copy[mask_gt30]
    fn_gt30 = int(np.sum(fn_df["z_gt"].to_numpy(dtype=float) > 30.0)) if not fn_df.empty else 0
    m_gt30 = compute_interval_subgroup_metrics(sub_gt30, n_fn=fn_gt30, alpha=alpha, drive_col=drive_col)
    m_gt30["subgroup"] = ">30 (grouped)"
    z_hat_bins_rows.append(m_gt30)
    categories["z_hat_prospective"] = z_hat_bins_rows

    # --------------------------------------------------------------------------
    # 2. Retrospective distance bins based on Z_gt (Diagnostic, v4 §6)
    # --------------------------------------------------------------------------
    z_gt_bins_rows = []
    z_gt_arr = pred_df["z_gt"].to_numpy(dtype=float)
    pred_df_copy["z_gt_bin"] = assign_distance_bin(z_gt_arr)

    for b_name, _, _ in CANONICAL_DISTANCE_BINS:
        mask = pred_df_copy["z_gt_bin"] == b_name
        sub_p = pred_df_copy[mask]
        fn_count = int(np.sum(assign_distance_bin(fn_df["z_gt"].to_numpy(dtype=float)) == b_name)) if not fn_df.empty else 0
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=fn_count, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = b_name
        z_gt_bins_rows.append(m_dict)

    mask_gt30_ret = pred_df_copy["z_gt"] > 30.0
    sub_gt30_ret = pred_df_copy[mask_gt30_ret]
    fn_gt30_ret = int(np.sum(fn_df["z_gt"].to_numpy(dtype=float) > 30.0)) if not fn_df.empty else 0
    m_gt30_ret = compute_interval_subgroup_metrics(sub_gt30_ret, n_fn=fn_gt30_ret, alpha=alpha, drive_col=drive_col)
    m_gt30_ret["subgroup"] = ">30 (grouped)"
    z_gt_bins_rows.append(m_gt30_ret)
    categories["z_gt_retrospective"] = z_gt_bins_rows

    # --------------------------------------------------------------------------
    # 3. Truncation & Touch-edge flags (Decision D84)
    # --------------------------------------------------------------------------
    trunc_rows = []
    t_arr = pred_df["truncated"].to_numpy(dtype=float)
    trunc_bins = [
        ("No Truncation (0.0)", t_arr == 0.0, fn_df["truncated"] == 0.0 if "truncated" in fn_df.columns else pd.Series(False, index=fn_df.index)),
        ("Mild (0.0 < t <= 0.15)", (t_arr > 0.0) & (t_arr <= 0.15), (fn_df["truncated"] > 0.0) & (fn_df["truncated"] <= 0.15) if "truncated" in fn_df.columns else pd.Series(False, index=fn_df.index)),
        ("Moderate/Severe (0.15 < t <= 0.50)", (t_arr > 0.15) & (t_arr <= 0.50), (fn_df["truncated"] > 0.15) & (fn_df["truncated"] <= 0.50) if "truncated" in fn_df.columns else pd.Series(False, index=fn_df.index)),
    ]
    for name, p_mask, fn_mask in trunc_bins:
        sub_p = pred_df[p_mask]
        fn_count = int(fn_mask.sum())
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=fn_count, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = name
        trunc_rows.append(m_dict)

    # Touch-edge categories derived from valid flags (Decision D84)
    vw = pred_df["valid_w"].to_numpy() == 1
    vh = pred_df["valid_h"].to_numpy() == 1
    vg = pred_df["valid_g"].to_numpy() == 1

    touch_bins = [
        ("No Edge Touch (Pattern 111)", vw & vh & vg),
        ("Touch Horizontal (valid_w=0)", (~vw) & (vh | vg)),
        ("Touch Vertical (valid_h=0 or valid_g=0)", vw & ((~vh) | (~vg))),
        ("Touch Multi-edge (valid_w=0 and (valid_h=0 or valid_g=0))", (~vw) & ((~vh) | (~vg))),
    ]
    for name, p_mask in touch_bins:
        sub_p = pred_df[p_mask]
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=0, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = name
        trunc_rows.append(m_dict)

    categories["truncation_and_edges"] = trunc_rows

    # --------------------------------------------------------------------------
    # 4. Occlusion levels
    # --------------------------------------------------------------------------
    occ_rows = []
    occ_arr = pred_df["occluded"].to_numpy(dtype=int)
    occ_levels = [
        ("0 (Fully visible)", 0),
        ("1 (Partly occluded)", 1),
        ("2 (Largely occluded)", 2),
    ]
    for name, val in occ_levels:
        sub_p = pred_df[occ_arr == val]
        fn_count = int(np.sum(fn_df["occluded"] == val)) if "occluded" in fn_df.columns else 0
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=fn_count, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = name
        occ_rows.append(m_dict)
    categories["occlusion"] = occ_rows

    # --------------------------------------------------------------------------
    # 5. Viewing angle theta bins (Decision D19)
    # --------------------------------------------------------------------------
    theta_rows = []
    alpha_arr = pred_df["alpha"].to_numpy(dtype=float)
    th_bins = assign_theta_bin(alpha_arr)
    pred_df_copy["theta_bin"] = th_bins

    for b_name in ["Side (<30°)", "Diagonal (30–60°)", "Front/Rear (>60°)"]:
        sub_p = pred_df_copy[pred_df_copy["theta_bin"] == b_name]
        fn_count = int(np.sum(assign_theta_bin(fn_df["alpha"].to_numpy(dtype=float)) == b_name)) if "alpha" in fn_df.columns and not fn_df.empty else 0
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=fn_count, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = b_name
        theta_rows.append(m_dict)
    categories["viewing_angle_theta"] = theta_rows

    # --------------------------------------------------------------------------
    # 6. KITTI Difficulty (Nested & Disjoint)
    # --------------------------------------------------------------------------
    diff_rows = []
    diff_configs = [
        ("Easy (nested)", {"Easy"}),
        ("Moderate (nested)", {"Easy", "Moderate"}),
        ("Hard (nested)", {"Easy", "Moderate", "Hard"}),
        ("---", set()),
        ("Easy (disjoint)", {"Easy"}),
        ("Moderate (disjoint)", {"Moderate"}),
        ("Hard (disjoint)", {"Hard"}),
    ]
    for name, d_set in diff_configs:
        if not d_set:
            diff_rows.append({"subgroup": "---"})
            continue
        p_mask = pred_df["difficulty"].isin(d_set)
        fn_mask = fn_df["difficulty"].isin(d_set) if "difficulty" in fn_df.columns else pd.Series(False, index=fn_df.index)
        sub_p = pred_df[p_mask]
        fn_count = int(fn_mask.sum())
        m_dict = compute_interval_subgroup_metrics(sub_p, n_fn=fn_count, alpha=alpha, drive_col=drive_col)
        m_dict["subgroup"] = name
        diff_rows.append(m_dict)
    categories["difficulty"] = diff_rows

    # --------------------------------------------------------------------------
    # 7. Fallback Pattern 000 (Decision D74)
    # --------------------------------------------------------------------------
    fb_rows = []
    fb_mask = pred_df["fallback_flag"].to_numpy(dtype=bool)
    sub_fb = pred_df[fb_mask]
    sub_norm = pred_df[~fb_mask]

    m_fb = compute_interval_subgroup_metrics(sub_fb, n_fn=0, alpha=alpha, drive_col=drive_col)
    m_fb["subgroup"] = "Fallback (Pattern 000)"
    fb_rows.append(m_fb)

    m_norm = compute_interval_subgroup_metrics(sub_norm, n_fn=len(fn_df), alpha=alpha, drive_col=drive_col)
    m_norm["subgroup"] = "Fused Geometric (>=1 cue)"
    fb_rows.append(m_norm)
    categories["fallback_pattern_000"] = fb_rows

    return categories


def compute_drive_level_coverage_table(
    pred_df: pd.DataFrame,
    alpha: float = 0.10,
    drive_col: str = "drive",
) -> dict[str, Any]:
    """
    Compute drive-level coverage breakdown across all 10 drives of Split T (Decision D50, D75).
    Reports pooled, macro (all 10 drives), and macro (drives with n >= 30, k=8 drives).
    """
    unique_drives = sorted(pred_df[drive_col].unique())
    drive_rows = []
    cov_macro_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    cov_macro_ge30_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    width_macro_acc: dict[str, list[float]] = {m: [] for m in METHODS}

    for d in unique_drives:
        sub_d = pred_df[pred_df[drive_col] == d]
        n_d = len(sub_d)
        row: dict[str, Any] = {
            "drive": d,
            "n": n_d,
            "low_n": bool(n_d < 30),
        }

        m_dict = compute_interval_subgroup_metrics(sub_d, n_fn=0, alpha=alpha, drive_col=drive_col)
        for m in METHODS:
            row[f"{m}_coverage"] = m_dict[m]["coverage"]
            row[f"{m}_mean_width"] = m_dict[m]["mean_width"]
            row[f"{m}_winkler"] = m_dict[m]["mean_winkler"]
            row[f"{m}_crossings"] = m_dict[m]["crossings"]

            if m_dict[m]["coverage"] is not None:
                cov_macro_acc[m].append(m_dict[m]["coverage"])
                if n_d >= 30:
                    cov_macro_ge30_acc[m].append(m_dict[m]["coverage"])
            if m_dict[m]["mean_width"] is not None:
                width_macro_acc[m].append(m_dict[m]["mean_width"])

        drive_rows.append(row)

    # Compute summary macro and pooled metrics
    pooled_dict = compute_interval_subgroup_metrics(pred_df, n_fn=0, alpha=alpha, drive_col=drive_col)

    summary: dict[str, Any] = {
        "n_total": len(pred_df),
        "n_drives": len(unique_drives),
        "n_drives_ge30": len([d for d in unique_drives if (pred_df[drive_col] == d).sum() >= 30]),
    }

    for m in METHODS:
        summary[m] = {
            "pooled_coverage": pooled_dict[m]["coverage"],
            "macro_coverage": round(float(np.mean(cov_macro_acc[m])), 4) if cov_macro_acc[m] else None,
            "macro_coverage_ge30": round(float(np.mean(cov_macro_ge30_acc[m])), 4) if cov_macro_ge30_acc[m] else None,
            "mean_width": pooled_dict[m]["mean_width"],
            "mean_winkler": pooled_dict[m]["mean_winkler"],
            "crossings": pooled_dict[m]["crossings"],
        }

    return {
        "drive_rows": drive_rows,
        "summary": summary,
    }


def compute_exchangeability_ks_diagnostics(
    df_c: pd.DataFrame,
    df_t: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Perform 2-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp) between Split C and Split T (Decision D79).
    Compares observable test-time distributions to identify breakdown of exchangeability.
    """
    features_to_test = [
        ("z_hat_f", "Độ sâu dự đoán Ẑ (m)"),
        ("z_gt", "Độ sâu thực tế Z_gt (m)"),
        ("confidence", "Độ tin cậy detector (confidence)"),
        ("valid_w", "Cờ hợp lệ Cue Chiều rộng (valid_w)"),
        ("valid_h", "Cờ hợp lệ Cue Chiều cao (valid_h)"),
        ("valid_g", "Cờ hợp lệ Cue Cạnh dưới (valid_g)"),
        ("abs_r", "Sai số log-residual |r| = |ln Z_gt - ln Z_base|"),
    ]

    results: list[dict[str, Any]] = []

    for feat_key, feat_name in features_to_test:
        if feat_key == "abs_r":
            val_c = np.abs(df_c["r_actual"].to_numpy(dtype=float)) if "r_actual" in df_c.columns else None
            val_t = np.abs(df_t["r_actual"].to_numpy(dtype=float)) if "r_actual" in df_t.columns else None
        else:
            val_c = df_c[feat_key].to_numpy(dtype=float) if feat_key in df_c.columns else None
            val_t = df_t[feat_key].to_numpy(dtype=float) if feat_key in df_t.columns else None

        if val_c is None or val_t is None:
            continue

        # Filter NaNs
        clean_c = val_c[np.isfinite(val_c)]
        clean_t = val_t[np.isfinite(val_t)]

        if len(clean_c) == 0 or len(clean_t) == 0:
            continue

        ks_res = stats.ks_2samp(clean_c, clean_t)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        results.append({
            "feature_key": feat_key,
            "feature_name": feat_name,
            "n_c": len(clean_c),
            "n_t": len(clean_t),
            "mean_c": round(float(np.mean(clean_c)), 4),
            "mean_t": round(float(np.mean(clean_t)), 4),
            "std_c": round(float(np.std(clean_c)), 4),
            "std_t": round(float(np.std(clean_t)), 4),
            "ks_statistic": round(ks_stat, 4),
            "p_value": p_val,
            "significant_diff": bool(p_val < 0.05),
        })

    return results


def cluster_bootstrap_coverage_ci(
    pred_df: pd.DataFrame,
    method: str = "cqr",
    n_boot: int = 1000,
    seed: int = 42,
    drive_col: str = "drive",
) -> dict[str, Any]:
    """
    Cluster bootstrap confidence interval (10 clusters) for pooled coverage and mean width (Decisions D18, D20, D54).
    """
    drives = np.asarray(pred_df[drive_col].unique())
    k = len(drives)
    rng = np.random.default_rng(seed)

    cov_samples = []
    width_samples = []

    z_gt = pred_df["z_gt"].to_numpy(dtype=float)
    z_lo = pred_df[f"z_lo_{method}"].to_numpy(dtype=float)
    z_hi = pred_df[f"z_hi_{method}"].to_numpy(dtype=float)
    drive_arr = pred_df[drive_col].to_numpy()

    # Pre-index objects by drive
    drive_indices = {d: np.where(drive_arr == d)[0] for d in drives}

    for _ in range(n_boot):
        boot_drives = rng.choice(drives, size=k, replace=True)
        boot_idx = np.concatenate([drive_indices[d] for d in boot_drives])

        cov = np.mean((z_lo[boot_idx] <= z_gt[boot_idx]) & (z_gt[boot_idx] <= z_hi[boot_idx]))
        cov_samples.append(cov)

        valid_w = (z_lo[boot_idx] > 0) & np.isfinite(z_lo[boot_idx]) & np.isfinite(z_hi[boot_idx])
        if np.any(valid_w):
            width = np.mean(z_hi[boot_idx][valid_w] / z_lo[boot_idx][valid_w])
            width_samples.append(width)

    cov_ci = [float(np.percentile(cov_samples, 2.5)), float(np.percentile(cov_samples, 97.5))]
    width_ci = [float(np.percentile(width_samples, 2.5)), float(np.percentile(width_samples, 97.5))] if width_samples else [float("nan"), float("nan")]

    return {
        "method": method,
        "n_clusters": k,
        "coverage_ci_95": [round(cov_ci[0], 4), round(cov_ci[1], 4)],
        "width_ci_95": [round(width_ci[0], 3), round(width_ci[1], 3)],
        "note": f"CI thô ({k} cụm)",
    }
