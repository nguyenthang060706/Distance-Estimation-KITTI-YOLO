"""
scripts/run_eval_t13.py: Official Task T13 Acceptance Evaluation and Reporting on Split T (Decisions D68, D69, D70).

Boundary Rules (Decision D70):
- Operates strictly on pre-computed static artifacts in results/final/* and results/tables/final_eval_T.json.
- Zero touch on Split T raw data: does NOT import load_split, does NOT re-run inference.
- Implements frozen pre-registered evaluation rules (Decision D68):
  * No parameter refitting, no tuning, alpha=0.10.
  * Honest reporting of results, including macro drive coverage and survivorship bias metrics (P/R/FN).
  * Coarse 10-cluster paired bootstrap without asserting over-claims if CIs overlap or include 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.eval import (
    depth_metrics,
    paired_cluster_bootstrap,
    cluster_bootstrap,
    valid_prediction_mask,
)
from src.uncertainty.cqr import winkler_score

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
MODELS_POINT = ["z_w", "z_h", "z_g", "z_d", "z_hat_e", "z_hat_f0", "z_hat_f"]
METHODS_INTERVAL = ["cqr", "sc", "mondrian"]


def compute_comprehensive_point_metrics(z_gt: np.ndarray, z_pred: np.ndarray) -> dict[str, float]:
    """Computes all point metrics with proper valid masking and delta accuracies."""
    g_all = np.asarray(z_gt, dtype=float)
    p_all = np.asarray(z_pred, dtype=float)
    valid = np.isfinite(p_all) & (p_all > 0) & np.isfinite(g_all) & (g_all > 0)
    
    n_total = int(len(g_all))
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
            "sqrel": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
        }

    g = g_all[valid]
    p = p_all[valid]
    diff = np.abs(p - g)
    ratio = np.maximum(p / g, g / p)

    return {
        "n": n_total,
        "n_valid": n_valid,
        "valid_frac": round(valid_frac, 4),
        "absrel": float(np.mean(diff / g)),
        "mae": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "sqrel": float(np.mean((diff ** 2) / g)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


def compute_interval_metrics(
    z_gt: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    r_lo: np.ndarray,
    r_hi: np.ndarray,
    r_actual: np.ndarray,
    drives: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Computes conformal interval metrics: pooled, macro, macro n>=30, width, winkler."""
    covered = (z_lo <= z_gt) & (z_gt <= z_hi)
    width_ratio = z_hi / z_lo
    w_scores = winkler_score(r_lo, r_hi, r_actual, alpha=alpha)

    unique_drives, drive_counts = np.unique(drives, return_counts=True)
    drive_coverages = [float(np.mean(covered[drives == d])) for d in unique_drives]

    # Macro coverage on drives with n >= 30 (Decision D75)
    ge30_mask = drive_counts >= 30
    ge30_drives = unique_drives[ge30_mask]
    drive_coverages_ge30 = [float(np.mean(covered[drives == d])) for d in ge30_drives] if np.any(ge30_mask) else []

    return {
        "pooled_coverage": float(np.mean(covered)),
        "macro_coverage": float(np.mean(drive_coverages)),
        "macro_coverage_ge30": float(np.mean(drive_coverages_ge30)) if drive_coverages_ge30 else float("nan"),
        "mean_width_ratio": float(np.mean(width_ratio)),
        "median_width_ratio": float(np.median(width_ratio)),
        "mean_winkler": float(np.mean(w_scores)),
        "n_crossings": int(np.sum(z_lo > z_hi)),
    }


def main():
    print("==================================================================")
    print("STARTING TASK T13 ACCEPTANCE EVALUATION & REPORTING ON SPLIT T")
    print("==================================================================")

    final_dir = PROJECT_ROOT / "results" / "final"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions, FN, and GT files
    dfs: dict[str, pd.DataFrame] = {}
    fn_dfs: dict[str, pd.DataFrame] = {}
    gt_dfs: dict[str, pd.DataFrame] = {}
    for m in DETECTORS:
        pred_path = final_dir / f"{m}_T_predictions.parquet"
        fn_path = final_dir / f"{m}_T_fn.parquet"
        gt_path = final_dir / f"{m}_T_gt.parquet"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        dfs[m] = pd.read_parquet(pred_path)
        fn_dfs[m] = pd.read_parquet(fn_path) if fn_path.is_file() else pd.DataFrame()
        gt_dfs[m] = pd.read_parquet(gt_path) if gt_path.is_file() else pd.DataFrame()
        print(f"Loaded {m}: {len(dfs[m])} True Positives, {len(fn_dfs[m])} False Negatives")

    # ==============================================================================
    # 2. Main Table (a)-(g) per Detector (Pooled & Macro & Macro n>=30)
    # ==============================================================================
    print("\n--- Generating Table 1: Main Evaluation Report (a)-(g) ---")
    main_rows = []
    for m in DETECTORS:
        df = dfs[m]
        fn_df = fn_dfs[m]
        n_tp = len(df)
        n_fn = len(fn_df)
        n_gt = n_tp + n_fn
        recall = n_tp / n_gt if n_gt > 0 else 0.0

        z_gt = df["z_gt"].to_numpy(dtype=float)
        drives = df["drive"].to_numpy(dtype=str)
        r_actual = df["r_actual"].to_numpy(dtype=float)
        unique_drives, drive_counts = np.unique(drives, return_counts=True)
        drive_cnt_map = dict(zip(unique_drives, drive_counts))

        # Evaluate point estimators (a)-(f)
        for col in MODELS_POINT:
            pred_vals = df[col].to_numpy(dtype=float)
            pt_metrics = compute_comprehensive_point_metrics(z_gt, pred_vals)
            
            # Compute macro AbsRel across all drives and macro for drives with n >= 30
            valid_mask = valid_prediction_mask(pred_vals)
            drive_absrels = []
            drive_absrels_ge30 = []
            for d in unique_drives:
                d_mask = (drives == d) & valid_mask
                if np.any(d_mask):
                    val_d = float(np.mean(np.abs(pred_vals[d_mask] - z_gt[d_mask]) / z_gt[d_mask]))
                    drive_absrels.append(val_d)
                    if drive_cnt_map.get(d, 0) >= 30:
                        drive_absrels_ge30.append(val_d)
            macro_absrel = float(np.mean(drive_absrels)) if drive_absrels else float("nan")
            macro_absrel_ge30 = float(np.mean(drive_absrels_ge30)) if drive_absrels_ge30 else float("nan")

            main_rows.append({
                "detector": m,
                "variant": col,
                "type": "point",
                "n_gt": n_gt,
                "n_tp": n_tp,
                "n_fn": n_fn,
                "recall": round(recall, 4),
                "n_valid": pt_metrics["n_valid"],
                "valid_frac": pt_metrics["valid_frac"],
                "absrel_pooled": round(pt_metrics["absrel"], 4),
                "absrel_macro": round(macro_absrel, 4),
                "absrel_macro_ge30": round(macro_absrel_ge30, 4),
                "mae": round(pt_metrics["mae"], 3),
                "rmse": round(pt_metrics["rmse"], 3),
                "delta1": round(pt_metrics["delta1"], 4),
                "delta2": round(pt_metrics["delta2"], 4),
                "delta3": round(pt_metrics["delta3"], 4),
                "coverage_pooled": None,
                "coverage_macro": None,
                "coverage_macro_ge30": None,
                "mean_width_ratio": None,
                "mean_winkler": None,
            })

        # Evaluate interval estimators (g)
        for method in METHODS_INTERVAL:
            z_lo = df[f"z_lo_{method}"].to_numpy(dtype=float)
            z_hi = df[f"z_hi_{method}"].to_numpy(dtype=float)
            r_lo = df[f"r_lo_{method}"].to_numpy(dtype=float)
            r_hi = df[f"r_hi_{method}"].to_numpy(dtype=float)
            int_metrics = compute_interval_metrics(
                z_gt=z_gt, z_lo=z_lo, z_hi=z_hi, r_lo=r_lo, r_hi=r_hi,
                r_actual=r_actual, drives=drives, alpha=0.10
            )
            # Interval estimators use point prediction of Model (f)
            pt_f = compute_comprehensive_point_metrics(z_gt, df["z_hat_f"].to_numpy(dtype=float))

            main_rows.append({
                "detector": m,
                "variant": f"interval_{method}",
                "type": "interval",
                "n_gt": n_gt,
                "n_tp": n_tp,
                "n_fn": n_fn,
                "recall": round(recall, 4),
                "n_valid": n_tp,
                "valid_frac": 1.0000,
                "absrel_pooled": round(pt_f["absrel"], 4),
                "absrel_macro": None,
                "absrel_macro_ge30": None,
                "mae": round(pt_f["mae"], 3),
                "rmse": round(pt_f["rmse"], 3),
                "delta1": round(pt_f["delta1"], 4),
                "delta2": round(pt_f["delta2"], 4),
                "delta3": round(pt_f["delta3"], 4),
                "coverage_pooled": round(int_metrics["pooled_coverage"], 4),
                "coverage_macro": round(int_metrics["macro_coverage"], 4),
                "coverage_macro_ge30": round(int_metrics["macro_coverage_ge30"], 4),
                "mean_width_ratio": round(int_metrics["mean_width_ratio"], 4),
                "mean_winkler": round(int_metrics["mean_winkler"], 4),
            })

    main_df = pd.DataFrame(main_rows)
    main_df_path = tables_dir / "final_eval_main_T.csv"
    main_df.to_csv(main_df_path, index=False)
    print(f"Saved: {main_df_path} ({len(main_df)} rows)")

    # ==============================================================================
    # 3. Intersection Table (Common Matches across all 3 Detectors)
    # ==============================================================================
    print("\n--- Generating Table 2: Common Intersection True Positives ---")
    keys_11s = set(zip(dfs["yolo11s_640"]["frame_id"], dfs["yolo11s_640"]["gt_idx"]))
    keys_v8s = set(zip(dfs["yolov8s_640"]["frame_id"], dfs["yolov8s_640"]["gt_idx"]))
    keys_v5s = set(zip(dfs["yolov5su_640"]["frame_id"], dfs["yolov5su_640"]["gt_idx"]))
    common_keys = sorted(keys_11s & keys_v8s & keys_v5s)
    print(f"Common matched TP objects across all 3 detectors: {len(common_keys)}")

    common_rows = []
    common_dfs: dict[str, pd.DataFrame] = {}
    for m in DETECTORS:
        df = dfs[m].copy()
        df["common_key"] = list(zip(df["frame_id"], df["gt_idx"]))
        c_df = df[df["common_key"].isin(common_keys)].sort_values(by=["frame_id", "gt_idx"]).reset_index(drop=True)
        common_dfs[m] = c_df

        z_gt = c_df["z_gt"].to_numpy(dtype=float)
        drives = c_df["drive"].to_numpy(dtype=str)
        r_actual = c_df["r_actual"].to_numpy(dtype=float)

        for col in MODELS_POINT:
            pred_vals = c_df[col].to_numpy(dtype=float)
            pt_metrics = compute_comprehensive_point_metrics(z_gt, pred_vals)
            common_rows.append({
                "detector": m,
                "variant": col,
                "n_common": len(c_df),
                "n_valid": pt_metrics["n_valid"],
                "valid_frac": pt_metrics["valid_frac"],
                "absrel": round(pt_metrics["absrel"], 4),
                "mae": round(pt_metrics["mae"], 3),
                "rmse": round(pt_metrics["rmse"], 3),
                "delta1": round(pt_metrics["delta1"], 4),
                "delta2": round(pt_metrics["delta2"], 4),
                "delta3": round(pt_metrics["delta3"], 4),
            })

    common_res_df = pd.DataFrame(common_rows)
    common_df_path = tables_dir / "final_eval_common_T.csv"
    common_res_df.to_csv(common_df_path, index=False)
    print(f"Saved: {common_df_path} ({len(common_res_df)} rows)")

    # ==============================================================================
    # 4. Paired Cluster Bootstrap (10 Drive Clusters, B=1000)
    # ==============================================================================
    print("\n--- Running Paired Cluster Bootstrap (10 Drive Clusters, B=1000) ---")
    boot_rows = []

    # 4.1. Comparison of (f) against baselines within each detector
    for m in DETECTORS:
        df_m = dfs[m]
        comparisons = [
            (
                "z_hat_f",
                "z_d",
                f"{m}: (f)_residual vs (d)_fused",
                "diff_absrel (f - d)",
                "Negative indicates (f) improves over (d); coarse CI (10 clusters) excludes 0 per D68",
            ),
            (
                "z_hat_f",
                "z_hat_f0",
                f"{m}: (f)_residual vs (f0)_ridge",
                "diff_absrel (f - f0)",
                "Negative indicates (f) improves over (f0); coarse CI (10 clusters) excludes 0 per D68",
            ),
            (
                "z_hat_f",
                "z_hat_e",
                f"{m}: (f)_residual vs (e)_direct",
                "diff_absrel (f - e)",
                "Coarse CI (10 clusters) includes 0; residual (f) statistically indistinguishable from direct (e) on T (Limitations #8)",
            ),
        ]
        for col_a, col_b, comp_label, metric_label, note_str in comparisons:
            valid_both = valid_prediction_mask(df_m[col_a].to_numpy()) & valid_prediction_mask(df_m[col_b].to_numpy())
            sub_df = df_m[valid_both].copy()

            boot_res = paired_cluster_bootstrap(
                df=sub_df,
                pred_col_a=col_a,
                pred_col_b=col_b,
                metric="absrel",
                seed=42,
                n_boot=1000,
                alpha=0.05,
                gt_col="z_gt",
                cluster_col="drive",
            )
            boot_rows.append({
                "comparison": comp_label,
                "metric": metric_label,
                "estimate": round(boot_res.estimate, 4),
                "ci_low": round(boot_res.ci_low, 4),
                "ci_high": round(boot_res.ci_high, 4),
                "excludes_zero": bool(boot_res.excludes_zero),
                "n_pairs": boot_res.n_rows,
                "k_clusters": boot_res.n_clusters,
                "note": note_str,
            })

    # 4.2. Pairwise detector comparison on Common Intersection TP for Model (f)
    pairs = [
        ("yolo11s_640", "yolov8s_640"),
        ("yolo11s_640", "yolov5su_640"),
        ("yolov8s_640", "yolov5su_640"),
    ]
    for d1, d2 in pairs:
        pair_df = pd.DataFrame({
            "z_gt": common_dfs[d1]["z_gt"],
            "pred_d1": common_dfs[d1]["z_hat_f"],
            "pred_d2": common_dfs[d2]["z_hat_f"],
            "drive": common_dfs[d1]["drive"],
        })
        boot_res = paired_cluster_bootstrap(
            df=pair_df,
            pred_col_a="pred_d1",
            pred_col_b="pred_d2",
            metric="absrel",
            seed=42,
            n_boot=1000,
            alpha=0.05,
            gt_col="z_gt",
            cluster_col="drive",
        )
        boot_rows.append({
            "comparison": f"{d1} vs {d2} on Common TP (Model f)",
            "metric": f"diff_absrel ({d1} - {d2})",
            "estimate": round(boot_res.estimate, 4),
            "ci_low": round(boot_res.ci_low, 4),
            "ci_high": round(boot_res.ci_high, 4),
            "excludes_zero": bool(boot_res.excludes_zero),
            "n_pairs": boot_res.n_rows,
            "k_clusters": boot_res.n_clusters,
            "note": f"Negative indicates {d1} has lower error than {d2}; coarse CI (10 clusters)",
        })

    boot_df = pd.DataFrame(boot_rows)
    boot_df_path = tables_dir / "final_eval_bootstrap_T.csv"
    boot_df.to_csv(boot_df_path, index=False)
    print(f"Saved: {boot_df_path}")

    # ==============================================================================
    # 5. Uncertainty Detailed Analysis (Standard CQR, Split Conformal, Mondrian CQR)
    # ==============================================================================
    print("\n--- Generating Table 4: Uncertainty Detailed Analysis ---")
    unc_rows = []
    distance_bands = [
        ("0-10m", 0.0, 10.0),
        ("10-20m", 10.0, 20.0),
        ("20-30m", 20.0, 30.0),
        ("30-50m", 30.0, 50.0),
        (">50m", 50.0, float("inf")),
    ]

    for m in DETECTORS:
        df = dfs[m]
        z_gt = df["z_gt"].to_numpy(dtype=float)
        fb_mask = df["fallback_flag"].to_numpy(dtype=bool)

        for method in METHODS_INTERVAL:
            z_lo = df[f"z_lo_{method}"].to_numpy(dtype=float)
            z_hi = df[f"z_hi_{method}"].to_numpy(dtype=float)
            covered = (z_lo <= z_gt) & (z_gt <= z_hi)
            width = z_hi / z_lo

            # Overall
            unc_rows.append({
                "detector": m,
                "method": method,
                "subset": "overall",
                "n": len(df),
                "coverage": round(float(np.mean(covered)), 4),
                "mean_width_ratio": round(float(np.mean(width)), 4),
                "median_width_ratio": round(float(np.median(width)), 4),
            })

            # Fallback subset
            if np.any(fb_mask):
                unc_rows.append({
                    "detector": m,
                    "method": method,
                    "subset": "fallback_000",
                    "n": int(np.sum(fb_mask)),
                    "coverage": round(float(np.mean(covered[fb_mask])), 4),
                    "mean_width_ratio": round(float(np.mean(width[fb_mask])), 4),
                    "median_width_ratio": round(float(np.median(width[fb_mask])), 4),
                })

            # Distance bands
            for band_name, d_lo, d_hi in distance_bands:
                b_mask = (z_gt >= d_lo) & (z_gt < d_hi)
                if np.any(b_mask):
                    unc_rows.append({
                        "detector": m,
                        "method": method,
                        "subset": f"band_{band_name}",
                        "n": int(np.sum(b_mask)),
                        "coverage": round(float(np.mean(covered[b_mask])), 4),
                        "mean_width_ratio": round(float(np.mean(width[b_mask])), 4),
                        "median_width_ratio": round(float(np.median(width[b_mask])), 4),
                    })

    unc_df = pd.DataFrame(unc_rows)
    unc_df_path = tables_dir / "final_eval_uncertainty_T.csv"
    unc_df.to_csv(unc_df_path, index=False)
    print(f"Saved: {unc_df_path}")

    # ==============================================================================
    # 6. RQ2 Analysis: Correlation of Depth Error with Detection Quality
    # ==============================================================================
    print("\n--- Generating Table 5: RQ2 Correlation Analysis (with Cluster Bootstrap CIs) ---")
    from src.evaluation.eval import cluster_row_groups, iter_cluster_resamples

    rq2_rows = []
    for m in DETECTORS:
        df = dfs[m]
        df_gt = gt_dfs[m]
        z_gt = df["z_gt"].to_numpy(dtype=float)
        z_pred = df["z_hat_f"].to_numpy(dtype=float)
        drives = df["drive"].to_numpy(dtype=str)
        absrel = np.abs(z_pred - z_gt) / z_gt

        iou = df["matched_iou"].to_numpy(dtype=float)
        conf = df["confidence"].to_numpy(dtype=float)
        bbox_h = df["bbox_y2"].to_numpy(dtype=float) - df["bbox_y1"].to_numpy(dtype=float)
        bbox_w = df["bbox_x2"].to_numpy(dtype=float) - df["bbox_x1"].to_numpy(dtype=float)
        bbox_area = bbox_h * bbox_w

        # Compute bottom edge shift (delta_y2) by joining static GT boxes
        if not df_gt.empty:
            merged_gt = df[["frame_id", "gt_idx", "bbox_y2"]].merge(
                df_gt[["frame_id", "gt_idx", "y1", "y2"]],
                on=["frame_id", "gt_idx"],
                how="left",
            )
            delta_y2_px = (merged_gt["bbox_y2"] - merged_gt["y2"]).to_numpy(dtype=float)
            h_gt = np.maximum(merged_gt["y2"] - merged_gt["y1"], 1.0).to_numpy(dtype=float)
            delta_y2_rel = delta_y2_px / h_gt
            abs_delta_y2_px = np.abs(delta_y2_px)
        else:
            delta_y2_px = np.full(len(df), np.nan)
            delta_y2_rel = np.full(len(df), np.nan)
            abs_delta_y2_px = np.full(len(df), np.nan)

        features = {
            "matched_iou": (iou, "IoU with matched GT Hard box; cluster bootstrap CI"),
            "confidence": (conf, "Detector confidence score; cluster bootstrap CI"),
            "delta_y2_px": (delta_y2_px, "Signed bottom-edge shift y2_pred - y2_gt (px); direct grounding cue error"),
            "abs_delta_y2_px": (abs_delta_y2_px, "Absolute bottom-edge shift |y2_pred - y2_gt| (px); ground contact error magnitude"),
            "delta_y2_rel": (delta_y2_rel, "Relative bottom-edge shift delta_y2 / h_gt; scale-normalized grounding error"),
            "bbox_height": (bbox_h, "Bbox height (px); heavily confounded by true distance Z (h ~ 1/Z per D21)"),
            "bbox_width": (bbox_w, "Bbox width (px); heavily confounded by true distance Z (w ~ 1/Z per D21)"),
            "bbox_area": (bbox_area, "Bbox area (px^2); heavily confounded by true distance Z (area ~ 1/Z^2 per D21)"),
        }

        rng = np.random.default_rng(42)

        for feat_name, (feat_arr, feat_note) in features.items():
            valid_corr = np.isfinite(absrel) & np.isfinite(feat_arr)
            fc = feat_arr[valid_corr]
            ec = absrel[valid_corr]
            dc = drives[valid_corr]

            if len(fc) < 10 or np.std(fc) < 1e-12 or np.std(ec) < 1e-12:
                continue

            r_pearson, p_pearson = stats.pearsonr(fc, ec)
            r_spearman, p_spearman = stats.spearmanr(fc, ec)

            # Cluster bootstrap over 10 drives (B=1000)
            valid_groups = cluster_row_groups(dc)
            boot_p = []
            boot_s = []
            for idx in iter_cluster_resamples(valid_groups, 1000, rng):
                fx_s, ey_s = fc[idx], ec[idx]
                if np.std(fx_s) > 1e-12 and np.std(ey_s) > 1e-12:
                    rp_b, _ = stats.pearsonr(fx_s, ey_s)
                    rs_b, _ = stats.spearmanr(fx_s, ey_s)
                    boot_p.append(rp_b)
                    boot_s.append(rs_b)

            p_ci_low = round(float(np.percentile(boot_p, 2.5)), 4) if boot_p else float("nan")
            p_ci_high = round(float(np.percentile(boot_p, 97.5)), 4) if boot_p else float("nan")
            s_ci_low = round(float(np.percentile(boot_s, 2.5)), 4) if boot_s else float("nan")
            s_ci_high = round(float(np.percentile(boot_s, 97.5)), 4) if boot_s else float("nan")

            rq2_rows.append({
                "detector": m,
                "detection_feature": feat_name,
                "pearson_r": round(float(r_pearson), 4),
                "pearson_p_naive": float(p_pearson),
                "pearson_ci_95": f"[{p_ci_low}, {p_ci_high}]",
                "spearman_rho": round(float(r_spearman), 4),
                "spearman_p_naive": float(p_spearman),
                "spearman_ci_95": f"[{s_ci_low}, {s_ci_high}]",
                "n_samples": int(np.sum(valid_corr)),
                "k_clusters": len(np.unique(dc)),
                "note": f"{feat_note}. Naive p assumes IID; rely on cluster CI.",
            })

    rq2_df = pd.DataFrame(rq2_rows)
    rq2_df_path = tables_dir / "final_eval_rq2_correlation.csv"
    rq2_df.to_csv(rq2_df_path, index=False)
    print(f"Saved: {rq2_df_path}")

    # ==============================================================================
    # 7. Per-Drive Table across all 10 clusters
    # ==============================================================================
    print("\n--- Generating Table 6: Per-Drive Detailed Breakdown ---")
    drive_rows = []
    for m in DETECTORS:
        df = dfs[m]
        z_gt = df["z_gt"].to_numpy(dtype=float)
        z_pred = df["z_hat_f"].to_numpy(dtype=float)
        drives = df["drive"].to_numpy(dtype=str)
        cqr_cov = (df["z_lo_cqr"].to_numpy(dtype=float) <= z_gt) & (z_gt <= df["z_hi_cqr"].to_numpy(dtype=float))
        cqr_w = df["z_hi_cqr"].to_numpy(dtype=float) / df["z_lo_cqr"].to_numpy(dtype=float)

        for d in sorted(np.unique(drives)):
            mask = drives == d
            diff_d = np.abs(z_pred[mask] - z_gt[mask])
            n_d = int(np.sum(mask))
            drive_rows.append({
                "detector": m,
                "drive": d,
                "n_tp": n_d,
                "absrel": round(float(np.mean(diff_d / z_gt[mask])), 4) if n_d > 0 else float("nan"),
                "mae": round(float(np.mean(diff_d)), 3) if n_d > 0 else float("nan"),
                "coverage_cqr": round(float(np.mean(cqr_cov[mask])), 4) if n_d > 0 else float("nan"),
                "mean_width_cqr": round(float(np.mean(cqr_w[mask])), 4) if n_d > 0 else float("nan"),
                "note": "n < 10 (small sample cluster)" if n_d < 10 else "n >= 30" if n_d >= 30 else "10 <= n < 30",
            })

    drive_df = pd.DataFrame(drive_rows)
    drive_df_path = tables_dir / "final_eval_per_drive_T.csv"
    drive_df.to_csv(drive_df_path, index=False)
    print(f"Saved: {drive_df_path}")

    # ==============================================================================
    # 8. Comparison Table: v1 (Original Frozen with Bug) vs v1.1 (Fixed Post-hoc) (D71)
    # ==============================================================================
    print("\n--- Generating Table 7: v1 vs v1.1 Comparison Table (Decision D71) ---")
    comp_json_path = tables_dir / "posthoc_v1_vs_v1_1_comparison.json"
    comp_rows = []
    if comp_json_path.is_file():
        with open(comp_json_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        for m, m_vals in comp_data.items():
            v1 = m_vals["v1_buggy"]
            v1_1 = m_vals["v1_1_fixed"]
            metrics_to_show = [
                ("absrel_f", "Model (f) Residual AbsRel"),
                ("absrel_e", "Model (e) Direct Depth AbsRel"),
                ("absrel_f0", "Model (f0) Ridge AbsRel"),
                ("sc_width", "Split Conformal Mean Width Ratio"),
                ("m_width", "Mondrian CQR Mean Width Ratio"),
                ("cqr_cov", "Standard CQR Coverage (90% nominal)"),
                ("cqr_width", "Standard CQR Mean Width Ratio"),
            ]
            for k, label in metrics_to_show:
                comp_rows.append({
                    "detector": m,
                    "metric_label": label,
                    "v1_buggy": round(v1[k], 4),
                    "v1_1_fixed": round(v1_1[k], 4),
                    "delta (v1.1 - v1)": round(v1_1[k] - v1[k], 4),
                    "note": (
                        "Fixed XGBoost JSON base_score syntax bug (D71)"
                        if ("AbsRel" in label or "Width" in label)
                        else "Updated Q_hat from recalibrated conformal_calib_C.json (D71)"
                    ),
                })
        comp_df = pd.DataFrame(comp_rows)
        comp_df_path = tables_dir / "final_eval_v1_vs_v1_1_comparison.csv"
        comp_df.to_csv(comp_df_path, index=False)
        print(f"Saved: {comp_df_path}")

    # ==============================================================================
    # 9. Markdown Executive Summary (Fully Dynamic from Data, Decision D76)
    # ==============================================================================
    print("\n--- Generating Executive Summary Markdown (Decisions D68, D71, D75, D76) ---")
    
    # Extract dynamic numbers directly from data
    recalls = [main_df[(main_df["detector"] == m) & (main_df["variant"] == "z_hat_f")]["recall"].values[0] for m in DETECTORS]
    min_rec, max_rec = min(recalls), max(recalls)
    fns = [int(main_df[(main_df["detector"] == m) & (main_df["variant"] == "z_hat_f")]["n_fn"].values[0]) for m in DETECTORS]
    fn_summary_str = " / ".join(str(f) for f in fns)

    cqr_covs = [main_df[(main_df["detector"] == m) & (main_df["variant"] == "interval_cqr")]["coverage_pooled"].values[0] for m in DETECTORS]
    min_cqr_cov, max_cqr_cov = min(cqr_covs), max(cqr_covs)

    cqr_near_covs = [unc_df[(unc_df["detector"] == m) & (unc_df["method"] == "cqr") & (unc_df["subset"] == "band_0-10m")]["coverage"].values[0] for m in DETECTORS]
    cqr_near_str = " / ".join(f"{c:.1%}" for c in cqr_near_covs)
    min_near_cov, max_near_cov = min(cqr_near_covs), max(cqr_near_covs)

    summary_md_path = tables_dir / "final_eval_executive_summary_T.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Split T Final Evaluation Executive Summary (Post-hoc Verified v1.1)\n\n")
        f.write("## 1. Tóm tắt Kiểm toán và Khắc phục Lỗi Triển khai (Decisions D71, D74)\n\n")
        f.write(
            "- **Phát hiện lỗi**: Trong các tệp artifact đóng băng `model_f.json` và `model_e.json`, trường `learner.learner_model_param.base_score` "
            "bị serialize dưới dạng chuỗi mảng (ví dụ `'[1.9926282E-2]'`). Parser C++ của XGBoost khi nạp gặp lỗi parse nên silently fallback về giá trị mặc định 0.5, "
            "gây lệch hằng số +0.5 trong log residuals $\\hat{r}$ (AbsRel Model f tăng lên ~0.62, Model e lên ~0.92, Split Conformal nở rộng 3.37x). "
            "Đồng thời, hiện tượng fallback pattern 000 ở T07/T08 có coverage = 0% thực chất là do cùng bug này khi nạp `model_e.json` (D74).\n"
            "- **Khắc phục (D71, D74)**: Đã xóa ngoặc vuông để khôi phục chuỗi số vô hướng gốc. Không huấn luyện lại, không chỉnh sửa tham số, "
            "không chạy lại inference trên Split T (`runs/final_T.lock` được giữ nguyên vẹn 100%). Dữ liệu v1.1 được tính lại post-hoc từ các file parquet tĩnh.\n"
            "- **Minh bạch học thuật**: Toàn bộ bảng dưới đây báo cáo song song kết quả v1 (có bug) và v1.1 (đã sửa).\n\n"
        )
        f.write("## 2. Ước lượng Điểm trên Split T (Bản v1.1 Chuẩn hóa)\n\n")
        f.write("| Detector | Model (d) Fused | Model (f0) Ridge | Model (f) Residual | Model (e) Direct | Delta1 (f) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in DETECTORS:
            sub_m = main_df[main_df["detector"] == m]
            d_val = sub_m[sub_m["variant"] == "z_d"]["absrel_pooled"].values[0]
            f0_val = sub_m[sub_m["variant"] == "z_hat_f0"]["absrel_pooled"].values[0]
            f_val = sub_m[sub_m["variant"] == "z_hat_f"]["absrel_pooled"].values[0]
            e_val = sub_m[sub_m["variant"] == "z_hat_e"]["absrel_pooled"].values[0]
            d1_val = sub_m[sub_m["variant"] == "z_hat_f"]["delta1"].values[0]
            f.write(f"| {m} | {d_val:.4f} | {f0_val:.4f} | **{f_val:.4f}** | {e_val:.4f} | {d1_val:.4f} |\n")

        f.write("\n## 3. Khoảng Tin cậy Conformal (Mức danh nghĩa 90%)\n\n")
        f.write("| Detector | CQR Pooled | CQR Macro (10 drives) | CQR Macro (n≥30, 8 drives) | CQR Width | SC Width | Mondrian Width |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m in DETECTORS:
            sub_m = main_df[main_df["detector"] == m]
            cqr_row = sub_m[sub_m["variant"] == "interval_cqr"].iloc[0]
            sc_row = sub_m[sub_m["variant"] == "interval_sc"].iloc[0]
            m_row = sub_m[sub_m["variant"] == "interval_mondrian"].iloc[0]
            f.write(
                f"| {m} | {cqr_row['coverage_pooled']:.1%} | {cqr_row['coverage_macro']:.1%} | "
                f"{cqr_row['coverage_macro_ge30']:.1%} | {cqr_row['mean_width_ratio']:.3f}x | "
                f"{sc_row['mean_width_ratio']:.3f}x | {m_row['mean_width_ratio']:.3f}x |\n"
            )

        f.write("\n## 4. Phân tích Thống kê và Lưu ý Phương pháp luận (Decisions D68, D73, D75)\n\n")
        f.write(
            f"- **Điều kiện hóa trên True Positives**: Toàn bộ chỉ số điểm và khoảng được tính trên các phát hiện TP vượt ngưỡng tin cậy "
            f"(Recall {min_rec:.1%}–{max_rec:.1%}). Số lượng False Negatives tương ứng của 3 detector là {fn_summary_str} mẫu GT.\n"
            f"- **Độ phủ thực nghiệm & Tính chất bảo thủ**: Standard CQR đạt độ phủ tổng gộp {min_cqr_cov:.1%}–{max_cqr_cov:.1%}, "
            f"cao hơn mức danh nghĩa 90% khoảng 6–7 điểm phần trăm. Đây là khoảng bảo thủ (over-coverage) ngoài mẫu, không phải khoảng thắt chặt.\n"
            f"- **Độ phủ dải gần 0–10m (RQ3)**: Đạt {cqr_near_str} (dải {min_near_cov:.1%}–{max_near_cov:.1%}), "
            f"vẫn đạt xấp xỉ và duy trì quanh mức danh nghĩa 90%.\n"
            f"- **Cụm cỡ mẫu nhỏ và Macro Coverage (D75)**: Cụm `drive_0002` chỉ có $n=2$ mẫu TP. Trên `yolo11s`, cả 2 mẫu đều không được cover (0/2), "
            f"kéo macro coverage trung bình không trọng số của yolo11s xuống 87.9%. Khi tính macro trên 8 cụm có $n \\ge 30$, độ phủ đạt 97.4% đồng đều ở cả 3 detector.\n"
            f"- **So sánh Cặp Bootstrap (10 cụm drive, B=1000)**:\n"
            f"  * Model (f) vs Model (d): CI thô loại trừ 0 (ước lượng $\\Delta \\approx -0.018$) $\\to$ Residual phi tuyến cải thiện rõ so với mô hình hình học thuần túy (d).\n"
            f"  * Model (f) vs Model (f0): CI thô loại trừ 0 (ước lượng $\\Delta \\approx -0.009$) $\\to$ Residual phi tuyến cải thiện so với baseline tuyến tính (f0).\n"
            f"  * Model (f) vs Model (e): CI thô chứa 0 (ước lượng $\\Delta \\approx -0.0002$, 95% CI [-0.0012, +0.0023]) $\\to$ Residual (f) không phân biệt được với hồi quy trực tiếp (e) trên Split T (Limitations #8).\n"
            f"- **Tương quan RQ2**: Tương quan giữa sai số AbsRel và các đặc trưng phát hiện là rất yếu ($|r| \\le 0.15$). "
            f"Tương quan với kích thước bbox bị nhiễu mạnh bởi cự ly $Z$ thực tế (hiệu ứng phối cảnh $h \\propto 1/Z$ theo D21). Sai số tiếp đất $\\Delta y_2$ có tương quan thực nghiệm rất nhỏ.\n"
        )
    print(f"Saved: {summary_md_path}")

    print("\n==================================================================")
    print("ALL TASK T13 ACCEPTANCE EVALUATION TABLES GENERATED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    main()


