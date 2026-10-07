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
    """Computes conformal interval metrics: pooled and macro coverage, width, winkler."""
    covered = (z_lo <= z_gt) & (z_gt <= z_hi)
    width_ratio = z_hi / z_lo
    w_scores = winkler_score(r_lo, r_hi, r_actual, alpha=alpha)

    unique_drives = np.unique(drives)
    drive_coverages = [float(np.mean(covered[drives == d])) for d in unique_drives]

    return {
        "pooled_coverage": float(np.mean(covered)),
        "macro_coverage": float(np.mean(drive_coverages)),
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

    # 1. Load predictions and FN files
    dfs: dict[str, pd.DataFrame] = {}
    fn_dfs: dict[str, pd.DataFrame] = {}
    for m in DETECTORS:
        pred_path = final_dir / f"{m}_T_predictions.parquet"
        fn_path = final_dir / f"{m}_T_fn.parquet"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        dfs[m] = pd.read_parquet(pred_path)
        fn_dfs[m] = pd.read_parquet(fn_path) if fn_path.is_file() else pd.DataFrame()
        print(f"Loaded {m}: {len(dfs[m])} True Positives, {len(fn_dfs[m])} False Negatives")

    # ==============================================================================
    # 2. Main Table (a)-(g) per Detector (Pooled & Macro)
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

        # Evaluate point estimators (a)-(f)
        for col in MODELS_POINT:
            pred_vals = df[col].to_numpy(dtype=float)
            pt_metrics = compute_comprehensive_point_metrics(z_gt, pred_vals)
            
            # Compute macro AbsRel across 10 drives
            valid_mask = valid_prediction_mask(pred_vals)
            drive_absrels = []
            for d in np.unique(drives):
                d_mask = (drives == d) & valid_mask
                if np.any(d_mask):
                    drive_absrels.append(np.mean(np.abs(pred_vals[d_mask] - z_gt[d_mask]) / z_gt[d_mask]))
            macro_absrel = float(np.mean(drive_absrels)) if drive_absrels else float("nan")

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
                "mae": round(pt_metrics["mae"], 3),
                "rmse": round(pt_metrics["rmse"], 3),
                "delta1": round(pt_metrics["delta1"], 4),
                "delta2": round(pt_metrics["delta2"], 4),
                "delta3": round(pt_metrics["delta3"], 4),
                "coverage_pooled": None,
                "coverage_macro": None,
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
                "mae": round(pt_f["mae"], 3),
                "rmse": round(pt_f["rmse"], 3),
                "delta1": round(pt_f["delta1"], 4),
                "delta2": round(pt_f["delta2"], 4),
                "delta3": round(pt_f["delta3"], 4),
                "coverage_pooled": round(int_metrics["pooled_coverage"], 4),
                "coverage_macro": round(int_metrics["macro_coverage"], 4),
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

    # 4.1. Comparison (d) Fused Cues vs (f) Residual Model within each detector
    for m in DETECTORS:
        df_m = dfs[m]
        # Keep valid rows for both (d) and (f)
        valid_both = valid_prediction_mask(df_m["z_d"].to_numpy()) & valid_prediction_mask(df_m["z_hat_f"].to_numpy())
        sub_df = df_m[valid_both].copy()

        boot_res = paired_cluster_bootstrap(
            df=sub_df,
            pred_col_a="z_d",
            pred_col_b="z_hat_f",
            metric="absrel",
            seed=42,
            n_boot=1000,
            alpha=0.05,
            gt_col="z_gt",
            cluster_col="drive",
        )
        boot_rows.append({
            "comparison": f"{m}: (d)_fused vs (f)_residual",
            "metric": "diff_absrel (f - d)",
            "estimate": round(boot_res.estimate, 4),
            "ci_low": round(boot_res.ci_low, 4),
            "ci_high": round(boot_res.ci_high, 4),
            "excludes_zero": bool(boot_res.excludes_zero),
            "n_pairs": boot_res.n_rows,
            "k_clusters": boot_res.n_clusters,
            "note": "Coarse CI (10 clusters) per Decision D68",
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
            "metric": f"diff_absrel ({d2} - {d1})",
            "estimate": round(boot_res.estimate, 4),
            "ci_low": round(boot_res.ci_low, 4),
            "ci_high": round(boot_res.ci_high, 4),
            "excludes_zero": bool(boot_res.excludes_zero),
            "n_pairs": boot_res.n_rows,
            "k_clusters": boot_res.n_clusters,
            "note": "Coarse CI (10 clusters) per Decision D68",
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
    print("\n--- Generating Table 5: RQ2 Correlation Analysis ---")
    rq2_rows = []
    for m in DETECTORS:
        df = dfs[m]
        z_gt = df["z_gt"].to_numpy(dtype=float)
        z_pred = df["z_hat_f"].to_numpy(dtype=float)
        absrel = np.abs(z_pred - z_gt) / z_gt

        iou = df["matched_iou"].to_numpy(dtype=float)
        conf = df["confidence"].to_numpy(dtype=float)
        bbox_h = df["bbox_y2"].to_numpy(dtype=float) - df["bbox_y1"].to_numpy(dtype=float)
        bbox_w = df["bbox_x2"].to_numpy(dtype=float) - df["bbox_x1"].to_numpy(dtype=float)
        bbox_area = bbox_h * bbox_w

        # Metrics for correlation
        features = {
            "matched_iou": iou,
            "confidence": conf,
            "bbox_height": bbox_h,
            "bbox_width": bbox_w,
            "bbox_area": bbox_area,
        }

        for feat_name, feat_arr in features.items():
            valid_corr = np.isfinite(absrel) & np.isfinite(feat_arr)
            r_pearson, p_pearson = stats.pearsonr(feat_arr[valid_corr], absrel[valid_corr])
            r_spearman, p_spearman = stats.spearmanr(feat_arr[valid_corr], absrel[valid_corr])

            rq2_rows.append({
                "detector": m,
                "detection_feature": feat_name,
                "pearson_r": round(float(r_pearson), 4),
                "pearson_p": float(p_pearson),
                "spearman_rho": round(float(r_spearman), 4),
                "spearman_p": float(p_spearman),
                "n": int(np.sum(valid_corr)),
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
            drive_rows.append({
                "detector": m,
                "drive": d,
                "n_tp": int(np.sum(mask)),
                "absrel": round(float(np.mean(diff_d / z_gt[mask])), 4),
                "mae": round(float(np.mean(diff_d)), 3),
                "coverage_cqr": round(float(np.mean(cqr_cov[mask])), 4),
                "mean_width_cqr": round(float(np.mean(cqr_w[mask])), 4),
            })

    drive_df = pd.DataFrame(drive_rows)
    drive_df_path = tables_dir / "final_eval_per_drive_T.csv"
    drive_df.to_csv(drive_df_path, index=False)
    print(f"Saved: {drive_df_path}")

    print("\n==================================================================")
    print("ALL TASK T13 ACCEPTANCE EVALUATION TABLES GENERATED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    main()
