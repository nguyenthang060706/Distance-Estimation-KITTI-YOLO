"""
scripts/run_coverage_stability.py: 20-Resplit Stability & Conditional Coverage for Conformal Variants (T08).

Decisions D24, D26, D46, D47, D48, D49, D50, D51:
1. Evaluates three conformal methods on the same models:
   - Split Conformal (|r - r_hat|)
   - Standard CQR (Romano et al. 2019)
   - Mondrian CQR (adaptive Q_hat per Ẑ bin)
2. 20 resplits of B∪C by drive (seed 0–19, fit ≈ 50%, calib ≈ 25%, eval ≈ 25%).
   Reports mean ± std AND all 20 individual values for each method and detector.
3. Evaluates conditional coverage across pre-registered subgroups:
   - Ẑ prospective bins ([0,10), [10,20), [20,30), [30,inf))
   - Z_gt retrospective bands (diagnostic only)
   - Truncation, Occlusion, Border touch, Viewing angle θ (D19), Difficulty, Fallback flag (D51).
4. Outputs:
   - results/tables/coverage_stability_20resplits.{json,md}
   - results/tables/coverage_conditional_dev.{json,md}
   - logs run to runs/pipeline_log.jsonl
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import yaml

import numpy as np
import pandas as pd
import xgboost as xgb

# Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import append_jsonl, make_log_record
from src.geometry.fusion import fit_fusion_weights, fuse_depths_vectorised
from src.residual.feature_extractor import extract_inference_features
from src.residual.models import (
    build_feature_matrices,
    fit_e,
    predict_e,
    predict_f,
)
from src.uncertainty.cqr import (
    compute_nonconformity_scores,
    compute_split_conformal_scores,
    conformalize,
    conformalize_mondrian,
    fit_quantile_xgb,
    predict_interval,
    predict_interval_mondrian,
    predict_split_interval,
    winkler_score,
    MondrianBinning,
)
from src.uncertainty.resplit import generate_resplit, get_bc_car_hard_counts

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]


def get_git_info() -> tuple[str, bool]:
    """Retrieve git HEAD commit SHA and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
        return commit, len(status) > 0
    except Exception:
        return "unknown", False


def fit_model_f_fixed(
    X_mat: np.ndarray,
    y_arr: np.ndarray,
    best_params: dict[str, Any],
    random_state: int = 42,
    n_jobs: int = 1,
) -> xgb.XGBRegressor:
    """Fit Model (f) with pre-registered fixed hyperparameters from manifest."""
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    params.update(best_params)
    model = xgb.XGBRegressor(**params)
    model.fit(X_mat, y_arr)
    return model


def evaluate_intervals_on_eval(
    z_gt: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    r_actual: np.ndarray,
    r_lo: np.ndarray,
    r_hi: np.ndarray,
    drives: np.ndarray,
    n_crossings: int,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics on eval set."""
    covered = (z_gt >= z_lo) & (z_gt <= z_hi)
    pooled_cov = float(np.mean(covered))

    # Macro coverage across unique eval drives
    unique_d = np.unique(drives)
    drive_covs = [float(np.mean(covered[drives == d])) for d in unique_d if np.sum(drives == d) > 0]
    macro_cov = float(np.mean(drive_covs)) if drive_covs else float("nan")

    # Width ratio Z_hi / Z_lo
    valid_bounds = (z_lo > 0) & (z_hi < np.inf) & (~np.isnan(z_lo)) & (~np.isnan(z_hi))
    if np.any(valid_bounds):
        widths = z_hi[valid_bounds] / z_lo[valid_bounds]
        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))
    else:
        mean_width = float("nan")
        median_width = float("nan")

    # Winkler score in log-residual space
    w_scores = winkler_score(r_lo, r_hi, r_actual, alpha=alpha)
    mean_winkler = float(np.mean(w_scores))
    median_winkler = float(np.median(w_scores))

    return {
        "pooled_coverage": pooled_cov,
        "macro_coverage": macro_cov,
        "mean_width_ratio": mean_width,
        "median_width_ratio": median_width,
        "mean_winkler": mean_winkler,
        "median_winkler": median_winkler,
        "n_crossings": int(n_crossings),
        "covered_mask": covered,
    }


def compute_subgroup_coverage(
    covered_mask: np.ndarray,
    z_gt: np.ndarray,
    z_hat: np.ndarray,
    eval_df: pd.DataFrame,
    features_df: pd.DataFrame,
    fallback_flags: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Compute conditional coverage across pre-registered subgroups."""
    subgroups: dict[str, dict[str, Any]] = {}

    # 1. Ẑ prospective bins: [0,10), [10,20), [20,30), [30,inf)
    z_hat_groups = {
        "0-10m": (z_hat >= 0.0) & (z_hat < 10.0),
        "10-20m": (z_hat >= 10.0) & (z_hat < 20.0),
        "20-30m": (z_hat >= 20.0) & (z_hat < 30.0),
        ">=30m": (z_hat >= 30.0),
    }
    subgroups["z_hat_bin"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in z_hat_groups.items()
    }

    # 2. Z_gt retrospective bands: 0-10m, 10-20m, 20-30m, 30-50m, >50m
    z_gt_groups = {
        "0-10m": (z_gt >= 0.0) & (z_gt < 10.0),
        "10-20m": (z_gt >= 10.0) & (z_gt < 20.0),
        "20-30m": (z_gt >= 20.0) & (z_gt < 30.0),
        "30-50m": (z_gt >= 30.0) & (z_gt < 50.0),
        ">50m": (z_gt >= 50.0),
    }
    subgroups["z_gt_bin"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in z_gt_groups.items()
    }

    # 3. Truncated
    trunc = eval_df["truncated"].to_numpy(dtype=float)
    trunc_groups = {
        "non_truncated": trunc == 0.0,
        "truncated": trunc > 0.0,
    }
    subgroups["truncated"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in trunc_groups.items()
    }

    # 4. Occluded: 0, 1, 2
    occ = eval_df["occluded"].to_numpy(dtype=int)
    occ_groups = {
        "occ_0": occ == 0,
        "occ_1": occ == 1,
        "occ_2": occ == 2,
    }
    subgroups["occluded"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in occ_groups.items()
    }

    # 5. Touch edge (touch_left | touch_right | touch_top | touch_bottom)
    t_left = features_df["touch_left"].to_numpy(dtype=bool)
    t_right = features_df["touch_right"].to_numpy(dtype=bool)
    t_top = features_df["touch_top"].to_numpy(dtype=bool)
    t_bot = features_df["touch_bottom"].to_numpy(dtype=bool)
    touch_any = t_left | t_right | t_top | t_bot
    touch_groups = {
        "no_edge_touch": ~touch_any,
        "touches_edge": touch_any,
    }
    subgroups["touch_edge"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in touch_groups.items()
    }

    # 6. Viewing angle theta = min(|alpha|, pi - |alpha|) (Decision D19)
    alpha = eval_df["alpha"].to_numpy(dtype=float)
    theta = np.minimum(np.abs(alpha), np.pi - np.abs(alpha))
    theta_groups = {
        "side_0_30deg": theta < (np.pi / 6.0),
        "diagonal_30_60deg": (theta >= (np.pi / 6.0)) & (theta < (np.pi / 3.0)),
        "front_rear_60_90deg": theta >= (np.pi / 3.0),
    }
    subgroups["theta_bin"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in theta_groups.items()
    }

    # 7. Difficulty: Easy, Moderate, Hard
    diff = eval_df["difficulty"].to_numpy(dtype=str)
    diff_groups = {
        "Easy": diff == "Easy",
        "Moderate": diff == "Moderate",
        "Hard": diff == "Hard",
    }
    subgroups["difficulty"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in diff_groups.items()
    }

    # 8. Fallback flag (pattern 000) (Decision D51)
    fb = np.asarray(fallback_flags, dtype=bool)
    fb_groups = {
        "normal_cues": ~fb,
        "fallback_pattern_000": fb,
    }
    subgroups["fallback_flag"] = {
        k: {
            "n": int(np.sum(m)),
            "coverage": float(np.mean(covered_mask[m])) if np.sum(m) > 0 else None,
        }
        for k, m in fb_groups.items()
    }

    return subgroups


def run_resplits_for_detector(
    model_key: str,
    seeds: list[int],
    alpha: float = 0.1,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Execute 20 resplits across split conformal, CQR, and Mondrian CQR for one detector."""
    print(f"\n==================================================================")
    print(f"Running 20-Resplit Evaluation for {model_key} (seeds 0..{len(seeds)-1})")
    print(f"==================================================================")

    data_dir = PROJECT_ROOT / "results" / "datasets"
    runs_dir = PROJECT_ROOT / "runs" / "residual"

    # Load B and C parquets and pool
    df_feat_b = pd.read_parquet(data_dir / f"{model_key}_B_features.parquet")
    df_feat_c = pd.read_parquet(data_dir / f"{model_key}_C_features.parquet")
    df_cues_b = pd.read_parquet(data_dir / f"{model_key}_B_cues.parquet")
    df_cues_c = pd.read_parquet(data_dir / f"{model_key}_C_cues.parquet")
    df_eval_b = pd.read_parquet(data_dir / f"{model_key}_B_eval.parquet")
    df_eval_c = pd.read_parquet(data_dir / f"{model_key}_C_eval.parquet")

    features_bc = pd.concat([df_feat_b, df_feat_c], ignore_index=True)
    cues_bc = pd.concat([df_cues_b, df_cues_c], ignore_index=True)
    eval_bc = pd.concat([df_eval_b, df_eval_c], ignore_index=True)

    n_total = len(features_bc)
    assert len(cues_bc) == n_total and len(eval_bc) == n_total

    # Load best_params_f from manifest
    man_path = runs_dir / model_key / "manifest.json"
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    best_params_f = manifest.get("best_params_f", {"n_estimators": 200, "max_depth": 3, "min_child_weight": 10})

    # Ground truth car counts across 22 drives
    car_counts = get_bc_car_hard_counts()

    # Pre-extract base inference features
    combined = features_bc.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined[col] = cues_bc[col]
    base_feats = extract_inference_features(combined)

    z_cues_all = cues_bc[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask_all = cues_bc[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)
    z_gt_all = eval_bc["z_gt"].to_numpy(dtype=float)
    drive_all = eval_bc["drive"].to_numpy(dtype=str)
    pattern_000_all = (~cues_bc["valid_w"]) & (~cues_bc["valid_h"]) & (~cues_bc["valid_g"])

    per_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        resplit = generate_resplit(seed, car_counts=car_counts)
        mask_fit = eval_bc["drive"].isin(resplit["fit_drives"]).to_numpy()
        mask_calib = eval_bc["drive"].isin(resplit["calib_drives"]).to_numpy()
        mask_eval = eval_bc["drive"].isin(resplit["eval_drives"]).to_numpy()

        n_fit = int(np.sum(mask_fit))
        n_calib = int(np.sum(mask_calib))
        n_eval = int(np.sum(mask_eval))

        # 1. Fit geometric fusion on fit partition
        fw_fit = fit_fusion_weights(
            Z_cues=z_cues_all[mask_fit],
            Z_gt=z_gt_all[mask_fit],
            valid_mask=valid_mask_all[mask_fit],
            drive_ids=drive_all[mask_fit],
        )
        z_d = fuse_depths_vectorised(z_cues_all, valid_mask_all, fw_fit)

        # 2. Fit Model (e) on fit partition and predict fallback Z_base
        model_e = fit_e(base_feats[mask_fit], np.log(z_gt_all[mask_fit]), random_state=42, n_jobs=n_jobs)
        z_hat_e, _ = predict_e(model_e, base_feats)
        z_base = np.where(~pattern_000_all, z_d, z_hat_e)

        r = np.log(z_gt_all) - np.log(z_base)

        # 3. Build feature matrices with derived ln_z_base
        feat_mats = build_feature_matrices(base_feats, z_base=z_base)
        X_f = feat_mats["f"]

        # 4. Fit Model (f) and Quantile models on fit partition
        m_f = fit_model_f_fixed(X_f[mask_fit].to_numpy(dtype=float), r[mask_fit], best_params_f, random_state=42, n_jobs=n_jobs)
        m_q05 = fit_quantile_xgb(X_f[mask_fit], r[mask_fit], q=0.05, best_params_f=best_params_f, random_state=42, n_jobs=n_jobs)
        m_q95 = fit_quantile_xgb(X_f[mask_fit], r[mask_fit], q=0.95, best_params_f=best_params_f, random_state=42, n_jobs=n_jobs)

        # 5. Predict on all samples
        z_hat_f, r_hat_f = predict_f(m_f, X_f, z_base=z_base)
        q_lo = m_q05.predict(X_f.to_numpy(dtype=float))
        q_hi = m_q95.predict(X_f.to_numpy(dtype=float))

        # 6. Conformal Calibration on calib partition
        # Method 1: Split Conformal
        sc_scores = compute_split_conformal_scores(r_hat_f[mask_calib], r[mask_calib])
        q_hat_sc = conformalize(sc_scores, alpha=alpha)

        # Method 2: Standard CQR
        cqr_scores = compute_nonconformity_scores(q_lo[mask_calib], q_hi[mask_calib], r[mask_calib])
        q_hat_cqr = conformalize(cqr_scores, alpha=alpha)

        # Method 3: Mondrian CQR
        binning = MondrianBinning(z_hat_f[mask_calib], base_edges=[0.0, 10.0, 20.0, 30.0], min_samples=50)
        calib_bins = binning.assign_bins(z_hat_f[mask_calib])
        q_hat_mondrian = conformalize_mondrian(cqr_scores, calib_bins, n_bins=binning.n_bins, alpha=alpha)

        # 7. Evaluation on eval partition
        z_gt_eval = z_gt_all[mask_eval]
        z_base_eval = z_base[mask_eval]
        r_actual_eval = r[mask_eval]
        drives_eval = drive_all[mask_eval]

        # Method 1 (Split)
        z_lo_sc, z_hi_sc, r_lo_sc, r_hi_sc, n_cross_sc = predict_split_interval(
            z_base_eval, r_hat_f[mask_eval], q_hat_sc
        )
        eval_sc = evaluate_intervals_on_eval(
            z_gt_eval, z_lo_sc, z_hi_sc, r_actual_eval, r_lo_sc, r_hi_sc, drives_eval, n_cross_sc, alpha=alpha
        )
        cov_sc_subgroups = compute_subgroup_coverage(
            eval_sc["covered_mask"], z_gt_eval, z_hat_f[mask_eval], eval_bc[mask_eval], base_feats[mask_eval], pattern_000_all[mask_eval]
        )

        # Method 2 (CQR)
        z_lo_cqr, z_hi_cqr, r_lo_cqr, r_hi_cqr, n_cross_cqr = predict_interval(
            z_base_eval, q_lo[mask_eval], q_hi[mask_eval], q_hat_cqr
        )
        eval_cqr = evaluate_intervals_on_eval(
            z_gt_eval, z_lo_cqr, z_hi_cqr, r_actual_eval, r_lo_cqr, r_hi_cqr, drives_eval, n_cross_cqr, alpha=alpha
        )
        cov_cqr_subgroups = compute_subgroup_coverage(
            eval_cqr["covered_mask"], z_gt_eval, z_hat_f[mask_eval], eval_bc[mask_eval], base_feats[mask_eval], pattern_000_all[mask_eval]
        )

        # Method 3 (Mondrian)
        eval_bins = binning.assign_bins(z_hat_f[mask_eval])
        z_lo_m, z_hi_m, r_lo_m, r_hi_m, n_cross_m = predict_interval_mondrian(
            z_base_eval, q_lo[mask_eval], q_hi[mask_eval], q_hat_mondrian, eval_bins
        )
        eval_m = evaluate_intervals_on_eval(
            z_gt_eval, z_lo_m, z_hi_m, r_actual_eval, r_lo_m, r_hi_m, drives_eval, n_cross_m, alpha=alpha
        )
        cov_m_subgroups = compute_subgroup_coverage(
            eval_m["covered_mask"], z_gt_eval, z_hat_f[mask_eval], eval_bc[mask_eval], base_feats[mask_eval], pattern_000_all[mask_eval]
        )

        # Strip covered_mask from serialized result
        del eval_sc["covered_mask"]
        del eval_cqr["covered_mask"]
        del eval_m["covered_mask"]

        seed_entry = {
            "seed": seed,
            "n_redraws": resplit["n_redraws"],
            "n_fit": n_fit,
            "n_calib": n_calib,
            "n_eval": n_eval,
            "n_drives": resplit["n_drives"],
            "top1_shares": resplit["top1_shares"],
            "split_conformal": {
                "q_hat": float(q_hat_sc),
                "metrics": eval_sc,
                "subgroups": cov_sc_subgroups,
            },
            "cqr": {
                "q_hat": float(q_hat_cqr),
                "metrics": eval_cqr,
                "subgroups": cov_cqr_subgroups,
            },
            "mondrian_cqr": {
                "n_bins": binning.n_bins,
                "bin_labels": binning.labels,
                "q_hat_per_bin": {int(k): float(v) for k, v in q_hat_mondrian.items()},
                "metrics": eval_m,
                "subgroups": cov_m_subgroups,
            },
        }
        per_seed_results.append(seed_entry)

        print(
            f"  Seed {seed:2d} (n_eval={n_eval:4d}) | "
            f"Split Cov: {eval_sc['pooled_coverage']:.4f} | "
            f"CQR Cov: {eval_cqr['pooled_coverage']:.4f} | "
            f"Mondrian Cov: {eval_m['pooled_coverage']:.4f} | "
            f"Q_cqr: {q_hat_cqr:+.4f}"
        )

    # Compute summary statistics across the 20 seeds
    methods = ["split_conformal", "cqr", "mondrian_cqr"]
    summary: dict[str, Any] = {}

    for meth in methods:
        covs = [r[meth]["metrics"]["pooled_coverage"] for r in per_seed_results]
        macro_covs = [r[meth]["metrics"]["macro_coverage"] for r in per_seed_results]
        widths = [r[meth]["metrics"]["mean_width_ratio"] for r in per_seed_results]
        winklers = [r[meth]["metrics"]["mean_winkler"] for r in per_seed_results]
        crossings = [r[meth]["metrics"]["n_crossings"] for r in per_seed_results]

        summary[meth] = {
            "mean_pooled_coverage": float(np.mean(covs)),
            "std_pooled_coverage": float(np.std(covs)),
            "min_pooled_coverage": float(np.min(covs)),
            "max_pooled_coverage": float(np.max(covs)),
            "mean_macro_coverage": float(np.mean(macro_covs)),
            "std_macro_coverage": float(np.std(macro_covs)),
            "mean_width_ratio": float(np.mean(widths)),
            "std_width_ratio": float(np.std(widths)),
            "mean_winkler": float(np.mean(winklers)),
            "std_winkler": float(np.std(winklers)),
            "total_crossings": int(np.sum(crossings)),
            "all_20_coverages": covs,
        }

    return {
        "model_key": model_key,
        "n_seeds": len(seeds),
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "summary": summary,
        "per_seed": per_seed_results,
    }


def generate_stability_md(results: list[dict[str, Any]]) -> str:
    """Generate Markdown report for 20 resplits stability (Decision D26)."""
    lines = [
        "# Đánh giá Độ ổn định Conformal qua 20 Lần chia lại B∪C (T08)",
        "",
        "> Đánh giá out-of-sample trên 20 phân hoạch ngẫu nhiên (seed 0–19) của 22 cụm drive có Car Hard.",
        "> Tỷ lệ phân hoạch: fit ≈ 50%, calib ≈ 25%, eval ≈ 25% (mỗi phần >= 4 cụm, top1_share <= 0.50).",
        "> Tuân thủ Decisions D24, D26, D46, D47, D48, D50, D51.",
        "",
        "## 1. Bảng tổng hợp Mean ± Std qua 20 Resplits",
        "",
        "| Detector | Phương án | Mean Cov ± Std | [Min, Max] Cov | Macro Cov ± Std | Mean Width (Z_hi/Z_lo) | Mean Winkler | Total Crossings |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for res in results:
        m = res["model_key"]
        for meth_name, meth_label in [
            ("split_conformal", "Split Conformal"),
            ("cqr", "Standard CQR"),
            ("mondrian_cqr", "Mondrian CQR"),
        ]:
            s = res["summary"][meth_name]
            lines.append(
                f"| `{m}` | {meth_label} | **{s['mean_pooled_coverage']*100:.2f}% ± {s['std_pooled_coverage']*100:.2f}%** | "
                f"[{s['min_pooled_coverage']*100:.2f}%, {s['max_pooled_coverage']*100:.2f}%] | "
                f"{s['mean_macro_coverage']*100:.2f}% ± {s['std_macro_coverage']*100:.2f}% | "
                f"{s['mean_width_ratio']:.3f} | {s['mean_winkler']:.4f} | {s['total_crossings']} |"
            )

    lines.extend([
        "",
        "## 2. Chi tiết 20 Giá trị Coverage cho từng Seed (Yêu cầu D26)",
        "",
        "> Ghi chú: Coverage trên chính tập calib là tầm thường (trivial 90%) do định nghĩa hiệu chỉnh,",
        "> vì vậy đánh giá trên tập eval độc lập là bắt buộc để quan sát dao động ngoài mẫu.",
        "",
    ])

    for res in results:
        m = res["model_key"]
        lines.extend([
            f"### Detector `{m}`",
            "",
            "| Seed | n_redraws | n_eval | Split Conformal | Standard CQR | Mondrian CQR | Q̂ (CQR) |",
            "|---|---|---|---|---|---|---|",
        ])
        for s in res["per_seed"]:
            lines.append(
                f"| {s['seed']:2d} | {s['n_redraws']:2d} | {s['n_eval']:4d} | "
                f"{s['split_conformal']['metrics']['pooled_coverage']*100:.2f}% | "
                f"{s['cqr']['metrics']['pooled_coverage']*100:.2f}% | "
                f"{s['mondrian_cqr']['metrics']['pooled_coverage']*100:.2f}% | "
                f"{s['cqr']['q_hat']:+.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_conditional_md(results: list[dict[str, Any]]) -> str:
    """Generate Markdown report for conditional coverage across subgroups (Decision D26)."""
    lines = [
        "# Đánh giá Độ phủ có điều kiện (Conditional Coverage) — T08",
        "",
        "> Trung bình độ phủ có điều kiện qua 20 lần chia lại eval out-of-sample.",
        "> Các nhóm: Ẑ dự đoán, Z thật (chẩn đoán), Truncated, Occluded, Chạm biên, Góc θ (D19), Difficulty, Fallback (D51).",
        "",
    ]

    subgroup_categories = [
        ("z_hat_bin", "Dải cự ly theo Ẑ dự đoán (Prospective Bins)"),
        ("z_gt_bin", "Dải cự ly theo Z thật (Retrospective Bins — Chẩn đoán)"),
        ("truncated", "Mức độ cắt biên (Truncated)"),
        ("occluded", "Mức độ che khuất (Occluded)"),
        ("touch_edge", "Chạm biên ảnh (Touch Edge)"),
        ("theta_bin", "Góc hướng quan sát θ (Decision D19)"),
        ("difficulty", "Mức độ khó KITTI (Difficulty)"),
        ("fallback_flag", "Nhóm Fallback Pattern 000 (Decision D51)"),
    ]

    for res in results:
        m = res["model_key"]
        lines.extend([
            f"## Detector `{m}`",
            "",
        ])

        for cat_key, cat_title in subgroup_categories:
            lines.extend([
                f"### {cat_title}",
                "",
                "| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |",
                "|---|---|---|---|---|",
            ])

            # Get group keys from seed 0
            group_keys = list(res["per_seed"][0]["cqr"]["subgroups"][cat_key].keys())
            for gk in group_keys:
                # Average n and coverage across the 20 seeds
                n_vals = [s["cqr"]["subgroups"][cat_key][gk]["n"] for s in res["per_seed"]]
                sc_covs = [s["split_conformal"]["subgroups"][cat_key][gk]["coverage"] for s in res["per_seed"] if s["split_conformal"]["subgroups"][cat_key][gk]["coverage"] is not None]
                cqr_covs = [s["cqr"]["subgroups"][cat_key][gk]["coverage"] for s in res["per_seed"] if s["cqr"]["subgroups"][cat_key][gk]["coverage"] is not None]
                m_covs = [s["mondrian_cqr"]["subgroups"][cat_key][gk]["coverage"] for s in res["per_seed"] if s["mondrian_cqr"]["subgroups"][cat_key][gk]["coverage"] is not None]

                mean_n = float(np.mean(n_vals))
                str_sc = f"{np.mean(sc_covs)*100:.2f}%" if sc_covs else "N/A"
                str_cqr = f"{np.mean(cqr_covs)*100:.2f}%" if cqr_covs else "N/A"
                str_m = f"{np.mean(m_covs)*100:.2f}%" if m_covs else "N/A"

                lines.append(f"| `{gk}` | {mean_n:.1f} | {str_sc} | {str_cqr} | {str_m} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run 20-resplit coverage stability and conditional evaluation (T08).")
    parser.add_argument("--config", type=str, default="configs/residual/coverage_prereg_v1.yaml")
    parser.add_argument("--model", type=str, default="all", choices=["all", "yolo11s_640", "yolov8s_640", "yolov5su_640"])
    parser.add_argument("--seeds", type=str, default="0-19", help="Seed range or comma-separated list (e.g. 0-19)")
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    # Parse seeds
    if "-" in args.seeds:
        start_s, end_s = map(int, args.seeds.split("-"))
        seeds = list(range(start_s, end_s + 1))
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    detectors = DETECTORS if args.model == "all" else [args.model]

    # Load prereg config
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    alpha = float(cfg.get("conformal_spec", {}).get("alpha", 0.1))

    git_commit, is_dirty = get_git_info()
    print(f"Git HEAD: {git_commit} (dirty: {is_dirty})")
    print(f"Detectors to evaluate: {detectors}")
    print(f"Seeds: {len(seeds)} seeds ({seeds[0]}..{seeds[-1]})")

    all_results: list[dict[str, Any]] = []
    for model_key in detectors:
        res = run_resplits_for_detector(model_key=model_key, seeds=seeds, alpha=alpha, n_jobs=args.n_jobs)
        all_results.append(res)

    out_table_dir = PROJECT_ROOT / "results" / "tables"
    out_table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save stability JSON and MD
    json_path = out_table_dir / "coverage_stability_20resplits.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Stability table JSON: {json_path}")

    md_stability = generate_stability_md(all_results)
    md_stab_path = out_table_dir / "coverage_stability_20resplits.md"
    with open(md_stab_path, "w", encoding="utf-8") as f:
        f.write(md_stability)
    print(f"[Saved] Stability table Markdown: {md_stab_path}")

    # 2. Save conditional JSON and MD
    cond_path = out_table_dir / "coverage_conditional_dev.json"
    # Filter per-seed to just summary subgroups for compactness
    with open(cond_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Saved] Conditional table JSON: {cond_path}")

    md_conditional = generate_conditional_md(all_results)
    md_cond_path = out_table_dir / "coverage_conditional_dev.md"
    with open(md_cond_path, "w", encoding="utf-8") as f:
        f.write(md_conditional)
    print(f"[Saved] Conditional table Markdown: {md_cond_path}")

    # 3. Append to runs/pipeline_log.jsonl
    log_rec = make_log_record(
        split="BC_resplit",
        split_hash="",
        seed=42,
        n_boot=len(seeds),
        tag="T08-Coverage-Stability-20Resplits",
        extra={
            "detectors": detectors,
            "n_seeds": len(seeds),
            "alpha": alpha,
            "mean_coverages_cqr": {r["model_key"]: r["summary"]["cqr"]["mean_pooled_coverage"] for r in all_results},
            "mean_coverages_mondrian": {r["model_key"]: r["summary"]["mondrian_cqr"]["mean_pooled_coverage"] for r in all_results},
            "mean_coverages_split": {r["model_key"]: r["summary"]["split_conformal"]["mean_pooled_coverage"] for r in all_results},
            "git_commit": git_commit,
            "git_dirty": is_dirty,
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_rec)
    print(f"[Logged] Recorded run in runs/pipeline_log.jsonl")


if __name__ == "__main__":
    main()
