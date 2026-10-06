"""
scripts/run_cqr.py: End-to-end execution of Conformal Quantile Regression (T07).

Decisions D13, D24, D26, D32, D35, D36, D43, D46, D47, D48, D49, D50, D51:
- Fits XGBoost quantile regression models (q=0.05, 0.95) on Split B using fixed best_params_f from manifest (D24).
- Target r = ln(Z_gt) - ln(Z_base_oof) built strictly on out-of-fold baseline from B_oof.parquet (D43).
- Uses shared frozen pipeline apply_frozen_pipeline for Split C inference (D49).
- Leave-One-Drive-Out (LODO) calibration across 10 Car Hard drives on Split C.
- Conformal quantile computed via exact finite-sample order statistic with float tolerance (D46).
- Interval [q_lo - Q_hat, q_hi + Q_hat] with sorted quantiles and explicit crossing reporting (D47).
- Winkler interval score computed in log-space r (D47).
- Evaluates Gate [85%, 95%], reports macro on drives with n >= 30, and analyzes fallback_flag (D48).
- Records coverage across standardized distance bands (D3, E1) by both Z_gt and Z_hat.
- Records raw quantile coverage (Q_hat = 0) as baseline.
- Documents drive-level heterogeneity (0057, 0004 under-covering) and fallback Z_e under-prediction (D50, D51).
- Serializes full Split C calibration parameter Q_hat into cqr_calib_C.json for T10/T11.
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

import numpy as np
import pandas as pd

# Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import append_jsonl, make_log_record
from src.pipeline.apply_frozen import apply_frozen_pipeline
from src.residual.feature_extractor import extract_inference_features
from src.residual.models import build_feature_matrices
from src.uncertainty.cqr import (
    assert_disjoint_drives,
    compute_nonconformity_scores,
    conformalize,
    fit_quantile_xgb,
    predict_interval,
    save_quantile_model,
    sort_quantiles,
    winkler_score,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]

BANDS = [
    ("0-10m", 0.0, 10.0),
    ("10-20m", 10.0, 20.0),
    ("20-30m", 20.0, 30.0),
    ("30-50m", 30.0, 50.0),
    (">50m", 50.0, float("inf")),
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", False


def get_split_hashes() -> tuple[str, str]:
    p = PROJECT_ROOT / "splits" / "split_metadata.json"
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta["splits"]["B"]["hash"], meta["splits"]["C"]["hash"]
    return "", ""


def run_cqr_for_detector(
    model_key: str,
    alpha: float = 0.1,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, Any]:
    print(f"\n================================================================================")
    print(f"  RUNNING CQR FOR DETECTOR: {model_key} (alpha={alpha}, nominal={1-alpha:.1%})")
    print(f"================================================================================")

    data_dir = PROJECT_ROOT / "results" / "datasets"
    model_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    best_params_f = manifest_data["best_params_f"]
    print(f"Loaded best_params_f from manifest: {best_params_f}")

    # Step 1: Load Split B OOF and features (Decision D43)
    oof_b_path = data_dir / f"{model_key}_B_oof.parquet"
    feat_b_path = data_dir / f"{model_key}_B_features.parquet"
    eval_b_path = data_dir / f"{model_key}_B_eval.parquet"
    cues_b_path = data_dir / f"{model_key}_B_cues.parquet"

    oof_b = pd.read_parquet(oof_b_path)
    feat_b = pd.read_parquet(feat_b_path)
    eval_b = pd.read_parquet(eval_b_path)
    cues_b = pd.read_parquet(cues_b_path)

    z_base_oof = oof_b["z_base"].to_numpy(dtype=float)
    z_gt_b = eval_b["z_gt"].to_numpy(dtype=float)
    r_b = np.log(z_gt_b) - np.log(z_base_oof)

    # Base features for B
    combined_b = feat_b.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined_b[col] = cues_b[col]
    base_feats_b = extract_inference_features(combined_b)
    feat_mats_b = build_feature_matrices(base_feats_b, z_base=z_base_oof)
    X_b_f = feat_mats_b["f"].to_numpy(dtype=float)

    # Step 2: Fit Quantile Models on full Split B (Decision D24: NO re-gridding)
    seed_q05 = seed
    seed_q95 = seed + 1
    print(f">>> Fitting XGBoost quantile q=0.05 on Split B (seed={seed_q05}, {len(X_b_f)} samples)...")
    model_q05 = fit_quantile_xgb(
        X=X_b_f,
        r=r_b,
        q=0.05,
        best_params_f=best_params_f,
        random_state=seed_q05,
        n_jobs=n_jobs,
    )

    print(f">>> Fitting XGBoost quantile q=0.95 on Split B (seed={seed_q95}, {len(X_b_f)} samples)...")
    model_q95 = fit_quantile_xgb(
        X=X_b_f,
        r=r_b,
        q=0.95,
        best_params_f=best_params_f,
        random_state=seed_q95,
        n_jobs=n_jobs,
    )

    q05_path = model_dir / "model_q05.json"
    q95_path = model_dir / "model_q95.json"
    save_quantile_model(model_q05, q05_path)
    save_quantile_model(model_q95, q95_path)
    sha_q05 = sha256_file(q05_path)
    sha_q95 = sha256_file(q95_path)
    print(f"[Saved] model_q05: {q05_path.name} (SHA: {sha_q05[:16]}...)")
    print(f"[Saved] model_q95: {q95_path.name} (SHA: {sha_q95[:16]}...)")

    # Step 3: Out-of-sample inference on Split C via apply_frozen_pipeline (Decision D49)
    print(f">>> Executing frozen pipeline for Split C...")
    c_data = apply_frozen_pipeline(model_key=model_key, split="C")
    z_base_c = c_data["z_base"]
    fallback_c = c_data["fallback_flag"]
    eval_c = c_data["eval_df"]
    z_gt_c = eval_c["z_gt"].to_numpy(dtype=float)
    r_actual_c = c_data["r_actual"]
    drives_c = eval_c["drive"].to_numpy(dtype=str)
    n_c = len(eval_c)

    X_c_f = c_data["feat_mats"]["f"].to_numpy(dtype=float)
    q05_raw_c = model_q05.predict(X_c_f)
    q95_raw_c = model_q95.predict(X_c_f)
    q05_c, q95_c = sort_quantiles(q05_raw_c, q95_raw_c)

    # Raw quantile baseline on C (Q_hat = 0)
    z_lo_raw, z_hi_raw, _, _, _ = predict_interval(z_base_c, q05_c, q95_c, Q_hat=0.0)
    covered_raw = (z_lo_raw <= z_gt_c) & (z_gt_c <= z_hi_raw)
    raw_coverage = float(np.mean(covered_raw))

    # Step 4: LODO Calibration on Split C (10 drives with Car Hard)
    unique_drives = np.unique(drives_c)
    print(f">>> Running LODO calibration on Split C across {len(unique_drives)} drives...")

    z_lo_lodo = np.zeros(n_c, dtype=float)
    z_hi_lodo = np.zeros(n_c, dtype=float)
    r_lo_lodo = np.zeros(n_c, dtype=float)
    r_hi_lodo = np.zeros(n_c, dtype=float)
    q_hat_per_drive: dict[str, float] = {}
    drive_stats: list[dict[str, Any]] = []

    for d in unique_drives:
        test_mask = drives_c == d
        calib_mask = ~test_mask

        # Guard
        assert_disjoint_drives(drives_c[calib_mask], drives_c[test_mask])

        # Calibration on C \ {d}
        scores_calib = compute_nonconformity_scores(
            q_lo=q05_c[calib_mask],
            q_hi=q95_c[calib_mask],
            r_actual=r_actual_c[calib_mask],
        )
        q_hat_d = conformalize(scores_calib, alpha=alpha)
        q_hat_per_drive[d] = q_hat_d

        # Test on held drive d
        z_lo_d, z_hi_d, r_lo_d, r_hi_d, n_cross_d = predict_interval(
            z_base=z_base_c[test_mask],
            q_lo=q05_c[test_mask],
            q_hi=q95_c[test_mask],
            Q_hat=q_hat_d,
        )

        z_lo_lodo[test_mask] = z_lo_d
        z_hi_lodo[test_mask] = z_hi_d
        r_lo_lodo[test_mask] = r_lo_d
        r_hi_lodo[test_mask] = r_hi_d

        n_d = int(np.sum(test_mask))
        covered_d = (z_lo_d <= z_gt_c[test_mask]) & (z_gt_c[test_mask] <= z_hi_d)
        cov_d = float(np.mean(covered_d))
        width_ratio_d = float(np.mean(z_hi_d / z_lo_d))
        winkler_d = float(np.mean(winkler_score(r_lo_d, r_hi_d, r_actual_c[test_mask], alpha=alpha)))

        drive_stats.append({
            "drive": str(d),
            "n": n_d,
            "q_hat": float(q_hat_d),
            "coverage": cov_d,
            "mean_width_ratio": width_ratio_d,
            "mean_winkler": winkler_d,
            "n_crossings": n_cross_d,
        })

    # Step 5: Metrics Aggregation
    covered_all = (z_lo_lodo <= z_gt_c) & (z_gt_c <= z_hi_lodo)
    pooled_coverage = float(np.mean(covered_all))
    macro_coverage_all = float(np.mean([ds["coverage"] for ds in drive_stats]))

    drives_ge_30 = [ds for ds in drive_stats if ds["n"] >= 30]
    n_samples_ge_30 = int(np.sum([ds["n"] for ds in drives_ge_30]))
    share_samples_ge_30 = n_samples_ge_30 / n_c
    macro_coverage_ge_30 = (
        float(np.mean([ds["coverage"] for ds in drives_ge_30])) if drives_ge_30 else float("nan")
    )

    width_ratios_all = z_hi_lodo / z_lo_lodo
    mean_width_ratio = float(np.mean(width_ratios_all))
    median_width_ratio = float(np.median(width_ratios_all))

    winkler_all = winkler_score(r_lo_lodo, r_hi_lodo, r_actual_c, alpha=alpha)
    mean_winkler = float(np.mean(winkler_all))
    median_winkler = float(np.median(winkler_all))

    total_crossings = int(np.sum(r_lo_lodo > r_hi_lodo))

    # Fallback diagnostics (pattern 000, D51)
    n_fallback = int(np.sum(fallback_c))
    fallback_cov = float(np.mean(covered_all[fallback_c])) if n_fallback > 0 else float("nan")
    normal_cov = float(np.mean(covered_all[~fallback_c])) if (n_c - n_fallback) > 0 else float("nan")
    mean_abs_r_fallback = float(np.mean(np.abs(r_actual_c[fallback_c]))) if n_fallback > 0 else float("nan")
    mean_abs_r_normal = float(np.mean(np.abs(r_actual_c[~fallback_c])))

    # Distance bands analysis (D3, E1)
    z_pred_mid = np.sqrt(z_lo_lodo * z_hi_lodo)
    band_results: list[dict[str, Any]] = []
    for b_name, b_min, b_max in BANDS:
        mask_gt = (z_gt_c >= b_min) & (z_gt_c < b_max)
        n_gt_b = int(np.sum(mask_gt))
        cov_gt_b = float(np.mean(covered_all[mask_gt])) if n_gt_b > 0 else float("nan")

        mask_pred = (z_pred_mid >= b_min) & (z_pred_mid < b_max)
        n_pred_b = int(np.sum(mask_pred))
        cov_pred_b = float(np.mean(covered_all[mask_pred])) if n_pred_b > 0 else float("nan")

        band_results.append({
            "band": b_name,
            "n_gt": n_gt_b,
            "coverage_by_gt": cov_gt_b,
            "n_pred": n_pred_b,
            "coverage_by_pred": cov_pred_b,
        })

    # Q_hat sign and residual shift analysis (B vs C shift)
    all_q_hats = [ds["q_hat"] for ds in drive_stats]
    mean_q_hat_lodo = float(np.mean(all_q_hats))
    median_q_hat_lodo = float(np.median(all_q_hats))

    mean_r_c = float(np.mean(r_actual_c))
    mean_r_b = float(np.mean(r_b))
    shift_r = mean_r_c - mean_r_b

    # Gate verification (D48: [85%, 95%])
    gate_passed = bool(0.85 <= pooled_coverage <= 0.95)

    print(f"\n--- Results Summary for {model_key} ---")
    print(f"  Raw Quantile Cov (Q_hat=0):{raw_coverage:.4f} (demonstrates under-coverage without calib)")
    print(f"  Pooled LODO Coverage:      {pooled_coverage:.4f} (Gate: [0.85, 0.95] -> {'PASS' if gate_passed else 'DEVIATION'})")
    print(f"  Macro LODO Coverage (all): {macro_coverage_all:.4f} (across 10 drives)")
    print(f"  Macro LODO Coverage (>=30):{macro_coverage_ge_30:.4f} (across {len(drives_ge_30)} drives, {share_samples_ge_30:.1%} samples)")
    print(f"  Width Ratio (Z_hi / Z_lo): mean={mean_width_ratio:.3f}, median={median_width_ratio:.3f}")
    print(f"  Winkler Score (log-space): mean={mean_winkler:.4f}, median={median_winkler:.4f}")
    print(f"  Quantile Crossings:        {total_crossings} / {n_c} ({total_crossings / n_c:.2%})")
    print(f"  Fallback (pattern 000):    n={n_fallback}, cov={fallback_cov:.4f} (mean |r|={mean_abs_r_fallback:.2f} vs normal={mean_abs_r_normal:.2f})")
    print(f"  Q_hat (LODO mean / med):   {mean_q_hat_lodo:+.5f} / {median_q_hat_lodo:+.5f}")
    print(f"  Mean r on C vs B (shift):  mean_r_C={mean_r_c:+.5f}, mean_r_B={mean_r_b:+.5f} (diff={shift_r:+.5f})")

    # Step 6: Full calibration on 100% Split C (for T10/T11 deployment)
    scores_full_c = compute_nonconformity_scores(q05_c, q95_c, r_actual_c)
    q_hat_full_c = conformalize(scores_full_c, alpha=alpha)
    scores_bytes = np.asarray(scores_full_c, dtype=np.float64).tobytes()
    scores_sha256 = hashlib.sha256(scores_bytes).hexdigest()

    calib_c_data = {
        "model_key": model_key,
        "split": "C",
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "q_hat_full_c": float(q_hat_full_c),
        "n_calib": len(scores_full_c),
        "scores_sha256": scores_sha256,
        "created_at": datetime.now().isoformat(),
    }
    calib_c_path = model_dir / "cqr_calib_C.json"
    with open(calib_c_path, "w", encoding="utf-8") as f:
        json.dump(calib_c_data, f, indent=2)
    sha_calib_c = sha256_file(calib_c_path)
    print(f"[Saved] Full Split C calib: {calib_c_path.name} (Q_hat={q_hat_full_c:.5f}, SHA: {sha_calib_c[:16]}...)")

    # Step 7: Update manifest.json
    manifest_data["updated_at"] = datetime.now().isoformat()
    manifest_data["models"]["model_q05"] = {"file": "model_q05.json", "sha256": sha_q05}
    manifest_data["models"]["model_q95"] = {"file": "model_q95.json", "sha256": sha_q95}
    manifest_data["models"]["cqr_calib_C"] = {"file": "cqr_calib_C.json", "sha256": sha_calib_c}
    manifest_data["cqr"] = {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "seed_q05": seed_q05,
        "seed_q95": seed_q95,
        "raw_coverage_qhat0": raw_coverage,
        "pooled_coverage_lodo": pooled_coverage,
        "macro_coverage_lodo_all": macro_coverage_all,
        "macro_coverage_lodo_ge_30": macro_coverage_ge_30,
        "share_samples_ge_30": share_samples_ge_30,
        "mean_width_ratio": mean_width_ratio,
        "median_width_ratio": median_width_ratio,
        "mean_winkler": mean_winkler,
        "total_crossings": total_crossings,
        "q_hat_full_c": float(q_hat_full_c),
        "gate_passed": gate_passed,
        "fallback_n": n_fallback,
        "fallback_mean_abs_r": mean_abs_r_fallback,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[Updated] Manifest file: {manifest_path}")

    # Return structured detector summary
    return {
        "model_key": model_key,
        "n_samples": n_c,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "raw_coverage": raw_coverage,
        "pooled_coverage": pooled_coverage,
        "macro_coverage_all": macro_coverage_all,
        "macro_coverage_ge_30": macro_coverage_ge_30,
        "share_samples_ge_30": share_samples_ge_30,
        "mean_width_ratio": mean_width_ratio,
        "median_width_ratio": median_width_ratio,
        "mean_winkler": mean_winkler,
        "median_winkler": median_winkler,
        "total_crossings": total_crossings,
        "crossing_rate": total_crossings / n_c,
        "n_fallback": n_fallback,
        "fallback_coverage": fallback_cov,
        "normal_coverage": normal_cov,
        "mean_abs_r_fallback": mean_abs_r_fallback,
        "mean_abs_r_normal": mean_abs_r_normal,
        "mean_q_hat_lodo": mean_q_hat_lodo,
        "median_q_hat_lodo": median_q_hat_lodo,
        "mean_r_c": mean_r_c,
        "mean_r_b": mean_r_b,
        "shift_r": shift_r,
        "q_hat_full_c": float(q_hat_full_c),
        "gate_passed": gate_passed,
        "drive_stats": drive_stats,
        "band_results": band_results,
        "sha_q05": sha_q05,
        "sha_q95": sha_q95,
        "sha_calib_c": sha_calib_c,
    }


def generate_results_table_md(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Kết quả Conformal Quantile Regression (CQR) trên Dev Split C (T07)",
        "",
        "> Đánh giá Leave-One-Drive-Out (LODO) trên 10 cụm drive có Car Hard của Split C.",
        "> Thống kê khoảng tin cậy cự ly danh nghĩa 90% (alpha = 0.1).",
        "> Tuân thủ Decisions D18, D26, D46, D47, D48, D49, D50, D51.",
        "",
        "## 1. Bảng tổng hợp các detector (LODO Calibration trên Split C)",
        "",
        "| Detector | N (C) | Raw Cov (Q̂=0) | Pooled Cov | Macro Cov (10 cụm) | Macro Cov (n >= 30) | Mean Width (Z_hi/Z_lo) | Mean Winkler | Crossings | Q_hat (full C) | Gate [85%, 95%] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        gate_str = "✅ PASS" if r["gate_passed"] else "⚠️ DEVIATION"
        macro_ge30_str = f"{r['macro_coverage_ge_30']:.4f}" if not np.isnan(r["macro_coverage_ge_30"]) else "N/A"
        lines.append(
            f"| `{r['model_key']}` | {r['n_samples']} | {r['raw_coverage']:.4f} | "
            f"**{r['pooled_coverage']:.4f}** | {r['macro_coverage_all']:.4f} | {macro_ge30_str} | "
            f"{r['mean_width_ratio']:.3f} | {r['mean_winkler']:.4f} | "
            f"{r['total_crossings']} ({r['crossing_rate']:.1%}) | "
            f"{r['q_hat_full_c']:+.5f} | {gate_str} |"
        )

    lines.extend([
        "",
        "## 2. Phân tích chẩn đoán độ dịch chuyển & Fallback (Decisions D50, D51)",
        "",
        "> Cảnh báo: Nhóm fallback_flag=True (pattern 000) có |r| trung bình ~2.56 (do Model e under-predict Z_e ≈ 0.5m so với Z_gt ≈ 6–8.5m),",
        "> nằm ngoài khoảng CQR ([r_lo, r_hi] ≈ [-0.2, +0.4]), dẫn tới độ phủ 0.000 ở cả 3 detector.",
        "",
        "| Detector | Mean r (B OOF) | Mean r (C) | Residual Shift (C - B) | Q_hat LODO (Mean) | Fallback n | Fallback Cov | Fallback Mean |r| | Normal Mean |r| |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for r in results:
        lines.append(
            f"| `{r['model_key']}` | {r['mean_r_b']:+.4f} | {r['mean_r_c']:+.4f} | "
            f"**{r['shift_r']:+.4f}** | {r['mean_q_hat_lodo']:+.5f} | "
            f"{r['n_fallback']} | **{r['fallback_coverage']:.4f}** | {r['mean_abs_r_fallback']:.2f} | {r['mean_abs_r_normal']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. Coverage theo dải cự ly Z (D3, E1)",
        "",
        "> Đánh giá độ phủ có điều kiện theo dải cự ly: theo Z thật (chẩn đoán) và theo Ẑ dự đoán (midpoint).",
        "",
    ])

    for r in results:
        lines.extend([
            f"### Detector `{r['model_key']}`",
            "",
            "| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | n (Ẑ dự đoán) | Coverage (theo Ẑ dự đoán) |",
            "|---|---|---|---|---|",
        ])
        for b in r["band_results"]:
            cov_gt_str = f"{b['coverage_by_gt']:.4f}" if not np.isnan(b["coverage_by_gt"]) else "N/A (n=0)"
            cov_pred_str = f"{b['coverage_by_pred']:.4f}" if not np.isnan(b["coverage_by_pred"]) else "N/A (n=0)"
            lines.append(
                f"| `{b['band']}` | {b['n_gt']} | {cov_gt_str} | {b['n_pred']} | {cov_pred_str} |"
            )
        lines.append("")

    lines.extend([
        "## 4. Chi tiết per-drive (10 cụm Car Hard trên Split C)",
        "",
        "> Lưu ý hiện tượng Drive Heterogeneity (D50): 8 drive over-cover (96–100%), trong khi 2 drive lớn",
        "> `0057` (n=250–260) và `0004` (n=309–325) chiếm ~39% mẫu bị under-cover (67–79%), kéo pooled coverage xuống 87.1–88.0%.",
        "",
    ])

    for r in results:
        lines.extend([
            f"### Detector `{r['model_key']}`",
            "",
            "| Drive ID | n | Q_hat (LODO) | Coverage | Mean Width (Z_hi/Z_lo) | Mean Winkler | Crossings |",
            "|---|---|---|---|---|---|---|",
        ])
        for ds in r["drive_stats"]:
            lines.append(
                f"| `{ds['drive']}` | {ds['n']} | {ds['q_hat']:+.5f} | "
                f"{ds['coverage']:.4f} | {ds['mean_width_ratio']:.3f} | {ds['mean_winkler']:.4f} | {ds['n_crossings']} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CQR core evaluation on Split C (T07)")
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DETECTORS,
        help="List of detectors to evaluate (default: all 3)",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker threads (default: 1)")
    args = parser.parse_args()

    git_commit, is_dirty = get_git_info()
    split_b_hash, split_c_hash = get_split_hashes()
    results = []

    for model_key in args.detectors:
        res = run_cqr_for_detector(
            model_key=model_key,
            alpha=args.alpha,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        results.append(res)

    # Save structured table JSON
    out_table_dir = PROJECT_ROOT / "results" / "tables"
    out_table_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_table_dir / "cqr_coverage_dev.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Structured table: {json_path}")

    # Save Markdown table
    md_content = generate_results_table_md(results)
    md_path = out_table_dir / "cqr_coverage_dev.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Markdown table: {md_path}")

    # Append to runs/pipeline_log.jsonl with real split hashes (AGENT_RULES §5)
    log_rec = make_log_record(
        split="C",
        split_hash=split_c_hash,
        seed=args.seed,
        n_boot=0,
        tag="T07-CQR-Dev-LODO",
        extra={
            "split_b_hash": split_b_hash,
            "detectors": args.detectors,
            "alpha": args.alpha,
            "pooled_coverages": {r["model_key"]: r["pooled_coverage"] for r in results},
            "gate_passed": {r["model_key"]: r["gate_passed"] for r in results},
            "raw_coverages": {r["model_key"]: r["raw_coverage"] for r in results},
            "git_commit": git_commit,
            "git_dirty": is_dirty,
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_rec)
    print(f"[Logged] Recorded run with split_hash in runs/pipeline_log.jsonl")


if __name__ == "__main__":
    main()
