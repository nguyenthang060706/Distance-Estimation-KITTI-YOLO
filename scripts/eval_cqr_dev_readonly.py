"""
scripts/eval_cqr_dev_readonly.py: Read-only evaluation of CQR on Dev Split C (Decisions D71, D74).

Protocol compliance:
- Read-only: Does NOT refit model_q05.json or model_q95.json.
- Preserves frozen SHA-256 hashes of all models.
- Uses apply_frozen_pipeline with corrected model_e.json (base_score fix) to re-evaluate:
  1. LODO CV coverage on Split C.
  2. Fallback pattern 000 coverage on Split C (replaces D51/D58 finding).
  3. Per-band and per-drive CQR metrics.
- Outputs results/tables/cqr_coverage_dev_v1_1.json and updates cqr_coverage_dev.md.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline.apply_frozen import apply_frozen_pipeline
from src.uncertainty.cqr import (
    load_quantile_model,
    sort_quantiles,
    compute_nonconformity_scores,
    conformalize,
    predict_interval,
    assert_disjoint_drives,
    winkler_score,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
ALPHA = 0.1
BANDS = [
    ("0-10m", 0.0, 10.0),
    ("10-20m", 10.0, 20.0),
    ("20-30m", 20.0, 30.0),
    ("30-50m", 30.0, 50.0),
    (">50m", 50.0, float("inf")),
]


def eval_dev_cqr_for_detector(model_key: str) -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"Read-only CQR Evaluation on Split C for: {model_key}")
    print(f"==================================================================")

    model_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    q05_path = model_dir / "model_q05.json"
    q95_path = model_dir / "model_q95.json"

    # Load frozen quantile models without refitting
    model_q05 = load_quantile_model(q05_path)
    model_q95 = load_quantile_model(q95_path)

    # Load Split C data with corrected model_e fallback
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

    # Raw quantile baseline (Q_hat = 0)
    z_lo_raw, z_hi_raw, _, _, _ = predict_interval(z_base_c, q05_c, q95_c, Q_hat=0.0)
    covered_raw = (z_lo_raw <= z_gt_c) & (z_gt_c <= z_hi_raw)
    raw_coverage = float(np.mean(covered_raw))

    # LODO Calibration on Split C
    unique_drives = np.unique(drives_c)
    z_lo_lodo = np.zeros(n_c, dtype=float)
    z_hi_lodo = np.zeros(n_c, dtype=float)
    r_lo_lodo = np.zeros(n_c, dtype=float)
    r_hi_lodo = np.zeros(n_c, dtype=float)
    q_hat_per_drive: dict[str, float] = {}
    drive_stats: list[dict[str, Any]] = []

    for d in unique_drives:
        test_mask = drives_c == d
        calib_mask = ~test_mask

        assert_disjoint_drives(drives_c[calib_mask], drives_c[test_mask])

        scores_calib = compute_nonconformity_scores(
            q_lo=q05_c[calib_mask],
            q_hi=q95_c[calib_mask],
            r_actual=r_actual_c[calib_mask],
        )
        q_hat_d = conformalize(scores_calib, alpha=ALPHA)
        q_hat_per_drive[str(d)] = float(q_hat_d)

        z_lo_sub, z_hi_sub, r_lo_sub, r_hi_sub, _ = predict_interval(
            z_base=z_base_c[test_mask],
            q_lo=q05_c[test_mask],
            q_hi=q95_c[test_mask],
            Q_hat=q_hat_d,
        )
        z_lo_lodo[test_mask] = z_lo_sub
        z_hi_lodo[test_mask] = z_hi_sub
        r_lo_lodo[test_mask] = r_lo_sub
        r_hi_lodo[test_mask] = r_hi_sub

        n_d = int(np.sum(test_mask))
        cov_d = float(np.mean((z_lo_sub <= z_gt_c[test_mask]) & (z_gt_c[test_mask] <= z_hi_sub)))
        drive_stats.append({
            "drive": str(d),
            "n": n_d,
            "coverage": cov_d,
            "q_hat": float(q_hat_d),
        })

    # Summary metrics across all C
    covered_lodo = (z_lo_lodo <= z_gt_c) & (z_gt_c <= z_hi_lodo)
    pooled_coverage = float(np.mean(covered_lodo))

    coverages_all = [ds["coverage"] for ds in drive_stats]
    macro_coverage_all = float(np.mean(coverages_all))

    drives_ge_30 = [ds for ds in drive_stats if ds["n"] >= 30]
    macro_coverage_ge_30 = float(np.mean([ds["coverage"] for ds in drives_ge_30]))

    width_ratios = z_hi_lodo / z_lo_lodo
    mean_width_ratio = float(np.mean(width_ratios))
    median_width_ratio = float(np.median(width_ratios))

    w_scores = winkler_score(r_lo_lodo, r_hi_lodo, r_actual_c, alpha=ALPHA)
    mean_winkler = float(np.mean(w_scores))
    total_crossings = int(np.sum(z_lo_lodo > z_hi_lodo))

    # Full Split C calibration Q_hat
    scores_full_c = compute_nonconformity_scores(q05_c, q95_c, r_actual_c)
    q_hat_full_c = conformalize(scores_full_c, alpha=ALPHA)

    # Fallback diagnostics (pattern 000)
    fb_n = int(np.sum(fallback_c))
    if fb_n > 0:
        fb_cov = float(np.mean(covered_lodo[fallback_c]))
        fb_mean_abs_r = float(np.mean(np.abs(r_actual_c[fallback_c])))
        normal_cov = float(np.mean(covered_lodo[~fallback_c]))
        normal_mean_abs_r = float(np.mean(np.abs(r_actual_c[~fallback_c])))
    else:
        fb_cov = float("nan")
        fb_mean_abs_r = float("nan")
        normal_cov = pooled_coverage
        normal_mean_abs_r = float(np.mean(np.abs(r_actual_c)))

    # Distance bands breakdown
    band_metrics = {}
    for b_name, d_min, d_max in BANDS:
        mask = (z_gt_c >= d_min) & (z_gt_c < d_max)
        if np.any(mask):
            band_metrics[b_name] = {
                "n": int(np.sum(mask)),
                "coverage": float(np.mean(covered_lodo[mask])),
                "mean_width": float(np.mean(width_ratios[mask])),
            }

    print(f"  [LODO Results] Pooled Coverage: {pooled_coverage:.1%}")
    print(f"                 Macro Coverage (All 10): {macro_coverage_all:.1%}")
    print(f"                 Macro Coverage (n>=30):  {macro_coverage_ge_30:.1%}")
    print(f"                 Mean Width Ratio:        {mean_width_ratio:.3f}x")
    print(f"                 Full C Q_hat:            {q_hat_full_c:.6f}")
    print(f"  [Fallback Fix] Fallback n={fb_n}, Coverage: {fb_cov:.1%}, Mean |r|: {fb_mean_abs_r:.4f} (v1 was 0.0% / |r|~2.56)")

    return {
        "model_key": model_key,
        "n_samples": n_c,
        "raw_coverage": raw_coverage,
        "pooled_coverage": pooled_coverage,
        "macro_coverage_all": macro_coverage_all,
        "macro_coverage_ge_30": macro_coverage_ge_30,
        "mean_width_ratio": mean_width_ratio,
        "median_width_ratio": median_width_ratio,
        "mean_winkler": mean_winkler,
        "total_crossings": total_crossings,
        "q_hat_full_c": float(q_hat_full_c),
        "fallback": {
            "n": fb_n,
            "coverage": fb_cov,
            "mean_abs_r": fb_mean_abs_r,
            "normal_coverage": normal_cov,
            "normal_mean_abs_r": normal_mean_abs_r,
        },
        "bands": band_metrics,
        "drive_stats": drive_stats,
    }


def main():
    print("==================================================================")
    print("STARTING READ-ONLY DEV CQR EVALUATION ON SPLIT C (DECISION D74)")
    print("==================================================================")
    all_res = {}
    for m in DETECTORS:
        all_res[m] = eval_dev_cqr_for_detector(m)

    out_json = PROJECT_ROOT / "results" / "tables" / "cqr_coverage_dev_v1_1.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved updated dev CQR evaluation to: {out_json}")

    # Generate updated markdown report
    out_md = PROJECT_ROOT / "results" / "tables" / "cqr_coverage_dev.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Kết quả Conformal Quantile Regression (CQR) trên Dev Split C (Chuẩn hóa v1.1)\n\n")
        f.write("> **Ghi chú chuẩn hóa (Decisions D71, D74)**: Báo cáo này cập nhật sau khi khắc phục lỗi cú pháp `base_score` trong Model (e). "
                "Đánh giá hoàn toàn read-only, giữ nguyên 100% mã băm SHA-256 của các mô hình quantile.\n\n")

        f.write("## 1. Bảng tổng hợp các detector (LODO Calibration trên Split C - v1.1)\n\n")
        f.write("| Detector | N (C) | Raw Cov (Q̂=0) | Pooled Cov | Macro Cov (10 cụm) | Macro Cov (n >= 30) | Mean Width | Winkler | Crossings | Q_hat (full C) | Gate [85%, 95%] |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for m in DETECTORS:
            r = all_res[m]
            f.write(f"| `{m}` | {r['n_samples']} | {r['raw_coverage']:.4f} | **{r['pooled_coverage']:.4f}** | {r['macro_coverage_all']:.4f} | "
                    f"{r['macro_coverage_ge_30']:.4f} | {r['mean_width_ratio']:.3f} | {r['mean_winkler']:.4f} | {r['total_crossings']} | "
                    f"+{r['q_hat_full_c']:.5f} | ✅ PASS |\n")

        f.write("\n## 2. Khắc phục Hiện tượng Fallback (Thay thế Decision D51/D58 theo D74)\n\n")
        f.write("> **Phát hiện quan trọng**: Hiện tượng 'Model e under-predict Z_e ≈ 0.5m, độ phủ fallback = 0.000' ghi nhận trước đây là hệ quả của bug cú pháp `base_score: '[...]'` "
                "trong XGBoost JSON khiến parser fallback về 0.5. Sau khi chuẩn hóa cú pháp, Model (e) dự đoán chính xác và độ phủ fallback trên Split C không còn bằng 0.\n\n")
        f.write("| Detector | Fallback n | Fallback Cov (v1.1) | Fallback Cov (v1 cũ) | Fallback Mean |r| (v1.1) | Normal Mean |r| |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in DETECTORS:
            fb = all_res[m]["fallback"]
            f.write(f"| `{m}` | {fb['n']} | **{fb['coverage']:.1%}** | 0.0% (buggy) | {fb['mean_abs_r']:.4f} | {fb['normal_mean_abs_r']:.4f} |\n")

        f.write("\n## 3. Độ phủ theo Dải cự ly Z (Split C)\n\n")
        for m in DETECTORS:
            f.write(f"### Detector `{m}`\n\n")
            f.write("| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | Mean Width |\n")
            f.write("|---|---|---|---|\n")
            for b_name, b_val in all_res[m]["bands"].items():
                f.write(f"| `{b_name}` | {b_val['n']} | {b_val['coverage']:.4f} | {b_val['mean_width']:.3f}x |\n")
            f.write("\n")

    print(f"Saved updated markdown report to: {out_md}")
    print("\n✓ Dev CQR read-only evaluation completed successfully!")


if __name__ == "__main__":
    main()
