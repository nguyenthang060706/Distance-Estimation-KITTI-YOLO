"""
scripts/recompute_final_T_posthoc.py: Post-hoc recomputation of frozen Split T predictions (Decision D71).

Protocol compliance:
- Strictly operates on static post-inference parquet files saved during Task T12:
  * results/final/{model_key}_T_predictions.parquet (backed up to *_v1_buggy.parquet)
  * results/final/{model_key}_T_detections.parquet
  * results/final/{model_key}_T_matches.parquet
  * results/final/{model_key}_T_gt.parquet
- Zero interaction with Split T raw data (does not touch splits/T.txt, does not rerun YOLO).
- Does NOT delete runs/final_T.lock.
- Loads corrected model_f.json, model_e.json (base_score syntax fixed, no retrain/no tuning).
- Loads updated conformal_calib_C.json from Split C calibration.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import src.residual.models
from src.residual.models import (
    load_model_f,
    load_model_e,
    load_model_f0,
    predict_f,
    predict_e,
    predict_f0,
    build_feature_matrices,
)
from src.residual.feature_extractor import extract_inference_features
from src.uncertainty.cqr import (
    load_quantile_model,
    predict_interval,
    predict_split_interval,
    predict_interval_mondrian,
    MondrianBinning,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]


def recompute_for_detector(model_key: str) -> dict[str, dict[str, float]]:
    print(f"\n==================================================================")
    print(f"Post-hoc Recomputing Split T Predictions for: {model_key}")
    print(f"==================================================================")

    final_dir = PROJECT_ROOT / "results" / "final"
    runs_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    pred_path = final_dir / f"{model_key}_T_predictions.parquet"
    buggy_path = final_dir / f"{model_key}_T_predictions_v1_buggy.parquet"

    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    # Step 1: Backup original (buggy v1) predictions if not already backed up
    if not buggy_path.is_file():
        shutil.copy2(pred_path, buggy_path)
        print(f"  [Backup] Preserved original v1 predictions to: {buggy_path.name}")
    else:
        print(f"  [Backup] Existing v1 backup found at: {buggy_path.name}")

    old_df = pd.read_parquet(buggy_path)
    df = old_df.copy()

    # Step 2: Prepare base features from stored static columns and detections metadata
    det_path = final_dir / f"{model_key}_T_detections.parquet"
    det_df = pd.read_parquet(det_path)
    merged_meta = df[["frame_id", "pred_idx"]].merge(
        det_df[["frame_id", "pred_idx", "fx", "fy", "cx", "cy", "img_w", "img_h"]],
        on=["frame_id", "pred_idx"],
        how="left",
    )

    combined = pd.DataFrame({
        "x1": df["bbox_x1"],
        "y1": df["bbox_y1"],
        "x2": df["bbox_x2"],
        "y2": df["bbox_y2"],
        "confidence": df["confidence"],
        "z_w": df["z_w"],
        "z_h": df["z_h"],
        "z_g": df["z_g"],
        "valid_w": df["valid_w"],
        "valid_h": df["valid_h"],
        "valid_g": df["valid_g"],
        "fx": merged_meta["fx"],
        "fy": merged_meta["fy"],
        "cx": merged_meta["cx"],
        "cy": merged_meta["cy"],
        "img_w": merged_meta["img_w"],
        "img_h": merged_meta["img_h"],
    })
    base_feats = extract_inference_features(combined)

    # Step 3: Recompute Model (e) depth
    model_e = load_model_e(runs_dir / "model_e.json")
    z_hat_e, ln_z_hat_e = predict_e(model_e, base_feats)

    # Step 4: Handle fallback pattern 000
    pattern_000 = df["fallback_flag"].to_numpy(dtype=bool)
    z_d = df["z_d"].to_numpy(dtype=float)
    z_base = np.where(~pattern_000, z_d, z_hat_e)

    # Step 5: Build feature matrices and recompute Model (f) & Model (f0)
    feat_mats = build_feature_matrices(base_feats, z_base=z_base)

    model_f = load_model_f(runs_dir / "model_f.json")
    z_hat_f, r_hat_f = predict_f(model_f, feat_mats["f"], z_base)

    model_f0 = load_model_f0(runs_dir / "model_f0.joblib")
    z_hat_f0, r_hat_f0 = predict_f0(model_f0, feat_mats["f0"], z_base)

    # Step 6: Recompute conformal prediction intervals from updated conformal_calib_C.json
    calib_path = runs_dir / "conformal_calib_C.json"
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    q05_model = load_quantile_model(runs_dir / "model_q05.json")
    q95_model = load_quantile_model(runs_dir / "model_q95.json")
    X_mat = feat_mats["f"].to_numpy(dtype=float)
    q05_pred = q05_model.predict(X_mat)
    q95_pred = q95_model.predict(X_mat)

    # 1. Standard CQR
    q_hat_cqr = calib["standard_cqr"]["q_hat"]
    z_lo_cqr, z_hi_cqr, r_lo_cqr, r_hi_cqr, n_cross_cqr = predict_interval(
        z_base, q05_pred, q95_pred, q_hat_cqr
    )

    # 2. Split Conformal
    q_hat_sc = calib["split_conformal"]["q_hat"]
    z_lo_sc, z_hi_sc, r_lo_sc, r_hi_sc, n_cross_sc = predict_split_interval(
        z_base, r_hat_f, q_hat_sc
    )

    # 3. Mondrian CQR
    mondrian_spec = calib["mondrian_cqr"]
    binning = MondrianBinning.__new__(MondrianBinning)
    binning.edges = list(mondrian_spec["bin_edges"])
    binning.min_samples = int(mondrian_spec.get("min_samples_per_bin", 50))
    binning.labels = list(mondrian_spec.get("bin_labels", [
        f">={int(binning.edges[i])}m" if i == len(binning.edges) - 1 else f"{int(binning.edges[i])}-{int(binning.edges[i+1])}m"
        for i in range(len(binning.edges))
    ]))
    binning.n_bins = len(binning.labels)
    mondrian_q_hats = {int(k): float(v) for k, v in mondrian_spec["q_hat_per_bin"].items()}
    mondrian_bins = binning.assign_bins(z_hat_f)
    z_lo_m, z_hi_m, r_lo_m, r_hi_m, n_cross_m = predict_interval_mondrian(
        z_base=z_base,
        q_lo=q05_pred,
        q_hi=q95_pred,
        q_hat_per_bin=mondrian_q_hats,
        bin_indices=mondrian_bins,
    )

    # Step 7: Update columns in DataFrame
    z_gt = df["z_gt"].to_numpy(dtype=float)
    r_actual = np.log(z_gt) - np.log(z_base)

    df["z_base"] = z_base
    df["z_hat_e"] = z_hat_e
    df["z_hat_f0"] = z_hat_f0
    df["r_hat_f0"] = r_hat_f0
    df["z_hat_f"] = z_hat_f
    df["r_hat_f"] = r_hat_f
    df["r_actual"] = r_actual

    df["z_lo_cqr"] = z_lo_cqr
    df["z_hi_cqr"] = z_hi_cqr
    df["r_lo_cqr"] = r_lo_cqr
    df["r_hi_cqr"] = r_hi_cqr

    df["z_lo_sc"] = z_lo_sc
    df["z_hi_sc"] = z_hi_sc
    df["r_lo_sc"] = r_lo_sc
    df["r_hi_sc"] = r_hi_sc

    df["z_lo_mondrian"] = z_lo_m
    df["z_hi_mondrian"] = z_hi_m
    df["r_lo_mondrian"] = r_lo_m
    df["r_hi_mondrian"] = r_hi_m
    df["mondrian_bin"] = mondrian_bins

    # Step 8: Save clean predictions parquet
    df.to_parquet(pred_path, index=False)
    print(f"  [Saved] Updated {pred_path.name} with clean post-hoc predictions.")

    # Step 9: Compare metrics before and after
    def calc_metrics(source_df: pd.DataFrame) -> dict[str, float]:
        zg = source_df["z_gt"].to_numpy(dtype=float)
        zf = source_df["z_hat_f"].to_numpy(dtype=float)
        ze = source_df["z_hat_e"].to_numpy(dtype=float)
        zd = source_df["z_d"].to_numpy(dtype=float)
        zf0 = source_df["z_hat_f0"].to_numpy(dtype=float)

        w_cqr = source_df["z_hi_cqr"] / source_df["z_lo_cqr"]
        w_sc = source_df["z_hi_sc"] / source_df["z_lo_sc"]
        w_m = source_df["z_hi_mondrian"] / source_df["z_lo_mondrian"]

        cov_cqr = (source_df["z_lo_cqr"] <= zg) & (zg <= source_df["z_hi_cqr"])
        cov_sc = (source_df["z_lo_sc"] <= zg) & (zg <= source_df["z_hi_sc"])
        cov_m = (source_df["z_lo_mondrian"] <= zg) & (zg <= source_df["z_hi_mondrian"])

        valid_d = np.isfinite(zd) & (zd > 0)
        absrel_d_val = float(np.mean(np.abs(zd[valid_d] - zg[valid_d]) / zg[valid_d])) if np.any(valid_d) else float("nan")

        return {
            "absrel_d": absrel_d_val,
            "absrel_f0": float(np.mean(np.abs(zf0 - zg) / zg)),
            "absrel_f": float(np.mean(np.abs(zf - zg) / zg)),
            "absrel_e": float(np.mean(np.abs(ze - zg) / zg)),
            "cqr_cov": float(np.mean(cov_cqr)),
            "cqr_width": float(np.mean(w_cqr)),
            "sc_cov": float(np.mean(cov_sc)),
            "sc_width": float(np.mean(w_sc)),
            "m_cov": float(np.mean(cov_m)),
            "m_width": float(np.mean(w_m)),
        }

    m_old = calc_metrics(old_df)
    m_new = calc_metrics(df)

    print(f"\n  --- Comparison on Split T ({model_key}) ---")
    print(f"  Model (d) Fused AbsRel:      {m_new['absrel_d']:.4f} (unchanged)")
    print(f"  Model (f0) Ridge AbsRel:     {m_new['absrel_f0']:.4f} (v1: {m_old['absrel_f0']:.4f})")
    print(f"  Model (f) Residual AbsRel:   {m_new['absrel_f']:.4f} (v1: {m_old['absrel_f']:.4f})  <-- FIXED!")
    print(f"  Model (e) Direct AbsRel:     {m_new['absrel_e']:.4f} (v1: {m_old['absrel_e']:.4f})  <-- FIXED!")
    print(f"  Standard CQR Coverage/Width: {m_new['cqr_cov']:.1%} / {m_new['cqr_width']:.3f}x (v1: {m_old['cqr_cov']:.1%} / {m_old['cqr_width']:.3f}x)")
    print(f"  Split Conformal Width:       {m_new['sc_width']:.3f}x (v1: {m_old['sc_width']:.3f}x) <-- FIXED!")
    print(f"  Mondrian CQR Width:          {m_new['m_width']:.3f}x (v1: {m_old['m_width']:.3f}x) <-- FIXED!")

    return {"v1_buggy": m_old, "v1_1_fixed": m_new}


def main():
    print("==================================================================")
    print("STARTING OFFICIAL POST-HOC RECOMPUTATION ON SPLIT T (DECISION D71)")
    print("==================================================================")
    all_res = {}
    for m in DETECTORS:
        all_res[m] = recompute_for_detector(m)

    out_summary = PROJECT_ROOT / "results" / "tables" / "posthoc_v1_vs_v1_1_comparison.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved comparison summary to: {out_summary}")

    # Decision D71 & AGENT_RULES §5: Append log record to runs/pipeline_log.jsonl
    log_file = PROJECT_ROOT / "runs" / "pipeline_log.jsonl"
    from datetime import datetime, timezone
    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "RECOMPUTE_FINAL_T_POSTHOC",
        "decision": "D71",
        "git_commit": git_commit,
        "tag": "final-config-v1.1",
        "status": "COMPLETED",
        "detectors": {
            m: {
                "absrel_f_v1": all_res[m]["v1_buggy"]["absrel_f"],
                "absrel_f_v1_1": all_res[m]["v1_1_fixed"]["absrel_f"],
                "absrel_e_v1_1": all_res[m]["v1_1_fixed"]["absrel_e"],
                "cqr_cov_v1_1": all_res[m]["v1_1_fixed"]["cqr_cov"],
            }
            for m in DETECTORS
        },
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")
    print(f"Logged post-hoc execution to: {log_file.name}")
    print("\n✓ Post-hoc recomputation completed successfully for all 3 detectors!")


if __name__ == "__main__":
    main()
