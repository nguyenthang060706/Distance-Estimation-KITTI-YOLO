"""
scripts/run_final_T.py: One-time acceptance runner for Split T with 4 safety guards (Decisions D27, D49, D59, D60).

Safety Guards:
- Guard 1: Head commit must strictly match git tag 'final-config-v1'.
- Guard 2: Working directory must be completely clean (ignoring runs/ and results/).
- Guard 3: All split hashes and model SHAs must match configs/pipeline_frozen_v1.yaml.
- Guard 4: One-way atomic lockfile runs/final_T.lock created with exclusive flag 'x', requires --confirm FINAL_T_RUN.

Dry-Run Mode:
- --dry-run C: Runs full inference and evaluation pipeline on Split C, verifies against runs/dryrun_golden_C.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from scripts.run_inference import run_inference_core
from src.pipeline.build_dataset import build_population_from_dfs
from src.residual.feature_extractor import (
    extract_inference_features,
)
from src.pipeline.geometry_stage import add_cues, fuse_with_weights, load_geometry_priors
from src.geometry.geometric_cues import BORDER_EPS
from src.geometry.fusion import load_fusion_weights
from src.residual.models import (
    build_feature_matrices,
    load_model_f,
    predict_f,
    load_model_f0,
    predict_f0,
    load_model_e,
    predict_e,
)
from src.uncertainty.cqr import (
    load_quantile_model,
    predict_interval,
    predict_split_interval,
    predict_interval_mondrian,
    MondrianBinning,
    winkler_score,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]


def log_final_t_event(event_dict: dict[str, Any]) -> None:
    """Append one JSON line to runs/final_T_log.jsonl (Decision D65, AGENT_RULES §5)."""
    log_path = PROJECT_ROOT / "runs" / "final_T_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 4 Safety Guards (Decision D27)
# ==============================================================================

def check_guard_1_tag():
    """Guard 1: Check that HEAD commit matches annotated tag final-config-v1."""
    try:
        tag_commit = subprocess.check_output(
            ["git", "rev-parse", "final-config-v1^{commit}"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Guard 1 FAILED: Git tag 'final-config-v1' does not exist! Pipeline is not frozen.")

    try:
        head_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Guard 1 FAILED: Could not resolve HEAD commit.")

    if tag_commit != head_commit:
        raise RuntimeError(
            f"Guard 1 FAILED: HEAD ({head_commit[:8]}) does not match tag final-config-v1 ({tag_commit[:8]})! "
            f"Split T execution requires exact match."
        )
    print("  [Guard 1 PASS] HEAD matches tag final-config-v1 perfectly.")


def check_guard_2_clean_tree():
    """Guard 2: Check that git working directory is completely clean."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if status:
        # Ignore runtime files in runs/ and results/
        dirty_lines = [
            line for line in status.splitlines()
            if not (
                line[3:].replace("\\", "/").startswith("runs/")
                or line[3:].replace("\\", "/").startswith("results/")
            )
        ]
        if dirty_lines:
            raise RuntimeError(
                f"Guard 2 FAILED: Working tree is dirty!\n"
                f"Dirty files:\n" + "\n".join(dirty_lines)
            )
    print("  [Guard 2 PASS] Git working tree is completely clean.")


def check_guard_3_hashes(frozen_cfg_path: Path):
    """Guard 3: Verify split hashes and checkpoint/model SHAs against pipeline_frozen_v1.yaml."""
    if not frozen_cfg_path.is_file():
        raise FileNotFoundError(f"Guard 3 FAILED: Missing frozen configuration at {frozen_cfg_path}")

    with open(frozen_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Verify split file hashes
    for s_name in ["A", "B", "C", "T"]:
        s_info = cfg["splits"][s_name]
        p = PROJECT_ROOT / s_info["path"]
        actual_sha = sha256_file(p)
        if actual_sha != s_info["sha256"]:
            raise ValueError(f"Guard 3 FAILED: Split {s_name} file SHA mismatch!")

    # 2. Verify detector checkpoints and calibrations
    for m in DETECTORS:
        det_info = cfg["detectors"][m]
        ckpt_path = PROJECT_ROOT / det_info["checkpoint"]["path"]
        actual_ckpt_sha = sha256_file(ckpt_path)
        if actual_ckpt_sha != det_info["checkpoint"]["sha256"]:
            raise ValueError(f"Guard 3 FAILED: Checkpoint SHA mismatch for {m}!")

        calib_path = PROJECT_ROOT / det_info["conformal_calib_C"]["path"]
        actual_calib_sha = sha256_file(calib_path)
        if actual_calib_sha != det_info["conformal_calib_C"]["sha256"]:
            raise ValueError(f"Guard 3 FAILED: Conformal calibration SHA mismatch for {m}!")

        # Verify all trained model weights (Decision D63)
        for model_name, model_spec in det_info.get("models", {}).items():
            model_file_path = PROJECT_ROOT / model_spec["path"]
            actual_model_sha = sha256_file(model_file_path)
            if actual_model_sha != model_spec["sha256"]:
                raise ValueError(
                    f"Guard 3 FAILED: Model {model_name} SHA mismatch for {m}!\n"
                    f"  Expected: {model_spec['sha256']}\n"
                    f"  Actual:   {actual_model_sha}"
                )

    # 3. Verify config files
    for cfg_rel, expected_sha in cfg.get("config_files_sha256", {}).items():
        actual_cfg_sha = sha256_file(PROJECT_ROOT / cfg_rel)
        if actual_cfg_sha != expected_sha:
            raise ValueError(f"Guard 3 FAILED: Config file SHA mismatch for {cfg_rel}!")

    print("  [Guard 3 PASS] All split hashes, checkpoint SHAs, trained model SHAs, calibrations, and config SHAs match frozen configuration.")


def check_guard_preflight(output_dir: Path) -> None:
    """Preflight check before locking: verifies disk space, write permissions, and GPU (Decision D67)."""
    # 1. Directory write permission
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / ".preflight_write_test.tmp"
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("preflight ok")
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Preflight FAILED: Output directory {output_dir} is not writable: {e}")

    # 2. Disk space check (require at least 1 GB)
    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 1.0:
        raise RuntimeError(f"Preflight FAILED: Insufficient disk space on {output_dir}. Free: {free_gb:.2f} GB (required >= 1.0 GB).")

    # 3. GPU availability check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [Preflight PASS] GPU detected: {gpu_name} (CUDA available)")
        else:
            print("  [Preflight WARNING] CUDA not available, inference will run on CPU.")
    except ImportError:
        print("  [Preflight WARNING] PyTorch not importable in preflight check.")

    print(f"  [Preflight PASS] Disk space {free_gb:.2f} GB available, directory {output_dir} writable.")


def check_and_create_guard_4_lock(dry_run: bool, confirm_flag: str | None):
    """Guard 4: Atomic lockfile creation with 'x' flag."""
    if dry_run:
        print("  [Guard 4 BYPASS] Dry-run mode: lockfile will not be created.")
        return

    if confirm_flag != "FINAL_T_RUN":
        raise ValueError(
            "Guard 4 FAILED: Running on Split T requires explicit confirmation flag: --confirm FINAL_T_RUN"
        )

    lock_file = PROJECT_ROOT / "runs" / "final_T.lock"
    try:
        with open(lock_file, "x", encoding="utf-8") as f:
            f.write(f"Split T execution locked.\nTimestamp: {pd.Timestamp.now().isoformat()}\n")
    except FileExistsError:
        raise FileExistsError(
            "Guard 4 FAILED: Lockfile runs/final_T.lock already exists! "
            "Split T has already been executed. Re-running is strictly prohibited by Decision D27."
        )
    print("  [Guard 4 PASS] Created atomic lockfile runs/final_T.lock successfully.")


# ==============================================================================
# Full Per-Object Pipeline Execution (Decision D60)
# ==============================================================================

def execute_pipeline_for_detector(
    model_key: str,
    split: str,
    output_dir: Path,
    allow_test: bool,
    conf_min: float = 0.05,
    iou_nms: float = 0.7,
    iou_match: float = 0.5,
    dontcare_mode: str = "iou",
) -> dict[str, Any]:
    """
    Executes end-to-end detector -> features -> cues -> Z_d -> Z_base -> Z_hat_f -> intervals.
    Saves full per-object artifacts to output_dir (Decision D60).
    """
    runs_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    base_model = model_key.replace("_640", "")

    # Step 1: Run inference core
    print(f"\n--- [{model_key}] Step 1: Running detector inference on Split {split} ---")
    det_out, match_out, gt_out = run_inference_core(
        model_name=base_model,
        split=split,
        output_dir=str(output_dir),
        conf_min=conf_min,
        iou_nms=iou_nms,
        iou_match=iou_match,
        dontcare_mode=dontcare_mode,
        allow_test=allow_test,
    )

    df_dets = pd.read_parquet(det_out)
    df_matches = pd.read_parquet(match_out) if match_out.is_file() else None
    df_gt = pd.read_parquet(gt_out) if gt_out.is_file() else None

    # Filter to True Positives passing threshold (Decisions D6, D15, D23)
    if df_matches is not None and df_gt is not None:
        print(f"[{model_key}] Step 2: Building population (pass_thr TP join)...")
        features_df, eval_df, fn_df = build_population_from_dfs(
            df_dets=df_dets,
            df_matches=df_matches,
            df_gt=df_gt,
            population="pass_thr",
        )
    else:
        features_df = df_dets[df_dets["pass_thr"] == True].copy().reset_index(drop=True)
        eval_df = None
        fn_df = None

    n_tp = len(features_df)
    print(f"[{model_key}] True Positives meeting pass_thr: {n_tp}")

    # Step 3: Compute geometric cues
    print(f"[{model_key}] Step 3: Computing geometric cues...")
    priors, _ = load_geometry_priors()
    df_tp_cues = add_cues(features_df, priors=priors, eps=BORDER_EPS)

    combined = features_df.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined[col] = df_tp_cues[col]

    base_feats = extract_inference_features(combined)

    # Step 4: Geometric fusion Z_d and Fallback Z_base
    print(f"[{model_key}] Step 4: Geometric fusion and fallback handling...")
    fw_path = runs_dir / "full_fw.json"
    full_fw = load_fusion_weights(fw_path)
    z_d = fuse_with_weights(df_tp_cues, full_fw)

    # Direct Model (e) depth prediction for all samples (Decision D13, D34, and T13 ablation support)
    model_e_path = runs_dir / "model_e.json"
    model_e = load_model_e(model_e_path)
    z_hat_e, ln_z_hat_e = predict_e(model_e, base_feats)

    valid_w = df_tp_cues["valid_w"].to_numpy(dtype=bool)
    valid_h = df_tp_cues["valid_h"].to_numpy(dtype=bool)
    valid_g = df_tp_cues["valid_g"].to_numpy(dtype=bool)
    pattern_000 = (~valid_w) & (~valid_h) & (~valid_g)
    n_fb = int(np.sum(pattern_000))

    if n_fb > 0:
        z_base = np.where(~pattern_000, z_d, z_hat_e)
    else:
        z_base = z_d.copy()

    # Step 5: Residual prediction (f) and baseline linear (f0)
    print(f"[{model_key}] Step 5: Predicting depth with Model (f) and Model (f0)...")
    feat_mats = build_feature_matrices(base_feats, z_base=z_base)
    model_f = load_model_f(runs_dir / "model_f.json")
    z_hat_f, r_hat_f = predict_f(model_f, feat_mats["f"], z_base)

    model_f0 = load_model_f0(runs_dir / "model_f0.joblib")
    z_hat_f0, r_hat_f0 = predict_f0(model_f0, feat_mats["f0"], z_base)

    # Step 6: Conformal interval predictions (3 methods)
    print(f"[{model_key}] Step 6: Constructing conformal prediction intervals...")
    calib_path = runs_dir / "conformal_calib_C.json"
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    q05_model = load_quantile_model(runs_dir / "model_q05.json")
    q95_model = load_quantile_model(runs_dir / "model_q95.json")
    X_mat = feat_mats["f"].to_numpy(dtype=float)
    q05_pred = q05_model.predict(X_mat)
    q95_pred = q95_model.predict(X_mat)

    # 1. Standard CQR (Primary)
    q_hat_cqr = calib["standard_cqr"]["q_hat"]
    z_lo_cqr, z_hi_cqr, r_lo_cqr, r_hi_cqr, n_cross_cqr = predict_interval(
        z_base, q05_pred, q95_pred, q_hat_cqr
    )

    # 2. Split Conformal (Descriptive baseline)
    q_hat_sc = calib["split_conformal"]["q_hat"]
    z_lo_sc, z_hi_sc, r_lo_sc, r_hi_sc, n_cross_sc = predict_split_interval(
        z_base, r_hat_f, q_hat_sc
    )

    # 3. Mondrian CQR (Descriptive baseline - Decoupled from test data, Decision D67)
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

    # Step 7: Build full per-object DataFrame (Decision D60, D65 for T13)
    per_obj_df = pd.DataFrame({
        "frame_id": features_df["frame_id"],
        "pred_idx": features_df["pred_idx"],
        "confidence": features_df["confidence"],
        "bbox_x1": features_df["x1"],
        "bbox_y1": features_df["y1"],
        "bbox_x2": features_df["x2"],
        "bbox_y2": features_df["y2"],
        "z_w": df_tp_cues["z_w"],
        "z_h": df_tp_cues["z_h"],
        "z_g": df_tp_cues["z_g"],
        "valid_w": df_tp_cues["valid_w"],
        "valid_h": df_tp_cues["valid_h"],
        "valid_g": df_tp_cues["valid_g"],
        "z_d": z_d,
        "z_base": z_base,
        "fallback_flag": pattern_000,
        "z_hat_e": z_hat_e,
        "z_hat_f0": z_hat_f0,
        "r_hat_f0": r_hat_f0,
        "z_hat_f": z_hat_f,
        "r_hat_f": r_hat_f,
        "z_lo_cqr": z_lo_cqr,
        "z_hi_cqr": z_hi_cqr,
        "r_lo_cqr": r_lo_cqr,
        "r_hi_cqr": r_hi_cqr,
        "z_lo_sc": z_lo_sc,
        "z_hi_sc": z_hi_sc,
        "r_lo_sc": r_lo_sc,
        "r_hi_sc": r_hi_sc,
        "z_lo_mondrian": z_lo_m,
        "z_hi_mondrian": z_hi_m,
        "r_lo_mondrian": r_lo_m,
        "r_hi_mondrian": r_hi_m,
        "mondrian_bin": mondrian_bins,
    })

    # Join GT evaluation info if available
    eval_metrics: dict[str, Any] = {}
    if eval_df is not None:
        z_gt = eval_df["z_gt"].to_numpy(dtype=float)
        drives = eval_df["drive"].to_numpy(dtype=str)

        per_obj_df["drive"] = drives
        per_obj_df["gt_idx"] = eval_df["gt_idx"]
        per_obj_df["z_gt"] = z_gt
        per_obj_df["matched_iou"] = eval_df["matched_iou"]
        per_obj_df["difficulty"] = eval_df["difficulty"]
        per_obj_df["truncated"] = eval_df["truncated"]
        per_obj_df["occluded"] = eval_df["occluded"]
        per_obj_df["alpha"] = eval_df["alpha"]

        r_actual = np.log(z_gt) - np.log(z_base)
        per_obj_df["r_actual"] = r_actual

        # Compute summary metrics
        covered_cqr = (z_lo_cqr <= z_gt) & (z_gt <= z_hi_cqr)
        unique_drives = np.unique(drives)

        eval_metrics = {
            "absrel_f": float(np.mean(np.abs(z_hat_f - z_gt) / z_gt)),
            "standard_cqr": {
                "pooled_coverage": float(np.mean(covered_cqr)),
                "macro_coverage": float(np.mean([np.mean(covered_cqr[drives == d]) for d in unique_drives])),
                "mean_width_ratio": float(np.mean(z_hi_cqr / z_lo_cqr)),
                "mean_winkler": float(np.mean(winkler_score(r_lo_cqr, r_hi_cqr, r_actual, alpha=0.1))),
                "n_crossings": n_cross_cqr,
            },
            "split_conformal": {
                "pooled_coverage": float(np.mean((z_lo_sc <= z_gt) & (z_gt <= z_hi_sc))),
                "macro_coverage": float(np.mean([np.mean(((z_lo_sc <= z_gt) & (z_gt <= z_hi_sc))[drives == d]) for d in unique_drives])),
                "mean_width_ratio": float(np.mean(z_hi_sc / z_lo_sc)),
                "n_crossings": n_cross_sc,
            },
            "mondrian_cqr": {
                "pooled_coverage": float(np.mean((z_lo_m <= z_gt) & (z_gt <= z_hi_m))),
                "macro_coverage": float(np.mean([np.mean(((z_lo_m <= z_gt) & (z_gt <= z_hi_m))[drives == d]) for d in unique_drives])),
                "mean_width_ratio": float(np.mean(z_hi_m / z_lo_m)),
                "n_crossings": n_cross_m,
            },
        }

    # Save per-object parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_parquet_path = output_dir / f"{model_key}_{split}_predictions.parquet"
    per_obj_df.to_parquet(pred_parquet_path, index=False)
    print(f"[{model_key}] Saved per-object predictions: {pred_parquet_path.name}")

    # Save False Negatives if matches available
    if fn_df is not None:
        fn_path = output_dir / f"{model_key}_{split}_fn.parquet"
        fn_df.to_parquet(fn_path, index=False)
        print(f"[{model_key}] Saved {len(fn_df)} False Negatives: {fn_path.name}")

    return {
        "model_key": model_key,
        "split": split,
        "n_tp": n_tp,
        "n_fallback": n_fb,
        "q_hat_cqr": q_hat_cqr,
        "metrics": eval_metrics,
        "per_obj_df": per_obj_df,
    }


# ==============================================================================
# Dry-run Verification against Golden File (Decision D59)
# ==============================================================================

def verify_dryrun_against_golden(results: dict[str, Any]) -> bool:
    golden_path = PROJECT_ROOT / "runs" / "dryrun_golden_C.json"
    if not golden_path.is_file():
        raise FileNotFoundError(f"Missing golden baseline file at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    all_passed = True
    print("\n==================================================================")
    print("VERIFYING DRY-RUN RESULTS AGAINST GOLDEN FILE (DECISION D59)")
    print("==================================================================")

    for m in DETECTORS:
        res = results[m]
        g = golden[m]
        met = res["metrics"]["standard_cqr"]

        diff_n = abs(res["n_tp"] - g["n_samples"])
        diff_q = abs(res["q_hat_cqr"] - g["q_hat"])
        diff_cov = abs(met["pooled_coverage"] - g["pooled_coverage"])
        rel_width = abs(met["mean_width_ratio"] - g["mean_width_ratio"]) / g["mean_width_ratio"]
        rel_winkler = abs(met["mean_winkler"] - g["mean_winkler"]) / g["mean_winkler"]

        pass_n = (diff_n == 0)
        pass_q = (diff_q <= 1e-6)
        pass_cov = (diff_cov <= 1e-4)
        pass_width = (rel_width <= 1e-4)
        pass_winkler = (rel_winkler <= 1e-4)

        status_str = "PASS" if (pass_n and pass_q and pass_cov and pass_width and pass_winkler) else "FAIL"
        if status_str == "FAIL":
            all_passed = False

        print(f"\n--- Detector `{m}` [{status_str}] ---")
        print(f"  Count TP:       Dry-Run={res['n_tp']}, Golden={g['n_samples']} (Diff={diff_n}) {'✓' if pass_n else '✗'}")
        print(f"  Q_hat CQR:      Dry-Run={res['q_hat_cqr']:.6f}, Golden={g['q_hat']:.6f} (Diff={diff_q:.2e}) {'✓' if pass_q else '✗'}")
        print(f"  Pooled Cov:     Dry-Run={met['pooled_coverage']*100:.2f}%, Golden={g['pooled_coverage']*100:.2f}% (Diff={diff_cov*100:.3f}%) {'✓' if pass_cov else '✗'}")
        print(f"  Mean Width:     Dry-Run={met['mean_width_ratio']:.4f}, Golden={g['mean_width_ratio']:.4f} (RelDiff={rel_width:.2e}) {'✓' if pass_width else '✗'}")
        print(f"  Mean Winkler:   Dry-Run={met['mean_winkler']:.4f}, Golden={g['mean_winkler']:.4f} (RelDiff={rel_winkler:.2e}) {'✓' if pass_winkler else '✗'}")

    return all_passed


# ==============================================================================
# Main Runner Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run final acceptance evaluation on Split T (or dry-run on C).")
    parser.add_argument("--dry-run", choices=["C"], default=None, help="Run dry-run verification on Split C")
    parser.add_argument("--confirm", type=str, default=None, help="Must be 'FINAL_T_RUN' to execute on Split T")
    parser.add_argument("--config", type=str, default="configs/pipeline_frozen_v1.yaml", help="Path to frozen config")
    args = parser.parse_args()

    frozen_cfg_path = PROJECT_ROOT / args.config
    if not frozen_cfg_path.is_file():
        raise FileNotFoundError(f"Missing frozen config at: {frozen_cfg_path}")

    with open(frozen_cfg_path, "r", encoding="utf-8") as f:
        frozen_cfg = yaml.safe_load(f)

    pipe_params = frozen_cfg.get("pipeline_parameters", {})
    conf_min = float(pipe_params.get("conf_min", 0.05))
    iou_nms = float(pipe_params.get("iou_nms", 0.7))
    iou_match = float(pipe_params.get("iou_match", 0.5))
    dontcare_mode = str(pipe_params.get("dontcare_mode", "iou"))

    if args.dry_run == "C":
        print("==================================================================")
        print("STARTING DRY-RUN VERIFICATION ON SPLIT C (DECISION D59)")
        print("==================================================================")
        print("\nVerifying Guard 3 (Hashes & Model SHAs) during Dry-run...")
        check_guard_3_hashes(frozen_cfg_path)
        check_and_create_guard_4_lock(dry_run=True, confirm_flag=None)

        dry_results = {}
        dry_out_dir = PROJECT_ROOT / "runs" / "dryrun_C"
        for m in DETECTORS:
            res = execute_pipeline_for_detector(
                model_key=m,
                split="C",
                output_dir=dry_out_dir,
                allow_test=True,  # Dry-run on C uses runner pipeline
                conf_min=conf_min,
                iou_nms=iou_nms,
                iou_match=iou_match,
                dontcare_mode=dontcare_mode,
            )
            dry_results[m] = res

        passed = verify_dryrun_against_golden(dry_results)
        if not passed:
            print("\n❌ DRY-RUN FAILED: Results deviated from Golden Baseline beyond allowed tolerances!")
            sys.exit(1)
        else:
            print("\n✓ DRY-RUN PASSED: All metrics match Golden Baseline within strict tolerances!")
            print("  Note: Dry-run serves as regression check (Decision D64).")
            sys.exit(0)

    # Official Split T Execution
    print("==================================================================")
    print("STARTING OFFICIAL FINAL TEST EVALUATION ON SPLIT T (DECISION D27)")
    print("==================================================================")

    # 1. Check all 4 safety guards and preflight requirements
    print("\nChecking Safety Guards...")
    check_guard_1_tag()
    check_guard_2_clean_tree()
    check_guard_3_hashes(frozen_cfg_path)
    final_out_dir = PROJECT_ROOT / "results" / "final"
    check_guard_preflight(final_out_dir)
    check_and_create_guard_4_lock(dry_run=False, confirm_flag=args.confirm)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"

    # Log START event (Decisions D65, D67)
    t_start_time = time.time()
    log_final_t_event({
        "event": "START",
        "timestamp": pd.Timestamp.now().isoformat(),
        "split": "T",
        "git_commit": git_commit,
        "tag": "final-config-v1",
        "seed": int(pipe_params.get("seed", 42)),
        "config_file": str(frozen_cfg_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "pipeline_parameters": pipe_params,
        "detectors": DETECTORS,
        "status": "STARTED",
    })

    # 2. Execute on Split T with exception handling (Decision D67)
    final_results = {}
    try:
        for m in DETECTORS:
            res = execute_pipeline_for_detector(
                model_key=m,
                split="T",
                output_dir=final_out_dir,
                allow_test=True,
                conf_min=conf_min,
                iou_nms=iou_nms,
                iou_match=iou_match,
                dontcare_mode=dontcare_mode,
            )
            final_results[m] = res
    except Exception as e:
        total_elapsed = time.time() - t_start_time
        tb_str = traceback.format_exc()
        log_final_t_event({
            "event": "FAILED",
            "timestamp": pd.Timestamp.now().isoformat(),
            "split": "T",
            "elapsed_seconds": round(total_elapsed, 2),
            "error": repr(e),
            "traceback": tb_str,
            "status": "FAILED",
        })
        print(f"\n❌ Execution on Split T FAILED: {repr(e)}")
        raise

    total_elapsed = time.time() - t_start_time

    # 3. Save summary report
    summary_path = PROJECT_ROOT / "results" / "tables" / "final_eval_T.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        m: {
            "n_tp": final_results[m]["n_tp"],
            "n_fallback": final_results[m]["n_fallback"],
            "metrics": final_results[m]["metrics"],
        }
        for m in DETECTORS
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Log COMPLETED event (Decision D65)
    log_final_t_event({
        "event": "COMPLETED",
        "timestamp": pd.Timestamp.now().isoformat(),
        "split": "T",
        "elapsed_seconds": round(total_elapsed, 2),
        "detectors_summary": {
            m: {
                "n_tp": final_results[m]["n_tp"],
                "n_fallback": final_results[m]["n_fallback"],
                "q_hat_cqr": final_results[m]["q_hat_cqr"],
                "absrel_f": final_results[m]["metrics"].get("absrel_f"),
                "cqr_pooled_coverage": final_results[m]["metrics"].get("standard_cqr", {}).get("pooled_coverage"),
            }
            for m in DETECTORS
        },
        "status": "SUCCESS",
    })

    print(f"\n✓ Split T evaluation completed successfully in {total_elapsed:.1f}s! Results stored in {final_out_dir}")


if __name__ == "__main__":
    main()
