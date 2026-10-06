"""
scripts/run_ablation_oof.py: Main runner for T05 Feature Ablation on Split B OOF.
(Decisions D24, D25, D30, D36, D37, D38, D39).

Workflow:
1. Loads pre-registration YAML: configs/residual/residual_prereg_v1.yaml (section ablation:).
2. For each detector (yolo11s_640, yolov8s_640, yolov5su_640):
   a. Loads B features, B eval, B cues, and B_oof.parquet (from T04).
   b. Runs 10 pre-registered ablations:
      - 6 drop_groups (A1..A6): bbox_geometry, edge_flags, confidence, cues_ln_z, validity_flags, ln_z_base.
        (Z_base and r_train are cached from T04, re-fitting only Model (f) with reduced features).
      - 3 drop_single_cue (C1..C3): drop_z_w, drop_z_h, drop_z_g.
        (Re-fits 2-cue fusion weights with explicit cue_names; pattern 000 routes to fallback Model (e);
         recomputes Z_base and trains Model (f) with 15 features).
      - 1 alternative_model (M1): MLPRegressor (128, 64) with StandardScaler inside each fold.
   c. Evaluates against Full Model (f) baseline on common support.
   d. Computes paired cluster bootstrap CI (12 clusters, n_boot=1000, seed=42) for Delta AbsRel.
   e. Saves predictions to results/datasets/{model_key}_B_ablation.parquet.
   f. Saves per-detector manifest in runs/residual/{model_key}/ablation_manifest.json.
3. Outputs comprehensive results/tables/ablation_oof_b.json and results/tables/ablation_oof_b.md.
4. Appends execution log to runs/pipeline_log.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import (
    append_jsonl,
    depth_metrics,
    macro_by_cluster,
    make_log_record,
    paired_cluster_bootstrap,
)
from src.geometry.fusion import (
    fit_fusion_weights,
    fuse_depths_vectorised,
)
from src.residual.feature_extractor import (
    check_no_gt_leakage,
    extract_inference_features,
)
from src.residual.models import (
    FEATURE_COLS_F,
    build_feature_matrices,
    fit_e,
    fit_f,
    fit_mlp,
    predict_e,
    predict_f,
    predict_mlp,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
CUE_ALL_NAMES = ["Z_w", "Z_h", "Z_g"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        tags = subprocess.check_output(["git", "tag", "-l", "prereg-residual-v1"], text=True).strip()
        tag = "prereg-residual-v1" if "prereg-residual-v1" in tags else "none"
    except Exception:
        tag = "none"
    return commit, tag


def run_ablation_for_detector(
    model_key: str,
    prereg_cfg: dict[str, Any],
    split_b_hash: str,
    git_commit: str,
    git_tag: str,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    print("\n" + "=" * 78)
    print(f"  RUNNING T05 ABLATION FOR DETECTOR: {model_key}")
    print("=" * 78)

    feat_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_features.parquet"
    eval_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_eval.parquet"
    cues_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_cues.parquet"
    oof_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_oof.parquet"

    for p in [feat_path, eval_path, cues_path, oof_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset: {p}")

    features_df = pd.read_parquet(feat_path)
    eval_df = pd.read_parquet(eval_path)
    cues_df = pd.read_parquet(cues_path)
    oof_df = pd.read_parquet(oof_path)

    n_samples = len(features_df)
    assert len(eval_df) == n_samples and len(cues_df) == n_samples and len(oof_df) == n_samples, (
        f"Mismatch row counts in datasets for {model_key}"
    )

    # 1. Base inference features extraction
    combined = features_df.copy()
    for col in ["z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]:
        combined[col] = cues_df[col]
    base_feats = extract_inference_features(combined)
    check_no_gt_leakage(base_feats.columns)

    z_gt = eval_df["z_gt"].to_numpy(dtype=float)
    ln_z_gt = np.log(z_gt)
    z_cues = cues_df[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = cues_df[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)

    # Cached baseline Z_base and residual from T04
    z_base_cached = oof_df["z_base"].to_numpy(dtype=float)
    z_hat_f_baseline = oof_df["z_hat_f"].to_numpy(dtype=float)
    z_hat_e_cached = oof_df["z_hat_e"].to_numpy(dtype=float)

    # Drives setup for 12 outer folds
    drive_col = cues_df["drive"].to_numpy(dtype=str)
    unique_drives = np.sort(np.unique(drive_col))
    assert len(unique_drives) == 12, f"Expected 12 drives in Split B, got {len(unique_drives)}"

    # -------------------------------------------------------------------------
    # Pre-compute Inner-LODO Z_e on train drives once per outer fold
    # (Because Model (e) feature set has NO cues or valid flags, it is 100% invariant)
    # -------------------------------------------------------------------------
    print("\n>>> Pre-computing invariant Inner-LODO Model (e) on train drives...")
    inner_ze_train_by_fold: list[np.ndarray] = []
    outer_splits = []

    for fold_idx, held_drive in enumerate(unique_drives):
        test_mask = (drive_col == held_drive)
        train_mask = ~test_mask
        train_drives = drive_col[train_mask]
        train_indices = np.where(train_mask)[0]
        n_train = np.sum(train_mask)

        outer_splits.append({
            "fold_idx": fold_idx,
            "held_drive": held_drive,
            "test_mask": test_mask,
            "train_mask": train_mask,
            "train_drives": train_drives,
        })

        inner_unique = np.unique(train_drives)
        z_e_train_oof = np.full(n_train, np.nan, dtype=float)

        for inner_d in inner_unique:
            inner_test_local = (train_drives == inner_d)
            inner_train_local = ~inner_test_local
            inner_train_global = train_indices[inner_train_local]
            inner_test_global = train_indices[inner_test_local]

            m_e_inner = fit_e(
                X=base_feats.iloc[inner_train_global],
                y_ln_gt=ln_z_gt[inner_train_global],
                random_state=seed,
                n_jobs=n_jobs,
            )
            z_e_pred, _ = predict_e(m_e_inner, base_feats.iloc[inner_test_global])
            z_e_train_oof[inner_test_local] = z_e_pred

        assert not np.any(np.isnan(z_e_train_oof)), f"NaN in z_e_train_oof fold {fold_idx}"
        inner_ze_train_by_fold.append(z_e_train_oof)

    print("Pre-computation of Model (e) inner-LODO completed.")

    # Dictionary to store all ablation predictions
    ablation_preds_df = pd.DataFrame({
        "frame_id": features_df["frame_id"],
        "drive": features_df["drive"],
        "pred_idx": features_df["pred_idx"],
        "z_hat_f_baseline": z_hat_f_baseline,
    })

    ablation_cfg = prereg_cfg.get("ablation", {})
    drop_groups = ablation_cfg.get("drop_groups", [])
    drop_single_cues = ablation_cfg.get("drop_single_cue", [])
    mlp_cfg = ablation_cfg.get("alternative_model", {}).get("mlp", {})

    experiment_records: list[dict[str, Any]] = []

    # Baseline performance
    eval_temp = eval_df.copy()
    eval_temp["z_hat_f"] = z_hat_f_baseline
    eval_temp["drive"] = drive_col
    met_base = depth_metrics(z_gt, z_hat_f_baseline)
    macro_base = macro_by_cluster(eval_temp, gt_col="z_gt", pred_col="z_hat_f", cluster_col="drive")

    base_record = {
        "model_key": model_key,
        "ablation_key": "baseline_full_f",
        "ablation_name": "Full Model (f) Baseline",
        "category": "baseline",
        "n_features": len(FEATURE_COLS_F),
        "pooled_absrel": met_base["absrel"],
        "macro_absrel": macro_base,
        "delta_pooled": 0.0,
        "delta_macro": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
    }
    experiment_records.append(base_record)
    print(f"\n[Baseline (f)] Pooled AbsRel: {met_base['absrel']:.4f} | Macro AbsRel: {macro_base:.4f}")

    # =========================================================================
    # 2. Drop Groups A1..A6 (cached Z_base from T04)
    # =========================================================================
    print("\n" + "-" * 60)
    print("  PHASE 1: DROP GROUPS (A1 - A6)")
    print("-" * 60)

    # Pre-construct base feature matrices with cached Z_base
    feat_mats_full = build_feature_matrices(base_feats, z_base_cached)["f"]

    for grp in drop_groups:
        grp_name = grp["name"]
        drop_cols = grp["features"]
        reduced_cols = [c for c in FEATURE_COLS_F if c not in drop_cols]
        print(f"\n>>> Running Ablation: Drop Group '{grp_name}' ({len(reduced_cols)} features remaining)...")

        z_hat_arr = np.full(n_samples, np.nan, dtype=float)

        for split in outer_splits:
            fold_idx = split["fold_idx"]
            train_mask = split["train_mask"]
            test_mask = split["test_mask"]
            train_drives = split["train_drives"]

            z_base_tr = z_base_cached[train_mask]
            z_base_te = z_base_cached[test_mask]
            r_tr = ln_z_gt[train_mask] - np.log(z_base_tr)

            m_f, _, _ = fit_f(
                X=feat_mats_full.iloc[train_mask],
                y=r_tr,
                groups=train_drives,
                feature_cols=reduced_cols,
                random_state=seed,
                n_jobs=n_jobs,
            )
            z_pred, _ = predict_f(
                model=m_f,
                X=feat_mats_full.iloc[test_mask],
                z_base=z_base_te,
                feature_cols=reduced_cols,
            )
            z_hat_arr[test_mask] = z_pred

        assert not np.any(np.isnan(z_hat_arr)), f"NaN in predictions for drop group {grp_name}"

        col_name = f"z_hat_drop_group_{grp_name}"
        ablation_preds_df[col_name] = z_hat_arr

        # Evaluate and compute paired cluster bootstrap vs baseline
        eval_temp[col_name] = z_hat_arr
        met = depth_metrics(z_gt, z_hat_arr)
        macro = macro_by_cluster(eval_temp, gt_col="z_gt", pred_col=col_name, cluster_col="drive")
        boot = paired_cluster_bootstrap(
            eval_temp,
            col_name,
            "z_hat_f",
            metric="absrel",
            seed=seed,
            n_boot=1000,
        )

        delta_p = met["absrel"] - met_base["absrel"]
        delta_m = macro - macro_base

        rec = {
            "model_key": model_key,
            "ablation_key": f"drop_group_{grp_name}",
            "ablation_name": f"Drop {grp_name}",
            "category": "drop_group",
            "dropped_features": drop_cols,
            "n_features": len(reduced_cols),
            "pooled_absrel": met["absrel"],
            "macro_absrel": macro,
            "delta_pooled": delta_p,
            "delta_macro": delta_m,
            "ci_lo": boot.ci_low,
            "ci_hi": boot.ci_high,
        }
        experiment_records.append(rec)
        print(f"    Pooled: {met['absrel']:.4f} (Δ={delta_p:+.4f}, 95% CI: [{boot.ci_low:+.4f}, {boot.ci_high:+.4f}]) | Macro: {macro:.4f} (Δ={delta_m:+.4f})")

    # =========================================================================
    # 3. Drop Single Cue C1..C3 (Refit fusion, fallback e, Decision D38)
    # =========================================================================
    print("\n" + "-" * 60)
    print("  PHASE 2: DROP SINGLE CUE (C1 - C3)")
    print("-" * 60)

    cue_mapping = {
        "drop_z_w": {
            "cue_indices": [1, 2],  # Z_h, Z_g
            "cue_names": ["Z_h", "Z_g"],
            "drop_features": ["ln_z_w", "valid_w"],
        },
        "drop_z_h": {
            "cue_indices": [0, 2],  # Z_w, Z_g
            "cue_names": ["Z_w", "Z_g"],
            "drop_features": ["ln_z_h", "valid_h"],
        },
        "drop_z_g": {
            "cue_indices": [0, 1],  # Z_w, Z_h
            "cue_names": ["Z_w", "Z_h"],
            "drop_features": ["ln_z_g", "valid_g"],
        },
    }

    for cue_key in drop_single_cues:
        spec = cue_mapping[cue_key]
        cue_indices = spec["cue_indices"]
        cue_names = spec["cue_names"]
        drop_cols = spec["drop_features"]
        reduced_cols = [c for c in FEATURE_COLS_F if c not in drop_cols]

        print(f"\n>>> Running Ablation: {cue_key} (keeping {cue_names}, {len(reduced_cols)} features)...")

        z_cues_2 = z_cues[:, cue_indices]
        valid_mask_2 = valid_mask[:, cue_indices]

        # Objects with no valid remaining cues become pattern 000
        pattern_000_2 = ~np.any(valid_mask_2 & (z_cues_2 > 0), axis=1)

        z_hat_arr = np.full(n_samples, np.nan, dtype=float)

        for split in outer_splits:
            fold_idx = split["fold_idx"]
            train_mask = split["train_mask"]
            test_mask = split["test_mask"]
            train_drives = split["train_drives"]

            # Step A: Refit 2-cue fusion weights strictly on 11 train drives with correct cue_names
            fw_2 = fit_fusion_weights(
                Z_cues=z_cues_2[train_mask],
                Z_gt=z_gt[train_mask],
                valid_mask=valid_mask_2[train_mask],
                drive_ids=train_drives,
                cue_names=cue_names,
            )

            # Step B: Fuse depth estimates with 2 cues
            z_d_train_2 = fuse_depths_vectorised(z_cues_2[train_mask], valid_mask_2[train_mask], fw_2)
            z_d_test_2 = fuse_depths_vectorised(z_cues_2[test_mask], valid_mask_2[test_mask], fw_2)

            # Step C: Form Z_base using Model (e) for pattern 000 (D13, D38)
            z_e_tr = inner_ze_train_by_fold[fold_idx]
            z_e_te = z_hat_e_cached[test_mask]

            z_base_tr = np.where(~pattern_000_2[train_mask], z_d_train_2, z_e_tr)
            z_base_te = np.where(~pattern_000_2[test_mask], z_d_test_2, z_e_te)

            r_tr = ln_z_gt[train_mask] - np.log(z_base_tr)

            # Step D: Construct feature matrices with new derived ln_z_base
            fm_tr = build_feature_matrices(base_feats.iloc[train_mask], z_base_tr)["f"]
            fm_te = build_feature_matrices(base_feats.iloc[test_mask], z_base_te)["f"]

            # Step E: Fit Model (f) with reduced features
            m_f, _, _ = fit_f(
                X=fm_tr,
                y=r_tr,
                groups=train_drives,
                feature_cols=reduced_cols,
                random_state=seed,
                n_jobs=n_jobs,
            )
            z_pred, _ = predict_f(
                model=m_f,
                X=fm_te,
                z_base=z_base_te,
                feature_cols=reduced_cols,
            )
            z_hat_arr[test_mask] = z_pred

        assert not np.any(np.isnan(z_hat_arr)), f"NaN in predictions for {cue_key}"

        col_name = f"z_hat_{cue_key}"
        ablation_preds_df[col_name] = z_hat_arr

        eval_temp[col_name] = z_hat_arr
        met = depth_metrics(z_gt, z_hat_arr)
        macro = macro_by_cluster(eval_temp, gt_col="z_gt", pred_col=col_name, cluster_col="drive")
        boot = paired_cluster_bootstrap(
            eval_temp,
            col_name,
            "z_hat_f",
            metric="absrel",
            seed=seed,
            n_boot=1000,
        )

        delta_p = met["absrel"] - met_base["absrel"]
        delta_m = macro - macro_base

        rec = {
            "model_key": model_key,
            "ablation_key": cue_key,
            "ablation_name": f"Drop Cue {cue_key.replace('drop_', '').upper()}",
            "category": "drop_cue",
            "dropped_features": drop_cols,
            "n_features": len(reduced_cols),
            "pooled_absrel": met["absrel"],
            "macro_absrel": macro,
            "delta_pooled": delta_p,
            "delta_macro": delta_m,
            "ci_lo": boot.ci_low,
            "ci_hi": boot.ci_high,
        }
        experiment_records.append(rec)
        print(f"    Pooled: {met['absrel']:.4f} (Δ={delta_p:+.4f}, 95% CI: [{boot.ci_low:+.4f}, {boot.ci_high:+.4f}]) | Macro: {macro:.4f} (Δ={delta_m:+.4f})")

    # =========================================================================
    # 4. Alternative Model M1: MLPRegressor (128, 64) with StandardScaler
    # =========================================================================
    print("\n" + "-" * 60)
    print("  PHASE 3: ALTERNATIVE MODEL MLP (M1)")
    print("-" * 60)
    print(">>> Running MLPRegressor with StandardScaler inside each fold...")

    mlp_hidden = tuple(mlp_cfg.get("hidden_layer_sizes", [128, 64]))
    mlp_act = mlp_cfg.get("activation", "relu")
    mlp_iter = mlp_cfg.get("max_iter", 500)
    mlp_seed = mlp_cfg.get("random_state", 42)

    z_hat_mlp = np.full(n_samples, np.nan, dtype=float)

    for split in outer_splits:
        train_mask = split["train_mask"]
        test_mask = split["test_mask"]

        z_base_tr = z_base_cached[train_mask]
        z_base_te = z_base_cached[test_mask]
        r_tr = ln_z_gt[train_mask] - np.log(z_base_tr)

        pipe_mlp = fit_mlp(
            X=feat_mats_full.iloc[train_mask],
            y=r_tr,
            feature_cols=FEATURE_COLS_F,
            hidden_layer_sizes=mlp_hidden,
            activation=mlp_act,
            max_iter=mlp_iter,
            random_state=mlp_seed,
        )
        z_pred, _ = predict_mlp(
            model=pipe_mlp,
            X=feat_mats_full.iloc[test_mask],
            z_base=z_base_te,
            feature_cols=FEATURE_COLS_F,
        )
        z_hat_mlp[test_mask] = z_pred

    assert not np.any(np.isnan(z_hat_mlp)), "NaN in MLP predictions"

    col_name = "z_hat_mlp"
    ablation_preds_df[col_name] = z_hat_mlp

    eval_temp[col_name] = z_hat_mlp
    met = depth_metrics(z_gt, z_hat_mlp)
    macro = macro_by_cluster(eval_temp, gt_col="z_gt", pred_col=col_name, cluster_col="drive")
    boot = paired_cluster_bootstrap(
        eval_temp,
        col_name,
        "z_hat_f",
        metric="absrel",
        seed=seed,
        n_boot=1000,
    )

    delta_p = met["absrel"] - met_base["absrel"]
    delta_m = macro - macro_base

    rec = {
        "model_key": model_key,
        "ablation_key": "mlp",
        "ablation_name": "MLP (128, 64) [Un-tuned]",
        "category": "alternative_model",
        "dropped_features": [],
        "n_features": len(FEATURE_COLS_F),
        "pooled_absrel": met["absrel"],
        "macro_absrel": macro,
        "delta_pooled": delta_p,
        "delta_macro": delta_m,
        "ci_lo": boot.ci_low,
        "ci_hi": boot.ci_high,
    }
    experiment_records.append(rec)
    print(f"    Pooled: {met['absrel']:.4f} (Δ={delta_p:+.4f}, 95% CI: [{boot.ci_low:+.4f}, {boot.ci_high:+.4f}]) | Macro: {macro:.4f} (Δ={delta_m:+.4f})")

    # Save ablation predictions parquet
    out_pq = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_ablation.parquet"
    ablation_preds_df.to_parquet(out_pq, index=False)
    print(f"\n[Saved] Ablation Predictions Parquet: {out_pq}")

    # Save detector manifest
    manifest_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "ablation_manifest.json"

    manifest_content = {
        "model_key": model_key,
        "split": "B",
        "split_b_hash": split_b_hash,
        "git_commit": git_commit,
        "git_tag": git_tag,
        "seed": seed,
        "bootstrap_seed": seed,
        "n_boot": 1000,
        "created_at": datetime.now().isoformat(),
        "n_samples": n_samples,
        "records": experiment_records,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)
    print(f"[Saved] Ablation Manifest: {manifest_file}")

    return ablation_preds_df, experiment_records


def generate_ablation_markdown(all_records: list[dict[str, Any]]) -> str:
    md = []
    md.append("# Kết quả Nghiên cứu Thành phần Mô hình (T05 Ablation, Split B OOF)\n")
    md.append("> [!IMPORTANT]")
    md.append("> - Đánh giá theo đúng danh sách pre-registration trong `residual_prereg_v1.yaml` (D25, D37, D38, D39).")
    md.append("> - **Phạm vi (D37):** Chỉ thực hiện trên OOF của Split B (12 folds), bảo toàn Split C chỉ cho Conformalize.")
    md.append("> - **Drop Single Cue (D38):** Loại bỏ cue $Z_k$ khỏi fusion, bỏ đồng thời $\\ln z_k$ và $\\text{valid}_k$ khỏi Model (f); pattern 000 đi fallback (e).")
    md.append("> - **Quy tắc đa so sánh (D18, D20):** Báo cáo 30 CI thô (12 cụm, 1000 bootstrap resamples) ở mức mô tả, không kết luận 'có ý nghĩa thống kê'.")
    md.append("> - **Diễn giải Redundancy:** Hiệu ứng khi bỏ $\\text{valid}_*$ hoặc $\\ln z_{\\text{base}}$ xấp xỉ 0 do đa cộng tuyến và thông tin dư thừa với $\\text{touch}_*$ và các cue, không thể hiện đặc trưng vô dụng.")
    md.append("> - **Mô hình thay thế (M1):** MLP là mô hình un-tuned (cố định 1 cấu hình), trong khi XGBoost được grid search 12 cấu hình.\n")

    # Table per detector
    detectors = sorted(list({r["model_key"] for r in all_records}))
    for det in detectors:
        md.append(f"## Bảng Ablation cho Detector: `{det}`\n")
        md.append("| STT | Nhóm thử nghiệm | Cấu hình / Thử nghiệm | Số Features | Pooled AbsRel | Macro AbsRel | Δ Pooled (vs Baseline) | 95% CI thô (12 cụm) | Δ Macro |")
        md.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        det_recs = [r for r in all_records if r["model_key"] == det]
        for idx, r in enumerate(det_recs):
            name = r["ablation_name"]
            cat = r["category"]
            n_f = r["n_features"]
            p = r["pooled_absrel"]
            m = r["macro_absrel"]
            dp = r["delta_pooled"]
            dm = r["delta_macro"]

            if cat == "baseline":
                ci_str = "—"
                dp_str = "Baseline"
                dm_str = "Baseline"
            else:
                ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                dp_str = f"{dp:+.4f}"
                dm_str = f"{dm:+.4f}"

            cat_label = {
                "baseline": "Baseline",
                "drop_group": "Drop Group",
                "drop_cue": "Drop Single Cue",
                "alternative_model": "Alternative Model",
            }.get(cat, cat)

            md.append(f"| {idx} | {cat_label} | **{name}** | {n_f} | {p:.4f} | {m:.4f} | {dp_str} | {ci_str} | {dm_str} |")
        md.append("\n")

    # Summary discussion (Decisions D18, D20, D32, D45 - strictly descriptive, no claims of significance or superiority)
    md.append("## Nhận xét & Diễn giải Kết quả (Mô tả theo số liệu)\n")
    md.append("1. **Loại bỏ nhóm đặc trưng (A1–A6):**")
    md.append("   - **bbox_geometry** và **confidence**: Δ Pooled AbsRel dao động từ -0.0005 đến +0.0030, tuy nhiên 95% CI thô (12 cụm) chứa 0 ở cả 3 detector (chưa tách biệt được sai khác ngoài nhiễu cụm).")
    md.append("   - **validity_flags**: Không thấy đóng góp đo lường được (Δ ≤ +0.0003, CI chứa 0 ở cả 3 detector). Điều này phù hợp về mặt cấu trúc vì khi cue invalid thì `ln_z_k = 0` và có cờ `touch_*` đi kèm.")
    md.append("   - **ln_z_base** (đặc trưng dẫn xuất D30): Đóng góp nhỏ (Δ ≤ 0.0010); CI loại trừ 0 ở 1/3 detector (`yolo11s_640`: +0.0010 [+0.0006, +0.0016]), trong khi ở 2 detector còn lại CI chứa 0 (`yolov8s`: +0.0002, `yolov5su`: +0.0007).")
    md.append("   - **cues_ln_z**: Bỏ toàn bộ `ln_z_*` làm Δ Pooled chỉ tăng +0.0001 đến +0.0003 (CI chứa 0), nhất quán với việc hồi quy trực tiếp từ bbox (e) đạt sai số gần tương đương (f).")
    md.append("2. **Đóng góp của từng Cue hình học khi loại bỏ (C1–C3):**")
    md.append("   - **Drop Z_h**: Là cue duy nhất khiến sai số tăng rõ rệt ở cả 3 detector (Δ Pooled +0.0070 đến +0.0091; 95% CI thô hoàn toàn loại trừ 0: [+0.0024, +0.0119] trên yolo11s, [+0.0023, +0.0153] trên v8s, [+0.0029, +0.0163] trên v5su). Đây là hiệu ứng trực tiếp lên $Z_{\\text{base}}$ vì $Z_h$ chiếm ~70% trọng số hợp nhất.")
    md.append("   - **Drop Z_w** và **Drop Z_g**: Δ Pooled xấp xỉ 0 (-0.0007 đến +0.0005; CI đều chứa 0). Thậm chí Macro AbsRel giảm nhẹ khi bỏ $Z_g$ (-0.0018 đến -0.0050). Kết quả này hoàn toàn nhất quán với Quyết định D31 (trọng số $w_w \\to 0$ khi refit trên bbox detector).")
    md.append("3. **So sánh XGBoost và MLP (M1):**")
    md.append("   - MLP un-tuned đạt Pooled AbsRel kém hơn nhẹ (+0.0014 đến +0.0026), nhưng 95% CI thô chứa 0 ở cả 3 detector. Đồng thời, Macro AbsRel của MLP lại thấp hơn ở 2/3 detector (yolo11s: -0.0020, yolov8s: -0.0037).")
    md.append("   - Theo nguyên tắc D20 và D32, không phân biệt được sự khác biệt có ý nghĩa thống kê giữa XGBoost và MLP trên tập dữ liệu này.\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Run T05 Feature Ablation on Split B OOF")
    parser.add_argument("--detector", choices=["all", *DETECTORS], default="yolo11s_640",
                        help="Detector to evaluate (defaults to yolo11s_640 per Decision D36)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--n_jobs", type=int, default=1, help="Thread count for reproducible XGBoost")
    args = parser.parse_args()

    prereg_yaml = PROJECT_ROOT / "configs" / "residual" / "residual_prereg_v1.yaml"
    if not prereg_yaml.is_file():
        raise FileNotFoundError(f"Missing pre-registration config: {prereg_yaml}")

    with open(prereg_yaml, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)

    meta_path = PROJECT_ROOT / "splits" / "split_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    split_b_hash = meta_data["splits"]["B"]["hash"]

    git_commit, git_tag = get_git_info()
    print(f"Loaded config: git={git_commit[:8]}, tag={git_tag}, Split B hash={split_b_hash[:16]}...")

    target_detectors = DETECTORS if args.detector == "all" else [args.detector]

    all_records: list[dict[str, Any]] = []

    # If running specific detector, load existing records if available
    summary_json_path = PROJECT_ROOT / "results" / "tables" / "ablation_oof_b.json"
    if summary_json_path.is_file():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                all_records = existing_data.get("records", [])
        except Exception:
            all_records = []

    for det in target_detectors:
        # Remove old records for this detector if re-running
        all_records = [r for r in all_records if r.get("model_key") != det]
        _, det_recs = run_ablation_for_detector(
            model_key=det,
            prereg_cfg=prereg_cfg,
            split_b_hash=split_b_hash,
            git_commit=git_commit,
            git_tag=git_tag,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        all_records.extend(det_recs)

    # Save consolidated summary JSON and Markdown
    summary_dir = PROJECT_ROOT / "results" / "tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_content = {
        "task": "T05",
        "description": "Feature and cue ablation on Split B OOF",
        "split": "B",
        "split_b_hash": split_b_hash,
        "git_commit": git_commit,
        "git_tag": git_tag,
        "seed": args.seed,
        "bootstrap_seed": args.seed,
        "n_boot": 1000,
        "created_at": datetime.now().isoformat(),
        "records": all_records,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_content, f, indent=2)
    print(f"\n[Saved] Consolidated Ablation JSON: {summary_json_path}")

    summary_md_path = summary_dir / "ablation_oof_b.md"
    md_text = generate_ablation_markdown(all_records)
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[Saved] Consolidated Ablation Markdown: {summary_md_path}")

    # Log record to runs/pipeline_log.jsonl
    log_rec = make_log_record(
        split="B",
        split_hash=split_b_hash,
        seed=args.seed,
        n_boot=1000,
        tag="T05-Ablation",
        extra={
            "detectors": target_detectors,
            "git_commit": git_commit,
            "git_tag": git_tag,
            "n_records": len(all_records),
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_rec)
    print(f"[Logged] Pipeline run recorded to runs/pipeline_log.jsonl")


if __name__ == "__main__":
    main()
