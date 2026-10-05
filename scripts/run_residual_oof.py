"""
scripts/run_residual_oof.py: Main runner for residual models and nested LODO on Split B.
(Decisions D13, D16b, D24, D25, D28, D29, D30, D33, D34, D36).

Workflow:
1. For each detector:
   - Loads features, eval, and cues parquets for Split B.
   - Executes nested LODO (12 folds over drives, with inner-LODO for Z_e on pattern 000).
   - Saves results/datasets/{model_key}_B_oof.parquet.
   - Fits full Split B models and serializes to runs/residual/{model_key}/:
     * model_f0.joblib
     * model_f.json
     * model_e.json
     * manifest.json (SHA256, seed, split hash, git commit, tag prereg, hyperparams).
2. Evaluates Gate T04:
   - Primary rule: (f) AbsRel < (d) AbsRel on common support (where d is valid), pooled AND macro.
   - Evaluates (d), (f0), (f), (e) pooled & macro.
   - Evaluates macro on drives with n >= 30.
   - Computes paired cluster bootstrap (12 clusters, 1000 resamples) for (f) vs (d), (f) vs (f0), (f) vs (e).
   - Computes exact one-sided sign test across 12 drives.
   - Checks f0 approx f criterion (|Delta AbsRel| < 0.005 and CI contains 0).
   - Range breakdown: 0-10, 10-20, 20-30, 30-50, >50, and merged >30 (with n, n_valid, and '*' if n < 100).
   - Evaluates secondary row: all matched samples (d + fallback Z_e).
3. Writes results/tables/residual_oof_b.json and residual_oof_b.md.
4. Appends run record to runs/pipeline_log.jsonl.
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

# Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import (
    BAND_EDGES,
    BAND_LABELS,
    FAR_MERGED_LABEL,
    FAR_MERGED_START,
    MIN_N,
    append_jsonl,
    depth_metrics,
    macro_by_cluster,
    make_log_record,
    paired_cluster_bootstrap,
    sign_test_one_sided,
)
from src.pipeline.oof import fit_full_b_models, run_nested_lodo_b
from src.residual.models import (
    save_model_e,
    save_model_f,
    save_model_f0,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]


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


def run_model_pipeline(
    model_key: str,
    prereg_cfg: dict[str, Any],
    split_b_hash: str,
    git_commit: str,
    git_tag: str,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"  PROCESSING DETECTOR: {model_key}")
    print("=" * 78)

    feat_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_features.parquet"
    eval_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_eval.parquet"
    cues_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_cues.parquet"

    for p in [feat_path, eval_path, cues_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset: {p}")

    features_df = pd.read_parquet(feat_path)
    eval_df = pd.read_parquet(eval_path)
    cues_df = pd.read_parquet(cues_path)

    n_samples = len(features_df)
    n_gt = len(eval_df)
    print(f"Loaded Split B: {n_samples} matched detections across {cues_df['drive'].nunique()} drives.")

    # 1. Run nested LODO
    print("\n>>> Running Nested Leave-One-Drive-Out (LODO)...")
    oof_df, fold_records = run_nested_lodo_b(
        features_df=features_df,
        eval_df=eval_df,
        cues_df=cues_df,
        min_train_drives=3,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=True,
    )

    # Save OOF parquet
    oof_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"\n[Saved] OOF Parquet: {oof_path}")

    # Determine consensus / best hyperparams from fold records
    f_combos = [str(r["best_params_f"]) for r in fold_records]
    most_common_combo_str = max(set(f_combos), key=f_combos.count)
    best_params_f = next(r["best_params_f"] for r in fold_records if str(r["best_params_f"]) == most_common_combo_str)
    best_alpha_f0 = float(np.median([r["best_alpha_f0"] for r in fold_records]))

    print(f"Consensus best hyperparams for (f): {best_params_f}")
    print(f"Consensus best alpha for (f0): {best_alpha_f0}")

    # 2. Fit full Split B models
    print("\n>>> Fitting full models across all Split B drives...")
    model_f0, model_f, model_e, full_fw = fit_full_b_models(
        features_df=features_df,
        eval_df=eval_df,
        cues_df=cues_df,
        best_params_f=best_params_f,
        best_alpha_f0=best_alpha_f0,
        random_state=seed,
        n_jobs=n_jobs,
    )

    # Save full models
    model_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    f0_path = model_dir / "model_f0.joblib"
    f_path = model_dir / "model_f.json"
    e_path = model_dir / "model_e.json"
    manifest_path = model_dir / "manifest.json"

    save_model_f0(model_f0, f0_path)
    save_model_f(model_f, f_path)
    save_model_e(model_e, e_path)

    sha_f0 = sha256_file(f0_path)
    sha_f = sha256_file(f_path)
    sha_e = sha256_file(e_path)

    manifest_data = {
        "model_key": model_key,
        "split": "B",
        "split_b_hash": split_b_hash,
        "git_commit": git_commit,
        "git_tag": git_tag,
        "seed": seed,
        "n_jobs": n_jobs,
        "n_samples": n_samples,
        "created_at": datetime.now().isoformat(),
        "best_params_f": best_params_f,
        "best_alpha_f0": best_alpha_f0,
        "models": {
            "model_f0": {"file": "model_f0.joblib", "sha256": sha_f0},
            "model_f": {"file": "model_f.json", "sha256": sha_f},
            "model_e": {"file": "model_e.json", "sha256": sha_e},
        },
        "fold_summary": fold_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[Saved] Models and manifest in {model_dir}")

    # 3. Comprehensive Evaluation
    print("\n>>> Evaluating Gate T04 and metric comparisons...")

    # Join oof predictions with eval_df
    eval_joined = eval_df.copy()
    for col in ["z_d", "z_base", "z_hat_f0", "z_hat_f", "z_hat_e", "fallback_flag"]:
        eval_joined[col] = oof_df[col]

    z_gt_all = eval_joined["z_gt"].to_numpy(dtype=float)

    # Common support: where (d) is valid (pattern != 000)
    common_mask = ~eval_joined["fallback_flag"].to_numpy(dtype=bool)
    n_common = int(common_mask.sum())
    n_fallback = n_samples - n_common

    eval_common = eval_joined.loc[common_mask].copy()
    z_gt_com = eval_common["z_gt"].to_numpy(dtype=float)

    z_d_com = eval_common["z_d"].to_numpy(dtype=float)
    z_f0_com = eval_common["z_hat_f0"].to_numpy(dtype=float)
    z_f_com = eval_common["z_hat_f"].to_numpy(dtype=float)
    z_e_com = eval_common["z_hat_e"].to_numpy(dtype=float)

    # Pooled metrics on common support
    metrics_d_com = depth_metrics(z_gt_com, z_d_com)
    metrics_f0_com = depth_metrics(z_gt_com, z_f0_com)
    metrics_f_com = depth_metrics(z_gt_com, z_f_com)
    metrics_e_com = depth_metrics(z_gt_com, z_e_com)

    # Macro metrics on common support (unweighted average across 12 drives, D18)
    macro_d = macro_by_cluster(eval_common.assign(z_pred=z_d_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive")
    macro_f0 = macro_by_cluster(eval_common.assign(z_pred=z_f0_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive")
    macro_f = macro_by_cluster(eval_common.assign(z_pred=z_f_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive")
    macro_e = macro_by_cluster(eval_common.assign(z_pred=z_e_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive")

    # Macro metrics on drives with n >= 30
    macro_d_n30 = macro_by_cluster(eval_common.assign(z_pred=z_d_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive", min_valid=30)
    macro_f0_n30 = macro_by_cluster(eval_common.assign(z_pred=z_f0_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive", min_valid=30)
    macro_f_n30 = macro_by_cluster(eval_common.assign(z_pred=z_f_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive", min_valid=30)
    macro_e_n30 = macro_by_cluster(eval_common.assign(z_pred=z_e_com), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive", min_valid=30)

    # Full matched support (all 3523 samples): d + fallback Z_e vs f0, f, e
    # Notice: on pattern 000, baseline (d + fallback Z_e) is exactly z_base!
    z_d_full_matched = eval_joined["z_base"].to_numpy(dtype=float)
    metrics_d_full = depth_metrics(z_gt_all, z_d_full_matched)
    metrics_f0_full = depth_metrics(z_gt_all, eval_joined["z_hat_f0"].to_numpy())
    metrics_f_full = depth_metrics(z_gt_all, eval_joined["z_hat_f"].to_numpy())
    metrics_e_full = depth_metrics(z_gt_all, eval_joined["z_hat_e"].to_numpy())

    # Paired cluster bootstrap (1000 resamples, 12 clusters)
    print("Running paired cluster bootstrap...")
    boot_f_minus_d = paired_cluster_bootstrap(
        df=eval_common,
        pred_col_a="z_hat_f",
        pred_col_b="z_d",
        gt_col="z_gt",
        cluster_col="drive",
        metric="absrel",
        n_boot=1000,
        seed=seed,
    )
    boot_f_minus_f0 = paired_cluster_bootstrap(
        df=eval_common,
        pred_col_a="z_hat_f",
        pred_col_b="z_hat_f0",
        gt_col="z_gt",
        cluster_col="drive",
        metric="absrel",
        n_boot=1000,
        seed=seed,
    )
    boot_f_minus_e = paired_cluster_bootstrap(
        df=eval_common,
        pred_col_a="z_hat_f",
        pred_col_b="z_hat_e",
        gt_col="z_gt",
        cluster_col="drive",
        metric="absrel",
        n_boot=1000,
        seed=seed,
    )

    # Sign test across drives for (f) vs (d)
    drive_wins = 0
    drive_losses = 0
    drive_ties = 0
    for d_name, d_grp in eval_common.groupby("drive"):
        g_d = d_grp["z_gt"].to_numpy()
        d_val = np.mean(np.abs(d_grp["z_d"].to_numpy() - g_d) / g_d)
        f_val = np.mean(np.abs(d_grp["z_hat_f"].to_numpy() - g_d) / g_d)
        if f_val < d_val - 1e-6:
            drive_wins += 1
        elif f_val > d_val + 1e-6:
            drive_losses += 1
        else:
            drive_ties += 1

    p_value_sign_test = sign_test_one_sided(drive_wins, drive_losses)

    # Gate T04 checks:
    # 1. Primary rule: (f) AbsRel < (d) AbsRel on common support, pooled AND macro
    gate_pooled_pass = bool(metrics_f_com["absrel"] < metrics_d_com["absrel"])
    gate_macro_pass = bool(macro_f < macro_d)
    primary_gate_pass = gate_pooled_pass and gate_macro_pass

    # 2. Check f0 approx f rule: CI contains 0 AND |Delta AbsRel pooled| < 0.005
    delta_f_f0_pooled = float(metrics_f_com["absrel"] - metrics_f0_com["absrel"])
    f0_approx_f = bool((boot_f_minus_f0.ci_low <= 0.0 <= boot_f_minus_f0.ci_high) and (abs(delta_f_f0_pooled) < 0.005))

    # Range breakdown for Model (f) and Baseline (d) on common support
    bands_res = []
    for low, high, label in zip(BAND_EDGES[:-1], BAND_EDGES[1:], BAND_LABELS):
        b_mask = (z_gt_com >= low) & (z_gt_com < high)
        n_b = int(b_mask.sum())
        if n_b > 0:
            m_d_b = depth_metrics(z_gt_com[b_mask], z_d_com[b_mask])
            m_f_b = depth_metrics(z_gt_com[b_mask], z_f_com[b_mask])
            bands_res.append({
                "band": label,
                "n": n_b,
                "low_n": n_b < MIN_N,
                "absrel_d": m_d_b["absrel"],
                "absrel_f": m_f_b["absrel"],
                "delta": m_f_b["absrel"] - m_d_b["absrel"],
            })

    # Merged >30 row
    far_mask = (z_gt_com >= FAR_MERGED_START)
    n_far = int(far_mask.sum())
    if n_far > 0:
        m_d_far = depth_metrics(z_gt_com[far_mask], z_d_com[far_mask])
        m_f_far = depth_metrics(z_gt_com[far_mask], z_f_com[far_mask])
        bands_res.append({
            "band": FAR_MERGED_LABEL,
            "n": n_far,
            "low_n": n_far < MIN_N,
            "absrel_d": m_d_far["absrel"],
            "absrel_f": m_f_far["absrel"],
            "delta": m_f_far["absrel"] - m_d_far["absrel"],
        })

    result_summary = {
        "model_key": model_key,
        "n_samples": n_samples,
        "n_common": n_common,
        "n_fallback": n_fallback,
        "gate_status": {
            "primary_gate_pass": primary_gate_pass,
            "gate_pooled_pass": gate_pooled_pass,
            "gate_macro_pass": gate_macro_pass,
            "f0_approx_f": f0_approx_f,
        },
        "common_support": {
            "d": {"pooled": metrics_d_com, "macro": macro_d, "macro_n30": macro_d_n30},
            "f0": {"pooled": metrics_f0_com, "macro": macro_f0, "macro_n30": macro_f0_n30},
            "f": {"pooled": metrics_f_com, "macro": macro_f, "macro_n30": macro_f_n30},
            "e": {"pooled": metrics_e_com, "macro": macro_e, "macro_n30": macro_e_n30},
        },
        "full_matched_support": {
            "d_plus_ze": {"pooled": metrics_d_full},
            "f0": {"pooled": metrics_f0_full},
            "f": {"pooled": metrics_f_full},
            "e": {"pooled": metrics_e_full},
        },
        "bootstrap": {
            "f_minus_d": {
                "diff": boot_f_minus_d.estimate,
                "ci_low": boot_f_minus_d.ci_low,
                "ci_high": boot_f_minus_d.ci_high,
                "ci_label": "CI thô (12 cụm)",
            },
            "f_minus_f0": {
                "diff": boot_f_minus_f0.estimate,
                "ci_low": boot_f_minus_f0.ci_low,
                "ci_high": boot_f_minus_f0.ci_high,
                "ci_label": "CI thô (12 cụm)",
            },
            "f_minus_e": {
                "diff": boot_f_minus_e.estimate,
                "ci_low": boot_f_minus_e.ci_low,
                "ci_high": boot_f_minus_e.ci_high,
                "ci_label": "CI thô (12 cụm)",
            },
        },
        "sign_test": {
            "wins": drive_wins,
            "losses": drive_losses,
            "ties": drive_ties,
            "p_value": p_value_sign_test,
        },
        "bands": bands_res,
    }

    # Print summary table
    print("\n" + "=" * 78)
    print(f"  GATE T04 & EVALUATION SUMMARY: {model_key}")
    print("=" * 78)
    print(f"Common support samples: N = {n_common} (Pattern != 000, d valid)")
    print(f"Fallback samples:       N = {n_fallback} (Pattern == 000, fallback to Z_e)")
    print("-" * 78)
    print(f"{'Method':<20} | {'Pooled AbsRel':<14} | {'Macro AbsRel':<14} | {'Macro n>=30':<14}")
    print("-" * 78)
    print(f"{'(d) Z_d LODO OOF':<20} | {metrics_d_com['absrel']:<14.4f} | {macro_d:<14.4f} | {macro_d_n30:<14.4f}")
    print(f"{'(f0) Ridge':<20} | {metrics_f0_com['absrel']:<14.4f} | {macro_f0:<14.4f} | {macro_f0_n30:<14.4f}")
    print(f"{'(f) XGBoost Residual':<20} | {metrics_f_com['absrel']:<14.4f} | {macro_f:<14.4f} | {macro_f_n30:<14.4f}")
    print(f"{'(e) XGBoost Direct':<20} | {metrics_e_com['absrel']:<14.4f} | {macro_e:<14.4f} | {macro_e_n30:<14.4f}")
    print("-" * 78)
    print(f"Primary Gate ((f) < (d)): Pooled: {metrics_f_com['absrel']:.4f} vs {metrics_d_com['absrel']:.4f} -> {'PASS' if gate_pooled_pass else 'FAIL'}")
    print(f"                          Macro:  {macro_f:.4f} vs {macro_d:.4f} -> {'PASS' if gate_macro_pass else 'FAIL'}")
    print(f"Primary Gate Overall:     {'PASS' if primary_gate_pass else 'FAIL'}")
    print(f"f0 approx f check:        {'TRUE (mô hình đơn giản đủ)' if f0_approx_f else 'FALSE'}")
    print(f"Bootstrap (f - d):        {boot_f_minus_d.estimate:+.4f} [{boot_f_minus_d.ci_low:+.4f}, {boot_f_minus_d.ci_high:+.4f}] (CI thô (12 cụm))")
    print(f"Bootstrap (f - f0):       {boot_f_minus_f0.estimate:+.4f} [{boot_f_minus_f0.ci_low:+.4f}, {boot_f_minus_f0.ci_high:+.4f}]")
    print(f"Sign test (f vs d):       {drive_wins} wins / {drive_losses} losses / {drive_ties} ties (p = {p_value_sign_test:.4f})")
    print("=" * 78)

    return result_summary


def write_markdown_report(all_results: dict[str, Any], output_path: Path):
    lines = [
        "# Kết quả Đánh giá Mô hình Residual (T04, Split B OOF)",
        "",
        "> [!IMPORTANT]",
        "> Đánh giá theo đúng cấu hình pre-registration `prereg-residual-v1` (D25, D33, D34).",
        "> - Mọi tuning và selection chỉ thực hiện trên Split B (D24).",
        "> - Baseline (d) là Z_d OOF LODO theo drive (D29), không in-sample.",
        "> - So sánh chính thực hiện trên tập chung nơi (d) hợp lệ (pattern != 000).",
        "",
        "## 1. Bảng Tổng hợp Gate T04 & Các Phương pháp (Tập chung nơi d hợp lệ)",
        "",
        "| Detector | Chỉ số | (d) Z_d LODO | (f0) Ridge | (f) XGBoost | (e) Direct | Δ(f − d) | 95% CI thô (12 cụm) | Gate (f < d) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_key, res in all_results.items():
        com = res["common_support"]
        boot = res["bootstrap"]["f_minus_d"]
        gate = res["gate_status"]
        pass_str = "✅ ĐẠT" if gate["primary_gate_pass"] else "❌ KHÔNG ĐẠT"

        p_d = com["d"]["pooled"]["absrel"]
        p_f0 = com["f0"]["pooled"]["absrel"]
        p_f = com["f"]["pooled"]["absrel"]
        p_e = com["e"]["pooled"]["absrel"]

        m_d = com["d"]["macro"]
        m_f0 = com["f0"]["macro"]
        m_f = com["f"]["macro"]
        m_e = com["e"]["macro"]

        delta_pooled = p_f - p_d
        delta_macro = m_f - m_d
        ci_str = f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]"

        lines.append(f"| **{model_key}** | **Pooled AbsRel** | {p_d:.4f} | {p_f0:.4f} | **{p_f:.4f}** | {p_e:.4f} | {delta_pooled:+.4f} | {ci_str} | {pass_str} |")
        lines.append(f"| | **Macro AbsRel** | {m_d:.4f} | {m_f0:.4f} | **{m_f:.4f}** | {m_e:.4f} | {delta_macro:+.4f} | — | |")
        lines.append(f"| | Macro (n≥30) | {com['d']['macro_n30']:.4f} | {com['f0']['macro_n30']:.4f} | {com['f']['macro_n30']:.4f} | {com['e']['macro_n30']:.4f} | — | — | |")

    lines.extend([
        "",
        "## 2. Kiểm định f0 ≈ f (Mô hình tuyến tính đơn giản)",
        "",
        "| Detector | Δ AbsRel(f − f0) | 95% CI thô (12 cụm) | CI chứa 0 | |Δ| < 0.005 | Kết luận f0 ≈ f |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model_key, res in all_results.items():
        boot_f0 = res["bootstrap"]["f_minus_f0"]
        delta_f0 = boot_f0["diff"]
        ci_contains_zero = (boot_f0["ci_low"] <= 0.0 <= boot_f0["ci_high"])
        small_diff = abs(delta_f0) < 0.005
        f0_approx = res["gate_status"]["f0_approx_f"]
        f0_str = "Có (mô hình đơn giản đủ)" if f0_approx else "Không"
        lines.append(f"| **{model_key}** | {delta_f0:+.4f} | [{boot_f0['ci_low']:+.4f}, {boot_f0['ci_high']:+.4f}] | {'Có' if ci_contains_zero else 'Không'} | {'Có' if small_diff else 'Không'} | **{f0_str}** |")

    lines.extend([
        "",
        "## 3. Kiểm định Sign Test theo Drive ((f) vs (d))",
        "",
        "| Detector | Số Drive Thắng (Wins) | Số Drive Thua (Losses) | Hòa (Ties) | p-value (one-sided binom) | Đạt p < 0.05 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model_key, res in all_results.items():
        st = res["sign_test"]
        p_val = st["p_value"]
        p_pass = "Đạt" if p_val < 0.05 else "Chưa (p >= 0.05)"
        lines.append(f"| **{model_key}** | {st['wins']} | {st['losses']} | {st['ties']} | {p_val:.4f} | {p_pass} |")

    lines.extend([
        "",
        "## 4. Dòng Phụ Đối Chiếu: Toàn bộ Mẫu Matched (N = 3.523..3.427, d + fallback Z_e)",
        "",
        "| Detector | N matched / N GT | (d + Z_e fallback) AbsRel | (f0) Ridge AbsRel | (f) XGBoost AbsRel | (e) Direct AbsRel |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model_key, res in all_results.items():
        full = res["full_matched_support"]
        n_samp = res["n_samples"]
        lines.append(
            f"| **{model_key}** | {n_samp} / 4776 | "
            f"{full['d_plus_ze']['pooled']['absrel']:.4f} | "
            f"{full['f0']['pooled']['absrel']:.4f} | "
            f"**{full['f']['pooled']['absrel']:.4f}** | "
            f"{full['e']['pooled']['absrel']:.4f} |"
        )

    lines.extend([
        "",
        "## 5. Phân rã theo Dải Cự ly (Tập chung nơi d hợp lệ)",
        "",
        "| Detector | Dải (m) | n | Cờ low_n (*) | Baseline (d) AbsRel | Residual (f) AbsRel | Δ(f − d) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model_key, res in all_results.items():
        for b in res["bands"]:
            star = "\\*" if b["low_n"] else ""
            lines.append(
                f"| {model_key} | {b['band']} | {b['n']} | {star} | "
                f"{b['absrel_d']:.4f} | {b['absrel_f']:.4f} | {b['delta']:+.4f} |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Saved] Markdown Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run residual nested LODO OOF on Split B.")
    parser.add_argument("--model", type=str, default="yolo11s_640", choices=DETECTORS + ["all"],
                        help="Detector model key to run (Decision D36: yolo11s_640 first).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Thread count (default: 1 for determinism)")
    args = parser.parse_args()

    # Load prereg config
    prereg_yaml_path = PROJECT_ROOT / "configs" / "residual" / "residual_prereg_v1.yaml"
    with open(prereg_yaml_path, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)

    # Load split metadata
    split_meta_path = PROJECT_ROOT / "splits" / "split_metadata.json"
    with open(split_meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    split_b_hash = split_meta["splits"]["B"]["hash"]

    git_commit, git_tag = get_git_info()
    print(f"Git environment: commit={git_commit[:7]}, tag={git_tag}")

    models_to_run = [args.model] if args.model != "all" else DETECTORS

    # Check if existing results table exists to accumulate
    json_table_path = PROJECT_ROOT / "results" / "tables" / "residual_oof_b.json"
    md_table_path = PROJECT_ROOT / "results" / "tables" / "residual_oof_b.md"

    accumulated_results: dict[str, Any] = {}
    if json_table_path.is_file():
        try:
            with open(json_table_path, "r", encoding="utf-8") as f:
                accumulated_results = json.load(f)
        except Exception:
            accumulated_results = {}

    for m in models_to_run:
        res = run_model_pipeline(
            model_key=m,
            prereg_cfg=prereg_cfg,
            split_b_hash=split_b_hash,
            git_commit=git_commit,
            git_tag=git_tag,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        accumulated_results[m] = res

        # Log record
        log_record = make_log_record(
            split="B",
            split_hash=split_b_hash,
            seed=args.seed,
            n_boot=1000,
            tag=git_tag,
            extra={
                "task": "T04",
                "model_key": m,
                "n_samples": res["n_samples"],
                "n_common": res["n_common"],
                "primary_gate_pass": res["gate_status"]["primary_gate_pass"],
                "f_pooled_absrel": res["common_support"]["f"]["pooled"]["absrel"],
                "d_pooled_absrel": res["common_support"]["d"]["pooled"]["absrel"],
                "f_macro_absrel": res["common_support"]["f"]["macro"],
                "d_macro_absrel": res["common_support"]["d"]["macro"],
                "f0_approx_f": res["gate_status"]["f0_approx_f"],
            }
        )
        append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_record)

    # Save accumulated tables
    with open(json_table_path, "w", encoding="utf-8") as f:
        json.dump(accumulated_results, f, indent=2)
    print(f"\n[Saved] JSON Report: {json_table_path}")

    write_markdown_report(accumulated_results, md_table_path)


if __name__ == "__main__":
    main()
