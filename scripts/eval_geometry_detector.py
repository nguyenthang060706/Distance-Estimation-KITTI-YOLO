"""
scripts/eval_geometry_detector.py: Geometry stage execution on detector bboxes and error decomposition (Decisions D10, D14, D17, D18, D20, D29).

Outputs:
- results/datasets/{model_key}_{B|C}_cues.parquet
- results/tables/geometry_on_detector_bbox.json
- results/tables/geometry_on_detector_bbox.md
- runs/pipeline_log.jsonl (1 log entry)

Verifies:
- Regression test on GT bbox of full Split B (4776 objects): pooled OOF AbsRel ≈ 0.0609, in-sample ≈ 0.0605 (diff <= 0.002).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import (
    append_jsonl,
    band_masks,
    evaluate_report,
    macro_by_cluster,
    make_log_record,
    paired_cluster_bootstrap,
)
from src.geometry.geometric_cues import (
    BORDER_EPS,
    CameraIntrinsics,
    compute_cues_batch,
)
from src.geometry.fusion import (
    fit_fusion_weights,
    fuse_depths_vectorised,
)
from src.pipeline.geometry_stage import (
    add_cues,
    fit_fusion_lodo,
    fuse_with_weights,
    load_frozen_gt_fusion_weights,
    load_geometry_priors,
)
from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_split

MODEL_KEYS = ["yolov8s_640", "yolo11s_640", "yolov5su_640"]
SPLITS = ["B", "C"]
BAND_NAMES = ["0-10", "10-20", "20-30", "30-50", ">50", ">30"]


def get_git_info() -> tuple[str, bool]:
    """Retrieve current git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()
        return commit, len(status) > 0
    except Exception:
        return "unknown", False


# ---------------------------------------------------------------------------
# 1. Regression test on GT bbox of full Split B (4776 objects)
# ---------------------------------------------------------------------------
def run_gt_bbox_regression_check(priors) -> dict[str, float]:
    """
    Run geometric cues and log-space fusion on full Split B GT bboxes (4776 objects).
    Must reproduce pooled OOF AbsRel ≈ 0.0609 and in-sample ≈ 0.0605 (diff <= 0.002).
    """
    print("\n" + "=" * 80)
    print("STEP 1: REGRESSION CHECK ON FULL SPLIT B GT BBOXES (4,776 OBJECTS)")
    print("=" * 80)

    loader = KITTILoader("data/kitti")
    split_b_ids = load_split("splits", "B", allow_test=False)

    records = []
    for fid in tqdm(split_b_ids, desc="Loading Split B GT"):
        frame = loader.load_frame(fid)
        calib = frame.calib
        drive = frame.drive
        with Image.open(frame.image_path) as img:
            img_w, img_h = img.size

        intrinsics = CameraIntrinsics(
            fx=float(calib.fx), fy=float(calib.fy), cx=float(calib.cx), cy=float(calib.cy)
        )

        for obj in frame.objects:
            if obj.obj_class != "Car" or not obj.passes_hard_filter():
                continue
            if obj.depth <= 0:
                continue

            records.append({
                "frame_id": fid,
                "drive": drive,
                "x1": float(obj.bbox[0]),
                "y1": float(obj.bbox[1]),
                "x2": float(obj.bbox[2]),
                "y2": float(obj.bbox[3]),
                "z_gt": float(obj.depth),
                "fx": float(calib.fx),
                "fy": float(calib.fy),
                "cx": float(calib.cx),
                "cy": float(calib.cy),
                "img_w": img_w,
                "img_h": img_h,
            })

    df_gt = pd.DataFrame(records)
    n_gt = len(df_gt)
    print(f"Loaded {n_gt} GT Car Hard objects from Split B.")

    if n_gt != 4776:
        raise ValueError(f"Expected 4776 GT objects on Split B, got {n_gt}!")

    # Add cues
    df_gt_cues = add_cues(df_gt, priors=priors, eps=BORDER_EPS)

    # In-sample fusion
    z_cues = df_gt_cues[["z_w", "z_h", "z_g"]].to_numpy(dtype=float)
    valid_mask = df_gt_cues[["valid_w", "valid_h", "valid_g"]].to_numpy(dtype=bool)
    z_gt_arr = df_gt["z_gt"].to_numpy(dtype=float)
    drives = df_gt["drive"].to_numpy(dtype=str)

    fw_insample = fit_fusion_weights(z_cues, z_gt_arr, valid_mask, drives)
    z_d_insample = fuse_depths_vectorised(z_cues, valid_mask, fw_insample)

    valid_in = np.isfinite(z_d_insample) & (z_d_insample > 0)
    insample_absrel = float(np.mean(np.abs(z_d_insample[valid_in] - z_gt_arr[valid_in]) / z_gt_arr[valid_in]))

    # LODO OOF fusion
    z_d_oof, _, _ = fit_fusion_lodo(df_gt_cues, z_gt_arr, min_train_drives=3)
    valid_oof = np.isfinite(z_d_oof) & (z_d_oof > 0)
    oof_absrel = float(np.mean(np.abs(z_d_oof[valid_oof] - z_gt_arr[valid_oof]) / z_gt_arr[valid_oof]))

    print(f"GT Bbox In-Sample AbsRel: {insample_absrel:.4f} (target ≈ 0.0605, diff: {abs(insample_absrel - 0.0605):.4f})")
    print(f"GT Bbox LODO OOF AbsRel:  {oof_absrel:.4f} (target ≈ 0.0609, diff: {abs(oof_absrel - 0.0609):.4f})")

    diff_in = abs(insample_absrel - 0.0605)
    diff_oof = abs(oof_absrel - 0.0609)
    if diff_in > 0.002 or diff_oof > 0.002:
        raise ValueError(
            f"REGRESSION FAILURE: GT Bbox error exceeds tolerance 0.002!\n"
            f"  In-sample: {insample_absrel:.4f} vs 0.0605 (diff {diff_in:.4f})\n"
            f"  OOF:       {oof_absrel:.4f} vs 0.0609 (diff {diff_oof:.4f})"
        )

    print(">>> REGRESSION CHECK PASSED! Differences <= 0.002.")
    return {
        "n_objects": n_gt,
        "insample_absrel": round(insample_absrel, 4),
        "oof_absrel": round(oof_absrel, 4),
        "weights_insample": fw_insample.weights.round(4).tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Process Detections on Split B and Split C for each detector
# ---------------------------------------------------------------------------
def process_detector(
    model_key: str,
    priors,
    frozen_gt_weights,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"PROCESSING MODEL: {model_key}")
    print("=" * 80)

    # 1. Load Split B data
    feats_b_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_features.parquet"
    eval_b_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_eval.parquet"
    gt_b_path = PROJECT_ROOT / "results" / "predictions" / f"{model_key}_B_gt.parquet"
    fn_b_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_fn.parquet"

    feats_b = pd.read_parquet(feats_b_path)
    eval_b = pd.read_parquet(eval_b_path)
    gt_b = pd.read_parquet(gt_b_path)
    fn_b = pd.read_parquet(fn_b_path)

    # Add geometric cues to detector features
    cues_b = add_cues(feats_b, priors=priors, eps=BORDER_EPS)

    # Merge GT coordinates for the matched TP detections (exact same support)
    # Join on (frame_id, gt_idx)
    gt_b_sub = gt_b[["frame_id", "gt_idx", "x1", "y1", "x2", "y2"]].rename(
        columns={"x1": "x1_gt", "y1": "y1_gt", "x2": "x2_gt", "y2": "y2_gt"}
    )
    merged_b = cues_b.merge(
        eval_b[["frame_id", "pred_idx", "gt_idx", "z_gt", "difficulty"]],
        on=["frame_id", "pred_idx"],
        how="inner",
    ).merge(gt_b_sub, on=["frame_id", "gt_idx"], how="inner")

    # Compute cues on GT bbox for the exact same TP detections
    feats_gt_matched_b = merged_b.copy()
    feats_gt_matched_b["x1"] = feats_gt_matched_b["x1_gt"]
    feats_gt_matched_b["y1"] = feats_gt_matched_b["y1_gt"]
    feats_gt_matched_b["x2"] = feats_gt_matched_b["x2_gt"]
    feats_gt_matched_b["y2"] = feats_gt_matched_b["y2_gt"]
    cues_gt_b = add_cues(feats_gt_matched_b, priors=priors, eps=BORDER_EPS)

    # 2. Mask rate comparison on same TP objects
    n_tp_b = len(merged_b)
    mask_rates_b = {
        "n_tp": n_tp_b,
        "detector": {
            "mask_rate_w": float((~cues_b["valid_w"]).mean()),
            "mask_rate_h": float((~cues_b["valid_h"]).mean()),
            "mask_rate_g": float((~cues_b["valid_g"]).mean()),
            "mask_rate_all_invalid": float((~cues_b["valid_w"] & ~cues_b["valid_h"] & ~cues_b["valid_g"]).mean()),
            "valid_rate_all": float((cues_b["valid_w"] & cues_b["valid_h"] & cues_b["valid_g"]).mean()),
        },
        "gt_bbox": {
            "mask_rate_w": float((~cues_gt_b["valid_w"]).mean()),
            "mask_rate_h": float((~cues_gt_b["valid_h"]).mean()),
            "mask_rate_g": float((~cues_gt_b["valid_g"]).mean()),
            "mask_rate_all_invalid": float((~cues_gt_b["valid_w"] & ~cues_gt_b["valid_h"] & ~cues_gt_b["valid_g"]).mean()),
            "valid_rate_all": float((cues_gt_b["valid_w"] & cues_gt_b["valid_h"] & cues_gt_b["valid_g"]).mean()),
        },
    }

    # 3. LODO fusion on Detector bboxes (Decision D29) and comparison against GT-bbox weights (D17)
    z_gt_b = merged_b["z_gt"].to_numpy(dtype=float)
    z_d_det_oof_b, fw_det_full_b, lodo_fits_b = fit_fusion_lodo(cues_b, z_gt_b, min_train_drives=3)
    z_d_det_gtweights_b = fuse_with_weights(cues_b, frozen_gt_weights)

    # LODO fusion on GT bboxes for decomposition
    z_d_gt_oof_b, fw_gt_tp_b, _ = fit_fusion_lodo(cues_gt_b, z_gt_b, min_train_drives=3)

    # Attach predictions to merged_b
    merged_b["z_w_det"] = cues_b["z_w"]
    merged_b["z_h_det"] = cues_b["z_h"]
    merged_b["z_g_det"] = cues_b["z_g"]
    merged_b["z_d_det"] = z_d_det_oof_b
    merged_b["z_d_det_gtweights"] = z_d_det_gtweights_b

    merged_b["z_w_gt"] = cues_gt_b["z_w"]
    merged_b["z_h_gt"] = cues_gt_b["z_h"]
    merged_b["z_g_gt"] = cues_gt_b["z_g"]
    merged_b["z_d_gt"] = z_d_gt_oof_b

    # Save Split B cues parquet
    out_b_df = cues_b[["frame_id", "drive", "pred_idx", "z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]].copy()
    out_b_df["z_d"] = z_d_det_oof_b
    cues_b_parquet = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_cues.parquet"
    out_b_df.to_parquet(cues_b_parquet, index=False)
    print(f"Saved: {cues_b_parquet}")

    # 4. Process Split C
    feats_c_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_C_features.parquet"
    eval_c_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_C_eval.parquet"
    gt_c_path = PROJECT_ROOT / "results" / "predictions" / f"{model_key}_C_gt.parquet"
    fn_c_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_C_fn.parquet"

    feats_c = pd.read_parquet(feats_c_path)
    eval_c = pd.read_parquet(eval_c_path)
    gt_c = pd.read_parquet(gt_c_path)
    fn_c = pd.read_parquet(fn_c_path)

    cues_c = add_cues(feats_c, priors=priors, eps=BORDER_EPS)

    # Fuse Split C using weights fit on Split B (Decision D29)
    z_d_det_c = fuse_with_weights(cues_c, fw_det_full_b)
    z_d_det_gtweights_c = fuse_with_weights(cues_c, frozen_gt_weights)

    gt_c_sub = gt_c[["frame_id", "gt_idx", "x1", "y1", "x2", "y2"]].rename(
        columns={"x1": "x1_gt", "y1": "y1_gt", "x2": "x2_gt", "y2": "y2_gt"}
    )
    merged_c = cues_c.merge(
        eval_c[["frame_id", "pred_idx", "gt_idx", "z_gt", "difficulty"]],
        on=["frame_id", "pred_idx"],
        how="inner",
    ).merge(gt_c_sub, on=["frame_id", "gt_idx"], how="inner")

    feats_gt_matched_c = merged_c.copy()
    feats_gt_matched_c["x1"] = feats_gt_matched_c["x1_gt"]
    feats_gt_matched_c["y1"] = feats_gt_matched_c["y1_gt"]
    feats_gt_matched_c["x2"] = feats_gt_matched_c["x2_gt"]
    feats_gt_matched_c["y2"] = feats_gt_matched_c["y2_gt"]
    cues_gt_c = add_cues(feats_gt_matched_c, priors=priors, eps=BORDER_EPS)
    z_d_gt_c = fuse_with_weights(cues_gt_c, frozen_gt_weights)

    merged_c["z_w_det"] = cues_c["z_w"]
    merged_c["z_h_det"] = cues_c["z_h"]
    merged_c["z_g_det"] = cues_c["z_g"]
    merged_c["z_d_det"] = z_d_det_c
    merged_c["z_d_det_gtweights"] = z_d_det_gtweights_c
    merged_c["z_w_gt"] = cues_gt_c["z_w"]
    merged_c["z_h_gt"] = cues_gt_c["z_h"]
    merged_c["z_g_gt"] = cues_gt_c["z_g"]
    merged_c["z_d_gt"] = z_d_gt_c

    out_c_df = cues_c[["frame_id", "drive", "pred_idx", "z_w", "z_h", "z_g", "valid_w", "valid_h", "valid_g"]].copy()
    out_c_df["z_d"] = z_d_det_c
    cues_c_parquet = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_C_cues.parquet"
    out_c_df.to_parquet(cues_c_parquet, index=False)
    print(f"Saved: {cues_c_parquet}")

    # 5. Evaluate (a)-(d) on GT-bbox vs Detector-bbox
    def evaluate_methods(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
        methods = {
            "a_zw_det": "z_w_det",
            "b_zh_det": "z_h_det",
            "c_zg_det": "z_g_det",
            "d_zd_det_refit": "z_d_det",
            "d_zd_det_gtweights": "z_d_det_gtweights",
            "a_zw_gt": "z_w_gt",
            "b_zh_gt": "z_h_gt",
            "c_zg_gt": "z_g_gt",
            "d_zd_gt": "z_d_gt",
        }
        res: dict[str, Any] = {}
        for m_name, col in methods.items():
            rep = evaluate_report(df, gt_col="z_gt", pred_col=col)
            ov = rep[(rep["group_type"] == "overall") & (rep["group"] == "all")].iloc[0]
            macro_ar = macro_by_cluster(df, "absrel", gt_col="z_gt", pred_col=col, cluster_col="drive")
            res[m_name] = {
                "n": int(ov["n"]),
                "n_valid": int(ov["n_valid"]),
                "valid_frac": float(ov["valid_frac"]),
                "pooled_absrel": float(ov["absrel"]),
                "pooled_rmse": float(ov["rmse"]),
                "pooled_mae": float(ov["mae"]),
                "pooled_delta1": float(ov["delta1"]),
                "macro_absrel": float(macro_ar),
            }
        return res

    eval_results_b = evaluate_methods(merged_b, "B")
    eval_results_c = evaluate_methods(merged_c, "C")

    # 6. Paired cluster bootstrap between (d)-detector and (d)-GT-bbox
    # Filter common support
    common_b = merged_b.dropna(subset=["z_d_det", "z_d_gt"])
    boot_res = paired_cluster_bootstrap(
        df=common_b,
        pred_col_a="z_d_det",
        pred_col_b="z_d_gt",
        metric="absrel",
        seed=42,
        n_boot=2000,
        cluster_col="drive",
    )
    bootstrap_info = {
        "n_common": len(common_b),
        "n_clusters": boot_res.n_clusters,
        "estimate_diff": float(boot_res.estimate),
        "ci_low": float(boot_res.ci_low),
        "ci_high": float(boot_res.ci_high),
        "excludes_zero": bool(boot_res.excludes_zero),
        "note": "CI thô (12 cụm)",
    }

    # 7. FN and Recall by depth band from fn.parquet
    def compute_recall_by_band(tp_df: pd.DataFrame, fn_df: pd.DataFrame) -> dict[str, Any]:
        masks_tp = band_masks(tp_df["z_gt"].to_numpy())
        masks_fn = band_masks(fn_df["z_gt"].to_numpy())
        by_band = {}
        for b_name in BAND_NAMES:
            n_tp = int(masks_tp[b_name].sum())
            n_fn = int(masks_fn[b_name].sum())
            total = n_tp + n_fn
            rec = n_tp / total if total > 0 else float("nan")
            by_band[b_name] = {
                "n_tp": n_tp,
                "n_fn": n_fn,
                "total_gt": total,
                "recall": round(rec, 4),
                "low_n": total < 100,
            }
        return by_band

    recall_b = compute_recall_by_band(merged_b, fn_b)
    recall_c = compute_recall_by_band(merged_c, fn_c)

    return {
        "model_key": model_key,
        "mask_rates_split_B": mask_rates_b,
        "weights": {
            "frozen_gt_bbox": frozen_gt_weights.weights.round(4).tolist(),
            "refit_detector_B": fw_det_full_b.weights.round(4).tolist(),
            "refit_detector_cov": fw_det_full_b.cov_shrunk.round(5).tolist(),
        },
        "eval_split_B": eval_results_b,
        "eval_split_C": eval_results_c,
        "paired_bootstrap_B": bootstrap_info,
        "recall_by_band_B": recall_b,
        "recall_by_band_C": recall_c,
    }


def main():
    print("=" * 80)
    print("TASK T02: GEOMETRY STAGE ON DETECTOR BBOXES (D17, D29)")
    print("=" * 80)

    # Load calibrated priors
    priors, cfg = load_geometry_priors()
    frozen_gt_weights = load_frozen_gt_fusion_weights()

    # Step 1: GT bbox regression check
    regression_res = run_gt_bbox_regression_check(priors)

    # Step 2: Process all 3 detectors
    all_results = {}
    for m in MODEL_KEYS:
        all_results[m] = process_detector(m, priors, frozen_gt_weights)

    # Output results to JSON
    out_json = PROJECT_ROOT / "results" / "tables" / "geometry_on_detector_bbox.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    full_output = {
        "metadata": {
            "task": "T02",
            "priors": {
                "W_eff": priors.W_eff,
                "H_obj": priors.H_obj,
                "H_cam": priors.H_cam,
                "delta_horizon": priors.delta_horizon,
            },
            "gt_bbox_regression": regression_res,
        },
        "detectors": all_results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nWrote JSON report: {out_json}")

    # Output Markdown report
    out_md = PROJECT_ROOT / "results" / "tables" / "geometry_on_detector_bbox.md"
    md_lines = [
        "# Báo cáo Tác vụ T02: Hình học trên Bbox Detector (D17, D29)",
        "",
        "## 1. Kiểm tra Regression trên GT Bbox (Toàn bộ Split B, 4.776 xe)",
        f"- **Số lượng mẫu:** {regression_res['n_objects']} Car Hard",
        f"- **In-sample AbsRel:** {regression_res['insample_absrel']:.4f} (chuẩn Day 4: 0.0605, chênh lệch: {abs(regression_res['insample_absrel'] - 0.0605):.4f} <= 0.002) -> **ĐẠT**",
        f"- **LODO OOF AbsRel:** {regression_res['oof_absrel']:.4f} (chuẩn Day 6: 0.0609, chênh lệch: {abs(regression_res['oof_absrel'] - 0.0609):.4f} <= 0.002) -> **ĐẠT**",
        "",
        "## 2. Bảng Tỷ lệ Mask Viền Ảnh (Mask Rate) giữa GT Bbox vs Detector Bbox trên tập TP",
        "",
        "| Detector | N (TP) | Mask $Z_w$ (GT / Det) | Mask $Z_h$ (GT / Det) | Mask $Z_g$ (GT / Det) | All Invalid (GT / Det) |",
        "|---|---|---|---|---|---|",
    ]

    for m in MODEL_KEYS:
        mr = all_results[m]["mask_rates_split_B"]
        n_tp = mr["n_tp"]
        mg = mr["gt_bbox"]
        md = mr["detector"]
        md_lines.append(
            f"| `{m}` | {n_tp} | {mg['mask_rate_w']:.3f} / **{md['mask_rate_w']:.3f}** | "
            f"{mg['mask_rate_h']:.3f} / **{md['mask_rate_h']:.3f}** | "
            f"{mg['mask_rate_g']:.3f} / **{md['mask_rate_g']:.3f}** | "
            f"{mg['mask_rate_all_invalid']:.4f} / **{md['mask_rate_all_invalid']:.4f}** |"
        )

    md_lines.extend([
        "",
        "> [!NOTE]",
        "> Tỷ lệ mask của detector bbox thấp hơn GT bbox do detector dự đoán viền cách lề ảnh 1-3 px thay vì 0 px.",
        "> Theo quy định T02 mục Cấm: báo cáo trung thực hiện tượng này, KHÔNG tự ý đổi eps (eps=2.0 px giữ nguyên).",
        "",
        "## 3. Trọng số Hợp nhất: GT-Bbox Đối chứng vs Refit Detector (D17)",
        "",
        "| Detector | $w_w$ (Width) | $w_h$ (Height) | $w_g$ (Ground) | Ghi chú |",
        "|---|---|---|---|---|",
        f"| **GT-Bbox (Chuẩn D10)** | {frozen_gt_weights.weights[0]:.4f} | {frozen_gt_weights.weights[1]:.4f} | {frozen_gt_weights.weights[2]:.4f} | Cận dưới lý thuyết trên GT |",
    ])

    for m in MODEL_KEYS:
        w_refit = all_results[m]["weights"]["refit_detector_B"]
        md_lines.append(f"| `{m}` (Refit) | {w_refit[0]:.4f} | {w_refit[1]:.4f} | {w_refit[2]:.4f} | Refit trên TP Split B |")

    md_lines.extend([
        "",
        "## 4. Hiệu năng Ước lượng (a)–(d) trên Split B (Pooled vs Macro by Drive)",
        "",
        "| Model | Phương pháp | N | N valid | Valid % | Pooled AbsRel | Macro AbsRel | Pooled RMSE | Pooled $\\delta_1$ |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for m in MODEL_KEYS:
        ev = all_results[m]["eval_split_B"]
        for method_key, method_name in [
            ("d_zd_det_refit", "(d) Det Refit"),
            ("d_zd_det_gtweights", "(d) Det GT-weights"),
            ("b_zh_det", "(b) $Z_h$ Det"),
            ("c_zg_det", "(c) $Z_g$ Det"),
            ("a_zw_det", "(a) $Z_w$ Det"),
            ("d_zd_gt", "(d) GT Bbox"),
        ]:
            e = ev[method_key]
            md_lines.append(
                f"| `{m}` | {method_name} | {e['n']} | {e['n_valid']} | {e['valid_frac']*100:.1f}% | "
                f"**{e['pooled_absrel']:.4f}** | {e['macro_absrel']:.4f} | {e['pooled_rmse']:.2f} m | {e['pooled_delta1']*100:.1f}% |"
            )

    md_lines.extend([
        "",
        "## 5. Paired Cluster Bootstrap: (d)-Detector vs (d)-GT Bbox (Phân rã sai số)",
        "",
        "| Model | N chung | Cụm | Ước lượng $\\Delta$ (Det - GT) | 95% CI thô | Khác biệt loại 0? |",
        "|---|---|---|---|---|---|",
    ])

    for m in MODEL_KEYS:
        pb = all_results[m]["paired_bootstrap_B"]
        md_lines.append(
            f"| `{m}` | {pb['n_common']} | {pb['n_clusters']} | "
            f"+{pb['estimate_diff']:.4f} | [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}] | "
            f"{'Có' if pb['excludes_zero'] else 'Không'} ({pb['note']}) |"
        )

    md_lines.extend([
        "",
        "## 6. False Negatives (FN) và Recall theo Dải Khoảng cách trên Split B",
        "",
        "| Model | Dải khoảng cách | GT Cars | TP | FN | Recall | Cờ n < 100 |",
        "|---|---|---|---|---|---|---|",
    ])

    for m in MODEL_KEYS:
        rec_b = all_results[m]["recall_by_band_B"]
        for b_name in BAND_NAMES:
            r = rec_b[b_name]
            star = "*" if r["low_n"] else ""
            md_lines.append(
                f"| `{m}` | {b_name} | {r['total_gt']}{star} | {r['n_tp']} | {r['n_fn']} | {r['recall']*100:.1f}% | {star} |"
            )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote Markdown report: {out_md}")

    # 7. Write reproducibility log entry into runs/pipeline_log.jsonl
    commit, dirty = get_git_info()
    log_rec = make_log_record(
        split="B,C",
        split_hash="0f83c3547188c8a6ab29a389f27524a5e09d5e7f96fcaff5f7a4f1328c2aef92",
        seed=42,
        n_boot=2000,
        tag="geometry-v2",
        extra={
            "task": "T02",
            "git_commit": commit,
            "git_dirty": dirty,
            "models": MODEL_KEYS,
            "gt_regression": regression_res,
            "outputs": [str(out_json), str(out_md)],
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_rec)
    print("Logged run to runs/pipeline_log.jsonl.")
    print(">>> T02 EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
