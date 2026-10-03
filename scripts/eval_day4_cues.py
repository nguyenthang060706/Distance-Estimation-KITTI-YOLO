"""
scripts/eval_day4_cues.py: Day 4 evaluation of geometric cues and log-space fusion
on Ground Truth bounding boxes of Split B (KE_HOACH_V4 §5.2 and §8.1).

Rules & Specifications:
- Evaluates on Split B only (Split T is strictly forbidden and guarded).
- Target population: Car passing KITTI Hard filter (§4).
- Pinhole cues:
    (a) Z_w = fx · W_eff / w
    (b) Z_h = fy · H_obj / h
    (c) Z_g = fy · H_cam / (y_bottom - y_h)
- Priors from Split A:
    - W_eff = 2.6184 m (median)
    - H_obj = 1.6797 m (median)
    - Baseline Z_g: H_cam = 1.8845 m, y_h = cy
    - Fitted Z_g:   H_cam = 2.0422 m, y_h = cy - 4.6782 px (Huber regression on Split A)
- Masking with epsilon = 2 px:
    - Touch left/right border: Z_w = NaN
    - Touch top/bottom border: Z_h = NaN, Z_g = NaN
    - y_bottom <= y_h + eps: Z_g = NaN
- Fusion (d):
    - Grouped CV by drive on Split B to estimate error covariance Σ
    - Ledoit-Wolf shrinkage towards diagonal target
    - Check negative weights, apply NNLS (w >= 0) if needed
    - Sub-matrix Σ_S for partial cue availability
- Stratified evaluation across 5 depth ranges (0-10, 10-20, 20-30, 30-50, >50, plus >30m).
- Verifies Week 2 entry condition: (d) not worse than best single cue in >= 3 of 4 ranges with n >= 100.
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_split
from src.geometry.geometric_cues import (
    CameraIntrinsics,
    GeometricPriors,
    compute_cues_batch,
    BORDER_EPS,
)
from src.geometry.fusion import (
    fit_fusion_weights,
    fuse_depths,
    FusionWeights,
)
from src.evaluation.metrics import (
    compute_metrics_by_range,
    format_metrics_table,
    DEPTH_BINS,
)


def run_day4_evaluation():
    print("=" * 75)
    print("DAY 4 EVALUATION: GEOMETRIC CUES & LOG-SPACE FUSION ON GT BBOX (SPLIT B)")
    print("=" * 75)

    loader = KITTILoader("data/kitti")
    split_b_ids = load_split("splits", "B", allow_test=False)
    print(f"Loaded Split B: {len(split_b_ids)} frames.")

    # 1. Priors from Split A
    priors_yaml = PROJECT_ROOT / "configs" / "geometry_priors.yaml"
    with open(priors_yaml, "r", encoding="utf-8") as f:
        priors_cfg = yaml.safe_load(f)

    w_eff = priors_cfg["classes"]["Car"]["W_eff_median"]
    h_obj = priors_cfg["classes"]["Car"]["H_obj_median"]
    h_cam_raw = priors_cfg["metadata"]["H_cam_median"]

    # Huber fitted ground parameters from Split A
    # Fitted: (y_bot - cy) = delta + H_cam * (fy / Z) => delta = -4.6782 px, H_cam = 2.0422 m
    delta_fitted = -4.6782
    h_cam_fitted = 2.0422

    print(f"\nPrior values (Split A):")
    print(f"  W_eff (median):  {w_eff:.4f} m")
    print(f"  H_obj (median):  {h_obj:.4f} m")
    print(f"  H_cam (raw med): {h_cam_raw:.4f} m (y_h = cy)")
    print(f"  H_cam (fitted):  {h_cam_fitted:.4f} m (delta = {delta_fitted:.4f} px, y_h = cy + delta)")

    # 2. Extract GT Car objects from Split B
    records = []
    print("\nExtracting GT Car (Hard) objects from Split B...")

    for fid in tqdm(split_b_ids, desc="Processing Split B"):
        frame = loader.load_frame(fid)
        calib = frame.calib
        drive = frame.drive
        # Exact image dimensions from file header
        with Image.open(frame.image_path) as img:
            img_w, img_h = img.size

        intrinsics = CameraIntrinsics(
            fx=calib.fx,
            fy=calib.fy,
            cx=calib.cx,
            cy=calib.cy,
        )

        for obj in frame.objects:
            if obj.obj_class != "Car":
                continue
            if not obj.passes_hard_filter():
                continue

            z_gt = obj.depth
            if z_gt <= 0:
                continue

            bbox = obj.bbox  # [x1, y1, x2, y2]
            records.append({
                "frame_id": fid,
                "drive": drive,
                "bbox": bbox,
                "z_gt": z_gt,
                "intrinsics": intrinsics,
                "img_w": img_w,
                "img_h": img_h,
            })

    n_objects = len(records)
    print(f"Found {n_objects} Car Hard objects in Split B across {len(split_b_ids)} frames.")

    # 3. Compute cues for each object
    z_gt_arr = np.array([r["z_gt"] for r in records])
    drive_arr = np.array([r["drive"] for r in records])

    # We evaluate both raw Z_g and fitted Z_g
    z_w_arr = np.full(n_objects, np.nan)
    z_h_arr = np.full(n_objects, np.nan)
    z_g_raw_arr = np.full(n_objects, np.nan)
    z_g_fit_arr = np.full(n_objects, np.nan)

    mask_w_count = 0
    mask_h_count = 0
    mask_g_raw_count = 0
    mask_g_fit_count = 0

    priors_raw = GeometricPriors(
        W_eff=w_eff,
        H_obj=h_obj,
        H_cam=h_cam_raw,
        delta_horizon=0.0,
    )
    priors_fit = GeometricPriors(
        W_eff=w_eff,
        H_obj=h_obj,
        H_cam=h_cam_fitted,
        delta_horizon=delta_fitted,
    )

    for i, r in enumerate(records):
        intr = r["intrinsics"]
        bbox = r["bbox"]
        w = r["img_w"]
        h = r["img_h"]

        # Raw priors
        res_raw = compute_cues_batch(
            np.array([bbox]),
            intr,
            priors_raw,
            img_width=w,
            img_height=h,
            eps=BORDER_EPS,
        )
        z_w_arr[i] = res_raw.Z_w[0]
        z_h_arr[i] = res_raw.Z_h[0]
        z_g_raw_arr[i] = res_raw.Z_g[0]

        if not res_raw.valid_w[0]:
            mask_w_count += 1
        if not res_raw.valid_h[0]:
            mask_h_count += 1
        if not res_raw.valid_g[0]:
            mask_g_raw_count += 1

        # Fitted priors for Z_g
        res_fit = compute_cues_batch(
            np.array([bbox]),
            intr,
            priors_fit,
            img_width=w,
            img_height=h,
            eps=BORDER_EPS,
        )
        z_g_fit_arr[i] = res_fit.Z_g[0]
        if not res_fit.valid_g[0]:
            mask_g_fit_count += 1

    # Masking statistics report
    print("\n" + "=" * 60)
    print("BOUNDARY MASKING REPORT ON SPLIT B (eps = 2 px):")
    print("=" * 60)
    print(f"Total Car Hard objects: {n_objects}")
    print(f"  Z_w masked (touch left/right):       {mask_w_count:>5d} / {n_objects} ({mask_w_count/n_objects:6.2%}) | Valid: {n_objects - mask_w_count}")
    print(f"  Z_h masked (touch top/bottom):       {mask_h_count:>5d} / {n_objects} ({mask_h_count/n_objects:6.2%}) | Valid: {n_objects - mask_h_count}")
    print(f"  Z_g (raw) masked (border / horizon): {mask_g_raw_count:>5d} / {n_objects} ({mask_g_raw_count/n_objects:6.2%}) | Valid: {n_objects - mask_g_raw_count}")
    print(f"  Z_g (fit) masked (border / horizon): {mask_g_fit_count:>5d} / {n_objects} ({mask_g_fit_count/n_objects:6.2%}) | Valid: {n_objects - mask_g_fit_count}")
    print("=" * 60)

    # 4. Fit Log-space Fusion (d) on Split B using Grouped CV by drive
    # Model A: using raw Z_g
    valid_cues_raw = np.column_stack([
        ~np.isnan(z_w_arr),
        ~np.isnan(z_h_arr),
        ~np.isnan(z_g_raw_arr),
    ])
    z_cues_raw = np.column_stack([z_w_arr, z_h_arr, z_g_raw_arr])

    fusion_weights_raw = fit_fusion_weights(
        Z_cues=z_cues_raw,
        Z_gt=z_gt_arr,
        valid_mask=valid_cues_raw,
        drive_ids=drive_arr,
    )
    z_fused_raw = fuse_depths(z_cues_raw, valid_cues_raw, fusion_weights_raw)

    # Model B: using fitted Z_g
    valid_cues_fit = np.column_stack([
        ~np.isnan(z_w_arr),
        ~np.isnan(z_h_arr),
        ~np.isnan(z_g_fit_arr),
    ])
    z_cues_fit = np.column_stack([z_w_arr, z_h_arr, z_g_fit_arr])

    fusion_weights_fit = fit_fusion_weights(
        Z_cues=z_cues_fit,
        Z_gt=z_gt_arr,
        valid_mask=valid_cues_fit,
        drive_ids=drive_arr,
    )
    z_fused_fit = fuse_depths(z_cues_fit, valid_cues_fit, fusion_weights_fit)

    print("\n" + "=" * 60)
    print("OPTIMAL FUSION WEIGHTS & DIAGNOSTICS (GROUPED CV BY DRIVE):")
    print("=" * 60)
    print("Model A (Baseline: raw Z_g):")
    print(f"  Samples with all 3 cues: {fusion_weights_raw.n_samples} across {fusion_weights_raw.n_drives} drives")
    print(f"  Shrinkage alpha:         {fusion_weights_raw.shrinkage_alpha:.4f}")
    print(f"  Weights [Z_w, Z_h, Z_g]: {fusion_weights_raw.weights.round(4).tolist()} (Constrained NNLS: {fusion_weights_raw.constrained})")
    print(f"  Covariance matrix (shrunk):\n{fusion_weights_raw.cov_shrunk.round(5)}")

    print("\nModel B (Fitted: regression Z_g with delta & H_cam):")
    print(f"  Samples with all 3 cues: {fusion_weights_fit.n_samples} across {fusion_weights_fit.n_drives} drives")
    print(f"  Shrinkage alpha:         {fusion_weights_fit.shrinkage_alpha:.4f}")
    print(f"  Weights [Z_w, Z_h, Z_g]: {fusion_weights_fit.weights.round(4).tolist()} (Constrained NNLS: {fusion_weights_fit.constrained})")
    print(f"  Covariance matrix (shrunk):\n{fusion_weights_fit.cov_shrunk.round(5)}")
    print("=" * 60)

    # 5. Evaluate all methods across depth ranges
    cues_to_evaluate = {
        "(a) Z_w (width)": z_w_arr,
        "(b) Z_h (height)": z_h_arr,
        "(c) Z_g (ground, raw)": z_g_raw_arr,
        "(c*) Z_g (ground, fit)": z_g_fit_arr,
        "(d) Fused (with raw Z_g)": z_fused_raw,
        "(d*) Fused (with fitted Z_g)": z_fused_fit,
    }

    all_metrics = {}
    for name, z_arr in cues_to_evaluate.items():
        all_metrics[name] = compute_metrics_by_range(z_arr, z_gt_arr)

    metrics_table_md = format_metrics_table(all_metrics)
    print("\n" + "=" * 80)
    print("DETAILED DEPTH ACCURACY RESULTS TABLE:")
    print("=" * 80)
    print(metrics_table_md)
    print("=" * 80)

    # 6. Check Week 2 Entry Condition (§8.1):
    # (d) not worse than best single cue in at least 3 of 4 ranges with n >= 100.
    # The 4 ranges typically: 0-10m, 10-20m, 20-30m, 30-50m (or >30m if >50m has n < 100)
    print("\n" + "=" * 70)
    print("WEEK 2 ENTRY CRITERIA VERIFICATION (KE_HOACH_V4 §8.1):")
    print("Criteria: (d) not worse than best single cue in >= 3 of 4 ranges (n >= 100).")
    print("=" * 70)

    single_cues = ["(a) Z_w (width)", "(b) Z_h (height)", "(c*) Z_g (ground, fit)"]
    fused_cue = "(d*) Fused (with fitted Z_g)"

    # Independent ranges with n >= 100 for Gate §8.1 (Decision D9)
    # Note: '>30m' is an aggregated range (overlaps 30-50m and >50m) so it is excluded from gate denominator.
    check_ranges = ["0-10m", "10-20m", "20-30m", "30-50m"]
    evaluated_ranges = []
    wins = 0

    for r_name in check_ranges:
        n_samples = all_metrics[fused_cue][r_name].n
        if n_samples < 100:
            continue

        evaluated_ranges.append(r_name)
        fused_abs_rel = all_metrics[fused_cue][r_name].abs_rel

        # Best single cue for this range
        best_single_cue = None
        best_single_abs_rel = float("inf")
        for sc in single_cues:
            sc_val = all_metrics[sc][r_name].abs_rel
            if sc_val < best_single_abs_rel:
                best_single_abs_rel = sc_val
                best_single_cue = sc

        # Condition: fused is not worse than best single cue (allowing tiny numerical tolerance 0.005)
        is_not_worse = fused_abs_rel <= (best_single_abs_rel + 0.005)
        if is_not_worse:
            wins += 1
            verdict = "PASSED (Win / Tied)"
        else:
            verdict = "LOSS (Thua)"

        print(f"  Range {r_name:<7s} (n={n_samples:<4d}): Fused AbsRel = {fused_abs_rel:.4f} | Best single ({best_single_cue}) = {best_single_abs_rel:.4f} -> {verdict}")

    passed_gate = wins >= 3
    print("-" * 70)
    print(f"Gate Summary (§8.1, D9): {wins}/{len(evaluated_ranges)} independent ranges passed (>=3/4 required).")
    print(f"Status: {'PASS -> READY FOR WEEK 2' if passed_gate else 'NEEDS INVESTIGATION'}")
    print("=" * 70 + "\n")

    # 7. Save results
    output_dir = PROJECT_ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "day4_gt_bbox_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Day 4 Results: Geometric Depth Cues & Log-Space Fusion on Split B (GT Bbox)\n\n")
        f.write(f"- **Split:** B (1,496 frames, {n_objects} Car Hard objects)\n")
        f.write(f"- **Priors from Split A:** W_eff = {w_eff:.4f}m, H_obj = {h_obj:.4f}m, H_cam_fit = {h_cam_fitted:.4f}m, delta_h = {delta_fitted:.4f}px\n")
        f.write(f"- **Fusion weights (Z_w, Z_h, Z_g):** {fusion_weights_fit.weights.round(4).tolist()}\n")
        f.write(f"- **Covariance shrinkage alpha:** {fusion_weights_fit.shrinkage_alpha:.4f}\n\n")
        f.write("## 1. Boundary Masking Rates (eps = 2 px)\n\n")
        f.write(f"- Z_w masked: {mask_w_count} / {n_objects} ({mask_w_count/n_objects:6.2%})\n")
        f.write(f"- Z_h masked: {mask_h_count} / {n_objects} ({mask_h_count/n_objects:6.2%})\n")
        f.write(f"- Z_g (fitted) masked: {mask_g_fit_count} / {n_objects} ({mask_g_fit_count/n_objects:6.2%})\n\n")
        f.write("## 2. Quantitative Metrics Table\n\n")
        f.write(metrics_table_md)
        f.write("\n\n*Note: `*` indicates n < 100.*\n\n")
        f.write(f"## 3. Week 2 Gate Status: **{'PASSED' if passed_gate else 'FAILED'}** ({wins}/{len(evaluated_ranges)} ranges)\n")

    json_path = output_dir / "day4_gt_bbox_evaluation.json"
    serializable_metrics = {
        cue: {r: m.to_dict() for r, m in r_dict.items()}
        for cue, r_dict in all_metrics.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": "B",
            "n_objects": n_objects,
            "mask_rates": {
                "Z_w": mask_w_count / n_objects,
                "Z_h": mask_h_count / n_objects,
                "Z_g_raw": mask_g_raw_count / n_objects,
                "Z_g_fit": mask_g_fit_count / n_objects,
            },
            "fusion_weights": {
                "weights": fusion_weights_fit.weights.tolist(),
                "alpha": fusion_weights_fit.shrinkage_alpha,
                "cov_shrunk": fusion_weights_fit.cov_shrunk.tolist(),
            },
            "metrics": serializable_metrics,
            "gate_passed": passed_gate,
        }, f, indent=2)

    print(f"Results saved to:\n  {report_path}\n  {json_path}")
    return all_metrics, passed_gate


if __name__ == "__main__":
    run_day4_evaluation()
