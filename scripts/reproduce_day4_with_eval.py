"""
scripts/reproduce_day4_with_eval.py: Reproduce Day 4 results on Split B using
the new Day 5 evaluate_report() and paired_cluster_bootstrap() from src.evaluation.eval.
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

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
)
from src.evaluation.eval import (
    evaluate_report,
    paired_cluster_bootstrap,
    cluster_bootstrap,
)


def main():
    print("=" * 80)
    print("DAY 5: REPRODUCING DAY 4 REPORT WITH EVAL.PY & RUNNING CLUSTER BOOTSTRAP")
    print("=" * 80)

    loader = KITTILoader("data/kitti")
    split_b_ids = load_split("splits", "B", allow_test=False)

    geom_yaml = PROJECT_ROOT / "configs" / "geometry_params.yaml"
    with open(geom_yaml, "r", encoding="utf-8") as f:
        geom_cfg = yaml.safe_load(f)

    priors_a = geom_cfg["priors_split_A"]
    w_eff = float(priors_a["W_eff"])
    h_obj = float(priors_a["H_obj"])
    h_cam_fitted = float(priors_a["effective_ground_plane"]["H_cam_effective"])
    delta_fitted = float(priors_a["effective_ground_plane"]["delta_horizon"])

    priors_fit = GeometricPriors(
        W_eff=w_eff,
        H_obj=h_obj,
        H_cam=h_cam_fitted,
        delta_horizon=delta_fitted,
    )

    records = []
    print("Extracting Car Hard objects from Split B...")
    for fid in tqdm(split_b_ids, desc="Processing Split B"):
        frame = loader.load_frame(fid)
        with Image.open(frame.image_path) as img:
            img_w, img_h = img.size

        calib = frame.calib
        intrinsics = CameraIntrinsics(
            fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy
        )

        for obj in frame.objects:
            if obj.obj_class != "Car" or not obj.passes_hard_filter():
                continue
            if obj.depth <= 0:
                continue

            records.append({
                "frame_id": fid,
                "drive": frame.drive,
                "bbox": obj.bbox,
                "z_gt": obj.depth,
                "cls": obj.obj_class,
                "difficulty": obj.get_difficulty(),
                "intrinsics": intrinsics,
                "img_w": img_w,
                "img_h": img_h,
            })

    n = len(records)
    print(f"Total Car Hard objects: {n}")

    z_w = np.full(n, np.nan)
    z_h = np.full(n, np.nan)
    z_g = np.full(n, np.nan)
    z_gt = np.array([r["z_gt"] for r in records])
    drives = np.array([r["drive"] for r in records])

    for i, r in enumerate(records):
        res = compute_cues_batch(
            np.array([r["bbox"]]),
            r["intrinsics"],
            priors_fit,
            img_width=r["img_w"],
            img_height=r["img_h"],
            eps=BORDER_EPS,
        )
        z_w[i] = res.Z_w[0]
        z_h[i] = res.Z_h[0]
        z_g[i] = res.Z_g[0]

    valid_mask = np.column_stack([~np.isnan(z_w), ~np.isnan(z_h), ~np.isnan(z_g)])
    z_cues = np.column_stack([z_w, z_h, z_g])

    fusion_weights = fit_fusion_weights(z_cues, z_gt, valid_mask, drives)
    z_d = fuse_depths(z_cues, valid_mask, fusion_weights)

    df_b = pd.DataFrame({
        "frame_id": [r["frame_id"] for r in records],
        "drive": drives,
        "cls": [r["cls"] for r in records],
        "difficulty": [r["difficulty"] for r in records],
        "z_gt": z_gt,
        "z_w": z_w,
        "z_h": z_h,
        "z_g": z_g,
        "z_d": z_d,
    })

    print("\n" + "=" * 80)
    print("1. EVALUATE_REPORT FOR (d) FUSED PREDICTIONS (includes valid_frac and no-estimate):")
    print("=" * 80)
    rep_d = evaluate_report(df_b, pred_col="z_d")
    print(rep_d.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. EVALUATE_REPORT FOR (b) Z_h PREDICTIONS:")
    print("=" * 80)
    rep_h = evaluate_report(df_b, pred_col="z_h")
    print(rep_h.to_string(index=False))

    print("\n" + "=" * 80)
    print("3. EVALUATE_REPORT FOR (c) Z_g PREDICTIONS:")
    print("=" * 80)
    rep_g = evaluate_report(df_b, pred_col="z_g")
    print(rep_g.to_string(index=False))

    print("\n" + "=" * 80)
    print("4. PAIRED CLUSTER BOOTSTRAP (Resampling whole drives on common support):")
    print("=" * 80)

    # Paired test: z_d vs z_h on AbsRel
    print("\n--- Paired Bootstrap: Fused (d) vs Z_h on Split B (Overall) ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r_dh = paired_cluster_bootstrap(df_b, "z_d", "z_h", metric="absrel", seed=42, n_boot=1000)
        for warn in w:
            print(f"[Warning caught]: {warn.message}")
    print(f"Difference (d - Z_h): estimate={r_dh.estimate:+.4f}, 95% CI=[{r_dh.ci_low:+.4f}, {r_dh.ci_high:+.4f}]")
    print(f"Excludes zero: {r_dh.excludes_zero} | Clusters: {r_dh.n_clusters} drives | Rows on common support: {r_dh.n_rows}")

    # Paired test: z_d vs z_g on AbsRel
    print("\n--- Paired Bootstrap: Fused (d) vs Z_g on Split B (Overall) ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r_dg = paired_cluster_bootstrap(df_b, "z_d", "z_g", metric="absrel", seed=42, n_boot=1000)
        for warn in w:
            print(f"[Warning caught]: {warn.message}")
    print(f"Difference (d - Z_g): estimate={r_dg.estimate:+.4f}, 95% CI=[{r_dg.ci_low:+.4f}, {r_dg.ci_high:+.4f}]")
    print(f"Excludes zero: {r_dg.excludes_zero} | Clusters: {r_dg.n_clusters} drives | Rows on common support: {r_dg.n_rows}")

    # Per-band paired bootstrap
    print("\n--- Paired Bootstrap per Distance Band (Fused d vs Best Single Cue) ---")
    bands = ["0-10", "10-20", "20-30", "30-50"]
    for b in bands:
        if b == "0-10":
            df_sub = df_b[df_b["z_gt"] < 10].copy()
            best_sc = "z_g"
        elif b == "10-20":
            df_sub = df_b[(df_b["z_gt"] >= 10) & (df_b["z_gt"] < 20)].copy()
            best_sc = "z_h"
        elif b == "20-30":
            df_sub = df_b[(df_b["z_gt"] >= 20) & (df_b["z_gt"] < 30)].copy()
            best_sc = "z_h"
        elif b == "30-50":
            df_sub = df_b[(df_b["z_gt"] >= 30) & (df_b["z_gt"] < 50)].copy()
            best_sc = "z_h"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r_band = paired_cluster_bootstrap(df_sub, "z_d", best_sc, metric="absrel", seed=42, n_boot=1000)
        print(f"Band {b}m (d vs {best_sc}): diff={r_band.estimate:+.4f} [{r_band.ci_low:+.4f}, {r_band.ci_high:+.4f}] | excludes_0={r_band.excludes_zero} (clusters={r_band.n_clusters})")


if __name__ == "__main__":
    main()
