"""
scripts/diagnose_day4.py: Diagnostic analysis for Day 4 Geometric Depth Cues.

Addresses:
- 2.1: Gate counting on 4 independent ranges (n >= 100: 0-10, 10-20, 20-30, 30-50m).
- 2.2: Common support comparison & breakdown by cue pattern (e.g. '100' vs '111').
- 2.4: Per-image actual dimensions (PIL header read) with border tolerance eps=2px.
- 2.4: Clarification of in-sample vs out-of-fold and drive counts.
"""

from __future__ import annotations
import sys
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

BINS = [0, 10, 20, 30, 50, np.inf]
LABELS = ["0-10", "10-20", "20-30", "30-50", ">50"]
CUES = ["z_w", "z_h", "z_g", "z_d"]


def _band(z: pd.Series) -> pd.Series:
    return pd.cut(z, BINS, labels=LABELS, right=False)


def common_support_table(df: pd.DataFrame) -> pd.DataFrame:
    """df has z_gt, z_w, z_h, z_g, z_d (NaN if cue invalid). Keep only full 3 cues."""
    full = df.dropna(subset=CUES).assign(band=lambda d: _band(d["z_gt"]))
    rows = [
        {
            "band": b,
            "n": len(g),
            **{c: float(np.mean(np.abs(g[c] - g["z_gt"]) / g["z_gt"])) for c in CUES},
        }
        for b, g in full.groupby("band", observed=True)
    ]
    return pd.DataFrame(rows)


def fused_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """AbsRel of (d) broken down by valid cue combination pattern (e.g. '100' = only Z_w)."""
    d = df.dropna(subset=["z_d"]).copy()
    d["pattern"] = d[["z_w", "z_h", "z_g"]].notna().astype(int).astype(str).apply("".join, axis=1)
    d["band"] = _band(d["z_gt"])
    d["absrel"] = np.abs(d["z_d"] - d["z_gt"]) / d["z_gt"]
    return d.groupby(["band", "pattern"], observed=True)["absrel"].agg(["count", "mean"])


def full_support_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standard table where each cue is evaluated on its own available samples."""
    d = df.copy().assign(band=lambda x: _band(x["z_gt"]))
    rows = []
    for b, g in d.groupby("band", observed=True):
        row = {"band": b, "n_total": len(g)}
        for c in CUES:
            valid = g.dropna(subset=[c])
            row[f"{c}_n"] = len(valid)
            row[f"{c}_absrel"] = float(np.mean(np.abs(valid[c] - valid["z_gt"]) / valid["z_gt"])) if len(valid) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("RUNNING DAY 4 DIAGNOSTIC ANALYSIS (Common Support, Pattern Breakdown, Exact Sizes)")
    print("=" * 80)

    loader = KITTILoader("data/kitti")
    split_b_ids = load_split("splits", "B", allow_test=False)

    priors_yaml = PROJECT_ROOT / "configs" / "geometry_priors.yaml"
    with open(priors_yaml, "r", encoding="utf-8") as f:
        priors_cfg = yaml.safe_load(f)

    w_eff = priors_cfg["classes"]["Car"]["W_eff_median"]
    h_obj = priors_cfg["classes"]["Car"]["H_obj_median"]
    delta_fitted = -4.6782
    h_cam_fitted = 2.0422

    priors_fit = GeometricPriors(
        W_eff=w_eff,
        H_obj=h_obj,
        H_cam=h_cam_fitted,
        delta_horizon=delta_fitted,
    )

    records = []
    print("\nExtracting Car Hard objects from Split B with exact image dimensions...")
    for fid in tqdm(split_b_ids, desc="Loading Split B"):
        frame = loader.load_frame(fid)
        # Read exact image size from image file
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
                "intrinsics": intrinsics,
                "img_w": img_w,
                "img_h": img_h,
            })

    n = len(records)
    print(f"Total Car Hard objects: {n}")

    # Compute cues
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

    # Fit fusion weights on Split B
    valid_mask = np.column_stack([~np.isnan(z_w), ~np.isnan(z_h), ~np.isnan(z_g)])
    z_cues = np.column_stack([z_w, z_h, z_g])

    fusion_weights = fit_fusion_weights(z_cues, z_gt, valid_mask, drives)
    z_d = fuse_depths(z_cues, valid_mask, fusion_weights)

    df = pd.DataFrame({
        "frame_id": [r["frame_id"] for r in records],
        "drive": drives,
        "z_gt": z_gt,
        "z_w": z_w,
        "z_h": z_h,
        "z_g": z_g,
        "z_d": z_d,
    })

    print("\n" + "=" * 80)
    print("1. FULL SUPPORT TABLE (Standard per-cue evaluation):")
    print("=" * 80)
    df_full = full_support_table(df)
    print(df_full.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. COMMON SUPPORT TABLE (Evaluated strictly on samples with all 3 cues valid):")
    print("=" * 80)
    df_common = common_support_table(df)
    print(df_common.to_string(index=False))

    print("\n" + "=" * 80)
    print("3. FUSED DEPTH (d) BROKEN DOWN BY CUE VALIDITY PATTERN:")
    print("=" * 80)
    df_pattern = fused_by_pattern(df)
    print(df_pattern.to_string())

    print("\n" + "=" * 80)
    print("4. FOCUS ON 0-10m RANGE (Checking 69-car hypothesis):")
    print("=" * 80)
    d010 = df[df["z_gt"] < 10].copy()
    d010["pattern"] = d010[["z_w", "z_h", "z_g"]].notna().astype(int).astype(str).apply("".join, axis=1)
    d010["absrel_d"] = np.abs(d010["z_d"] - d010["z_gt"]) / d010["z_gt"]
    d010["absrel_w"] = np.abs(d010["z_w"] - d010["z_gt"]) / d010["z_gt"]
    d010["absrel_h"] = np.abs(d010["z_h"] - d010["z_gt"]) / d010["z_gt"]
    d010["absrel_g"] = np.abs(d010["z_g"] - d010["z_gt"]) / d010["z_gt"]

    print("Pattern distribution in 0-10m:")
    print(d010["pattern"].value_counts())
    print("\nAbsRel by pattern in 0-10m:")
    for pat, grp in d010.groupby("pattern"):
        print(f"  Pattern {pat}: n={len(grp)}, mean AbsRel(d) = {grp['absrel_d'].mean():.4f}")
        if grp["absrel_w"].notna().any():
            print(f"    mean AbsRel(Z_w) = {grp['absrel_w'].dropna().mean():.4f}")
        if grp["absrel_h"].notna().any():
            print(f"    mean AbsRel(Z_h) = {grp['absrel_h'].dropna().mean():.4f}")
        if grp["absrel_g"].notna().any():
            print(f"    mean AbsRel(Z_g) = {grp['absrel_g'].dropna().mean():.4f}")

    print("\n" + "=" * 80)
    print("5. GATE RE-CALCULATION (§8.1, Decision D9):")
    print("=" * 80)
    print("Independent ranges (n >= 100):")
    # Independent ranges
    indep_bands = ["0-10", "10-20", "20-30", "30-50"]
    for b in indep_bands:
        row_c = df_common[df_common["band"] == b]
        row_f = df_full[df_full["band"] == b]
        print(f"\nRange {b}m:")
        print(f"  Full Support:   n(d)={row_f['z_d_n'].values[0]}, AbsRel(d)={row_f['z_d_absrel'].values[0]:.4f} | Best single: min({row_f['z_w_absrel'].values[0]:.4f}, {row_f['z_h_absrel'].values[0]:.4f}, {row_f['z_g_absrel'].values[0]:.4f})")
        if len(row_c) > 0:
            print(f"  Common Support: n={row_c['n'].values[0]}, AbsRel(d)={row_c['z_d'].values[0]:.4f} | Z_w={row_c['z_w'].values[0]:.4f}, Z_h={row_c['z_h'].values[0]:.4f}, Z_g={row_c['z_g'].values[0]:.4f}")


if __name__ == "__main__":
    main()
