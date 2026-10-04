"""
scripts/eval_day6_analysis.py: Day 6 GT Bbox In-depth Analysis on Split B (splits-v2).

Implements:
1. Leave-one-drive-out (LODO) OOF cross-validation for fusion (d) on Split B (12 folds).
   - Centered covariance shrinkage (standard) vs. Uncentered second-moment matrix M = E[e e^T].
2. AbsRel stratified by viewing angle theta = min(|alpha|, pi - |alpha|) with 3 bins:
   - Front/rear (<30 deg), Diagonal (30-60 deg), Side (>60 deg) to test Hypothesis H1 on Z_w.
3. Per-drive error table, per-drive delta, and sign count (d vs Z_h, d vs Z_g).
4. Descriptive cluster bootstrap CI (coarse, 12 clusters).
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    paired_cluster_bootstrap,
    cluster_bootstrap,
)

BINS = [0, 10, 20, 30, 50, np.inf]
LABELS = ["0-10", "10-20", "20-30", "30-50", ">50"]


def compute_angle_theta(alpha: float) -> float:
    """Compute viewing angle theta = min(|alpha|, pi - |alpha|) in degrees."""
    abs_alpha = abs(alpha)
    folded = min(abs_alpha, np.pi - abs_alpha)
    return np.degrees(folded)


def fuse_single(zw: float, zh: float, zg: float, cov: np.ndarray) -> float:
    cues = []
    indices = []
    for idx, c in enumerate([zw, zh, zg]):
        if np.isfinite(c) and c > 0:
            cues.append(c)
            indices.append(idx)
    if not cues:
        return np.nan
    if len(cues) == 1:
        return cues[0]
    sub_cov = cov[np.ix_(indices, indices)]
    inv = np.linalg.pinv(sub_cov)
    ones = np.ones(len(cues))
    denom = ones.T @ inv @ ones
    if abs(denom) > 1e-12:
        w = inv @ ones / denom
    else:
        w = np.full(len(cues), 1.0 / len(cues))
    if np.any(w < 0):
        from scipy.optimize import nnls
        w, _ = nnls(sub_cov, ones)
        if np.sum(w) > 0:
            w = w / np.sum(w)
        else:
            w = np.full(len(cues), 1.0 / len(cues))
    return float(np.exp(np.sum(w * np.log(cues))))


def main():
    print("=" * 80)
    print("DAY 6: GT BBOX IN-DEPTH ANALYSIS ON SPLIT B (LODO OOF, VIEWING ANGLE, PER-DRIVE)")
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

    priors = GeometricPriors(
        W_eff=w_eff,
        H_obj=h_obj,
        H_cam=h_cam_fitted,
        delta_horizon=delta_fitted,
    )

    records = []
    print(f"Loading Split B ({len(split_b_ids)} frames) from disk...")
    for fid in tqdm(split_b_ids, desc="Loading Split B"):
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

            theta_deg = compute_angle_theta(obj.alpha)
            records.append({
                "frame_id": fid,
                "drive": frame.drive,
                "bbox": obj.bbox,
                "z_gt": obj.depth,
                "alpha": obj.alpha,
                "theta_deg": theta_deg,
                "intrinsics": intrinsics,
                "img_w": img_w,
                "img_h": img_h,
            })

    n = len(records)
    print(f"Total Car Hard objects on Split B: {n}")

    # Compute cues
    z_w = np.full(n, np.nan)
    z_h = np.full(n, np.nan)
    z_g = np.full(n, np.nan)

    for i, r in enumerate(records):
        x1, y1, x2, y2 = r["bbox"]
        cues = compute_cues_batch(
            np.array([[x1, y1, x2, y2]]),
            r["intrinsics"],
            priors,
            r["img_w"],
            r["img_h"],
        )
        z_w[i] = cues.Z_w[0]
        z_h[i] = cues.Z_h[0]
        z_g[i] = cues.Z_g[0]

    df = pd.DataFrame(records)
    df["z_w"] = z_w
    df["z_h"] = z_h
    df["z_g"] = z_g
    df["band"] = pd.cut(df["z_gt"], BINS, labels=LABELS, right=False)

    # Filter to drives with cars
    car_drives = sorted(df["drive"].unique())
    print(f"Unique drives with Car Hard: {len(car_drives)}")

    # ---------------------------------------------------------------------------
    # 1. Leave-One-Drive-Out (LODO) OOF Cross-Validation
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. LEAVE-ONE-DRIVE-OUT (LODO) OOF EVALUATION (12 FOLDS)")
    print("=" * 80)

    # In-sample fusion (frozen geometry-v2 covariance for reference)
    frozen_cov = np.array(geom_cfg["fusion_split_B"]["covariance_shrunk"])
    z_d_in_sample = np.full(n, np.nan)
    for i in range(n):
        z_d_in_sample[i] = fuse_single(z_w[i], z_h[i], z_g[i], frozen_cov)
    df["z_d_insample"] = z_d_in_sample

    # LODO OOF predictions
    z_d_oof_centered = np.full(n, np.nan)
    z_d_oof_uncentered = np.full(n, np.nan)
    oof_weights_centered = {}
    oof_weights_uncentered = {}

    for d_val in car_drives:
        train_mask = df["drive"] != d_val
        val_mask = df["drive"] == d_val

        # Sub-dataframe with all 3 cues valid for training
        train_full = df[train_mask & df["z_w"].notna() & df["z_h"].notna() & df["z_g"].notna()]
        log_e = np.column_stack([
            np.log(train_full["z_w"]) - np.log(train_full["z_gt"]),
            np.log(train_full["z_h"]) - np.log(train_full["z_gt"]),
            np.log(train_full["z_g"]) - np.log(train_full["z_gt"]),
        ])

        # Centered covariance fit (Ledoit-Wolf)
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(log_e)
        cov_c = lw.covariance_
        inv_cov_c = np.linalg.pinv(cov_c)
        ones = np.ones(3)
        w_c = inv_cov_c @ ones / (ones.T @ inv_cov_c @ ones)
        if np.any(w_c < 0):
            from scipy.optimize import nnls
            w_c, _ = nnls(cov_c, ones)
            w_c = w_c / np.sum(w_c)
        oof_weights_centered[d_val] = w_c

        # Uncentered second-moment matrix M = (1/N) * sum(e e^T) (includes bias!)
        M = (log_e.T @ log_e) / len(log_e)
        inv_M = np.linalg.pinv(M)
        w_u = inv_M @ ones / (ones.T @ inv_M @ ones)
        if np.any(w_u < 0):
            from scipy.optimize import nnls
            w_u, _ = nnls(M, ones)
            w_u = w_u / np.sum(w_u)
        oof_weights_uncentered[d_val] = w_u

        # Evaluate on val_mask
        for idx in df[val_mask].index:
            zw_val, zh_val, zg_val = df.loc[idx, ["z_w", "z_h", "z_g"]]
            z_d_oof_centered[idx] = fuse_single(zw_val, zh_val, zg_val, cov_c)
            z_d_oof_uncentered[idx] = fuse_single(zw_val, zh_val, zg_val, M)

    df["z_d_oof"] = z_d_oof_centered
    df["z_d_oof_uncentered"] = z_d_oof_uncentered

    # Compare Overall and Per-Band Metrics
    print(f"\n{'Range':<8} | {'N':>5} | {'In-Sample (d)':>14} | {'OOF Centered (d)':>16} | {'OOF Uncentered':>15} | {'Z_h (height)':>12}")
    print("-" * 78)
    for band_name in LABELS:
        sub = df[df["band"] == band_name]
        v_in = sub.dropna(subset=["z_d_insample"])
        v_oof = sub.dropna(subset=["z_d_oof"])
        v_unc = sub.dropna(subset=["z_d_oof_uncentered"])
        v_zh = sub.dropna(subset=["z_h"])

        ar_in = np.mean(np.abs(v_in["z_d_insample"] - v_in["z_gt"]) / v_in["z_gt"])
        ar_oof = np.mean(np.abs(v_oof["z_d_oof"] - v_oof["z_gt"]) / v_oof["z_gt"])
        ar_unc = np.mean(np.abs(v_unc["z_d_oof_uncentered"] - v_unc["z_gt"]) / v_unc["z_gt"])
        ar_zh = np.mean(np.abs(v_zh["z_h"] - v_zh["z_gt"]) / v_zh["z_gt"])
        star = "*" if len(v_in) < 100 else ""
        print(f"{band_name:<8} | {len(v_in):>4}{star} | {ar_in:>14.4f} | {ar_oof:>16.4f} | {ar_unc:>15.4f} | {ar_zh:>12.4f}")

    # Overall
    ar_in_all = np.mean(np.abs(df["z_d_insample"].dropna() - df.loc[df["z_d_insample"].notna(), "z_gt"]) / df.loc[df["z_d_insample"].notna(), "z_gt"])
    ar_oof_all = np.mean(np.abs(df["z_d_oof"].dropna() - df.loc[df["z_d_oof"].notna(), "z_gt"]) / df.loc[df["z_d_oof"].notna(), "z_gt"])
    ar_unc_all = np.mean(np.abs(df["z_d_oof_uncentered"].dropna() - df.loc[df["z_d_oof_uncentered"].notna(), "z_gt"]) / df.loc[df["z_d_oof_uncentered"].notna(), "z_gt"])
    ar_zh_all = np.mean(np.abs(df["z_h"].dropna() - df.loc[df["z_h"].notna(), "z_gt"]) / df.loc[df["z_h"].notna(), "z_gt"])
    print("-" * 78)
    print(f"{'Overall':<8} | {df['z_d_oof'].notna().sum():>5} | {ar_in_all:>14.4f} | {ar_oof_all:>16.4f} | {ar_unc_all:>15.4f} | {ar_zh_all:>12.4f}")

    # Mean weights across 12 folds
    mean_w_c = np.mean(list(oof_weights_centered.values()), axis=0)
    std_w_c = np.std(list(oof_weights_centered.values()), axis=0)
    mean_w_u = np.mean(list(oof_weights_uncentered.values()), axis=0)
    print(f"\nMean LODO Centered Weights [Z_w, Z_h, Z_g]: {mean_w_c.round(4).tolist()} (std: {std_w_c.round(4).tolist()})")
    print(f"Mean LODO Uncentered Weights [Z_w, Z_h, Z_g]: {mean_w_u.round(4).tolist()}")

    # ---------------------------------------------------------------------------
    # 2. AbsRel Stratified by Viewing Angle theta = min(|alpha|, pi - |alpha|)
    # KITTI Convention: |alpha| ≈ pi/2 is Front/Rear (>60 deg), alpha ≈ 0 or pi is Side (<30 deg)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. ERROR VS. VIEWING ANGLE (Side <30 deg, Diagonal 30-60 deg, Front/Rear >60 deg)")
    print("=" * 80)
    angle_bins = [0, 30, 60, 90.001]
    angle_labels = ["<30 deg (Side / Ngang)", "30-60 deg (Diagonal / Chéo)", ">60 deg (Front/Rear / Đầu-Đuôi)"]
    df["angle_bin"] = pd.cut(df["theta_deg"], angle_bins, labels=angle_labels, right=False)

    print(f"{'Angle Bin':<32} | {'N':>5} | {'Z_w (width)':>12} | {'Z_h (height)':>12} | {'Z_g (ground)':>12} | {'(d) Fused OOF':>13}")
    print("-" * 95)
    angle_results = {}
    for ab in angle_labels:
        sub = df[df["angle_bin"] == ab]
        zw_v = sub["z_w"].dropna()
        zh_v = sub["z_h"].dropna()
        zg_v = sub["z_g"].dropna()
        zd_v = sub["z_d_oof"].dropna()

        ar_zw = float(np.mean(np.abs(zw_v - sub.loc[zw_v.index, "z_gt"]) / sub.loc[zw_v.index, "z_gt"]))
        ar_zh = float(np.mean(np.abs(zh_v - sub.loc[zh_v.index, "z_gt"]) / sub.loc[zh_v.index, "z_gt"]))
        ar_zg = float(np.mean(np.abs(zg_v - sub.loc[zg_v.index, "z_gt"]) / sub.loc[zg_v.index, "z_gt"]))
        ar_zd = float(np.mean(np.abs(zd_v - sub.loc[zd_v.index, "z_gt"]) / sub.loc[zd_v.index, "z_gt"]))
        print(f"{ab:<32} | {len(sub):>5} | {ar_zw:>12.4f} | {ar_zh:>12.4f} | {ar_zg:>12.4f} | {ar_zd:>13.4f}")
        angle_results[ab] = {
            "n": len(sub),
            "z_w": round(ar_zw, 4),
            "z_h": round(ar_zh, 4),
            "z_g": round(ar_zg, 4),
            "z_d_oof": round(ar_zd, 4),
        }

    # ---------------------------------------------------------------------------
    # 3. Signed Error by Distance Band (Checking positive bias of Z_g at 30-50m)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. SIGNED RELATIVE ERROR (Bias: mean((Z_pred - Z_gt) / Z_gt)) BY BAND")
    print("=" * 80)
    print(f"{'Range':<8} | {'N':>5} | {'Bias Z_w':>10} | {'Bias Z_h':>10} | {'Bias Z_g':>10} | {'Bias Fused (d)':>14}")
    print("-" * 65)
    for band_name in LABELS:
        sub = df[df["band"] == band_name]
        zw_v = sub["z_w"].dropna()
        zh_v = sub["z_h"].dropna()
        zg_v = sub["z_g"].dropna()
        zd_v = sub["z_d_oof"].dropna()

        b_zw = np.mean((zw_v - sub.loc[zw_v.index, "z_gt"]) / sub.loc[zw_v.index, "z_gt"])
        b_zh = np.mean((zh_v - sub.loc[zh_v.index, "z_gt"]) / sub.loc[zh_v.index, "z_gt"])
        b_zg = np.mean((zg_v - sub.loc[zg_v.index, "z_gt"]) / sub.loc[zg_v.index, "z_gt"])
        b_zd = np.mean((zd_v - sub.loc[zd_v.index, "z_gt"]) / sub.loc[zd_v.index, "z_gt"])
        star = "*" if len(sub) < 100 else ""
        print(f"{band_name:<8} | {len(sub):>4}{star} | {b_zw:>+10.4f} | {b_zh:>+10.4f} | {b_zg:>+10.4f} | {b_zd:>+14.4f}")

    # ---------------------------------------------------------------------------
    # 4. Per-Drive Table & Sign Count (d vs Z_h, d vs Z_g)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. PER-DRIVE ERROR TABLE & SIGN COUNT (Split B v2, 12 Drives)")
    print("=" * 80)
    print(f"{'Drive':<28} | {'N Cars':>6} | {'(d) OOF':>8} | {'Z_h':>8} | {'Z_g':>8} | {'Delta (d - Z_h)':>15} | {'Win vs Z_h?':>11}")
    print("-" * 92)

    drive_stats = []
    for d in car_drives:
        sub = df[df["drive"] == d]
        v_d = sub["z_d_oof"].dropna()
        v_h = sub["z_h"].dropna()
        v_g = sub["z_g"].dropna()

        ar_d = float(np.mean(np.abs(v_d - sub.loc[v_d.index, "z_gt"]) / sub.loc[v_d.index, "z_gt"]))
        ar_h = float(np.mean(np.abs(v_h - sub.loc[v_h.index, "z_gt"]) / sub.loc[v_h.index, "z_gt"]))
        ar_g = float(np.mean(np.abs(v_g - sub.loc[v_g.index, "z_gt"]) / sub.loc[v_g.index, "z_gt"]))
        delta = ar_d - ar_h
        win_h = "WIN (d)" if delta < -0.0005 else ("TIED" if abs(delta) <= 0.0005 else "LOSS")

        drive_stats.append({
            "drive": d,
            "n": len(sub),
            "ar_d": ar_d,
            "ar_h": ar_h,
            "ar_g": ar_g,
            "delta_h": delta,
            "win_h": win_h,
        })
        print(f"{d:<28} | {len(sub):>6} | {ar_d:>8.4f} | {ar_h:>8.4f} | {ar_g:>8.4f} | {delta:>+15.4f} | {win_h:>11}")

    wins = sum(1 for s in drive_stats if s["win_h"] == "WIN (d)")
    losses = sum(1 for s in drive_stats if s["win_h"] == "LOSS")
    ties = sum(1 for s in drive_stats if s["win_h"] == "TIED")
    print("-" * 92)
    from scipy.stats import binomtest
    p_val_sign = float(binomtest(k=wins, n=len(car_drives), p=0.5, alternative="greater").pvalue)
    print(f"Sign Count across 12 Drives: Fused (d) wins in {wins}/12 drives (Loss: {losses}, Tied: {ties}, p-value: {p_val_sign:.4f})")

    # Macro averages across 12 drives (Decision D18 / D20)
    macro_d = float(np.mean([s["ar_d"] for s in drive_stats]))
    macro_h = float(np.mean([s["ar_h"] for s in drive_stats]))
    macro_g = float(np.mean([s["ar_g"] for s in drive_stats]))
    print(f"Macro AbsRel (unweighted 12 drives): (d) OOF = {macro_d:.4f} | Z_h = {macro_h:.4f} | Z_g = {macro_g:.4f}")

    drives_ge_30 = [s for s in drive_stats if s["n"] >= 30]
    macro_d_30 = float(np.mean([s["ar_d"] for s in drives_ge_30]))
    macro_h_30 = float(np.mean([s["ar_h"] for s in drives_ge_30]))
    print(f"Macro AbsRel (drives with n >= 30, k={len(drives_ge_30)}): (d) OOF = {macro_d_30:.4f} | Z_h = {macro_h_30:.4f}")

    # ---------------------------------------------------------------------------
    # 5. Paired Cluster Bootstrap (Descriptive CI, 12 clusters)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. PAIRED CLUSTER BOOTSTRAP (12 CLUSTERS, DESCRIPTIVE COARSE CI)")
    print("=" * 80)
    common = df.dropna(subset=["z_d_oof", "z_h"])
    boot_h = paired_cluster_bootstrap(
        df=common,
        pred_col_a="z_d_oof",
        pred_col_b="z_h",
        metric="absrel",
        seed=42,
        n_boot=2000,
    )
    print("Paired Bootstrap: (d) OOF vs Z_h (Common Support, n=4676):")
    print(f"  Estimate (d - Z_h): {boot_h.estimate:+.4f}")
    print(f"  95% CI (12 clusters, coarse): [{boot_h.ci_low:+.4f}, {boot_h.ci_high:+.4f}]")
    print(f"  CI excludes 0: {boot_h.excludes_zero}")

    # Save summary report to JSON
    report_out = PROJECT_ROOT / "results" / "tables" / "day6_gt_bbox_analysis.json"
    report_data = {
        "n_objects": n,
        "n_drives": len(car_drives),
        "mean_oof_weights_centered": mean_w_c.round(4).tolist(),
        "mean_oof_weights_uncentered": mean_w_u.round(4).tolist(),
        "pooled_absrel": {
            "in_sample": round(ar_in_all, 4),
            "oof_centered": round(ar_oof_all, 4),
            "oof_uncentered": round(ar_unc_all, 4),
            "z_h": round(ar_zh_all, 4),
        },
        "macro_absrel_12_drives": {
            "oof_centered": round(macro_d, 4),
            "z_h": round(macro_h, 4),
            "z_g": round(macro_g, 4),
        },
        "macro_absrel_drives_ge_30": {
            "oof_centered": round(macro_d_30, 4),
            "z_h": round(macro_h_30, 4),
            "n_drives": len(drives_ge_30),
        },
        "sign_count_vs_zh": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total_drives": len(car_drives),
            "p_value_one_sided": round(p_val_sign, 4),
        },
        "bootstrap_vs_zh": {
            "diff": round(boot_h.estimate, 4),
            "ci": [round(boot_h.ci_low, 4), round(boot_h.ci_high, 4)],
            "excludes_zero": bool(boot_h.excludes_zero),
        },
        "viewing_angle_stratification": angle_results,
        "decisions_applied": ["D18", "D19", "D20", "D21"],
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved Day 6 comprehensive analysis to: {report_out}")


if __name__ == "__main__":
    main()
