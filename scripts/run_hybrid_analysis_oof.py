"""
scripts/run_hybrid_analysis_oof.py: Thí nghiệm chuyên sâu so sánh Hybrid (f) vs Direct (e) vs Fused Geometry (d).
Thực thi theo pre-registration configs/residual/hybrid_prereg_v1.yaml (D131):
- (a) Phân tích nhóm con (Subgroups): theta (side/diag/front-rear), truncated, fallback, dải cự ly.
- (b) Đường cong học (Learning Curve): k in {2, 4, 6, 8, 11} drives huấn luyện.
- (c) Ngoại suy cự ly (Range Extrapolation): train Z <= 30m, eval Z > 30m.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS_E = [
    "w", "h", "w_h_ratio", "y_bottom_minus_cy", "cx_offset_norm",
    "touch_left", "touch_right", "touch_top", "touch_bottom", "confidence",
]

FEATURE_COLS_F = [
    "w", "h", "w_h_ratio", "y_bottom_minus_cy", "cx_offset_norm",
    "touch_left", "touch_right", "touch_top", "touch_bottom", "confidence",
    "ln_z_w", "ln_z_h", "ln_z_g", "valid_w", "valid_h", "valid_g", "ln_z_base",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if not np.any(valid_mask):
        return {"absrel": float("nan"), "mae": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "n": 0}
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    absrel = float(np.mean(np.abs(yp - yt) / yt))
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    delta = np.maximum(yp / yt, yt / yp)
    delta1 = float(np.mean(delta < 1.25))
    return {"absrel": absrel, "mae": mae, "rmse": rmse, "delta1": delta1, "n": int(len(yt))}


def run_subgroup_analysis(df: pd.DataFrame) -> dict[str, Any]:
    print("Running (a) Subgroup Analysis on existing B_oof...")
    # 1. Viewing angle theta
    alpha = df["alpha"].to_numpy()
    abs_alpha = np.abs(alpha)
    theta_rad = np.minimum(abs_alpha, np.pi - abs_alpha)
    theta_deg = theta_rad * 180.0 / np.pi

    theta_groups = {
        "side (<30 deg)": theta_deg < 30.0,
        "diagonal (30-60 deg)": (theta_deg >= 30.0) & (theta_deg <= 60.0),
        "front_rear (>60 deg)": theta_deg > 60.0,
    }
    theta_results = {}
    for name, mask in theta_groups.items():
        sub = df[mask]
        theta_results[name] = {
            "n": int(len(sub)),
            "d": compute_metrics(sub["z_gt"].to_numpy(), sub["z_d"].to_numpy()),
            "e": compute_metrics(sub["z_gt"].to_numpy(), sub["z_hat_e"].to_numpy()),
            "f": compute_metrics(sub["z_gt"].to_numpy(), sub["z_hat_f"].to_numpy()),
        }

    # 2. Truncation
    trunc_mask = df["truncated"] > 0.0
    trunc_results = {
        "untruncated (t=0)": {
            "n": int(np.sum(~trunc_mask)),
            "d": compute_metrics(df.loc[~trunc_mask, "z_gt"].to_numpy(), df.loc[~trunc_mask, "z_d"].to_numpy()),
            "e": compute_metrics(df.loc[~trunc_mask, "z_gt"].to_numpy(), df.loc[~trunc_mask, "z_hat_e"].to_numpy()),
            "f": compute_metrics(df.loc[~trunc_mask, "z_gt"].to_numpy(), df.loc[~trunc_mask, "z_hat_f"].to_numpy()),
        },
        "truncated (t>0)": {
            "n": int(np.sum(trunc_mask)),
            "d": compute_metrics(df.loc[trunc_mask, "z_gt"].to_numpy(), df.loc[trunc_mask, "z_d"].to_numpy()),
            "e": compute_metrics(df.loc[trunc_mask, "z_gt"].to_numpy(), df.loc[trunc_mask, "z_hat_e"].to_numpy()),
            "f": compute_metrics(df.loc[trunc_mask, "z_gt"].to_numpy(), df.loc[trunc_mask, "z_hat_f"].to_numpy()),
        },
    }

    # 3. Fallback pattern
    fb_mask = df["fallback_flag"]
    fallback_results = {
        "normal (valid cues)": {
            "n": int(np.sum(~fb_mask)),
            "d": compute_metrics(df.loc[~fb_mask, "z_gt"].to_numpy(), df.loc[~fb_mask, "z_d"].to_numpy()),
            "e": compute_metrics(df.loc[~fb_mask, "z_gt"].to_numpy(), df.loc[~fb_mask, "z_hat_e"].to_numpy()),
            "f": compute_metrics(df.loc[~fb_mask, "z_gt"].to_numpy(), df.loc[~fb_mask, "z_hat_f"].to_numpy()),
        },
        "fallback (pattern 000)": {
            "n": int(np.sum(fb_mask)),
            "d": compute_metrics(df.loc[fb_mask, "z_gt"].to_numpy(), df.loc[fb_mask, "z_d"].to_numpy()),
            "e": compute_metrics(df.loc[fb_mask, "z_gt"].to_numpy(), df.loc[fb_mask, "z_hat_e"].to_numpy()),
            "f": compute_metrics(df.loc[fb_mask, "z_gt"].to_numpy(), df.loc[fb_mask, "z_hat_f"].to_numpy()),
        },
    }

    # 4. Distance ranges
    dist_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 150)]
    dist_results = {}
    for lo, hi in dist_bins:
        name = f"{lo}-{hi}m" if hi < 100 else ">50m"
        mask = (df["z_gt"] >= lo) & (df["z_gt"] < hi)
        sub = df[mask]
        dist_results[name] = {
            "n": int(len(sub)),
            "d": compute_metrics(sub["z_gt"].to_numpy(), sub["z_d"].to_numpy()),
            "e": compute_metrics(sub["z_gt"].to_numpy(), sub["z_hat_e"].to_numpy()),
            "f": compute_metrics(sub["z_gt"].to_numpy(), sub["z_hat_f"].to_numpy()),
        }

    return {
        "viewing_angle_theta": theta_results,
        "truncation": trunc_results,
        "fallback": fallback_results,
        "distance_bins": dist_results,
    }


def run_learning_curve(full_df: pd.DataFrame) -> dict[str, Any]:
    print("Running (b) Learning Curve across drive counts k in [2, 4, 6, 8, 11]...")
    unique_drives = sorted(full_df["drive"].unique())
    k_list = [2, 4, 6, 8, 11]
    results = {}

    rng = np.random.RandomState(42)

    for k in k_list:
        preds_e_all = []
        preds_f_all = []
        targets_all = []
        preds_d_all = []

        for held_out in unique_drives:
            train_pool = [d for d in unique_drives if d != held_out]
            # Select k drives from train_pool deterministically with seed
            if k == len(train_pool):
                selected_drives = train_pool
            else:
                # Deterministic selection based on seed and held_out drive hash
                sub_seed = (42 + hash(held_out) + k * 101) % (2**31)
                sub_rng = np.random.RandomState(sub_seed)
                selected_drives = list(sub_rng.choice(train_pool, size=k, replace=False))

            train_mask = full_df["drive"].isin(selected_drives)
            test_mask = full_df["drive"] == held_out

            train_data = full_df[train_mask]
            test_data = full_df[test_mask]

            if len(test_data) == 0 or len(train_data) == 0:
                continue

            # Fit model (e): direct log depth
            X_train_e = train_data[FEATURE_COLS_E].to_numpy()
            y_train_e = np.log(train_data["z_gt"].to_numpy())
            X_test_e = test_data[FEATURE_COLS_E].to_numpy()

            reg_e = xgb.XGBRegressor(
                n_estimators=150, max_depth=3, min_child_weight=15,
                learning_rate=0.05, random_state=42, n_jobs=1
            )
            reg_e.fit(X_train_e, y_train_e)
            pred_e = np.exp(reg_e.predict(X_test_e))

            # Fit model (f): residual on log z_base
            X_train_f = train_data[FEATURE_COLS_F].to_numpy()
            y_train_f = np.log(train_data["z_gt"].to_numpy()) - np.log(train_data["z_base"].to_numpy())
            X_test_f = test_data[FEATURE_COLS_F].to_numpy()

            reg_f = xgb.XGBRegressor(
                n_estimators=150, max_depth=3, min_child_weight=15,
                learning_rate=0.05, random_state=42, n_jobs=1
            )
            reg_f.fit(X_train_f, y_train_f)
            pred_f = test_data["z_base"].to_numpy() * np.exp(reg_f.predict(X_test_f))

            preds_e_all.extend(pred_e)
            preds_f_all.extend(pred_f)
            preds_d_all.extend(test_data["z_d"].to_numpy())
            targets_all.extend(test_data["z_gt"].to_numpy())

        m_d = compute_metrics(np.array(targets_all), np.array(preds_d_all))
        m_e = compute_metrics(np.array(targets_all), np.array(preds_e_all))
        m_f = compute_metrics(np.array(targets_all), np.array(preds_f_all))

        results[f"k_{k}_drives"] = {
            "k": k,
            "d_absrel": m_d["absrel"],
            "e_absrel": m_e["absrel"],
            "f_absrel": m_f["absrel"],
            "delta_f_minus_e": m_f["absrel"] - m_e["absrel"],
            "n_eval": len(targets_all),
        }
        print(f"  k={k:2d} drives -> (d) AbsRel: {m_d['absrel']:.4f} | (e) Direct: {m_e['absrel']:.4f} | (f) Hybrid: {m_f['absrel']:.4f} | Diff: {m_f['absrel'] - m_e['absrel']:+.4f}")

    return results


def run_range_extrapolation(full_df: pd.DataFrame) -> dict[str, Any]:
    print("Running (c) Range Extrapolation (Train Z <= 30m, Eval Z > 30m)...")
    train_mask = full_df["z_gt"] <= 30.0
    test_mask = full_df["z_gt"] > 30.0

    train_data = full_df[train_mask]
    test_data = full_df[test_mask]

    # Fit Model (e) on near/mid only
    X_train_e = train_data[FEATURE_COLS_E].to_numpy()
    y_train_e = np.log(train_data["z_gt"].to_numpy())
    X_test_e = test_data[FEATURE_COLS_E].to_numpy()

    reg_e = xgb.XGBRegressor(
        n_estimators=150, max_depth=3, min_child_weight=15,
        learning_rate=0.05, random_state=42, n_jobs=1
    )
    reg_e.fit(X_train_e, y_train_e)
    pred_e = np.exp(reg_e.predict(X_test_e))

    # Fit Model (f) on near/mid only
    X_train_f = train_data[FEATURE_COLS_F].to_numpy()
    y_train_f = np.log(train_data["z_gt"].to_numpy()) - np.log(train_data["z_base"].to_numpy())
    X_test_f = test_data[FEATURE_COLS_F].to_numpy()

    reg_f = xgb.XGBRegressor(
        n_estimators=150, max_depth=3, min_child_weight=15,
        learning_rate=0.05, random_state=42, n_jobs=1
    )
    reg_f.fit(X_train_f, y_train_f)
    pred_f = test_data["z_base"].to_numpy() * np.exp(reg_f.predict(X_test_f))

    pred_d = test_data["z_d"].to_numpy()
    targets = test_data["z_gt"].to_numpy()

    m_d = compute_metrics(targets, pred_d)
    m_e = compute_metrics(targets, pred_e)
    m_f = compute_metrics(targets, pred_f)

    print(f"  Extrapolation Z > 30m (n={len(test_data)}):")
    print(f"    (d) Fused Geometry: AbsRel = {m_d['absrel']:.4f}, MAE = {m_d['mae']:.2f} m")
    print(f"    (e) Direct Model:   AbsRel = {m_e['absrel']:.4f}, MAE = {m_e['mae']:.2f} m")
    print(f"    (f) Hybrid Model:   AbsRel = {m_f['absrel']:.4f}, MAE = {m_f['mae']:.2f} m")

    return {
        "n_train_near": int(len(train_data)),
        "n_test_far": int(len(test_data)),
        "d_metrics": m_d,
        "e_metrics": m_e,
        "f_metrics": m_f,
        "delta_f_minus_e": m_f["absrel"] - m_e["absrel"],
    }


def main():
    print("=" * 70)
    print("THÍ NGHIỆM CHUYÊN SÂU HYBRID VS DIRECT (Split B OOF, prereg-hybrid-v1)")
    print("=" * 70)

    oof_path = REPO_ROOT / "results" / "datasets" / "yolo11s_640_B_oof.parquet"
    eval_path = REPO_ROOT / "results" / "datasets" / "yolo11s_640_B_eval.parquet"
    feat_path = REPO_ROOT / "results" / "datasets" / "yolo11s_640_B_features.parquet"
    cues_path = REPO_ROOT / "results" / "datasets" / "yolo11s_640_B_cues.parquet"

    df_oof = pd.read_parquet(oof_path)
    df_eval = pd.read_parquet(eval_path)
    df_feat = pd.read_parquet(feat_path)
    df_cues = pd.read_parquet(cues_path)

    # Combine into single unified dataframe
    merged = df_oof.copy()
    merged["alpha"] = df_eval["alpha"].values
    merged["truncated"] = df_eval["truncated"].values
    merged["occluded"] = df_eval["occluded"].values
    merged["z_gt"] = df_eval["z_gt"].values
    merged["matched_iou"] = df_eval["matched_iou"].values

    # Add geometric features needed for (e) and (f)
    for c in ["w", "h", "w_h_ratio", "y_bottom_minus_cy", "cx_offset_norm",
              "touch_left", "touch_right", "touch_top", "touch_bottom", "confidence"]:
        if c in df_feat.columns:
            merged[c] = df_feat[c].values
        elif c in df_cues.columns:
            merged[c] = df_cues[c].values

    # Derived and log features
    merged["w"] = (df_feat["x2"] - df_feat["x1"]).values
    merged["h"] = (df_feat["y2"] - df_feat["y1"]).values
    merged["w_h_ratio"] = merged["w"] / np.maximum(merged["h"], 1.0)
    merged["y_bottom_minus_cy"] = (df_feat["y2"] - df_feat["cy"]).values
    merged["cx_offset_norm"] = np.abs((df_feat["x1"] + df_feat["x2"]) / 2.0 - df_feat["cx"]).values / df_feat["img_w"].values
    merged["confidence"] = df_feat["confidence"].values
    merged["touch_left"] = (df_feat["x1"] <= 2).astype(float).values
    merged["touch_right"] = (df_feat["x2"] >= df_feat["img_w"] - 2).astype(float).values
    merged["touch_top"] = (df_feat["y1"] <= 2).astype(float).values
    merged["touch_bottom"] = (df_feat["y2"] >= df_feat["img_h"] - 2).astype(float).values

    merged["ln_z_w"] = np.log(np.maximum(df_cues["z_w"].values, 0.1))
    merged["ln_z_h"] = np.log(np.maximum(df_cues["z_h"].values, 0.1))
    merged["ln_z_g"] = np.log(np.maximum(df_cues["z_g"].values, 0.1))
    merged["valid_w"] = df_cues["valid_w"].astype(float).values
    merged["valid_h"] = df_cues["valid_h"].astype(float).values
    merged["valid_g"] = df_cues["valid_g"].astype(float).values
    merged["ln_z_base"] = np.log(np.maximum(merged["z_base"].values, 0.1))

    # Run 3 experiments
    subgroup_res = run_subgroup_analysis(merged)
    learning_res = run_learning_curve(merged)
    extrap_res = run_range_extrapolation(merged)

    # Save to JSON
    out_dir = REPO_ROOT / "results" / "tables"
    out_json = out_dir / "hybrid_analysis_oof_b.json"
    full_output = {
        "metadata": {
            "prereg_tag": "prereg-hybrid-v1",
            "model": "yolo11s_640",
            "n_objects": len(merged),
            "description": "In-depth comparison of hybrid (f) vs direct (e) vs pinhole geometry (d) on Split B OOF"
        },
        "subgroup_analysis": subgroup_res,
        "learning_curve": learning_res,
        "range_extrapolation": extrap_res,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Save summary Markdown
    out_md = out_dir / "hybrid_analysis_oof_b.md"
    md_lines = [
        "# Báo cáo Phân tích Chuyên sâu Hybrid (f) vs Direct (e) trên Split B OOF",
        "",
        "> Pre-registration: `prereg-hybrid-v1` (D131). Tuyệt đối không chạm Split T.",
        "",
        "## 1. Phân tích Nhóm con (Subgroup Analysis)",
        "",
        "### A. Theo Góc Quan sát $\\theta$ (Viewing Angle)",
        "| Phân nhóm $\\theta$ | $N$ | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\\Delta(f - e)$ |",
        "|---|---|---|---|---|---|",
    ]
    for grp, vals in subgroup_res["viewing_angle_theta"].items():
        md_lines.append(f"| {grp} | {vals['n']} | {vals['d']['absrel']:.4f} | {vals['e']['absrel']:.4f} | {vals['f']['absrel']:.4f} | {vals['f']['absrel'] - vals['e']['absrel']:+.4f} |")

    md_lines.extend([
        "",
        "### B. Theo Trạng thái Cắt Viền (Truncation)",
        "| Trạng thái | $N$ | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\\Delta(f - e)$ |",
        "|---|---|---|---|---|---|",
    ])
    for grp, vals in subgroup_res["truncation"].items():
        md_lines.append(f"| {grp} | {vals['n']} | {vals['d']['absrel']:.4f} | {vals['e']['absrel']:.4f} | {vals['f']['absrel']:.4f} | {vals['f']['absrel'] - vals['e']['absrel']:+.4f} |")

    md_lines.extend([
        "",
        "### C. Theo Dải Khoảng cách",
        "| Dải khoảng cách | $N$ | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\\Delta(f - e)$ |",
        "|---|---|---|---|---|---|",
    ])
    for grp, vals in subgroup_res["distance_bins"].items():
        md_lines.append(f"| {grp} | {vals['n']} | {vals['d']['absrel']:.4f} | {vals['e']['absrel']:.4f} | {vals['f']['absrel']:.4f} | {vals['f']['absrel'] - vals['e']['absrel']:+.4f} |")

    md_lines.extend([
        "",
        "## 2. Đường cong học theo Số drive Huấn luyện (Learning Curve)",
        "",
        "| Số drive huấn luyện ($k$) | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\\Delta(f - e)$ |",
        "|---|---|---|---|---|",
    ])
    for k_key, vals in learning_res.items():
        md_lines.append(f"| $k = {vals['k']}$ drives | {vals['d_absrel']:.4f} | {vals['e_absrel']:.4f} | {vals['f_absrel']:.4f} | {vals['delta_f_minus_e']:+.4f} |")

    md_lines.extend([
        "",
        "## 3. Ngoại suy Cự ly xa (Range Extrapolation: Train $Z \\le 30$ m, Eval $Z > 30$ m)",
        "",
        f"- Mẫu huấn luyện cự ly gần/trung ($Z \\le 30$ m): **N = {extrap_res['n_train_near']}**",
        f"- Mẫu kiểm tra cự ly xa ($Z > 30$ m): **N = {extrap_res['n_test_far']}**",
        "",
        "| Phương pháp | AbsRel ($Z > 30$ m) | MAE ($Z > 30$ m) | $\\delta < 1.25$ |",
        "|---|---|---|---|",
        f"| (d) Fused Geometry | {extrap_res['d_metrics']['absrel']:.4f} | {extrap_res['d_metrics']['mae']:.2f} m | {extrap_res['d_metrics']['delta1']*100:.1f}% |",
        f"| (e) Direct Regression | {extrap_res['e_metrics']['absrel']:.4f} | {extrap_res['e_metrics']['mae']:.2f} m | {extrap_res['e_metrics']['delta1']*100:.1f}% |",
        f"| (f) Hybrid Residual | {extrap_res['f_metrics']['absrel']:.4f} | {extrap_res['f_metrics']['mae']:.2f} m | {extrap_res['f_metrics']['delta1']*100:.1f}% |",
        "",
        f"> **Chênh lệch $\\Delta(f - e)$:** **{extrap_res['delta_f_minus_e']:+.4f}**",
    ])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved results to:\n  - {out_json}\n  - {out_md}")
    print("=" * 70)


if __name__ == "__main__":
    main()
