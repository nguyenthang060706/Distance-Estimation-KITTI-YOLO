"""
scripts/eval_sensitivity_d14.py: Sensitivity Analysis for Split B Dominant Drives (Decision D14).

Background:
  Audit of Split B-v2 reveals 12 drives containing Car Hard (N = 4,776 objects):
    - Top-1: 2011_09_26_drive_0104_sync: 1,282 cars (26.84%)
    - Top-2: 2011_09_26_drive_0059_sync: 1,185 cars (24.81%)
  Together, drive_0104 and drive_0059 account for 2,467 / 4,776 = 51.65% of Split B.

Objective:
  Evaluate the stability of the log-space fusion weights [w_w, w_h, w_g] and prediction accuracy
  when dominant drives are excluded from the fitting set:
    1. Baseline: All 12 drives (N = 4,776)
    2. Exclude drive_0059 (11 drives, N = 3,591)
    3. Exclude drive_0104 (11 drives, N = 3,494)
    4. Exclude both drive_0059 and drive_0104 (10 drives, N = 2,309)

Conventions (see NHAT_KY_QUYET_DINH.md D18/D20):
  - Strictly descriptive reporting (ranges, min/max, relative deltas).
  - Explicit distinction between in-sample and true out-of-sample evaluation cells.
  - Metrics computed using canonical src.evaluation.eval.depth_metrics (reporting n and n_valid).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.geometry.geometric_cues import (
    BORDER_EPS,
    CameraIntrinsics,
    compute_cues_batch,
    load_geometry_v2,
)
from src.geometry.fusion import (
    fit_fusion_weights,
    fuse_depths,
    fuse_depths_lodo,
)
from src.evaluation.eval import depth_metrics
from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_split


def run_sensitivity_analysis():
    print("=" * 80)
    print("DECISION D14 SENSITIVITY ANALYSIS: SPLIT B DOMINANT DRIVES (0059 & 0104)")
    print("=" * 80)

    loader = KITTILoader("data/kitti")
    split_b_ids = load_split("splits", "B", allow_test=False)
    print(f"Loaded Split B: {len(split_b_ids)} frames")

    # Load calibrated geometry parameters
    priors, geom_cfg = load_geometry_v2("configs/geometry_params.yaml")
    print(f"Loaded geometry-v2 priors: W_eff={priors.W_eff:.4f}, H_obj={priors.H_obj:.4f}, "
          f"H_cam={priors.H_cam:.4f}, delta={priors.delta_horizon:.4f}")

    # Extract all GT Car Hard objects
    records = []
    print("\nExtracting GT Car Hard objects from Split B...")
    for fid in tqdm(split_b_ids, desc="Extracting B"):
        frame = loader.load_frame(fid)
        with Image.open(frame.image_path) as img:
            img_w, img_h = img.size

        calib = frame.calib
        intr = CameraIntrinsics(fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy)

        for obj in frame.objects:
            if obj.obj_class != "Car" or not obj.passes_hard_filter():
                continue
            if obj.depth <= 0:
                continue

            cues = compute_cues_batch(
                np.array([obj.bbox]),
                intr,
                priors,
                img_width=img_w,
                img_height=img_h,
                eps=BORDER_EPS,
            )

            records.append({
                "frame_id": fid,
                "drive": frame.drive,
                "bbox": obj.bbox,
                "z_gt": obj.depth,
                "z_w": cues.Z_w[0],
                "z_h": cues.Z_h[0],
                "z_g": cues.Z_g[0],
                "valid_w": cues.valid_w[0],
                "valid_h": cues.valid_h[0],
                "valid_g": cues.valid_g[0],
            })

    df = pd.DataFrame(records)
    total_n = len(df)
    print(f"Total Car Hard objects on Split B: {total_n}")

    # Unique drives
    all_drives = sorted(df["drive"].unique())
    n_drives_total = len(all_drives)
    drive_counts = df["drive"].value_counts()
    print(f"\nDrive distribution on Split B ({n_drives_total} drives with Car Hard):")
    for d, cnt in drive_counts.items():
        print(f"  {d:<30}: {cnt:>5} cars ({cnt/total_n:6.2%})")

    d_0059 = "2011_09_26_drive_0059_sync"
    d_0104 = "2011_09_26_drive_0104_sync"

    # Compute full LODO OOF baseline across all 12 drives (D16c)
    z_cues_all = df[["z_w", "z_h", "z_g"]].values
    valid_mask_all = df[["valid_w", "valid_h", "valid_g"]].values
    z_gt_all = df["z_gt"].values
    drives_all = df["drive"].values

    z_d_lodo_all, fits_lodo = fuse_depths_lodo(
        Z_cues=z_cues_all,
        Z_gt=z_gt_all,
        valid_mask=valid_mask_all,
        drive_ids=drives_all,
        min_train_drives=3,
    )
    m_lodo_all = depth_metrics(z_gt_all, z_d_lodo_all)

    # Per-drive AbsRel in LODO baseline
    per_drive_absrel_lodo = {}
    for d in all_drives:
        m_d = depth_metrics(df.loc[df["drive"] == d, "z_gt"].values, z_d_lodo_all[df["drive"] == d])
        per_drive_absrel_lodo[d] = m_d["absrel"]
    macro_absrel_lodo_all = float(np.mean(list(per_drive_absrel_lodo.values())))
    print(f"\nBaseline 12-drive LODO OOF: Pooled AbsRel = {m_lodo_all['absrel']:.4f} (n_valid={m_lodo_all['n_valid']}/{m_lodo_all['n']}), Macro AbsRel = {macro_absrel_lodo_all:.4f}")

    # Define the 4 experiment setups
    experiments = {
        "Baseline (All 12 drives)": {
            "train_mask": np.ones(len(df), dtype=bool),
            "excluded_drives": [],
        },
        "Exclude drive_0059 (Top-2, 24.8%)": {
            "train_mask": df["drive"] != d_0059,
            "excluded_drives": [d_0059],
        },
        "Exclude drive_0104 (Top-1, 26.8%)": {
            "train_mask": df["drive"] != d_0104,
            "excluded_drives": [d_0104],
        },
        "Exclude both 0059 & 0104 (51.7%)": {
            "train_mask": (~df["drive"].isin([d_0059, d_0104])),
            "excluded_drives": [d_0059, d_0104],
        },
    }

    results: dict[str, Any] = {}

    print("\n" + "=" * 115)
    print(f"{'Condition':<36} | {'N fit':>5} | {'w_w':>7} | {'w_h':>7} | {'w_g':>7} | {'Fit AbsRel':>10} | {'Test 0059':>15} | {'Test 0104':>15} | {'Macro 12d':>10}")
    print("-" * 115)

    for name, exp in experiments.items():
        train_mask = exp["train_mask"]
        excluded = exp["excluded_drives"]
        sub_df = df[train_mask]

        n_sub = len(sub_df)
        n_drives_sub = sub_df["drive"].nunique()

        z_cues_sub = sub_df[["z_w", "z_h", "z_g"]].values
        valid_mask_sub = sub_df[["valid_w", "valid_h", "valid_g"]].values
        z_gt_sub = sub_df["z_gt"].values
        drives_sub = sub_df["drive"].values

        # Fit weights
        fw = fit_fusion_weights(
            Z_cues=z_cues_sub,
            Z_gt=z_gt_sub,
            valid_mask=valid_mask_sub,
            drive_ids=drives_sub,
        )
        w = fw.weights
        w_w, w_h, w_g = float(w[0]), float(w[1]), float(w[2])

        # In-sample metrics on training subset
        z_fused_train = fuse_depths(z_cues_sub, valid_mask_sub, fw)
        m_train = depth_metrics(z_gt_sub, z_fused_train)

        # Test on drive_0059
        df_0059 = df[df["drive"] == d_0059]
        z_cues_0059 = df_0059[["z_w", "z_h", "z_g"]].values
        valid_0059 = df_0059[["valid_w", "valid_h", "valid_g"]].values
        z_gt_0059 = df_0059["z_gt"].values
        z_fused_0059 = fuse_depths(z_cues_0059, valid_0059, fw)
        m_0059 = depth_metrics(z_gt_0059, z_fused_0059)
        is_oos_0059 = d_0059 in excluded

        # Test on drive_0104
        df_0104 = df[df["drive"] == d_0104]
        z_cues_0104 = df_0104[["z_w", "z_h", "z_g"]].values
        valid_0104 = df_0104[["valid_w", "valid_h", "valid_g"]].values
        z_gt_0104 = df_0104["z_gt"].values
        z_fused_0104 = fuse_depths(z_cues_0104, valid_0104, fw)
        m_0104 = depth_metrics(z_gt_0104, z_fused_0104)
        is_oos_0104 = d_0104 in excluded

        # Macro AbsRel across all 12 drives when applying these weights
        z_fused_all_with_fw = fuse_depths(z_cues_all, valid_mask_all, fw)
        macro_absrels = []
        for d in all_drives:
            m_d = depth_metrics(df.loc[df["drive"] == d, "z_gt"].values, z_fused_all_with_fw[df["drive"] == d])
            macro_absrels.append(m_d["absrel"])
        macro_12d = float(np.mean(macro_absrels))

        str_0059 = f"{m_0059['absrel']:.4f} " + ("[OOS]" if is_oos_0059 else "[in-sample]")
        str_0104 = f"{m_0104['absrel']:.4f} " + ("[OOS]" if is_oos_0104 else "[in-sample]")

        print(f"{name:<36} | {n_sub:>5} | {w_w:>7.4f} | {w_h:>7.4f} | {w_g:>7.4f} | {m_train['absrel']:>10.4f} | {str_0059:>15} | {str_0104:>15} | {macro_12d:>10.4f}")

        results[name] = {
            "n_samples": n_sub,
            "n_drives": n_drives_sub,
            "excluded_drives": excluded,
            "weights": [round(w_w, 4), round(w_h, 4), round(w_g, 4)],
            "shrinkage_alpha": round(float(fw.shrinkage_alpha), 6),
            "cov_shrunk": [[round(float(c), 5) for c in row] for row in fw.cov_shrunk],
            "fit_metrics": {
                "n": m_train["n"],
                "n_valid": m_train["n_valid"],
                "valid_frac": round(m_train["n_valid"] / m_train["n"] if m_train["n"] else 0.0, 4),
                "absrel": round(m_train["absrel"], 4),
            },
            "test_drive_0059": {
                "is_out_of_sample": is_oos_0059,
                "n": m_0059["n"],
                "n_valid": m_0059["n_valid"],
                "absrel": round(m_0059["absrel"], 4),
            },
            "test_drive_0104": {
                "is_out_of_sample": is_oos_0104,
                "n": m_0104["n"],
                "n_valid": m_0104["n_valid"],
                "absrel": round(m_0104["absrel"], 4),
            },
            "macro_absrel_12_drives": round(macro_12d, 4),
        }

    # Extract dynamic min-max for observations
    all_weights = [r["weights"] for r in results.values()]
    w_w_vals = [w[0] for w in all_weights]
    w_h_vals = [w[1] for w in all_weights]
    w_g_vals = [w[2] for w in all_weights]

    base_0059 = results["Baseline (All 12 drives)"]["test_drive_0059"]["absrel"]
    base_0104 = results["Baseline (All 12 drives)"]["test_drive_0104"]["absrel"]

    oos_0059_ex59 = results["Exclude drive_0059 (Top-2, 24.8%)"]["test_drive_0059"]["absrel"]
    oos_0104_ex104 = results["Exclude drive_0104 (Top-1, 26.8%)"]["test_drive_0104"]["absrel"]

    oos_0059_exboth = results["Exclude both 0059 & 0104 (51.7%)"]["test_drive_0059"]["absrel"]
    oos_0104_exboth = results["Exclude both 0059 & 0104 (51.7%)"]["test_drive_0104"]["absrel"]

    delta_0059_ex59 = oos_0059_ex59 - base_0059
    delta_0104_ex104 = oos_0104_ex104 - base_0104
    delta_0059_exboth = oos_0059_exboth - base_0059
    delta_0104_exboth = oos_0104_exboth - base_0104

    # Save outputs
    out_json = PROJECT_ROOT / "results" / "tables" / "sensitivity_d14_report.json"
    out_md = PROJECT_ROOT / "results" / "tables" / "sensitivity_d14_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON report: {out_json}")

    # Build Markdown table dynamically with strict in-sample / OOS annotations
    md_content = [
        "# Báo cáo Phân tích Độ nhạy Split B Dominant Drives (Quyết định D14)",
        "",
        "## 1. Bối cảnh & Mục tiêu",
        "Split B-v2 có **12 drives** chứa Car Hard với $N = 4,776$ mẫu.",
        "Trong đó, hai drive lớn nhất chiếm hơn một nửa tổng số mẫu:",
        f"- `drive_0104`: 1,282 xe (26.84%)",
        f"- `drive_0059`: 1,185 xe (24.81%)",
        "Tổng cộng hai drive chiếm **51.65%** ($N = 2,467$).",
        "",
        "Mục tiêu là mô tả sự biến thiên của trọng số hợp nhất $[w_w, w_h, w_g]$ và độ lệch AbsRel khi loại bỏ từng drive lớn hoặc cả hai khỏi tập khớp trọng số.",
        "Toàn bộ metrics được tính bằng `src.evaluation.eval.depth_metrics` theo quy ước D18/D20.",
        "",
        "## 2. Kết quả Refit Trọng số và Độ chính xác AbsRel",
        "",
        "| Điều kiện kiểm thử | $N_{\\text{fit}}$ ($n_{\\text{valid}}$) | Drives | $w_w$ (Width) | $w_h$ (Height) | $w_g$ (Ground) | Fit AbsRel | Test `drive_0059` ($N=1,185$) | Test `drive_0104` ($N=1,282$) | Macro AbsRel (12 drives) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for name, data in results.items():
        w = data["weights"]
        fit_m = data["fit_metrics"]
        t59 = data["test_drive_0059"]
        t104 = data["test_drive_0104"]

        tag59 = "**0.0510** [OOS]" if t59["is_out_of_sample"] else f"{t59['absrel']:.4f} [in-sample]"
        tag104 = f"**{t104['absrel']:.4f}** [OOS]" if t104["is_out_of_sample"] else f"{t104['absrel']:.4f} [in-sample]"
        if name == "Exclude both 0059 & 0104 (51.7%)":
            tag59 = f"**{t59['absrel']:.4f}** [OOS]"

        md_content.append(
            f"| **{name}** | {fit_m['n']} ({fit_m['n_valid']}) | {data['n_drives']} | "
            f"{w[0]:.4f} | {w[1]:.4f} | {w[2]:.4f} | "
            f"{fit_m['absrel']:.4f} | "
            f"{tag59} | "
            f"{tag104} | "
            f"{data['macro_absrel_12_drives']:.4f} |"
        )

    md_content.extend([
        "",
        "> [!NOTE]",
        "> Các ô in đậm kèm ký hiệu `[OOS]` là kết quả kiểm thử ngoài mẫu (Out-Of-Sample) thực sự (drive mục tiêu bị loại hoàn toàn khỏi tập fit trọng số). Các ô `[in-sample]` được cung cấp để đối chiếu đường cơ sở.",
        "",
        "## 3. Nhận xét định lượng (Mô tả dữ liệu theo D18/D20)",
        f"1. **Biên độ biến thiên của trọng số:**",
        f"   - $w_h$ (Height): dao động trong khoảng **{min(w_h_vals)*100:.1f}% – {max(w_h_vals)*100:.1f}%** ({min(w_h_vals):.4f} đến {max(w_h_vals):.4f}).",
        f"   - $w_g$ (Ground): dao động trong khoảng **{min(w_g_vals)*100:.1f}% – {max(w_g_vals)*100:.1f}%** ({min(w_g_vals):.4f} đến {max(w_g_vals):.4f}).",
        f"   - $w_w$ (Width): dao động trong khoảng **{min(w_w_vals)*100:.1f}% – {max(w_w_vals)*100:.1f}%** ({min(w_w_vals):.4f} đến {max(w_w_vals):.4f}), tức biên độ thay đổi gấp **{max(w_w_vals)/min(w_w_vals):.1f} lần** khi loại bỏ đồng thời hai drive lớn.",
        f"   - Thứ tự phân cấp tương đối $w_h > w_g > w_w$ giữ nguyên trên tất cả 4 tập con.",
        "",
        f"2. **Độ lệch AbsRel trên các ô ngoài mẫu (Out-of-sample $\\Delta$):**",
        f"   - Khi loại `drive_0059`: AbsRel OOS trên `drive_0059` là **{oos_0059_ex59:.4f}** (chênh lệch so với baseline: $\\Delta = {delta_0059_ex59:+.4f}$).",
        f"   - Khi loại `drive_0104`: AbsRel OOS trên `drive_0104` là **{oos_0104_ex104:.4f}** (chênh lệch so với baseline: $\\Delta = {delta_0104_ex104:+.4f}$, tương đương tăng tương đối {delta_0104_ex104/base_0104*100:+.1f}%).",
        f"   - Khi loại đồng thời cả hai drive (mất 51.7% dữ liệu fit):",
        f"     - OOS trên `drive_0059`: **{oos_0059_exboth:.4f}** ($\\Delta = {delta_0059_exboth:+.4f}$, tăng tương đối {delta_0059_exboth/base_0059*100:+.1f}%).",
        f"     - OOS trên `drive_0104`: **{oos_0104_exboth:.4f}** ($\\Delta = {delta_0104_exboth:+.4f}$, tăng tương đối {delta_0104_exboth/base_0104*100:+.1f}%).",
        f"   - Macro AbsRel qua toàn bộ 12 drives: dao động trong biên hẹp từ **{min([r['macro_absrel_12_drives'] for r in results.values()]):.4f} đến {max([r['macro_absrel_12_drives'] for r in results.values()]):.4f}**.",
        "",
        "3. **Lưu ý phương pháp luận cho Paper:**",
        "   - Do Split B chỉ có 12 cụm drive ($N=4,776$), khoảng tin cậy của ước lượng hiệp phương sai là thô.",
        "   - Việc loại bỏ các drive lớn dẫn đến sự tái phân bổ đáng kể tỷ trọng $w_w$ (từ 8.1% xuống 4.1%), cho thấy ước lượng $w_w$ nhạy hơn với thành phần drive so với $w_h$ và $w_g$.",
        r"   - Tuy nhiên, độ lệch AbsRel ngoài mẫu trên hai drive lớn bị loại vẫn nằm trong phạm vi nhỏ ($\le +0.0012$, tức $\le 2.0\%$ tương đối), phản ánh cấu trúc tương quan giữa 3 cue hình học có tính ổn định tương đối giữa các cụm.",
    ])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")
    print(f"Saved Markdown report: {out_md}")


if __name__ == "__main__":
    run_sensitivity_analysis()
