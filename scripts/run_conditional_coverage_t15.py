"""
scripts/run_conditional_coverage_t15.py: Official Task T15 Conditional Coverage & Exchangeability Diagnostics on Split T (Decisions D19, D47, D50, D54, D55, D70, D74, D75, D79).

Boundary Rules (Decision D70):
- Reads exclusively from pre-computed static artifacts in results/final/* and results/predictions/*.
- Zero touch on Split T raw data: does NOT import load_split, does NOT re-run inference.
- Implements frozen pre-registered evaluation rules (Decision D68):
  * No parameter refitting, no tuning, alpha=0.10.
  * Transparent reporting of survivorship bias metrics (n_TP, n_FN, n_GT, Recall).
  * Coarse 10-cluster paired bootstrap without asserting over-claims if CIs overlap or include 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.eval import append_jsonl, make_log_record
from src.pipeline.apply_frozen import apply_frozen_pipeline
from src.residual.models import load_model_f, predict_f
from src.uncertainty.conditional_coverage import (
    compute_conditional_coverage_breakdown,
    compute_drive_level_coverage_table,
    compute_exchangeability_ks_diagnostics,
    cluster_bootstrap_coverage_ci,
    METHODS,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
ALPHA = 0.10


def format_subgroup_table_md(rows: list[dict[str, Any]], title: str) -> str:
    """Format markdown table for a specific subgroup category."""
    lines = [
        f"### {title}",
        "",
        "> (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.",
        "",
        "| Phân nhóm | $n_{\\text{TP}}$ | $n_{\\text{FN}}$ | $n_{\\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|",
    ]

    for r in rows:
        sg = r.get("subgroup", "")
        if sg == "---":
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | :---: |")
            continue

        n_tp = r.get("n_tp", 0)
        n_fn = r.get("n_fn", 0)
        n_gt = r.get("n_gt", 0)
        rec = f"{r.get('recall', 0.0)*100:.1f}%" if n_gt > 0 else "0.0%"
        k = r.get("k_clusters", 0)
        flag = "*" if r.get("low_n", False) else ""

        cqr_cov = f"{r['cqr']['coverage']*100:.1f}%" if r['cqr']['coverage'] is not None else "N/A"
        cqr_w = f"{r['cqr']['mean_width']:.3f}" if r['cqr']['mean_width'] is not None else "N/A"
        cqr_wink = f"{r['cqr']['mean_winkler']:.4f}" if r['cqr']['mean_winkler'] is not None else "N/A"

        sc_cov = f"{r['sc']['coverage']*100:.1f}%" if r['sc']['coverage'] is not None else "N/A"
        sc_w = f"{r['sc']['mean_width']:.3f}" if r['sc']['mean_width'] is not None else "N/A"

        mon_cov = f"{r['mondrian']['coverage']*100:.1f}%" if r['mondrian']['coverage'] is not None else "N/A"
        mon_w = f"{r['mondrian']['mean_width']:.3f}" if r['mondrian']['mean_width'] is not None else "N/A"

        lines.append(
            f"| `{sg}` | {n_tp:,} | {n_fn:,} | {n_gt:,} | {rec} | "
            f"**{cqr_cov}** | {cqr_w} | {cqr_wink} | "
            f"{sc_cov} | {sc_w} | "
            f"{mon_cov} | {mon_w} | {k} | {flag} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_conditional_report_md(all_results: dict[str, Any]) -> str:
    """Generate Markdown report for conditional coverage across subgroups (Decisions D19, D54, D55, D74)."""
    lines = [
        "# Đánh giá Độ Phủ Có Điều Kiện trên Split T (Tác vụ T15)",
        "",
        "> **Bối cảnh phương pháp luận (Decisions D19, D47, D50, D54, D55, D70, D74, D75, D79):**",
        "> - **Zero-Touch Split T (D70):** Toàn bộ phân tích đọc trực tiếp từ các artifact nghiệm thu tĩnh tại `results/final/`.",
        "> - **Standard CQR là phương án chính tiên nghiệm (D55):** Split Conformal và Mondrian CQR đóng vai trò baseline đối chứng.",
        "> - **Minh bạch thiên lệch kẻ sống sót (D32, D70):** Mọi bảng phân rã đều công bố song song $n_{\\text{TP}}, n_{\\text{FN}}, n_{\\text{GT}}$ và Recall.",
        "> - **Cảnh báo cỡ mẫu nhỏ (D54):** Gắn cờ `*` khi $n_{\\text{TP}} < 100$ và ghi rõ số cụm drive $k$.",
        "> - **Mức ý nghĩa danh nghĩa:** $\\alpha = 0.10$ (độ phủ danh nghĩa $90.0\\%$).",
        "",
    ]

    for model_key in DETECTORS:
        m_data = all_results[model_key]
        cats = m_data["categories"]
        boot = m_data["bootstrap_ci_95"]

        lines.extend([
            f"## 1. Detector `{model_key}`",
            "",
            "> - **Bootstrap 95% CI Pooled Coverage (10 cụm):**",
            f">   * Standard CQR: [{boot['cqr']['coverage_ci_95'][0]*100:.2f}%, {boot['cqr']['coverage_ci_95'][1]*100:.2f}%] (Mean Width: [{boot['cqr']['width_ci_95'][0]:.3f}, {boot['cqr']['width_ci_95'][1]:.3f}]) — *{boot['cqr']['note']}*",
            f">   * Split Conformal: [{boot['sc']['coverage_ci_95'][0]*100:.2f}%, {boot['sc']['coverage_ci_95'][1]*100:.2f}%] (Mean Width: [{boot['sc']['width_ci_95'][0]:.3f}, {boot['sc']['width_ci_95'][1]:.3f}]) — *{boot['sc']['note']}*",
            f">   * Mondrian CQR: [{boot['mondrian']['coverage_ci_95'][0]*100:.2f}%, {boot['mondrian']['coverage_ci_95'][1]*100:.2f}%] (Mean Width: [{boot['mondrian']['width_ci_95'][0]:.3f}, {boot['mondrian']['width_ci_95'][1]:.3f}]) — *{boot['mondrian']['note']}*",
            "",
        ])

        # 1. Prospective Bins
        lines.append(format_subgroup_table_md(cats["z_hat_prospective"], "1.1 Dải cự ly theo Ẑ dự đoán (Prospective Distance Bins)"))

        # 2. Retrospective Bins
        lines.append(format_subgroup_table_md(cats["z_gt_retrospective"], "1.2 Dải cự ly theo Z thật (Retrospective Distance Bins — Chẩn đoán, v4 §6)"))

        # 3. Truncation and Touch Edges
        lines.append(format_subgroup_table_md(cats["truncation_and_edges"], "1.3 Mức độ cắt biên (Truncation) & Chạm viền ảnh (Touch Edges, D84)"))

        # 4. Occlusion
        lines.append(format_subgroup_table_md(cats["occlusion"], "1.4 Mức độ che khuất (Occlusion Levels)"))

        # 5. Viewing Angle Theta
        lines.append(format_subgroup_table_md(cats["viewing_angle_theta"], "1.5 Góc hướng quan sát θ (Viewing Angle Bins, D19)"))

        # 6. Difficulty
        lines.append(format_subgroup_table_md(cats["difficulty"], "1.6 Mức độ khó KITTI (Nested & Disjoint Difficulty)"))

        # 7. Fallback Pattern 000
        lines.append(format_subgroup_table_md(cats["fallback_pattern_000"], "1.7 Nhóm Fallback Pattern 000 (Decision D74)"))

    return "\n".join(lines)


def generate_drive_level_report_md(all_drive_data: dict[str, Any]) -> str:
    """Generate Markdown report for per-drive breakdown (Decisions D50, D75, D86)."""
    lines = [
        "# Bóc Tách Độ Phủ Theo Cụm Drive trên Split T (Tác vụ T15)",
        "",
        "> **Quy chuẩn Macro Kép & Tính Không Đồng Nhất (Decisions D50, D75, D79, D86):**",
        "> - Báo cáo chi tiết trên toàn bộ 10 cụm drive của Split T.",
        "> - Ngưỡng cờ cỡ mẫu cảnh báo: (*) cờ drive $n < 30$ (phục vụ đối chiếu Macro ge30 theo D50, D75, D86); các phân nhóm cự ly/danh mục sử dụng ngưỡng $n < 100$ theo D54.",
        "> - Công bố song song `Macro Coverage (10 drive)` và `Macro Coverage ge30 (8 drive)` loại trừ các drive $n < 30$ (ví dụ `drive_0002` $n=2$).",
        "> - Winkler score đo lường độ sắc nét khoảng trong không gian log ($r$).",
        "",
    ]

    for model_key in DETECTORS:
        d_res = all_drive_data[model_key]
        rows = d_res["drive_rows"]
        summ = d_res["summary"]

        lines.extend([
            f"## Detector `{model_key}` (Tổng $N = {summ['n_total']:,}$ TP)",
            "",
            "| Drive ID | $n_{\\text{TP}}$ | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | SC Winkler | Mondrian Cov | Mondrian Width | Mondrian Winkler | Crossings |",
            "|---|---|---|---|---|---|---|---|---|---|---|:---:|",
        ])

        for r in rows:
            d_id = r["drive"].replace("2011_09_26_drive_", "0926_").replace("2011_09_28_drive_", "0928_").replace("2011_09_29_drive_", "0929_").replace("_sync", "")
            n = r["n"]
            flag = " (*)" if r.get("low_n", False) else ""

            cqr_cov = f"{r['cqr_coverage']*100:.1f}%" if r['cqr_coverage'] is not None else "N/A"
            cqr_w = f"{r['cqr_mean_width']:.3f}" if r['cqr_mean_width'] is not None else "N/A"
            cqr_wink = f"{r['cqr_winkler']:.4f}" if r['cqr_winkler'] is not None else "N/A"

            sc_cov = f"{r['sc_coverage']*100:.1f}%" if r['sc_coverage'] is not None else "N/A"
            sc_w = f"{r['sc_mean_width']:.3f}" if r['sc_mean_width'] is not None else "N/A"
            sc_wink = f"{r['sc_winkler']:.4f}" if r['sc_winkler'] is not None else "N/A"

            mon_cov = f"{r['mondrian_coverage']*100:.1f}%" if r['mondrian_coverage'] is not None else "N/A"
            mon_w = f"{r['mondrian_mean_width']:.3f}" if r['mondrian_mean_width'] is not None else "N/A"
            mon_wink = f"{r['mondrian_winkler']:.4f}" if r['mondrian_winkler'] is not None else "N/A"

            lines.append(
                f"| `{d_id}` | {n}{flag} | **{cqr_cov}** | {cqr_w} | {cqr_wink} | "
                f"{sc_cov} | {sc_w} | {sc_wink} | "
                f"{mon_cov} | {mon_w} | {mon_wink} | {r['cqr_crossings']} |"
            )

        # Summary rows
        lines.extend([
            "|---|---|---|---|---|---|---|---|---|---|---|:---:|",
            f"| **POOLED** | **{summ['n_total']:,}** | **{summ['cqr']['pooled_coverage']*100:.2f}%** | {summ['cqr']['mean_width']:.3f} | {summ['cqr']['mean_winkler']:.4f} | "
            f"{summ['sc']['pooled_coverage']*100:.2f}% | {summ['sc']['mean_width']:.3f} | {summ['sc']['mean_winkler']:.4f} | "
            f"{summ['mondrian']['pooled_coverage']*100:.2f}% | {summ['mondrian']['mean_width']:.3f} | {summ['mondrian']['mean_winkler']:.4f} | {summ['cqr']['crossings']} |",
            f"| **MACRO (10 drives)** | --- | **{summ['cqr']['macro_coverage']*100:.2f}%** | --- | --- | "
            f"{summ['sc']['macro_coverage']*100:.2f}% | --- | --- | "
            f"{summ['mondrian']['macro_coverage']*100:.2f}% | --- | --- | --- |",
            f"| **MACRO ge30 (8 drives)** | --- | **{summ['cqr']['macro_coverage_ge30']*100:.2f}%** | --- | --- | "
            f"{summ['sc']['macro_coverage_ge30']*100:.2f}% | --- | --- | "
            f"{summ['mondrian']['macro_coverage_ge30']*100:.2f}% | --- | --- | --- |",
            "",
        ])

    return "\n".join(lines)


def generate_exchangeability_report_md(all_ks_data: dict[str, list[dict[str, Any]]]) -> str:
    """Generate Markdown report for Exchangeability Diagnostics (Decisions D26, D68, D79, D87)."""
    lines = [
        "# Chẩn Đoán Tính Khả Hoán $C \\leftrightarrow T$ (Exchangeability Diagnostics)",
        "",
        "> **Cơ Sở Lý Thuyết & Bản Chất Post-hoc / Exploratory của Hiện Tượng Over-coverage (Decisions D68, D79, D87):**",
        "> - **Đối chiếu tiên đoán Tiền đăng ký (D68):** Ban đầu, D68 dự báo nguy cơ *under-coverage* ngoài mẫu do phân tích độ ổn định 20 resplits (T08) chỉ đạt 84–87%. Tuy nhiên, kết quả thực tế trên Split T đạt độ phủ danh nghĩa vượt mức: **96.4%–97.1%** (over-coverage).",
        "> - **Tính chất Diễn giải Hậu nghiệm (Post-hoc / Exploratory):** Giả thuyết *'Split C có độ khó cao hơn Split T khiến ngưỡng sai số không tương đồng $\\hat{Q}$ bị nới rộng, dẫn đến bảo thủ ngoài mẫu'* là suy luận post-hoc được hình thành sau khi quan sát dữ liệu Split T, không phải kiểm chứng tiên nghiệm. Thông điệp phương pháp luận chính của RQ3 là: *Độ phủ biên không chuyển giao ổn định giữa các cụm khi số lượng cụm drive còn nhỏ (~10 cụm); hướng lệch (under hay over) phụ thuộc vào thành phần drive của tập hiệu chuẩn C so với tập kiểm định T.*",
        "> - **Cảnh báo Phương pháp luận về Tránh Lỗi Pseudo-replication (AGENT_RULES §6.2, Decisions D20, D73):** Bounding box trong KITTI gom theo các cụm driving sequence có tương quan chuỗi mạnh. Việc tính p-value giả định các hàng độc lập (i.i.d) tạo ra p-value ngụy tạo ($p \\approx 10^{-20}$). Do đó, bảng bên dưới chỉ báo cáo chỉ số thống kê Kolmogorov-Smirnov $D_{\\text{KS}} = \\sup |F_C(x) - F_T(x)|$ mang tính **mô tả phân kỳ phân bố thực nghiệm (descriptive empirical divergence)**; đối với các cờ nhị phân (`valid_*`), chỉ báo cáo tỷ lệ trung bình (mean proportion).",
        "",
    ]

    for model_key in DETECTORS:
        ks_rows = all_ks_data[model_key]

        lines.extend([
            f"## Detector `{model_key}`",
            "",
            "| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Stat (mô tả) | Ghi chú diễn giải |",
            "|---|---|---|---|---|---|---|---|---|",
        ])

        for r in ks_rows:
            ks_val_str = f"**{r['ks_statistic']:.4f}**" if r.get("ks_statistic") is not None else "---"
            note_str = r.get("note", "")
            lines.append(
                f"| **{r['feature_name']}** | {r['n_c']:,} | {r['n_t']:,} | "
                f"{r['mean_c']:.3f} | {r['mean_t']:.3f} | "
                f"{r['std_c']:.3f} | {r['std_t']:.3f} | "
                f"{ks_val_str} | {note_str} |"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("Task T15: Conditional Coverage & Exchangeability Diagnostics on Split T")
    print("=" * 70)

    final_dir = PROJECT_ROOT / "results" / "final"
    out_table_dir = PROJECT_ROOT / "results" / "tables"
    out_table_dir.mkdir(parents=True, exist_ok=True)

    all_conditional_results: dict[str, Any] = {}
    all_drive_results: dict[str, Any] = {}
    all_ks_results: dict[str, list[dict[str, Any]]] = {}

    for m in DETECTORS:
        print(f"\nProcessing Detector: {m}...")
        pred_path = final_dir / f"{m}_T_predictions.parquet"
        fn_path = final_dir / f"{m}_T_fn.parquet"

        if not pred_path.is_file() or not fn_path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {pred_path} or {fn_path}")

        # 1. Load Split T data
        df_t = pd.read_parquet(pred_path)
        fn_t = pd.read_parquet(fn_path)
        print(f"  Loaded Split T: {len(df_t)} TP, {len(fn_t)} FN across {df_t['drive'].nunique()} drives")

        # 2. Compute conditional coverage across 7 categories
        cats = compute_conditional_coverage_breakdown(df_t, fn_t, alpha=ALPHA, drive_col="drive")

        # 3. Compute 95% Cluster Bootstrap CIs for pooled coverage and width
        print("  Computing cluster bootstrap (B=1000) for pooled coverage and width...")
        boot_cis = {}
        for meth in METHODS:
            boot_cis[meth] = cluster_bootstrap_coverage_ci(df_t, method=meth, n_boot=1000, seed=42, drive_col="drive")

        all_conditional_results[m] = {
            "categories": cats,
            "bootstrap_ci_95": boot_cis,
        }

        # 4. Compute Drive-level table
        drive_table = compute_drive_level_coverage_table(df_t, alpha=ALPHA, drive_col="drive")
        all_drive_results[m] = drive_table

        # 5. Load Split C data for Exchangeability diagnostics
        print("  Loading Split C data for Exchangeability diagnostics...")
        res_c = apply_frozen_pipeline(m, "C", verify_stored_zd=True)
        runs_dir = PROJECT_ROOT / "runs" / "residual" / m
        model_f = load_model_f(runs_dir / "model_f.json")
        z_hat_f_c, r_hat_f_c = predict_f(model_f, res_c["feat_mats"]["f"], res_c["z_base"])

        df_c = res_c["eval_df"].copy()
        df_c["z_hat_f"] = z_hat_f_c
        df_c["r_actual"] = res_c["r_actual"]
        df_c["confidence"] = res_c["features_df"]["confidence"].to_numpy()
        df_c["valid_w"] = res_c["cues_df"]["valid_w"].to_numpy()
        df_c["valid_h"] = res_c["cues_df"]["valid_h"].to_numpy()
        df_c["valid_g"] = res_c["cues_df"]["valid_g"].to_numpy()

        ks_diagnostics = compute_exchangeability_ks_diagnostics(df_c, df_t)
        all_ks_results[m] = ks_diagnostics
        print(f"  Completed KS diagnostics for {m} across {len(ks_diagnostics)} features.")

    # --------------------------------------------------------------------------
    # Output Files
    # --------------------------------------------------------------------------
    print("\n--- Saving Output Tables & Manifests ---")

    # 1. Conditional Coverage Tables
    cond_json_path = out_table_dir / "coverage_conditional_T.json"
    cond_md_path = out_table_dir / "coverage_conditional_T.md"
    with open(cond_json_path, "w", encoding="utf-8") as f:
        json.dump(all_conditional_results, f, indent=2, ensure_ascii=False)
    cond_md_text = generate_conditional_report_md(all_conditional_results)
    with open(cond_md_path, "w", encoding="utf-8") as f:
        f.write(cond_md_text)
    print(f"  Wrote: {cond_json_path}")
    print(f"  Wrote: {cond_md_path}")

    # 2. Drive Level Coverage Tables
    drive_json_path = out_table_dir / "coverage_per_drive_T.json"
    drive_md_path = out_table_dir / "coverage_per_drive_T.md"
    with open(drive_json_path, "w", encoding="utf-8") as f:
        json.dump(all_drive_results, f, indent=2, ensure_ascii=False)
    drive_md_text = generate_drive_level_report_md(all_drive_results)
    with open(drive_md_path, "w", encoding="utf-8") as f:
        f.write(drive_md_text)
    print(f"  Wrote: {drive_json_path}")
    print(f"  Wrote: {drive_md_path}")

    # 3. Exchangeability Tables
    ks_json_path = out_table_dir / "exchangeability_c_vs_t.json"
    ks_md_path = out_table_dir / "exchangeability_c_vs_t.md"
    with open(ks_json_path, "w", encoding="utf-8") as f:
        json.dump(all_ks_results, f, indent=2, ensure_ascii=False)
    ks_md_text = generate_exchangeability_report_md(all_ks_results)
    with open(ks_md_path, "w", encoding="utf-8") as f:
        f.write(ks_md_text)
    print(f"  Wrote: {ks_json_path}")
    print(f"  Wrote: {ks_md_path}")

    # --------------------------------------------------------------------------
    # Pipeline Log
    # --------------------------------------------------------------------------
    print("\n--- Logging Event to pipeline_log.jsonl ---")
    split_meta_path = PROJECT_ROOT / "splits" / "split_metadata.json"
    split_t_hash = ""
    if split_meta_path.is_file():
        with open(split_meta_path, "r", encoding="utf-8") as f:
            s_meta = json.load(f)
            split_t_hash = s_meta.get("splits", {}).get("T", {}).get("hash", "")

    log_record = make_log_record(
        split="T",
        split_hash=split_t_hash,
        seed=42,
        n_boot=1000,
        tag="T15-Conditional-Coverage-T",
        extra={
            "task": "T15",
            "alpha": ALPHA,
            "models": DETECTORS,
            "yolo11s_cqr_pooled_cov": all_drive_results["yolo11s_640"]["summary"]["cqr"]["pooled_coverage"],
            "yolo11s_cqr_macro_cov": all_drive_results["yolo11s_640"]["summary"]["cqr"]["macro_coverage"],
            "yolo11s_cqr_macro_ge30": all_drive_results["yolo11s_640"]["summary"]["cqr"]["macro_coverage_ge30"],
            "yolov8s_cqr_pooled_cov": all_drive_results["yolov8s_640"]["summary"]["cqr"]["pooled_coverage"],
            "yolov5su_cqr_pooled_cov": all_drive_results["yolov5su_640"]["summary"]["cqr"]["pooled_coverage"],
            "output_files": [
                str(cond_json_path.relative_to(PROJECT_ROOT)),
                str(cond_md_path.relative_to(PROJECT_ROOT)),
                str(drive_json_path.relative_to(PROJECT_ROOT)),
                str(drive_md_path.relative_to(PROJECT_ROOT)),
                str(ks_json_path.relative_to(PROJECT_ROOT)),
                str(ks_md_path.relative_to(PROJECT_ROOT)),
            ],
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_record)
    print("  Appended event 'T15-Conditional-Coverage-T' to runs/pipeline_log.jsonl")

    print("\n" + "=" * 70)
    print("Task T15 Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
