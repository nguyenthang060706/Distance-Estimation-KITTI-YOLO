"""
scripts/eval_detector_performance.py: Evaluates detector performance (Precision, Recall, F1)
across distance bands and difficulty levels on KITTI Split B and Split C (§5.1, §6).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DISTANCE_BANDS = [
    ("0-10m", 0.0, 10.0),
    ("10-20m", 10.0, 20.0),
    ("20-30m", 20.0, 30.0),
    ("30-50m", 30.0, 50.0),
    (">50m", 50.0, 999.0),
]

MODELS = ["yolov8s", "yolo11s", "yolov5su"]


def evaluate_split(split: str, pred_dir: str = "results/predictions") -> dict:
    pred_path = Path(pred_dir)
    results = {"split": split, "models": {}}

    gt_dfs = {}
    match_dfs = {}
    det_dfs = {}

    for m in MODELS:
        m_key = f"{m}_640"
        gt_file = pred_path / f"{m_key}_{split}_gt.parquet"
        match_file = pred_path / f"{m_key}_{split}_matches.parquet"
        det_file = pred_path / f"{m_key}_{split}_detections.parquet"

        if not (gt_file.exists() and match_file.exists() and det_file.exists()):
            print(f"Warning: Artifacts for {m} on Split {split} not found in {pred_dir}")
            return results

        gt_dfs[m] = pd.read_parquet(gt_file)
        match_dfs[m] = pd.read_parquet(match_file)
        det_dfs[m] = pd.read_parquet(det_file)

    # All models should evaluate against the exact same GT objects on this split
    ref_gt = gt_dfs[MODELS[0]].copy()
    ref_gt["gt_key"] = ref_gt["frame_id"] + "_" + ref_gt["gt_idx"].astype(str)
    total_gt = len(ref_gt)

    # 1. Individual model performance
    model_stats = {}
    tp_gt_keys_per_model = {}

    for m in MODELS:
        m_matches = match_dfs[m].copy()
        m_dets = det_dfs[m]
        m_gt = gt_dfs[m].copy()

        # Ensure index alignment
        assert len(m_matches) == len(m_dets)

        m_gt["gt_key"] = m_gt["frame_id"] + "_" + m_gt["gt_idx"].astype(str)
        m_matches["gt_key"] = m_matches["frame_id"] + "_" + m_matches["matched_gt_idx"].astype(str)

        # Filter by pass_thr (Decision D6 threshold)
        pass_mask = m_dets["pass_thr"].to_numpy()
        statuses = m_matches["status"].to_numpy()

        # At conf_min (floor 0.05)
        tp_floor = np.sum(statuses == "TP")
        fp_floor = np.sum(statuses == "FP")
        recall_floor = tp_floor / total_gt if total_gt > 0 else 0.0

        # At pass_thr
        tp_thresh_mask = pass_mask & (statuses == "TP")
        fp_thresh_mask = pass_mask & (statuses == "FP")
        ign_nh_mask = pass_mask & (statuses == "IGNORED_NONHARD")
        ign_dc_mask = pass_mask & (statuses == "IGNORED_DONTCARE")

        tp = int(np.sum(tp_thresh_mask))
        fp = int(np.sum(fp_thresh_mask))
        ign_nh = int(np.sum(ign_nh_mask))
        ign_dc = int(np.sum(ign_dc_mask))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Matched GT keys at threshold
        matched_gts = set(m_matches.loc[tp_thresh_mask, "gt_key"])
        assert len(matched_gts) == tp, f"Mismatch between TP count {tp} and matched unique GTs {len(matched_gts)}"
        tp_gt_keys_per_model[m] = matched_gts

        # Distance band breakdown
        band_stats = {}
        for band_name, z_min, z_max in DISTANCE_BANDS:
            band_gt_mask = (m_gt["z_gt"] >= z_min) & (m_gt["z_gt"] < z_max)
            band_gt_keys = set(m_gt.loc[band_gt_mask, "gt_key"])
            n_band_gt = len(band_gt_keys)
            n_band_matched = len(band_gt_keys.intersection(matched_gts))
            band_rec = n_band_matched / n_band_gt if n_band_gt > 0 else 0.0
            band_stats[band_name] = {
                "n_gt": n_band_gt,
                "n_matched": n_band_matched,
                "recall": round(band_rec, 4),
            }

        # Difficulty breakdown (cumulative KITTI: Easy, Moderate, Hard)
        diff_stats = {}
        easy_mask = m_gt["difficulty"] == "Easy"
        mod_mask = m_gt["difficulty"].isin(["Easy", "Moderate"])
        hard_mask = np.ones(len(m_gt), dtype=bool)

        for diff_name, mask in [("Easy", easy_mask), ("Moderate", mod_mask), ("Hard", hard_mask)]:
            diff_keys = set(m_gt.loc[mask, "gt_key"])
            n_diff_gt = len(diff_keys)
            n_diff_matched = len(diff_keys.intersection(matched_gts))
            diff_rec = n_diff_matched / n_diff_gt if n_diff_gt > 0 else 0.0
            diff_stats[diff_name] = {
                "n_gt": n_diff_gt,
                "n_matched": n_diff_matched,
                "recall": round(diff_rec, 4),
            }

        model_stats[m] = {
            "total_gt": total_gt,
            "total_preds_floor": len(m_dets),
            "preds_passed_thr": int(np.sum(pass_mask)),
            "tp": tp,
            "fp": fp,
            "ign_nonhard": ign_nh,
            "ign_dontcare": ign_dc,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "recall_floor_005": round(recall_floor, 4),
            "bands": band_stats,
            "difficulty": diff_stats,
        }

    # 2. Common support analysis across all 3 detectors
    all_matched = [tp_gt_keys_per_model[m] for m in MODELS]
    common_gt = set.intersection(*all_matched)
    union_gt = set.union(*all_matched)

    common_stats = {
        "n_common_gt": len(common_gt),
        "pct_of_total_gt": round(len(common_gt) / total_gt * 100, 2) if total_gt > 0 else 0.0,
        "n_union_gt": len(union_gt),
        "pct_union_of_total_gt": round(len(union_gt) / total_gt * 100, 2) if total_gt > 0 else 0.0,
    }

    # Common support distance breakdown
    common_band_stats = {}
    for band_name, z_min, z_max in DISTANCE_BANDS:
        band_gt_mask = (ref_gt["z_gt"] >= z_min) & (ref_gt["z_gt"] < z_max)
        band_gt_keys = set(ref_gt.loc[band_gt_mask, "gt_key"])
        n_band_gt = len(band_gt_keys)
        n_common_band = len(band_gt_keys.intersection(common_gt))
        common_band_stats[band_name] = {
            "n_gt": n_band_gt,
            "n_common": n_common_band,
            "common_frac": round(n_common_band / n_band_gt, 4) if n_band_gt > 0 else 0.0,
        }
    common_stats["bands"] = common_band_stats

    results["models"] = model_stats
    results["common_support"] = common_stats
    return results


def format_markdown_report(eval_b: dict, eval_c: dict) -> str:
    md = []
    md.append("# Đánh giá Hiệu năng Detector trên KITTI Split B và Split C (§5.1, §6)")
    md.append("")
    md.append("Báo cáo đánh giá 3 mô hình detector (`yolov8s_640`, `yolo11s_640`, `yolov5su_640`) trên:")
    md.append("- **Split B (Residual Fitting):** 1.499 ảnh, 4.776 Car Hard.")
    md.append("- **Split C (Conformal Calibration):** 766 ảnh, 1.826 Car Hard.")
    md.append("- **Cấu hình:** FP32, imgsz=640, matching IoU 0.5 (Greedy theo điểm số, Decision D15 & D16), DontCare mode `iou`.")
    md.append("")

    for split_name, eval_data in [("Split B", eval_b), ("Split C", eval_c)]:
        md.append(f"## 1. Hiệu năng Tổng thể trên {split_name}")
        md.append("")
        md.append("| Detector | Conf Thr | Preds (Pass) | TP | FP | Ign Non-Hard | Ign DontCare | Precision (%) | Recall (%) | F1 | Recall Floor (0.05) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        models = eval_data.get("models", {})
        for m in MODELS:
            st = models.get(m, {})
            if not st:
                continue
            # Read threshold from config
            conf_cfg = yaml.safe_load(open("configs/detector/conf_thresholds.yaml", encoding="utf-8"))
            thr = conf_cfg["detectors"][m]["conf_threshold"]
            p_pct = st["precision"] * 100
            r_pct = st["recall"] * 100
            rf_pct = st["recall_floor_005"] * 100
            md.append(f"| **{m}** | {thr:.3f} | {st['preds_passed_thr']:,} | {st['tp']:,} | {st['fp']:,} | {st['ign_nonhard']:,} | {st['ign_dontcare']:,} | **{p_pct:.2f}%** | **{r_pct:.2f}%** | **{st['f1']:.4f}** | {rf_pct:.2f}% |")
        md.append("")

        md.append(f"### 1.1. Phân rã Recall theo Dải Khoảng cách trên {split_name}")
        md.append("")
        md.append("| Dải Khoảng cách | Tổng GT Car Hard | Recall yolov8s | Recall yolo11s | Recall yolov5su | Tập chung (Cả 3 bắt được) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        comm = eval_data.get("common_support", {})
        comm_bands = comm.get("bands", {})
        for band_name, _, _ in DISTANCE_BANDS:
            n_gt = comm_bands.get(band_name, {}).get("n_gt", 0)
            flag = " *" if n_gt < 100 else ""
            r_v8 = models.get("yolov8s", {}).get("bands", {}).get(band_name, {}).get("recall", 0.0) * 100
            r_11 = models.get("yolo11s", {}).get("bands", {}).get(band_name, {}).get("recall", 0.0) * 100
            r_v5 = models.get("yolov5su", {}).get("bands", {}).get(band_name, {}).get("recall", 0.0) * 100
            c_frac = comm_bands.get(band_name, {}).get("common_frac", 0.0) * 100
            n_comm = comm_bands.get(band_name, {}).get("n_common", 0)

            md.append(f"| **{band_name}**{flag} | {n_gt:,} | {r_v8:.2f}% | {r_11:.2f}% | {r_v5:.2f}% | **{c_frac:.2f}%** ({n_comm:,}) |")
        md.append("")

        md.append(f"### 1.2. Phân rã Recall theo Mức độ Khó KITTI trên {split_name} (Cumulative)")
        md.append("")
        md.append("| Độ khó KITTI | Tổng GT | Recall yolov8s | Recall yolo11s | Recall yolov5su |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for diff in ["Easy", "Moderate", "Hard"]:
            n_gt = models.get("yolov8s", {}).get("difficulty", {}).get(diff, {}).get("n_gt", 0)
            r_v8 = models.get("yolov8s", {}).get("difficulty", {}).get(diff, {}).get("recall", 0.0) * 100
            r_11 = models.get("yolo11s", {}).get("difficulty", {}).get(diff, {}).get("recall", 0.0) * 100
            r_v5 = models.get("yolov5su", {}).get("difficulty", {}).get(diff, {}).get("recall", 0.0) * 100
            md.append(f"| **{diff}** | {n_gt:,} | {r_v8:.2f}% | {r_11:.2f}% | {r_v5:.2f}% |")
        md.append("")

        md.append(f"### 1.3. Phân tích Tập Hỗ trợ Chung (Common Support) trên {split_name}")
        md.append(f"- Số GT Car Hard được cả 3 detector phát hiện đồng thời: **{comm.get('n_common_gt', 0):,}** / {eval_data.get('models', {}).get('yolov8s', {}).get('total_gt', 0):,} (**{comm.get('pct_of_total_gt', 0.0):.2f}%**).")
        md.append(f"- Hợp (Union) của các GT được ít nhất 1 detector phát hiện: **{comm.get('n_union_gt', 0):,}** (**{comm.get('pct_union_of_total_gt', 0.0):.2f}%**).")
        md.append("")

    md.append("## 2. Ghi chú Thảo luận & Hạn chế cho Paper")
    md.append("1. **Độ sụt giảm Recall ở ngưỡng F1 tối ưu:** Ngưỡng F1 chọn trên Split V tối ưu hóa F1 nhưng làm giảm Recall ~10–12% so với trần ở sàn 0.05. Sự sụt giảm tập trung chủ yếu ở dải cự ly xa (30–50 m và >50 m), nơi kích thước vật thể nhỏ và độ tin cậy detector thấp hơn.")
    md.append("2. **Đặc thù Split C v2:** Toàn bộ mẫu xe Car Hard trên Split C đều có cự ly $\\le 50$ m ($n_{>50m} = 0$). Đánh giá detector trên Split C phản ánh chân thực phân bố đô thị cự ly gần và trung bình.")
    md.append("3. **Tập khớp chung (Common Support):** Việc đánh giá ranging trên tập khớp chung loại bỏ hoàn toàn thiên lệch do recall detector khác nhau, bảo đảm so sánh công bằng giữa các thuật toán ước lượng cự ly.")
    md.append("")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector performance on Split B and C.")
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--output-md", default="results/tables/detector_eval_b_c.md")
    parser.add_argument("--output-json", default="results/tables/detector_eval_b_c.json")
    args = parser.parse_args()

    print("Evaluating detector performance on Split B...")
    eval_b = evaluate_split("B", pred_dir=args.pred_dir)

    print("Evaluating detector performance on Split C...")
    eval_c = evaluate_split("C", pred_dir=args.pred_dir)

    full_results = {
        "metadata": {
            "splits_version": "v2",
            "matching_iou": 0.5,
            "matching_algorithm": "Greedy (Decision D15 & D16)",
            "models": MODELS,
        },
        "Split_B": eval_b,
        "Split_C": eval_c,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    md_report = format_markdown_report(eval_b, eval_c)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\n✓ Evaluation complete!")
    print(f"  Markdown saved to: {args.output_md}")
    print(f"  JSON saved to:     {args.output_json}")


if __name__ == "__main__":
    main()
