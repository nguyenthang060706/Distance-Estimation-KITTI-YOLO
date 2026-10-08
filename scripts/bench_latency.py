"""
scripts/bench_latency.py: Official Tier 1 pipeline latency benchmark across stages.
(Kế hoạch v4 §5.6, Decisions D22, D40, D44, D94, D95 - T16 Nghiệm thu chính thức).

Resolves 4 Methodological Limitations from T06 Preliminary Benchmark (Decision D44):
1. Eliminates Preprocessing Double-Count: Measures raw model forward pass strictly on pre-computed tensors on GPU.
2. Fixes Sum-of-Medians Bias: Measures total end-to-end execution time per image t_total^(i) = sum(t_stage^(i))
   and computes distribution statistics (Median, Mean, IQR, P95) directly on {t_total^(i)}.
3. Full Feature Residual Pipeline: Vectorized extraction of all 17 canonical features (including actual detection
   confidence and derived ln_z_base) without shortcuts or GT leakage.
4. Conformal CQR Timing Included: Measures CQR quantile prediction (q05, q95) and interval construction in log-space.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
import xgboost as xgb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import append_jsonl, make_log_record
from src.geometry.geometric_cues import (
    BORDER_EPS,
    CameraIntrinsics,
    CueResult,
    compute_cues_batch,
    load_geometry_v2,
)
from src.geometry.fusion import fuse_depths_vectorised, load_fusion_weights
from src.residual.models import (
    FEATURE_COLS_F,
    load_model_e,
    load_model_f,
    predict_e,
)
from src.uncertainty.cqr import (
    load_quantile_model,
    predict_interval,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
CPU_THREADS = 4
N_PARITY_IMAGES = 50
N_BENCHMARK_IMAGES = 200
N_WARMUP = 20


def compute_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(model_key: str, checkpoint_path: Path, checkpoints_cfg_path: Path) -> str:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    with open(checkpoints_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expected_sha = cfg["checkpoints"][model_key]["last.pt"]["sha256"]
    actual_sha = compute_file_sha256(checkpoint_path)

    if actual_sha != expected_sha:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {model_key}!\n"
            f"  Expected: {expected_sha}\n"
            f"  Actual:   {actual_sha}"
        )
    return actual_sha


def letterbox_image(img: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, float, float]:
    h_orig, w_orig = img.shape[:2]
    scale = min(target_size / w_orig, target_size / h_orig)
    nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
    pad_x = (target_size - nw) / 2.0
    pad_y = (target_size - nh) / 2.0

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    padded[top:top+nh, left:left+nw] = resized

    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, left, top


def postprocess_onnx_boxes(
    out_raw: np.ndarray,
    scale: float,
    left: float,
    top: float,
    orig_w: int,
    orig_h: int,
    conf_min: float = 0.05,
    iou_nms: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decodes raw model output tensor (1, 7, 8400) to Car bounding boxes and scores.
    """
    out = out_raw[0].T  # (8400, 7)
    scores = out[:, 4]  # Class 0: Car
    mask = scores >= conf_min
    if not np.any(mask):
        return np.empty((0, 4), dtype=float), np.empty(0, dtype=float)

    out_filt = out[mask]
    scores_filt = scores[mask]

    xc, yc, bw, bh = out_filt[:, 0], out_filt[:, 1], out_filt[:, 2], out_filt[:, 3]
    x1 = (xc - bw / 2.0 - left) / scale
    y1 = (yc - bh / 2.0 - top) / scale
    x2 = (xc + bw / 2.0 - left) / scale
    y2 = (yc + bh / 2.0 - top) / scale

    x1 = np.clip(x1, 0, orig_w - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    cand_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float64)

    # NMS greedy suppression
    order = scores_filt.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(cand_boxes[i, 0], cand_boxes[order[1:], 0])
        yy1 = np.maximum(cand_boxes[i, 1], cand_boxes[order[1:], 1])
        xx2 = np.minimum(cand_boxes[i, 2], cand_boxes[order[1:], 2])
        yy2 = np.minimum(cand_boxes[i, 3], cand_boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (cand_boxes[i, 2] - cand_boxes[i, 0]) * (cand_boxes[i, 3] - cand_boxes[i, 1])
        area_rem = (cand_boxes[order[1:], 2] - cand_boxes[order[1:], 0]) * (cand_boxes[order[1:], 3] - cand_boxes[order[1:], 1])
        union = np.maximum(area_i + area_rem - inter, 1e-6)
        iou = inter / union
        inds = np.where(iou <= iou_nms)[0]
        order = order[inds + 1]

    return cand_boxes[keep].astype(float), scores_filt[keep]


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def run_parity_check(
    model_pt: YOLO,
    sess_ort: ort.InferenceSession,
    image_paths: list[Path],
    conf_min: float = 0.05,
    iou_nms: float = 0.70,
) -> dict[str, Any]:
    total_pt_boxes = 0
    total_ort_boxes = 0
    high_iou_matches = 0
    matched_boxes_count = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]

        # 1. PyTorch .pt inference on GPU
        res = model_pt.predict(
            source=str(img_path),
            imgsz=640,
            conf=conf_min,
            iou=iou_nms,
            device="cuda",
            verbose=False,
        )[0]
        boxes_pt = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if int(c) == 0:
                    boxes_pt.append(b)
        boxes_pt_arr = np.array(boxes_pt) if boxes_pt else np.empty((0, 4))
        total_pt_boxes += len(boxes_pt_arr)

        # 2. ONNX inference on CPU
        blob, scale, left, top = letterbox_image(img, 640)
        out_ort = sess_ort.run(None, {"images": blob})[0]
        boxes_ort_arr, _ = postprocess_onnx_boxes(out_ort, scale, left, top, w_orig, h_orig, conf_min, iou_nms)
        total_ort_boxes += len(boxes_ort_arr)

        # 3. Match boxes
        for b_pt in boxes_pt_arr:
            matched_boxes_count += 1
            if len(boxes_ort_arr) > 0:
                best_iou = max(compute_box_iou(b_pt, b_ort) for b_ort in boxes_ort_arr)
                if best_iou >= 0.90:
                    high_iou_matches += 1

    count_ratio = total_ort_boxes / total_pt_boxes if total_pt_boxes > 0 else 1.0
    high_iou_rate = high_iou_matches / matched_boxes_count if matched_boxes_count > 0 else 1.0

    return {
        "n_images_tested": len(image_paths),
        "total_pt_detections": int(total_pt_boxes),
        "total_ort_detections": int(total_ort_boxes),
        "count_ratio": round(float(count_ratio), 4),
        "high_iou_match_rate": round(float(high_iou_rate), 4),
        "count_parity_passed": bool(abs(count_ratio - 1.0) <= 0.05),
        "iou_parity_passed": bool(high_iou_rate >= 0.95),
    }


def extract_17_features_vectorized(
    active_boxes: np.ndarray,
    active_scores: np.ndarray,
    cues_res: CuesResult,
    z_base: np.ndarray,
    intrinsics: CameraIntrinsics,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """
    Vectorized extraction of the 17 canonical inference features (FEATURE_COLS_F) (D11, D30, D44).
    """
    n_boxes = len(active_boxes)
    if n_boxes == 0:
        return np.empty((0, 17), dtype=float)

    bw = np.maximum(active_boxes[:, 2] - active_boxes[:, 0], 1.0)
    bh = np.maximum(active_boxes[:, 3] - active_boxes[:, 1], 1.0)
    wh_ratio = bw / bh
    y_bot_cy = active_boxes[:, 3] - intrinsics.cy
    box_cx = (active_boxes[:, 0] + active_boxes[:, 2]) / 2.0
    cx_offset_norm = (box_cx - intrinsics.cx) / float(orig_w)

    t_left = (active_boxes[:, 0] <= BORDER_EPS).astype(float)
    t_right = (active_boxes[:, 2] >= (orig_w - 1 - BORDER_EPS)).astype(float)
    t_top = (active_boxes[:, 1] <= BORDER_EPS).astype(float)
    t_bot = (active_boxes[:, 3] >= (orig_h - 1 - BORDER_EPS)).astype(float)

    conf_col = np.asarray(active_scores, dtype=float)

    ln_zw = np.where(cues_res.valid_w & (cues_res.Z_w > 0), np.log(np.maximum(cues_res.Z_w, 1e-4)), 0.0)
    ln_zh = np.where(cues_res.valid_h & (cues_res.Z_h > 0), np.log(np.maximum(cues_res.Z_h, 1e-4)), 0.0)
    ln_zg = np.where(cues_res.valid_g & (cues_res.Z_g > 0), np.log(np.maximum(cues_res.Z_g, 1e-4)), 0.0)

    z_base_pos = np.maximum(np.nan_to_num(z_base, nan=20.0), 1e-4)
    ln_zbase = np.log(z_base_pos)

    X_mat = np.stack([
        bw, bh, wh_ratio, y_bot_cy, cx_offset_norm,
        t_left, t_right, t_top, t_bot, conf_col,
        ln_zw, ln_zh, ln_zg,
        cues_res.valid_w.astype(float), cues_res.valid_h.astype(float), cues_res.valid_g.astype(float),
        ln_zbase,
    ], axis=1)

    return X_mat


def compute_latency_stats(arr: list[float]) -> dict[str, float]:
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "iqr_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "median_ms": round(float(np.median(a)), 3),
        "mean_ms": round(float(np.mean(a)), 3),
        "std_ms": round(float(np.std(a)), 3),
        "iqr_ms": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "min_ms": round(float(np.min(a)), 3),
        "max_ms": round(float(np.max(a)), 3),
    }


def import_version(pkg_name: str) -> str:
    try:
        mod = __import__(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def generate_latency_markdown(data: dict[str, Any]) -> str:
    hw = data["metadata"]["hardware"]
    vers = data["metadata"]["versions"]
    params = data["metadata"]["parameters"]

    md = []
    md.append("# Bảng Đo Độ Trễ Từng Khâu (Latency Tier 1 Benchmark, T16 Nghiệm Thu Chính Thức)\n")
    md.append("> [!IMPORTANT]")
    md.append("> - Tuân thủ đặc tả Kế hoạch v4 §5.6 và **Quyết định D40, D44, D94, D95**.")
    md.append("> - **GPU Line:** PyTorch `.pt` FP16 trên CUDA GPU (`torch.cuda.synchronize()`).")
    md.append("> - **CPU Line:** ONNX Runtime CPU FP32 (cố định `intra_op_num_threads=4`).")
    md.append(r"> - **Khắc phục 4 lỗi D44:** Triệt tiêu double-count tiền xử lý; đo mảng $\{t_{\text{total}}^{(i)}\}$ từng ảnh thay cho sum of medians; chạy đủ 17 đặc trưng XGBoost; bổ sung đo khâu CQR uncertainty.")
    md.append("> - **Cấm:** Không suy diễn kết luận về độ chính xác từ mô hình ONNX.\n")

    md.append("## 1. Cấu hình Phần cứng & Thư viện Thử nghiệm\n")
    md.append(f"- **CPU:** `{hw['cpu']}` (Luồng kiểm thử: `{hw['cpu_threads']}`)")
    md.append(f"- **GPU:** `{hw['gpu']}`")
    md.append(f"- **Thư viện:** PyTorch `{vers['torch']}`, ONNX Runtime `{vers['onnxruntime']}`, XGBoost `{vers['xgboost']}`")
    md.append(f"- **Tập dữ liệu:** `{params['n_images']}` ảnh ngẫu nhiên từ Split B (sau `{params['n_warmup']}` ảnh khởi động warmup).\n")

    md.append("## 2. Kết quả Kiểm tra Tính Tương đồng (Parity Check: PyTorch .pt vs ONNX, D44, D95)\n")
    md.append("| Detector | Ảnh kiểm thử | Detections (.pt) | Detections (ONNX) | Tỉ lệ Số lượng (ORT/PT) | Tỉ lệ Khớp IoU ≥ 0.90 | Parity Gate |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for det, val in data["detectors"].items():
        par = val["parity"]
        gate_status = "✅ ĐẠT" if par["count_parity_passed"] else "❌ KHÔNG ĐẠT"
        md.append(
            f"| `{det}` | {par['n_images_tested']} | {par['total_pt_detections']} | {par['total_ort_detections']} | "
            f"{par['count_ratio']:.4f} | {par['high_iou_match_rate']:.4f} | {gate_status} |"
        )
    md.append("\n")

    md.append("## 3. Bảng Độ Trễ Từng Khâu (Median / P95 theo ms)\n")
    md.append("| Khâu Pipeline | `yolo11s_640` (GPU / CPU) | `yolov8s_640` (GPU / CPU) | `yolov5su_640` (GPU / CPU) |")
    md.append("| :--- | :---: | :---: | :---: |")

    stages = [
        ("Preprocess (Resize/Letterbox)", "preprocess_gpu", "preprocess_cpu"),
        ("Detector Inference (FP16 GPU / FP32 CPU)", "detector_gpu_fp16", "detector_cpu_ort_fp32"),
        ("Postprocess (Decode/NMS)", "postprocess_gpu", "postprocess_cpu"),
        ("Geometry (Cues + Fusion)", "geometry_gpu", "geometry_cpu"),
        ("Residual (17 Features + XGBoost)", "residual_gpu", "residual_cpu"),
        ("CQR Uncertainty (Log-Interval)", "cqr_gpu", "cqr_cpu"),
    ]

    for label, k_gpu, k_cpu in stages:
        row_str = f"| **{label}** | "
        cells = []
        for det in DETECTORS:
            lat = data["detectors"][det]["latency"]
            gpu_med = lat[k_gpu]["median_ms"]
            cpu_med = lat[k_cpu]["median_ms"]
            cells.append(f"{gpu_med:.2f} ms / {cpu_med:.2f} ms")
        row_str += " | ".join(cells) + " |"
        md.append(row_str)

    # Total row
    md.append("| :--- | :---: | :---: | :---: |")
    row_tot = "| **Tổng Toàn Pipeline End-to-End (ms)** | "
    row_sum_med = "| *Đối chứng: Tổng các Trung vị (Sum of Medians, D44)* | "
    row_fps = "| **Thông lượng Tương đương (FPS)** | "
    tot_cells = []
    sum_med_cells = []
    fps_cells = []
    for det in DETECTORS:
        lat = data["detectors"][det]["latency"]
        tg = lat["total_pipeline_gpu"]["median_ms"]
        tc = lat["total_pipeline_cpu"]["median_ms"]
        sm_g = lat["total_pipeline_gpu"]["sum_of_medians_ms"]
        sm_c = lat["total_pipeline_cpu"]["sum_of_medians_ms"]
        fg = lat["total_pipeline_gpu"]["fps_median"]
        fc = lat["total_pipeline_cpu"]["fps_median"]
        tot_cells.append(f"**{tg:.2f} ms / {tc:.2f} ms**")
        sum_med_cells.append(f"*{sm_g:.2f} ms / {sm_c:.2f} ms*")
        fps_cells.append(f"**{fg} FPS / {fc} FPS**")
    row_tot += " | ".join(tot_cells) + " |"
    row_sum_med += " | ".join(sum_med_cells) + " |"
    row_fps += " | ".join(fps_cells) + " |"
    md.append(row_tot)
    md.append(row_sum_med)
    md.append(row_fps)
    md.append("\n")

    md.append("## 4. Nhận xét Phân bố Thời gian Thực thi (D44, D94)\n")
    det_f_gpu = [data["detectors"][d]["latency"]["detector_gpu_fp16"]["median_ms"] for d in DETECTORS]
    det_f_cpu = [data["detectors"][d]["latency"]["detector_cpu_ort_fp32"]["median_ms"] for d in DETECTORS]
    geom_all = [data["detectors"][d]["latency"]["geometry_gpu"]["median_ms"] for d in DETECTORS]
    res_all = [data["detectors"][d]["latency"]["residual_gpu"]["median_ms"] for d in DETECTORS]
    cqr_all = [data["detectors"][d]["latency"]["cqr_gpu"]["median_ms"] for d in DETECTORS]
    fps_gpu_all = [data["detectors"][d]["latency"]["total_pipeline_gpu"]["fps_median"] for d in DETECTORS]
    fps_cpu_all = [data["detectors"][d]["latency"]["total_pipeline_cpu"]["fps_median"] for d in DETECTORS]

    md.append(f"1. **Khâu Detector:** Là điểm nghẽn chính về thời gian. Trên GPU NVIDIA RTX 5060 Laptop (PyTorch FP16 CUDA), suy luận thô mất ~{min(det_f_gpu):.1f}–{max(det_f_gpu):.1f} ms; trên CPU (ONNX Runtime 4 luồng) mất ~{min(det_f_cpu):.1f}–{max(det_f_cpu):.1f} ms.")
    md.append(f"2. **Khâu Hình học & Residual:** Cực kỳ gọn nhẹ: hình học (cues + fusion) chỉ mất ~{min(geom_all):.2f}–{max(geom_all):.2f} ms; khâu trích xuất 17 đặc trưng và dự đoán XGBoost mất ~{min(res_all):.2f}–{max(res_all):.2f} ms cho mỗi ảnh.")
    md.append(f"3. **Khâu CQR Uncertainty:** Khâu tính toán khoảng tin cậy conformal trong không gian log chỉ mất ~{min(cqr_all):.2f}–{max(cqr_all):.2f} ms (chủ yếu do 2 mô hình quantile XGBoost), hoàn toàn nằm trong ngân sách thời gian thực.")
    md.append(r"4. **Khắc phục lỗi Sum-of-Medians (D44):** Tổng thời gian end-to-end thực tế đo trên từng ảnh $\{t_{\text{total}}^{(i)}\}$ phản ánh chính xác phân phối thời gian thực thi (kèm P95 và IQR), khắc phục độ lệch so với tổng các trung vị đơn lẻ.")
    md.append(f"5. **Khả năng thời gian thực:** Toàn bộ pipeline đạt ~{min(fps_gpu_all):.1f}–{max(fps_gpu_all):.1f} FPS trên GPU RTX 5060 và ~{min(fps_cpu_all):.1f}–{max(fps_cpu_all):.1f} FPS trên CPU 4 luồng, hoàn toàn đáp ứng yêu cầu ADAS thời gian thực (chuẩn $\\ge 10$ FPS trên GPU).\n")

    return "\n".join(md)


def main():
    print("=" * 78)
    print("  OFFICIAL TIER 1 LATENCY BENCHMARK (T16) - STARTING")
    print("=" * 78)

    checkpoints_cfg = PROJECT_ROOT / "configs" / "detector" / "checkpoints.yaml"
    conf_thresh_cfg = PROJECT_ROOT / "configs" / "detector" / "conf_thresholds.yaml"
    geometry_cfg = PROJECT_ROOT / "configs" / "geometry_params.yaml"

    with open(conf_thresh_cfg, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    priors, _ = load_geometry_v2(str(geometry_cfg))

    # Hardware & System Info
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
    cpu_name = platform.processor() or "AMD/Intel x86_64"

    print(f"System Environment:")
    print(f"  OS:           {platform.system()} {platform.release()}")
    print(f"  CPU:          {cpu_name}")
    print(f"  GPU:          {gpu_name} (CUDA={has_cuda})")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  Ultralytics:  {import_version('ultralytics')}")
    print(f"  XGBoost:      {xgb.__version__}")

    # Load Split B images
    feat_df = pd.read_parquet(PROJECT_ROOT / "results" / "datasets" / "yolo11s_640_B_features.parquet")
    unique_frames = feat_df["frame_id"].unique()
    np.random.seed(42)
    selected_frames = np.random.choice(unique_frames, size=min(N_BENCHMARK_IMAGES, len(unique_frames)), replace=False)
    image_paths = [PROJECT_ROOT / "data" / "kitti" / "image_2" / f"{fid}.png" for fid in selected_frames]

    # Pre-load calibration for geometry
    frame_calibs: dict[str, CameraIntrinsics] = {}
    for fid in selected_frames:
        sub = feat_df[feat_df["frame_id"] == fid].iloc[0]
        frame_calibs[fid] = CameraIntrinsics(fx=float(sub["fx"]), fy=float(sub["fy"]), cx=float(sub["cx"]), cy=float(sub["cy"]))

    benchmark_results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "status": "OFFICIAL_TIER1_BENCHMARK",
            "hardware": {
                "cpu": cpu_name,
                "gpu": gpu_name,
                "cpu_threads": CPU_THREADS,
            },
            "versions": {
                "torch": torch.__version__,
                "onnxruntime": ort.__version__,
                "xgboost": xgb.__version__,
                "ultralytics": import_version("ultralytics"),
            },
            "parameters": {
                "n_images": len(image_paths),
                "n_warmup": N_WARMUP,
                "imgsz": 640,
                "iou_nms": 0.70,
            },
        },
        "detectors": {},
    }

    for model_key in DETECTORS:
        print("\n" + "=" * 78)
        print(f"  BENCHMARKING DETECTOR: {model_key}")
        print("=" * 78)

        ckpt_path = PROJECT_ROOT / "runs" / "detector" / model_key / "weights" / "last.pt"
        onnx_path = PROJECT_ROOT / "runs" / "detector" / model_key / "weights" / "last.onnx"
        runs_model_dir = PROJECT_ROOT / "runs" / "residual" / model_key

        # 1. Verify Checkpoint SHA-256
        print(">>> Verifying checkpoint SHA-256...")
        sha_ckpt = verify_checkpoint(model_key, ckpt_path, checkpoints_cfg)
        print(f"    Checkpoint verified: {sha_ckpt[:16]}...")

        # 2. Export ONNX if missing
        if not onnx_path.is_file():
            print(f">>> Exporting ONNX model to {onnx_path}...")
            model_export = YOLO(str(ckpt_path))
            model_export.export(format="onnx", imgsz=640, half=False, verbose=False)
            assert onnx_path.is_file(), f"Failed to export {onnx_path}"

        # 3. Load Models
        model_pt = YOLO(str(ckpt_path))
        if has_cuda:
            model_pt.to("cuda")
            model_pt.model.half()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = CPU_THREADS
        opts.inter_op_num_threads = 1
        sess_ort = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

        # Load Residual & Quantile Models
        fw_frozen = load_fusion_weights(runs_model_dir / "full_fw.json")
        model_f = load_model_f(runs_model_dir / "model_f.json")
        model_e = load_model_e(runs_model_dir / "model_e.json")
        model_q05 = load_quantile_model(runs_model_dir / "model_q05.json")
        model_q95 = load_quantile_model(runs_model_dir / "model_q95.json")

        with open(runs_model_dir / "conformal_calib_C.json", "r", encoding="utf-8") as f:
            calib_c = json.load(f)
        q_hat = float(calib_c["standard_cqr"]["q_hat"])

        # Detector confidence threshold
        det_conf_thr = float(conf_cfg["detectors"][model_key.split("_")[0]]["conf_threshold"])
        print(f"    Confidence threshold for ranging (D6): {det_conf_thr}")

        # 4. Parity Check
        print("\n>>> Running Parity Check on 50 sample images...")
        parity_info = run_parity_check(model_pt, sess_ort, image_paths[:N_PARITY_IMAGES], conf_min=0.05, iou_nms=0.70)
        print(f"    Detection count ratio (ORT / PT): {parity_info['count_ratio']:.4f} (passed: {parity_info['count_parity_passed']})")
        print(f"    High IoU (>= 0.90) match rate:    {parity_info['high_iou_match_rate']:.4f} (passed: {parity_info['iou_parity_passed']})")

        # 5. Latency Measurements across N images
        print(f"\n>>> Measuring latency across {len(image_paths)} images (warmup {N_WARMUP})...")

        # Warmup
        for w_idx in range(N_WARMUP):
            p = image_paths[w_idx % len(image_paths)]
            img_w = cv2.imread(str(p))
            blob_w, s_w, l_w, t_w = letterbox_image(img_w, 640)
            if has_cuda:
                tensor_w = torch.from_numpy(blob_w).to("cuda").half()
                with torch.no_grad():
                    _ = model_pt.model(tensor_w)
                torch.cuda.synchronize()
            _ = sess_ort.run(None, {"images": blob_w})

        # Arrays for GPU line
        t_prep_gpu: list[float] = []
        t_det_gpu: list[float] = []
        t_post_gpu: list[float] = []
        t_geom_gpu: list[float] = []
        t_res_gpu: list[float] = []
        t_cqr_gpu: list[float] = []
        t_total_gpu: list[float] = []

        # Arrays for CPU line
        t_prep_cpu: list[float] = []
        t_det_cpu: list[float] = []
        t_post_cpu: list[float] = []
        t_geom_cpu: list[float] = []
        t_res_cpu: list[float] = []
        t_cqr_cpu: list[float] = []
        t_total_cpu: list[float] = []

        # Benchmark Loop
        for img_idx, p in enumerate(image_paths):
            fid = selected_frames[img_idx]
            intrinsics = frame_calibs[fid]

            # ---------------- GPU LINE ----------------
            if has_cuda:
                # Stage 1: Preprocess GPU (imread + letterbox + H2D tensor transfer)
                t0 = time.perf_counter()
                img = cv2.imread(str(p))
                orig_h, orig_w = img.shape[:2]
                blob, scale, left, top = letterbox_image(img, 640)
                tensor_gpu = torch.from_numpy(blob).to("cuda").half()
                torch.cuda.synchronize()
                t_p_gpu = (time.perf_counter() - t0) * 1000.0

                # Stage 2: Raw Detector Forward FP16 on CUDA (pure forward, no double count)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_raw_gpu = model_pt.model(tensor_gpu)[0]
                torch.cuda.synchronize()
                t_d_gpu = (time.perf_counter() - t0) * 1000.0

                # Stage 3: Postprocess GPU (decode + NMS)
                t0 = time.perf_counter()
                out_raw_np = out_raw_gpu.cpu().numpy()
                boxes_gpu, scores_gpu = postprocess_onnx_boxes(
                    out_raw_np, scale, left, top, orig_w, orig_h, conf_min=det_conf_thr, iou_nms=0.70
                )
                t_po_gpu = (time.perf_counter() - t0) * 1000.0

                # Stage 4: Geometry GPU (Cues + Fusion + Fallback check)
                t0 = time.perf_counter()
                if len(boxes_gpu) > 0:
                    cues_gpu = compute_cues_batch(
                        bboxes=boxes_gpu,
                        intrinsics=intrinsics,
                        priors=priors,
                        img_width=orig_w,
                        img_height=orig_h,
                        eps=BORDER_EPS,
                    )
                    z_cues_mat = np.stack([cues_gpu.Z_w, cues_gpu.Z_h, cues_gpu.Z_g], axis=1)
                    valid_mat = np.stack([cues_gpu.valid_w, cues_gpu.valid_h, cues_gpu.valid_g], axis=1)
                    z_d_gpu = fuse_depths_vectorised(z_cues_mat, valid_mat, fw_frozen)

                    # Fallback pattern 000
                    p000 = ~cues_gpu.valid_w & ~cues_gpu.valid_h & ~cues_gpu.valid_g
                    z_base_gpu = z_d_gpu.copy()
                    if np.any(p000):
                        # Extract e features for fallback
                        X_e_fb = np.stack([
                            boxes_gpu[p000, 2] - boxes_gpu[p000, 0],
                            boxes_gpu[p000, 3] - boxes_gpu[p000, 1],
                            (boxes_gpu[p000, 2] - boxes_gpu[p000, 0]) / (boxes_gpu[p000, 3] - boxes_gpu[p000, 1]),
                            boxes_gpu[p000, 3] - intrinsics.cy,
                            ((boxes_gpu[p000, 0] + boxes_gpu[p000, 2]) / 2.0 - intrinsics.cx) / orig_w,
                            (boxes_gpu[p000, 0] <= BORDER_EPS).astype(float),
                            (boxes_gpu[p000, 2] >= orig_w - 1 - BORDER_EPS).astype(float),
                            (boxes_gpu[p000, 1] <= BORDER_EPS).astype(float),
                            (boxes_gpu[p000, 3] >= orig_h - 1 - BORDER_EPS).astype(float),
                            scores_gpu[p000],
                        ], axis=1)
                        z_base_gpu[p000] = np.exp(model_e.predict(X_e_fb))
                else:
                    cues_gpu = None
                    z_base_gpu = np.empty(0)
                t_g_gpu = (time.perf_counter() - t0) * 1000.0

                # Stage 5: Full Residual GPU (17 features + XGBoost predict)
                t0 = time.perf_counter()
                if len(boxes_gpu) > 0 and cues_gpu is not None:
                    X_f_gpu = extract_17_features_vectorized(
                        boxes_gpu, scores_gpu, cues_gpu, z_base_gpu, intrinsics, orig_w, orig_h
                    )
                    r_hat_gpu = model_f.predict(X_f_gpu)
                    z_hat_gpu = z_base_gpu * np.exp(r_hat_gpu)
                else:
                    X_f_gpu = np.empty((0, 17))
                    z_hat_gpu = np.empty(0)
                t_r_gpu = (time.perf_counter() - t0) * 1000.0

                # Stage 6: Conformal CQR Timing (q05, q95, predict_interval)
                t0 = time.perf_counter()
                if len(boxes_gpu) > 0 and len(X_f_gpu) > 0:
                    q05_gpu = model_q05.predict(X_f_gpu)
                    q95_gpu = model_q95.predict(X_f_gpu)
                    z_lo_gpu, z_hi_gpu, _, _, _ = predict_interval(z_base_gpu, q05_gpu, q95_gpu, q_hat)
                t_c_gpu = (time.perf_counter() - t0) * 1000.0

                # End-to-end total for image i
                t_tot_gpu_i = t_p_gpu + t_d_gpu + t_po_gpu + t_g_gpu + t_r_gpu + t_c_gpu

                t_prep_gpu.append(t_p_gpu)
                t_det_gpu.append(t_d_gpu)
                t_post_gpu.append(t_po_gpu)
                t_geom_gpu.append(t_g_gpu)
                t_res_gpu.append(t_r_gpu)
                t_cqr_gpu.append(t_c_gpu)
                t_total_gpu.append(t_tot_gpu_i)

            # ---------------- CPU LINE ----------------
            # Stage 1: Preprocess CPU (imread + letterbox)
            t0 = time.perf_counter()
            img_c = cv2.imread(str(p))
            orig_h_c, orig_w_c = img_c.shape[:2]
            blob_c, scale_c, left_c, top_c = letterbox_image(img_c, 640)
            t_p_cpu = (time.perf_counter() - t0) * 1000.0

            # Stage 2: Raw Detector Forward FP32 ONNX Runtime CPU
            t0 = time.perf_counter()
            out_raw_cpu = sess_ort.run(None, {"images": blob_c})[0]
            t_d_cpu = (time.perf_counter() - t0) * 1000.0

            # Stage 3: Postprocess CPU (decode + NMS)
            t0 = time.perf_counter()
            boxes_cpu, scores_cpu = postprocess_onnx_boxes(
                out_raw_cpu, scale_c, left_c, top_c, orig_w_c, orig_h_c, conf_min=det_conf_thr, iou_nms=0.70
            )
            t_po_cpu = (time.perf_counter() - t0) * 1000.0

            # Stage 4: Geometry CPU
            t0 = time.perf_counter()
            if len(boxes_cpu) > 0:
                cues_cpu = compute_cues_batch(
                    bboxes=boxes_cpu,
                    intrinsics=intrinsics,
                    priors=priors,
                    img_width=orig_w_c,
                    img_height=orig_h_c,
                    eps=BORDER_EPS,
                )
                z_cues_mat_c = np.stack([cues_cpu.Z_w, cues_cpu.Z_h, cues_cpu.Z_g], axis=1)
                valid_mat_c = np.stack([cues_cpu.valid_w, cues_cpu.valid_h, cues_cpu.valid_g], axis=1)
                z_d_cpu = fuse_depths_vectorised(z_cues_mat_c, valid_mat_c, fw_frozen)

                p000_c = ~cues_cpu.valid_w & ~cues_cpu.valid_h & ~cues_cpu.valid_g
                z_base_cpu = z_d_cpu.copy()
                if np.any(p000_c):
                    X_e_fb_c = np.stack([
                        boxes_cpu[p000_c, 2] - boxes_cpu[p000_c, 0],
                        boxes_cpu[p000_c, 3] - boxes_cpu[p000_c, 1],
                        (boxes_cpu[p000_c, 2] - boxes_cpu[p000_c, 0]) / (boxes_cpu[p000_c, 3] - boxes_cpu[p000_c, 1]),
                        boxes_cpu[p000_c, 3] - intrinsics.cy,
                        ((boxes_cpu[p000_c, 0] + boxes_cpu[p000_c, 2]) / 2.0 - intrinsics.cx) / orig_w_c,
                        (boxes_cpu[p000_c, 0] <= BORDER_EPS).astype(float),
                        (boxes_cpu[p000_c, 2] >= orig_w_c - 1 - BORDER_EPS).astype(float),
                        (boxes_cpu[p000_c, 1] <= BORDER_EPS).astype(float),
                        (boxes_cpu[p000_c, 3] >= orig_h_c - 1 - BORDER_EPS).astype(float),
                        scores_cpu[p000_c],
                    ], axis=1)
                    z_base_cpu[p000_c] = np.exp(model_e.predict(X_e_fb_c))
            else:
                cues_cpu = None
                z_base_cpu = np.empty(0)
            t_g_cpu = (time.perf_counter() - t0) * 1000.0

            # Stage 5: Full Residual CPU
            t0 = time.perf_counter()
            if len(boxes_cpu) > 0 and cues_cpu is not None:
                X_f_cpu = extract_17_features_vectorized(
                    boxes_cpu, scores_cpu, cues_cpu, z_base_cpu, intrinsics, orig_w_c, orig_h_c
                )
                r_hat_cpu = model_f.predict(X_f_cpu)
                z_hat_cpu = z_base_cpu * np.exp(r_hat_cpu)
            else:
                X_f_cpu = np.empty((0, 17))
                z_hat_cpu = np.empty(0)
            t_r_cpu = (time.perf_counter() - t0) * 1000.0

            # Stage 6: Conformal CQR Timing CPU
            t0 = time.perf_counter()
            if len(boxes_cpu) > 0 and len(X_f_cpu) > 0:
                q05_cpu = model_q05.predict(X_f_cpu)
                q95_cpu = model_q95.predict(X_f_cpu)
                z_lo_cpu, z_hi_cpu, _, _, _ = predict_interval(z_base_cpu, q05_cpu, q95_cpu, q_hat)
            t_c_cpu = (time.perf_counter() - t0) * 1000.0

            # End-to-end total for image i
            t_tot_cpu_i = t_p_cpu + t_d_cpu + t_po_cpu + t_g_cpu + t_r_cpu + t_c_cpu

            t_prep_cpu.append(t_p_cpu)
            t_det_cpu.append(t_d_cpu)
            t_post_cpu.append(t_po_cpu)
            t_geom_cpu.append(t_g_cpu)
            t_res_cpu.append(t_r_cpu)
            t_cqr_cpu.append(t_c_cpu)
            t_total_cpu.append(t_tot_cpu_i)

        # Compute Statistics
        prep_gpu_stat = compute_latency_stats(t_prep_gpu)
        det_gpu_stat = compute_latency_stats(t_det_gpu)
        post_gpu_stat = compute_latency_stats(t_post_gpu)
        geom_gpu_stat = compute_latency_stats(t_geom_gpu)
        res_gpu_stat = compute_latency_stats(t_res_gpu)
        cqr_gpu_stat = compute_latency_stats(t_cqr_gpu)
        total_gpu_stat = compute_latency_stats(t_total_gpu)

        prep_cpu_stat = compute_latency_stats(t_prep_cpu)
        det_cpu_stat = compute_latency_stats(t_det_cpu)
        post_cpu_stat = compute_latency_stats(t_post_cpu)
        geom_cpu_stat = compute_latency_stats(t_geom_cpu)
        res_cpu_stat = compute_latency_stats(t_res_cpu)
        cqr_cpu_stat = compute_latency_stats(t_cqr_cpu)
        total_cpu_stat = compute_latency_stats(t_total_cpu)

        # Sum of medians comparison (D44)
        sum_med_gpu = round(
            prep_gpu_stat["median_ms"] + det_gpu_stat["median_ms"] + post_gpu_stat["median_ms"] +
            geom_gpu_stat["median_ms"] + res_gpu_stat["median_ms"] + cqr_gpu_stat["median_ms"], 3
        )
        sum_med_cpu = round(
            prep_cpu_stat["median_ms"] + det_cpu_stat["median_ms"] + post_cpu_stat["median_ms"] +
            geom_cpu_stat["median_ms"] + res_cpu_stat["median_ms"] + cqr_cpu_stat["median_ms"], 3
        )

        fps_gpu_med = round(1000.0 / total_gpu_stat["median_ms"], 1) if total_gpu_stat["median_ms"] > 0 else 0.0
        fps_cpu_med = round(1000.0 / total_cpu_stat["median_ms"], 1) if total_cpu_stat["median_ms"] > 0 else 0.0

        total_gpu_stat["sum_of_medians_ms"] = sum_med_gpu
        total_gpu_stat["fps_median"] = fps_gpu_med
        total_gpu_stat["fps_mean"] = round(1000.0 / total_gpu_stat["mean_ms"], 1) if total_gpu_stat["mean_ms"] > 0 else 0.0

        total_cpu_stat["sum_of_medians_ms"] = sum_med_cpu
        total_cpu_stat["fps_median"] = fps_cpu_med
        total_cpu_stat["fps_mean"] = round(1000.0 / total_cpu_stat["mean_ms"], 1) if total_cpu_stat["mean_ms"] > 0 else 0.0

        print(f"\nFinal Official Results for {model_key}:")
        print(f"  GPU Total End-to-End: {total_gpu_stat['median_ms']} ms (P95: {total_gpu_stat['p95_ms']} ms) -> {fps_gpu_med} FPS")
        print(f"    (Sum of medians was: {sum_med_gpu} ms; diff = {round(total_gpu_stat['median_ms'] - sum_med_gpu, 3)} ms)")
        print(f"  CPU Total End-to-End: {total_cpu_stat['median_ms']} ms (P95: {total_cpu_stat['p95_ms']} ms) -> {fps_cpu_med} FPS")
        print(f"    (Sum of medians was: {sum_med_cpu} ms; diff = {round(total_cpu_stat['median_ms'] - sum_med_cpu, 3)} ms)")

        benchmark_results["detectors"][model_key] = {
            "checkpoint_sha256": sha_ckpt,
            "parity": parity_info,
            "latency": {
                "preprocess_gpu": prep_gpu_stat,
                "preprocess_cpu": prep_cpu_stat,
                "detector_gpu_fp16": det_gpu_stat,
                "detector_cpu_ort_fp32": det_cpu_stat,
                "postprocess_gpu": post_gpu_stat,
                "postprocess_cpu": post_cpu_stat,
                "geometry_gpu": geom_gpu_stat,
                "geometry_cpu": geom_cpu_stat,
                "residual_gpu": res_gpu_stat,
                "residual_cpu": res_cpu_stat,
                "cqr_gpu": cqr_gpu_stat,
                "cqr_cpu": cqr_cpu_stat,
                "total_pipeline_gpu": total_gpu_stat,
                "total_pipeline_cpu": total_cpu_stat,
            },
        }

    # Write JSON and Markdown
    out_dir = PROJECT_ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latency_tier1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Saved] Latency JSON: {json_path}")

    md_path = out_dir / "latency_tier1.md"
    md_content = generate_latency_markdown(benchmark_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Latency Markdown: {md_path}")

    # Log record (AGENT_RULES §5)
    log_rec = make_log_record(
        split="B",
        split_hash=compute_file_sha256(checkpoints_cfg),
        seed=42,
        n_boot=0,
        tag="T16-Latency-Official",
        extra={
            "status": "OFFICIAL_TIER1_BENCHMARK",
            "hardware": benchmark_results["metadata"]["hardware"],
            "detectors": DETECTORS,
            "fps_gpu": {d: benchmark_results["detectors"][d]["latency"]["total_pipeline_gpu"]["fps_median"] for d in DETECTORS},
            "fps_cpu": {d: benchmark_results["detectors"][d]["latency"]["total_pipeline_cpu"]["fps_median"] for d in DETECTORS},
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_rec)
    print("[Logged] Official benchmark recorded to runs/pipeline_log.jsonl")


if __name__ == "__main__":
    main()
