"""
scripts/run_inference.py: Batch inference pipeline for trained YOLO detectors on KITTI splits.

Implements Decision D4, D6, D11, D12, D15, D22:
- Supports splits: A (diagnostic), B (residual fitting), C (conformal calibration).
- Hard guard: Split T is strictly forbidden and raises PermissionError.
- Verifies checkpoint SHA-256 against configs/detector/checkpoints.yaml before running.
- Runs inference in FP32 (half=False) at imgsz=640, conf=0.05, iou=0.7.
- Flags predictions with pass_thr (Decision D6 threshold from conf_thresholds.yaml)
  to enable full recall-confidence analysis without re-running.
- Strictly separates outputs into two artifacts (Decision D11 & D22):
    1. {model}_{split}_detections.parquet: Test-time observable features only (no GT).
    2. {model}_{split}_matches.parquet: Evaluation-only match results (TP/FP/IGNORED).
- Appends run metadata to runs/inference_log.jsonl.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_split, compute_split_hash
from src.detection.matching import match_detections_frame, MatchStatus


MODEL_NAME_MAP = {
    "yolov8s": "yolov8s_640",
    "yolov8s_640": "yolov8s_640",
    "yolo11s": "yolo11s_640",
    "yolo11s_640": "yolo11s_640",
    "yolov5su": "yolov5su_640",
    "yolov5su_640": "yolov5su_640",
}


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_info() -> tuple[str, bool]:
    """Return (commit_hash, is_dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", False


def verify_checkpoint(model_key: str, checkpoint_path: Path, checkpoints_cfg_path: Path) -> str:
    """
    Verify checkpoint file exists and matches expected SHA-256 hash in config.
    """
    if not checkpoint_path.exists():
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


def run_inference_for_model(
    model_name: str,
    split: str,
    data_root: str = "data/kitti",
    splits_dir: str = "splits",
    output_dir: str = "results/predictions",
    checkpoints_cfg: str = "configs/detector/checkpoints.yaml",
    conf_thresholds_cfg: str = "configs/detector/conf_thresholds.yaml",
    conf_min: float = 0.05,
    iou_nms: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    dontcare_mode: str = "iou",
) -> tuple[Path, Path]:
    """
    Run full inference and matching for a single model on a single split.
    """
    # 1. Hard Guard against Split T
    if split.upper() == "T":
        raise PermissionError(
            "Access to Split T is strictly forbidden during Week 2 development (Decision D4, D22)! "
            "Split T is strictly reserved for one-time final test in Week 3."
        )

    model_key = MODEL_NAME_MAP.get(model_name)
    if not model_key:
        raise ValueError(f"Unknown model name: {model_name}. Allowed: {list(MODEL_NAME_MAP.keys())}")

    base_model = model_key.replace("_640", "")
    checkpoint_path = PROJECT_ROOT / "runs" / "detector" / model_key / "weights" / "last.pt"
    checkpoints_cfg_path = PROJECT_ROOT / checkpoints_cfg
    conf_cfg_path = PROJECT_ROOT / conf_thresholds_cfg

    # 2. Checkpoint SHA-256 verification
    print(f"\n[{model_key}] Verifying checkpoint SHA-256...")
    actual_sha = verify_checkpoint(model_key, checkpoint_path, checkpoints_cfg_path)
    print(f"[{model_key}] Checkpoint verified (SHA-256: {actual_sha[:16]}...)")

    # 3. Load Decision D6 confidence threshold
    with open(conf_cfg_path, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)
    model_conf_thr = float(conf_cfg["detectors"][base_model]["conf_threshold"])
    print(f"[{model_key}] Loaded Decision D6 confidence threshold: {model_conf_thr:.4f}")

    # 4. Device selection
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] Target device: {device} (FP32)")

    # 5. Load Split frame IDs
    frame_ids = load_split(splits_dir, split, allow_test=False)
    split_hash = compute_split_hash(frame_ids)
    print(f"[{model_key}] Running on Split {split} ({len(frame_ids)} frames, hash {split_hash[:12]}...)")

    # 6. Initialize Loader and YOLO model
    loader = KITTILoader(data_root)
    model = YOLO(str(checkpoint_path))

    detection_records = []
    match_records = []

    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for fid in tqdm(frame_ids, desc=f"Infer {model_key} on {split}"):
        frame = loader.load_frame(fid)
        img_path = frame.image_path

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        calib = frame.calib
        fx, fy = float(calib.fx), float(calib.fy)
        cx, cy = float(calib.cx), float(calib.cy)

        # Predict with FP32, conf=0.05, iou=0.7, imgsz=640
        res = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=conf_min,
            iou=iou_nms,
            half=False,
            device=device,
            verbose=False,
        )[0]

        pred_boxes = []
        pred_scores = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_all = res.boxes.xyxy.cpu().numpy()
            clss_all = res.boxes.cls.cpu().numpy().astype(int)
            confs_all = res.boxes.conf.cpu().numpy()

            for b, c, s in zip(boxes_all, clss_all, confs_all):
                if c == 0:  # Class 0 is Car
                    # Clip coordinates to original image bounds
                    x1 = max(0.0, min(float(b[0]), float(img_w - 1)))
                    y1 = max(0.0, min(float(b[1]), float(img_h - 1)))
                    x2 = max(0.0, min(float(b[2]), float(img_w - 1)))
                    y2 = max(0.0, min(float(b[3]), float(img_h - 1)))
                    if x2 > x1 and y2 > y1:
                        pred_boxes.append([x1, y1, x2, y2])
                        pred_scores.append(float(s))

        pred_boxes_arr = np.array(pred_boxes) if pred_boxes else np.empty((0, 4))
        pred_scores_arr = np.array(pred_scores) if pred_scores else np.empty(0)

        # 7. Record pure detection features (Decision D11: NO GT fields here!)
        for p_idx in range(len(pred_boxes_arr)):
            b = pred_boxes_arr[p_idx]
            s = pred_scores_arr[p_idx]
            detection_records.append({
                "frame_id": fid,
                "drive": frame.drive,
                "pred_idx": p_idx,
                "x1": round(float(b[0]), 2),
                "y1": round(float(b[1]), 2),
                "x2": round(float(b[2]), 2),
                "y2": round(float(b[3]), 2),
                "conf": round(float(s), 4),
                "pass_thr": bool(s >= model_conf_threshold),
                "fx": round(fx, 4),
                "fy": round(fy, 4),
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "img_w": int(img_w),
                "img_h": int(img_h),
            })

        # 8. Evaluation matching against GT (Decision D8, D15)
        gt_hard = [o.bbox for o in frame.objects if o.obj_class == "Car" and o.passes_hard_filter()]
        gt_non_hard = [o.bbox for o in frame.objects if o.obj_class == "Car" and not o.passes_hard_filter()]
        dontcares = frame.dontcare_boxes

        matches = match_detections_frame(
            pred_boxes=pred_boxes_arr,
            pred_scores=pred_scores_arr,
            gt_hard_boxes=np.array(gt_hard) if gt_hard else np.empty((0, 4)),
            gt_non_hard_boxes=np.array(gt_non_hard) if gt_non_hard else np.empty((0, 4)),
            dontcare_boxes=np.array(dontcares) if len(dontcares) > 0 else np.empty((0, 4)),
            iou_threshold=0.7,
            dontcare_mode=dontcare_mode,
            dontcare_threshold=0.5,
        )

        for m in matches:
            match_records.append({
                "frame_id": fid,
                "pred_idx": m.pred_idx,
                "status": m.status.value,
                "matched_gt_idx": m.matched_gt_idx if m.matched_gt_idx is not None else -1,
                "matched_iou": round(float(m.matched_iou), 4),
            })

    elapsed = time.time() - start_time
    print(f"[{model_key}] Completed inference in {elapsed:.1f}s ({len(frame_ids)/elapsed:.1f} FPS)")

    df_dets = pd.DataFrame(detection_records)
    df_matches = pd.DataFrame(match_records)

    # 9. Save two distinct artifacts
    det_out = out_dir / f"{model_key}_{split}_detections.parquet"
    match_out = out_dir / f"{model_key}_{split}_matches.parquet"

    df_dets.to_parquet(det_out, index=False)
    df_matches.to_parquet(match_out, index=False)
    print(f"[{model_key}] Saved detections to: {det_out} ({len(df_dets)} rows)")
    print(f"[{model_key}] Saved matches to:    {match_out} ({len(df_matches)} rows)")

    # 10. Append to inference log JSONL
    log_file = PROJECT_ROOT / "runs" / "inference_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = get_git_info()

    n_pass = int(df_dets["pass_thr"].sum()) if len(df_dets) > 0 else 0
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_key,
        "split": split,
        "n_frames": len(frame_ids),
        "split_hash": split_hash,
        "checkpoint_sha256": actual_sha,
        "conf_min": conf_min,
        "conf_threshold": model_conf_threshold,
        "iou_nms": iou_nms,
        "imgsz": imgsz,
        "half": False,
        "device": device,
        "ultralytics_version": "8.4.171",
        "torch_version": torch.__version__,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "total_detections": len(df_dets),
        "detections_passed_threshold": n_pass,
        "detections_file": str(det_out.relative_to(PROJECT_ROOT)),
        "matches_file": str(match_out.relative_to(PROJECT_ROOT)),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return det_out, match_out


def main():
    parser = argparse.ArgumentParser(description="Run detector inference and matching on KITTI splits.")
    parser.add_argument("--model", default="all", choices=["yolov8s", "yolo11s", "yolov5su", "all"])
    parser.add_argument("--split", required=True, choices=["A", "B", "C"], help="Split to infer: A, B, or C (T is forbidden)")
    parser.add_argument("--conf", type=float, default=0.05, help="Minimum confidence threshold (default 0.05)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default 0.7)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    parser.add_argument("--device", default=None, help="Device to run inference on (default cuda:0 if available)")
    parser.add_argument("--dontcare-mode", default="iou", choices=["iou", "area_pred"], help="DontCare overlap criterion")

    args = parser.parse_args()

    # Guard on Split T
    if args.split.upper() == "T":
        raise PermissionError(
            "Access to Split T is strictly forbidden during Week 2 development (Decision D4, D22)!"
        )

    models_to_run = ["yolov8s", "yolo11s", "yolov5su"] if args.model == "all" else [args.model]

    print("=" * 80)
    print(f"RUNNING INFERENCE ON SPLIT {args.split} FOR MODELS: {models_to_run}")
    print("=" * 80)

    for m in models_to_run:
        run_inference_for_model(
            model_name=m,
            split=args.split,
            conf_min=args.conf,
            iou_nms=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            dontcare_mode=args.dontcare_mode,
        )

    print("\n✓ All inferences completed successfully!")


if __name__ == "__main__":
    main()
