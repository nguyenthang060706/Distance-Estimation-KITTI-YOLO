"""
scripts/find_conf_thresholds.py: Finds optimal confidence threshold for each detector
using maximum F1 on class Car on Split V (§5.1, Decision D6).
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# UTF-8 stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_split


def box_iou(box1, box2):
    """Compute IoU between [x1, y1, x2, y2] and [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def find_best_conf_threshold(model_path: str, model_name: str, val_frame_ids: list, loader: KITTILoader, imgsz: int = 640):
    print(f"\n--- Finding Optimal Conf Threshold for {model_name} (Car Hard on Split V, D8 protocol) ---")
    model = YOLO(model_path)

    # 1. Collect all predictions and ground truths for Split V
    # In YOLO: class 0 is Car
    predictions = []  # list of (conf, is_tp)
    n_gt_hard_cars = 0

    for fid in tqdm(val_frame_ids, desc=f"Predicting {model_name}"):
        frame = loader.load_frame(fid)
        img_path = frame.image_path

        # Decision D8: Target population is Car passing Hard filter
        gt_hard_cars = [obj.bbox for obj in frame.objects if obj.obj_class == "Car" and obj.passes_hard_filter()]
        gt_non_hard_cars = [obj.bbox for obj in frame.objects if obj.obj_class == "Car" and not obj.passes_hard_filter()]
        dontcares = frame.dontcare_boxes
        n_gt_hard_cars += len(gt_hard_cars)

        # Run inference using imgsz from config
        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=0.01,   # very low to get full PR curve
            iou=0.7,
            device=0,
            verbose=False,
        )[0]

        # Extract predicted car boxes & confidences
        pred_cars = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for b, c, score in zip(boxes, clss, confs):
                if c == 0:  # Car
                    pred_cars.append((score, b))

        # Sort predictions by confidence desc
        pred_cars.sort(key=lambda x: x[0], reverse=True)

        # Greedy match IoU >= 0.5 according to D8 protocol:
        # Step 1: Match GT Hard first
        # Step 2: If not matched to GT Hard, check GT non-Hard -> Ignore (neither TP nor FP)
        # Step 3: Check DontCare -> Ignore (neither TP nor FP)
        # Step 4: Otherwise False Positive
        matched_gt_hard = set()
        for score, pred_box in pred_cars:
            # 1. Check match with GT Hard
            best_iou = 0.0
            best_idx = -1
            for g_idx, gt_box in enumerate(gt_hard_cars):
                if g_idx in matched_gt_hard:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx

            if best_iou >= 0.5:
                matched_gt_hard.add(best_idx)
                predictions.append((score, True))  # True Positive
                continue

            # 2. Check match with non-Hard GT cars (D8: ignored)
            is_non_hard = False
            for nh_box in gt_non_hard_cars:
                if box_iou(pred_box, nh_box) >= 0.5:
                    is_non_hard = True
                    break
            if is_non_hard:
                continue

            # 3. Check match with DontCare (D8: ignored)
            is_dc = False
            for dc_box in dontcares:
                if box_iou(pred_box, dc_box) >= 0.5:
                    is_dc = True
                    break
            if is_dc:
                continue

            # 4. Unmatched detection outside GT Hard, non-Hard, and DontCare -> False Positive
            predictions.append((score, False))

    # 2. Sweep thresholds from 0.05 to 0.90
    thresholds = np.linspace(0.05, 0.90, 86)
    best_f1 = 0.0
    best_thresh = 0.5
    best_p = 0.0
    best_r = 0.0

    print(f"Total GT Hard Cars on V: {n_gt_hard_cars}")
    print(f"Total Evaluated Predictions on V: {len(predictions)}")

    raw_f1 = np.zeros(len(thresholds))
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))

    for idx, t in enumerate(thresholds):
        tp = sum(1 for score, is_tp in predictions if score >= t and is_tp)
        fp = sum(1 for score, is_tp in predictions if score >= t and not is_tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / n_gt_hard_cars if n_gt_hard_cars > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precisions[idx] = p
        recalls[idx] = r
        raw_f1[idx] = f1

    # Decision D6: Smooth F1 curve using moving average window of 0.05 (±0.025)
    window_half = 0.025
    smoothed_f1 = np.zeros_like(raw_f1)
    for i, t in enumerate(thresholds):
        mask = (thresholds >= t - window_half) & (thresholds <= t + window_half)
        smoothed_f1[i] = np.mean(raw_f1[mask])

    best_idx = int(np.argmax(smoothed_f1))
    best_thresh = float(thresholds[best_idx])
    best_f1_smooth = float(smoothed_f1[best_idx])
    best_f1_raw = float(raw_f1[best_idx])
    best_p = float(precisions[best_idx])
    best_r = float(recalls[best_idx])

    # Retrieve best.pt epoch from results.csv if available
    best_epoch = None
    results_csv = Path(model_path).parent.parent / "results.csv"
    if results_csv.exists():
        try:
            import pandas as pd
            rdf = pd.read_csv(results_csv)
            rdf.columns = [c.strip() for c in rdf.columns]
            if "fitness" in rdf.columns:
                best_epoch = int(rdf["fitness"].idxmax())
            elif "metrics/mAP50-95(B)" in rdf.columns:
                best_epoch = int(rdf["metrics/mAP50-95(B)"].idxmax())
        except Exception:
            pass

    print(f"Optimal Threshold for {model_name} (D6 smoothed argmax): {best_thresh:.3f}")
    print(f"  Smoothed F1: {best_f1_smooth:.4f} | Raw F1: {best_f1_raw:.4f} | Precision: {best_p:.4f} | Recall: {best_r:.4f}")
    if best_epoch is not None:
        print(f"  Note: best.pt occurred at epoch {best_epoch} (D12 uses last.pt at epoch 100)")

    return {
        "conf_threshold": round(best_thresh, 3),
        "smoothed_f1": round(best_f1_smooth, 4),
        "raw_f1": round(best_f1_raw, 4),
        "precision": round(best_p, 4),
        "recall": round(best_r, 4),
        "n_gt_hard": n_gt_hard_cars,
        "checkpoint_used": "last.pt (Decision D12)",
        "best_epoch_reference": best_epoch,
    }



def get_git_info():
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).returncode != 0
    except Exception:
        dirty = False
    return commit, dirty


def main():
    loader = KITTILoader("data/kitti")
    val_frame_ids = load_split("splits", "V")
    print(f"Loaded {len(val_frame_ids)} frames from Split V.")

    # Read imgsz from train_config.yaml
    train_cfg_path = Path("configs/detector/train_config.yaml")
    imgsz = 640
    if train_cfg_path.exists():
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            tcfg = yaml.safe_load(f)
            imgsz = tcfg.get("imgsz", 640)
    print(f"Using imgsz={imgsz} from config.")

    # Decision D12: Target checkpoint is last.pt (epoch 100)
    models = {
        "yolov8s": f"runs/detector/yolov8s_{imgsz}/weights/last.pt",
        "yolo11s": f"runs/detector/yolo11s_{imgsz}/weights/last.pt",
        "yolov5su": f"runs/detector/yolov5su_{imgsz}/weights/last.pt",
    }


    git_commit, git_dirty = get_git_info()

    results = {
        "metadata": {
            "decision": "D6 (thresholds) + D8 (matching population: Car Hard, ignore non-Hard & DontCare)",
            "split": "V",
            "imgsz": imgsz,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "detectors": {},
    }
    for name, path in models.items():
        results["detectors"][name] = find_best_conf_threshold(path, name, val_frame_ids, loader, imgsz=imgsz)

    # Save to configs/detector/conf_thresholds.yaml
    output_path = Path("configs/detector/conf_thresholds.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 65)
    print("FINAL OPTIMAL CONFIDENCE THRESHOLDS (Decisions D6 & D8):")
    print("=" * 65)
    print(f"{'Detector':<12} | {'Conf Thresh':>12} | {'Max F1':>8} | {'Precision':>10} | {'Recall':>8}")
    print("-" * 65)
    for name, r in results["detectors"].items():
        print(f"{name:<12} | {r['conf_threshold']:>12.3f} | {r['max_f1']:>8.4f} | {r['precision']:>10.4f} | {r['recall']:>8.4f}")
    print("=" * 65)
    print(f"Saved thresholds to: {output_path}\n")


if __name__ == "__main__":
    main()
