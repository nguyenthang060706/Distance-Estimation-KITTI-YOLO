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


def find_best_conf_threshold(model_path: str, model_name: str, val_frame_ids: list, loader: KITTILoader):
    print(f"\n--- Finding Optimal Conf Threshold for {model_name} (Car on Split V) ---")
    model = YOLO(model_path)

    # 1. Collect all predictions and ground truths for Split V
    # In YOLO: class 0 is Car
    predictions = []  # list of (conf, iou_match, is_dontcare)
    n_gt_cars = 0

    for fid in tqdm(val_frame_ids, desc=f"Predicting {model_name}"):
        frame = loader.load_frame(fid)
        img_path = frame.image_path

        # Ground truth cars (without Hard filter for matching threshold selection on V, §5.1)
        gt_cars = [obj.bbox for obj in frame.objects if obj.obj_class == "Car"]
        dontcares = frame.dontcare_boxes
        n_gt_cars += len(gt_cars)

        # Run inference
        results = model.predict(
            source=img_path,
            imgsz=640,
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

        # Greedy match IoU >= 0.5
        matched_gt = set()
        for score, pred_box in pred_cars:
            # Check DontCare first (§5.1: detection matched to DontCare is ignored)
            is_dc = False
            for dc_box in dontcares:
                if box_iou(pred_box, dc_box) >= 0.5:
                    is_dc = True
                    break
            if is_dc:
                # Ignore DontCare detection
                continue

            best_iou = 0.0
            best_idx = -1
            for g_idx, gt_box in enumerate(gt_cars):
                if g_idx in matched_gt:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx

            if best_iou >= 0.5:
                matched_gt.add(best_idx)
                predictions.append((score, True))  # TP
            else:
                predictions.append((score, False)) # FP

    # 2. Sweep thresholds from 0.05 to 0.90
    thresholds = np.linspace(0.05, 0.90, 86)
    best_f1 = 0.0
    best_thresh = 0.5
    best_p = 0.0
    best_r = 0.0

    print(f"Total GT Cars on V: {n_gt_cars}")
    print(f"Total Predictions evaluated: {len(predictions)}")

    for t in thresholds:
        tp = sum(1 for score, is_tp in predictions if score >= t and is_tp)
        fp = sum(1 for score, is_tp in predictions if score >= t and not is_tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / n_gt_cars if n_gt_cars > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_p = float(p)
            best_r = float(r)

    print(f"Optimal Threshold for {model_name}: {best_thresh:.3f}")
    print(f"  Max F1: {best_f1:.4f} | Precision: {best_p:.4f} | Recall: {best_r:.4f}")

    return {
        "conf_threshold": round(best_thresh, 3),
        "max_f1": round(best_f1, 4),
        "precision": round(best_p, 4),
        "recall": round(best_r, 4),
        "n_gt": n_gt_cars,
    }


def main():
    loader = KITTILoader("data/kitti")
    val_frame_ids = load_split("splits", "V")
    print(f"Loaded {len(val_frame_ids)} frames from Split V.")

    models = {
        "yolov8s": "runs/detector/yolov8s_640/weights/best.pt",
        "yolo11s": "runs/detector/yolo11s_640/weights/best.pt",
        "yolov5su": "runs/detector/yolov5su_640/weights/best.pt",
    }

    results = {}
    for name, path in models.items():
        results[name] = find_best_conf_threshold(path, name, val_frame_ids, loader)

    # Save to configs/detector/conf_thresholds.yaml
    output_path = Path("configs/detector/conf_thresholds.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 65)
    print("FINAL OPTIMAL CONFIDENCE THRESHOLDS (Decision D6):")
    print("=" * 65)
    print(f"{'Detector':<12} | {'Conf Thresh':>12} | {'Max F1':>8} | {'Precision':>10} | {'Recall':>8}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<12} | {r['conf_threshold']:>12.3f} | {r['max_f1']:>8.4f} | {r['precision']:>10.4f} | {r['recall']:>8.4f}")
    print("=" * 65)
    print(f"Saved thresholds to: {output_path}\n")


if __name__ == "__main__":
    main()
