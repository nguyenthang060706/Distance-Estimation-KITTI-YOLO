"""
scripts/eval_detectors_val.py: Evaluates all 3 trained detectors on Split V.
Computes P, R, mAP50, mAP50-95 per class (Car, Van, Truck) and overall.
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# Set UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    # Decision D12: Target checkpoint is last.pt (epoch 100)
    models = [
        ("YOLOv8s (last)", "runs/detector/yolov8s_640/weights/last.pt"),
        ("YOLOv8s (best)", "runs/detector/yolov8s_640/weights/best.pt"),
        ("YOLO11s (last)", "runs/detector/yolo11s_640/weights/last.pt"),
        ("YOLO11s (best)", "runs/detector/yolo11s_640/weights/best.pt"),
        ("YOLOv5su (last)", "runs/detector/yolov5su_640/weights/last.pt"),
        ("YOLOv5su (best)", "runs/detector/yolov5su_640/weights/best.pt"),
    ]


    print(f"\n{'Model':<10} | {'Class':<8} | {'P':>7} | {'R':>7} | {'mAP50':>7} | {'mAP50-95':>9}")
    print("-" * 62)

    summary = {}
    for name, pt in models:
        if not Path(pt).exists():
            print(f"Weights not found: {pt}")
            continue
        m = YOLO(pt)
        metrics = m.val(
            data="data/yolo_kitti/dataset.yaml",
            imgsz=640,
            split="val",
            device=0,
            verbose=False,
            workers=0,  # avoid multiprocessing issue on Windows
        )
        summary[name] = {}
        for i, c in enumerate(["Car", "Van", "Truck"]):
            p = float(metrics.box.p[i])
            r = float(metrics.box.r[i])
            map50 = float(metrics.box.ap50[i])
            map50_95 = float(metrics.box.ap[i])
            summary[name][c] = {
                "P": p,
                "R": r,
                "mAP50": map50,
                "mAP50-95": map50_95,
            }
            print(f"{name:<10} | {c:<8} | {p:>7.4f} | {r:>7.4f} | {map50:>7.4f} | {map50_95:>9.4f}")
        all_map50 = float(metrics.box.map50)
        all_map = float(metrics.box.map)
        summary[name]["all"] = {"mAP50": all_map50, "mAP50-95": all_map}
        print(f"{name:<10} | {'ALL':<8} | {'-':>7} | {'-':>7} | {all_map50:>7.4f} | {all_map:>9.4f}")
        print("-" * 62)


if __name__ == "__main__":
    main()
