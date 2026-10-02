"""
scripts/train_queue.py: Sequential training queue for all 3 detectors.

Queue order (from KE_HOACH_V4 §8.1 Day 3):
1. YOLOv8s  (yolov8s.pt)
2. YOLO11s  (yolo11s.pt)
3. YOLOv5su (yolov5su.pt)

All detectors train under identical conditions (§2, train_config.yaml):
- Dataset: KITTI Split A (train) + Split V (val)
- Resolution: imgsz=640 (locked per Decision D5)
- Batch: 16 (FP16 AMP enabled)
- Epochs: 100 (patience=15 early stopping on V)
- Seed: 42
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# UTF-8 stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.train_detector import train_detector

DETECTOR_QUEUE = [
    {"name": "yolov8s_640",  "model": "yolov8s.pt"},
    {"name": "yolo11s_640",  "model": "yolo11s.pt"},
    {"name": "yolov5su_640", "model": "yolov5su.pt"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run sequential fine-tuning queue for all 3 detectors.")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="GPU device (default: 0)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8s.pt", "yolo11s.pt", "yolov5su.pt"],
        help="List of model weights to train",
    )
    return parser.parse_args()


def run_queue():
    args = parse_args()
    summary = []

    # Read patience from config so the printed value is accurate
    config_path = Path("configs/detector/train_config.yaml")
    patience_val = "?"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            _cfg = yaml.safe_load(f)
            patience_val = _cfg.get("patience", "?")

    print("\n" + "=" * 70)
    print("STARTING DETECTOR FINE-TUNING QUEUE (D7: patience=%s)" % patience_val)
    print(f"Models:   {args.models}")
    print(f"imgsz:    {args.imgsz}")
    print(f"Epochs:   {args.epochs} (patience={patience_val})")
    print(f"Batch:    {args.batch}")
    print(f"Device:   {args.device}")
    print("=" * 70 + "\n")

    for i, model_weight in enumerate(args.models, 1):
        stem = Path(model_weight).stem
        run_name = f"{stem}_{args.imgsz}"

        print("\n" + "#" * 70)
        print(f"[{i}/{len(args.models)}] STARTING TRAINING: {run_name} ({model_weight})")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#" * 70 + "\n")

        cmd_args = [
            "--config", "configs/detector/train_config.yaml",
            "--model", model_weight,
            "--data", "data/yolo_kitti/dataset.yaml",
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
            "--device", str(args.device),
            "--name", run_name,
            "--project", "runs/detector",
        ]

        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + cmd_args
        t_start = datetime.now()
        try:
            results = train_detector()
            status = "SUCCESS"
        except Exception as e:
            print(f"[ERROR] Training failed for {run_name}: {e}", file=sys.stderr)
            status = f"FAILED: {e}"
            results = None
        finally:
            sys.argv = old_argv

        t_end = datetime.now()
        duration_min = (t_end - t_start).total_seconds() / 60.0

        model_summary = {
            "model": model_weight,
            "run_name": run_name,
            "status": status,
            "duration_minutes": duration_min,
            "save_dir": str(getattr(results, "save_dir", "")),
        }
        summary.append(model_summary)

        print(f"\n[FINISHED {i}/{len(args.models)}] {run_name} in {duration_min:.2f} minutes.")

    # Save final queue summary
    summary_path = Path("runs/detector/queue_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL DETECTORS IN QUEUE COMPLETED!")
    print(f"Summary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_queue()
