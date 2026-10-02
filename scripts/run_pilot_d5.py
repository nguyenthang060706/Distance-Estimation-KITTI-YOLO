"""
scripts/run_pilot_d5.py: Runs Pilot D5 on YOLOv8s (30 epochs) to benchmark imgsz=640 vs imgsz=960.

Criteria (from Decision D5):
- Metric: Car mAP@0.5:0.95 on Split V.
- Tie-breaker: Pick smaller imgsz (640).
- Hardware note: Measures time/epoch and peak VRAM on RTX 5060 (8GB).
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.train_detector import train_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pilot D5 (imgsz benchmark) on YOLOv8s.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of pilot epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=640, choices=[640, 960], help="Image size to test")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    return parser.parse_args()


def run_pilot(imgsz: int, epochs: int = 30, batch: int = 16, device: str = "0"):
    print("\n" + "=" * 60)
    print(f"PILOT D5: Training YOLOv8s at imgsz={imgsz} for {epochs} epochs (batch={batch})")
    print("=" * 60 + "\n")

    run_name = f"pilot_yolov8s_{imgsz}"
    cmd_args = [
        "--config", "configs/detector/train_config.yaml",
        "--model", "yolov8s.pt",
        "--data", "data/yolo_kitti/dataset.yaml",
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--imgsz", str(imgsz),
        "--device", str(device),
        "--name", run_name,
        "--project", "runs/detector",
    ]

    # Backup sys.argv and run
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + cmd_args
    try:
        results = train_detector()
    finally:
        sys.argv = old_argv

    return results


if __name__ == "__main__":
    args = parse_args()
    run_pilot(imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device)
