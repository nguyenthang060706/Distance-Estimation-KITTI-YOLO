"""
src/detection/train_detector.py: Training script for 2D object detectors on KITTI (Splits A & V).

Key Design Points:
- Reads train_config.yaml.
- Converts nms_conf -> conf and nms_iou -> iou for Ultralytics.
- Strips non-Ultralytics config keys (classes, conf_threshold, etc.) to prevent errors.
- Logs experiment metadata (seed, split hashes, git commit, metrics) to runs/detector/train_log.jsonl.
- Supports CLI overrides for pilot runs (imgsz, epochs, model).
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if git fails."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def get_split_hashes(splits_meta_path: Path) -> dict:
    """Read split hashes from split_metadata.json."""
    if not splits_meta_path.exists():
        return {}
    with open(splits_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "seed": meta.get("seed"),
        "split_A_hash": meta.get("splits", {}).get("A", {}).get("hash"),
        "split_V_hash": meta.get("splits", {}).get("V", {}).get("hash"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO detector on KITTI Splits A & V.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detector/train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Detector model weight/architecture (e.g. yolov8s.pt, yolo11s.pt, yolov5su.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/yolo_kitti/dataset.yaml",
        help="Path to YOLO dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. 0, cpu)")
    parser.add_argument("--name", type=str, default=None, help="Custom experiment name")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detector",
        help="Directory to save runs",
    )
    return parser.parse_args()


def prepare_ultralytics_args(raw_config: dict, cli_args: argparse.Namespace) -> dict:
    """
    Prepare arguments for Ultralytics train():
    - Map nms_conf -> conf, nms_iou -> iou
    - Remove non-Ultralytics keys
    - Apply CLI overrides
    """
    cfg = dict(raw_config)

    # Key renaming for Ultralytics
    if "nms_conf" in cfg:
        cfg["conf"] = cfg.pop("nms_conf")
    if "nms_iou" in cfg:
        cfg["iou"] = cfg.pop("nms_iou")

    # Remove non-Ultralytics keys that cause crashes
    keys_to_remove = ["classes", "conf_threshold"]
    for k in keys_to_remove:
        cfg.pop(k, None)

    # CLI overrides
    if cli_args.epochs is not None:
        cfg["epochs"] = cli_args.epochs
    if cli_args.batch is not None:
        cfg["batch"] = cli_args.batch
    if cli_args.imgsz is not None:
        cfg["imgsz"] = cli_args.imgsz
    if cli_args.device is not None:
        cfg["device"] = cli_args.device

    # Ensure dataset path is absolute or correct relative path
    cfg["data"] = str(Path(cli_args.data).resolve())

    # Set project and run name (use absolute path to avoid Ultralytics nested default directories)
    cfg["project"] = Path(cli_args.project).resolve().as_posix()
    model_name = Path(cli_args.model).stem
    if cli_args.name is not None:
        cfg["name"] = cli_args.name
    else:
        cfg["name"] = f"{model_name}_{cfg.get('imgsz', 640)}"

    cfg["exist_ok"] = True

    return cfg


def log_experiment_to_jsonl(log_entry: dict, log_dir: Path):
    """Append experiment result entry to train_log.jsonl."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")
    print(f"Logged experiment details to {log_file}")


def train_detector():
    args = parse_args()
    config_path = Path(args.config)
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Check dataset existence
    data_yaml_path = Path(args.data)
    assert data_yaml_path.exists(), (
        f"dataset.yaml not found at {data_yaml_path}. "
        f"Run 'python scripts/make_yolo_dataset.py' first!"
    )

    # Prepare Ultralytics arguments
    train_args = prepare_ultralytics_args(raw_config, args)

    git_commit = get_git_commit()
    split_hashes = get_split_hashes(Path("splits/split_metadata.json"))

    print("\n" + "=" * 60)
    print(f"STARTING DETECTOR TRAINING: {args.model}")
    print(f"Git commit:     {git_commit}")
    print(f"Split metadata: {split_hashes}")
    print(f"Train args:     {train_args}")
    print("=" * 60 + "\n")

    # Import ultralytics here so script can show help without importing heavy torch/ultralytics
    from ultralytics import YOLO

    model = YOLO(args.model)

    start_time = datetime.now()
    results = model.train(**train_args)
    end_time = datetime.now()
    duration_sec = (end_time - start_time).total_seconds()

    # Extract metrics
    metrics_summary = {}
    try:
        if hasattr(results, "results_dict"):
            metrics_summary = {k: float(v) for k, v in results.results_dict.items()}
    except Exception as e:
        metrics_summary["error"] = str(e)

    # Log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "run_name": train_args.get("name"),
        "git_commit": git_commit,
        "split_hashes": split_hashes,
        "duration_seconds": duration_sec,
        "train_args": {k: v for k, v in train_args.items() if k not in ["data"]},
        "metrics": metrics_summary,
        "save_dir": str(getattr(results, "save_dir", "")),
    }

    log_experiment_to_jsonl(log_entry, Path(args.project))
    print(f"\n[DONE] Training complete in {duration_sec / 60:.2f} minutes.")
    print(f"Results saved to: {getattr(results, 'save_dir', '')}")
    return results


if __name__ == "__main__":
    train_detector()
