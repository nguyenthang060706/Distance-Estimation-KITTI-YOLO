"""
scripts/make_yolo_dataset.py: Converts KITTI Split A (train) and Split V (val) to YOLO format.

Rules & Decisions:
- 3 classes: Car (0), Van (1), Truck (2).
- Non-vehicle objects (Pedestrian, Cyclist, DontCare, Misc, etc.) excluded.
- NO Hard filter applied (includes all vehicles for training).
- Bounding boxes normalized by actual individual image dimensions.
- Copy images safely on Windows.
- Asserts split hashes against splits/split_metadata.json and writes SPLIT_HASH.json.
"""

import sys
import os
import shutil
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, ".")
from src.utils.kitti_loader import parse_label, VEHICLE_CLASSES
from src.utils.split_builder import load_splits, compute_split_hash

CLASS_MAPPING = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KITTI splits A & V to YOLO dataset format.")
    parser.add_argument("--data-root", type=str, default="data/kitti", help="Path to KITTI root")
    parser.add_argument("--splits-dir", type=str, default="splits", help="Path to splits directory")
    parser.add_argument("--output-dir", type=str, default="data/yolo_kitti", help="Path to output YOLO dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset")
    return parser.parse_args()


def convert_split(
    split_name: str,
    target_subset: str,
    frame_ids: list,
    data_root: Path,
    output_dir: Path,
):
    """
    Convert a split (e.g. A or V) to YOLO directory subset (train or val).
    """
    img_src_dir = data_root / "image_2"
    lbl_src_dir = data_root / "label_2"

    img_dst_dir = output_dir / "images" / target_subset
    lbl_dst_dir = output_dir / "labels" / target_subset

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {c: 0 for c in CLASS_MAPPING.keys()}
    total_boxes = 0

    print(f"\nProcessing Split {split_name} -> {target_subset} ({len(frame_ids)} frames)...")

    for fid in tqdm(frame_ids, desc=f"Converting {split_name}"):
        src_img_path = img_src_dir / f"{fid}.png"
        dst_img_path = img_dst_dir / f"{fid}.png"
        src_lbl_path = lbl_src_dir / f"{fid}.txt"
        dst_lbl_path = lbl_dst_dir / f"{fid}.txt"

        if not src_img_path.exists():
            raise FileNotFoundError(f"Missing image: {src_img_path}")
        if not src_lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {src_lbl_path}")

        # Get actual image dimensions
        with Image.open(src_img_path) as img:
            img_w, img_h = img.size

        # Copy image if not exists or size differs
        if not dst_img_path.exists() or dst_img_path.stat().st_size != src_img_path.stat().st_size:
            shutil.copy2(src_img_path, dst_img_path)

        # Parse labels (vehicle_only=True extracts Car, Van, Truck; no Hard filter)
        objects = parse_label(str(src_lbl_path), vehicle_only=True)

        yolo_lines = []
        for obj in objects:
            if obj.obj_class not in CLASS_MAPPING:
                continue

            cls_id = CLASS_MAPPING[obj.obj_class]
            x1, y1, x2, y2 = obj.bbox

            # Clamp coordinates to image boundaries
            x1 = max(0.0, min(float(img_w), float(x1)))
            y1 = max(0.0, min(float(img_h), float(y1)))
            x2 = max(0.0, min(float(img_w), float(x2)))
            y2 = max(0.0, min(float(img_h), float(y2)))

            w = x2 - x1
            h = y2 - y1

            # Ignore invalid degenerate boxes
            if w <= 1.0 or h <= 1.0:
                continue

            xc = x1 + w / 2.0
            yc = y1 + h / 2.0

            # Normalize to [0, 1]
            xc_norm = xc / img_w
            yc_norm = yc / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_lines.append(f"{cls_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            class_counts[obj.obj_class] += 1
            total_boxes += 1

        # Write label file (empty file if 0 objects, standard in YOLO)
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

    return {
        "n_frames": len(frame_ids),
        "total_boxes": total_boxes,
        "class_counts": class_counts,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    print("=== Creating YOLO Dataset from KITTI Splits A & V ===")
    print(f"Data root:   {data_root}")
    print(f"Splits dir:  {splits_dir}")
    print(f"Output dir:  {output_dir}")

    # 1. Assert split hashes against split_metadata.json
    meta_path = splits_dir / "split_metadata.json"
    assert meta_path.exists(), f"split_metadata.json not found in {splits_dir}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = load_splits(str(splits_dir))
    for s_name in ["A", "V"]:
        calc_hash = compute_split_hash(splits[s_name])
        expected_hash = meta["splits"][s_name]["hash"]
        assert calc_hash == expected_hash, (
            f"Split {s_name} hash mismatch! Calculated: {calc_hash}, Expected: {expected_hash}"
        )
        print(f"Split {s_name} hash verified: {calc_hash[:16]}... (matches split_metadata.json)")

    # 2. Convert Split A -> train, Split V -> val
    stats_a = convert_split("A", "train", splits["A"], data_root, output_dir)
    stats_v = convert_split("V", "val", splits["V"], data_root, output_dir)

    # 3. Create dataset.yaml
    dataset_yaml_path = output_dir / "dataset.yaml"
    # Using forward slashes for cross-platform compatibility
    yaml_content = f"""# Ultralytics YOLO Dataset Configuration for KITTI Vehicles
path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: Car
  1: Van
  2: Truck
"""
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\nCreated {dataset_yaml_path}")

    # 4. Create SPLIT_HASH.json
    split_hash_info = {
        "A": {
            "n_frames": len(splits["A"]),
            "hash": meta["splits"]["A"]["hash"],
            "stats": stats_a,
        },
        "V": {
            "n_frames": len(splits["V"]),
            "hash": meta["splits"]["V"]["hash"],
            "stats": stats_v,
        },
        "dataset_yaml": str(dataset_yaml_path.resolve()),
    }
    hash_file_path = output_dir / "SPLIT_HASH.json"
    with open(hash_file_path, "w", encoding="utf-8") as f:
        json.dump(split_hash_info, f, indent=2)
    print(f"Created {hash_file_path}")

    print("\n=== Dataset Summary ===")
    print(f"Train (A): {stats_a['n_frames']} frames, {stats_a['total_boxes']} vehicle boxes {stats_a['class_counts']}")
    print(f"Val   (V): {stats_v['n_frames']} frames, {stats_v['total_boxes']} vehicle boxes {stats_v['class_counts']}")
    print("\nDataset preparation complete! Ready for YOLO training.")


if __name__ == "__main__":
    main()
