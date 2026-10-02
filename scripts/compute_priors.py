"""
scripts/compute_priors.py: Computes geometric priors (W_eff, H_obj, H_cam, y_horizon)
strictly from Split A (train) labels, as required by KE_HOACH_V4 §4.2 and §5.2.
"""

import sys
import json
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_splits, compute_split_hash


def compute_priors(data_root: str = "data/kitti", splits_dir: str = "splits", output_path: str = "configs/geometry_priors.yaml"):
    loader = KITTILoader(data_root)
    splits = load_splits(splits_dir)
    frame_ids = splits["A"]
    split_a_hash = compute_split_hash(frame_ids)

    print(f"Computing geometric priors from Split A ({len(frame_ids)} frames, hash {split_a_hash[:16]}...)...")

    # Accumulators per class
    # Classes: Car, Van, Truck
    classes = ["Car", "Van", "Truck"]
    data = {c: {
        "W_eff": [],       # w * Z / fx
        "H_eff": [],       # h * Z / fy
        "dim_h": [],       # 3D height
        "dim_w": [],       # 3D width
        "dim_l": [],       # 3D length
        "H_cam_contact": [], # (y_bottom - cy) * Z / fy
        "cy_diff": [],     # y_bottom - cy
        "Z": [],
    } for c in classes}

    all_cy = []
    all_H_cam = []

    for fid in frame_ids:
        frame = loader.load_frame(fid)
        calib = frame.calib
        fx, fy = calib.fx, calib.fy
        cx, cy = calib.cx, calib.cy
        all_cy.append(cy)

        for obj in frame.objects:
            c = obj.obj_class
            if c not in classes:
                continue

            # Skip severe truncation / severe occlusion when calculating pristine priors
            # (matches standard practice to avoid partial bbox skewing priors)
            if obj.truncated > 0.30 or obj.occluded > 1:
                continue

            # bbox: [x1, y1, x2, y2]
            w = obj.bbox[2] - obj.bbox[0]
            h = obj.bbox[3] - obj.bbox[1]
            y_bottom = obj.bbox[3]
            Z = obj.depth  # location[2]

            if Z <= 1.0 or w <= 5.0 or h <= 5.0:
                continue

            W_eff = (w * Z) / fx
            H_eff = (h * Z) / fy
            dim_h, dim_w, dim_l = obj.dimensions  # KITTI 3D: height, width, length

            # Ground contact cue: (y_bottom - cy) / fy = H_cam / Z => H_cam = (y_bottom - cy) * Z / fy
            if y_bottom > cy + 2.0:
                h_cam = ((y_bottom - cy) * Z) / fy
                data[c]["H_cam_contact"].append(h_cam)
                all_H_cam.append(h_cam)

            data[c]["W_eff"].append(W_eff)
            data[c]["H_eff"].append(H_eff)
            data[c]["dim_h"].append(dim_h)
            data[c]["dim_w"].append(dim_w)
            data[c]["dim_l"].append(dim_l)
            data[c]["Z"].append(Z)

    priors = {
        "metadata": {
            "source_split": "A",
            "n_frames": len(frame_ids),
            "split_A_hash": split_a_hash,
            "y_horizon_cy_mean": float(np.mean(all_cy)),
            "H_cam_mean": float(np.mean(all_H_cam)),
            "H_cam_median": float(np.median(all_H_cam)),
            "H_cam_std": float(np.std(all_H_cam)),
        },
        "classes": {},
    }

    for c in classes:
        n = len(data[c]["W_eff"])
        if n == 0:
            continue
        priors["classes"][c] = {
            "n_samples": n,
            "W_eff_mean": float(np.mean(data[c]["W_eff"])),
            "W_eff_median": float(np.median(data[c]["W_eff"])),
            "W_eff_std": float(np.std(data[c]["W_eff"])),
            "H_obj_mean": float(np.mean(data[c]["H_eff"])),
            "H_obj_median": float(np.median(data[c]["H_eff"])),
            "H_obj_std": float(np.std(data[c]["H_eff"])),
            "dim_h_mean": float(np.mean(data[c]["dim_h"])),
            "dim_w_mean": float(np.mean(data[c]["dim_w"])),
            "dim_l_mean": float(np.mean(data[c]["dim_l"])),
        }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(priors, f, default_flow_style=False, sort_keys=False)

    print(f"\nPriors computed successfully and saved to {output_file}:")
    print(f"  H_cam median: {priors['metadata']['H_cam_median']:.3f} m (std: {priors['metadata']['H_cam_std']:.3f} m)")
    print(f"  y_horizon cy: {priors['metadata']['y_horizon_cy_mean']:.1f} px")
    for c, stat in priors["classes"].items():
        print(f"  [{c}] n={stat['n_samples']:<5d} | W_eff median={stat['W_eff_median']:.3f}m | H_obj median={stat['H_obj_median']:.3f}m | 3D (H,W,L)=({stat['dim_h_mean']:.2f}, {stat['dim_w_mean']:.2f}, {stat['dim_l_mean']:.2f})m")

    return priors


if __name__ == "__main__":
    compute_priors()
