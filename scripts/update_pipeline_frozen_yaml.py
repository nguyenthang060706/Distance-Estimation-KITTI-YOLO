import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import yaml

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

root = Path(__file__).resolve().parent.parent

# Splits
splits = {}
for s, fname in [("A", "splits/A.txt"), ("B", "splits/B.txt"), ("C", "splits/C.txt"), ("T", "splits/T.txt")]:
    p = root / fname
    splits[s] = {
        "path": fname,
        "sha256": sha256_file(p),
        "lines": len(p.read_text(encoding="utf-8").strip().splitlines()),
    }
meta_p = root / "splits/split_metadata.json"
splits["metadata"] = {
    "path": "splits/split_metadata.json",
    "sha256": sha256_file(meta_p),
}

# Detector thresholds
with open(root / "configs/detector/conf_thresholds.yaml", "r", encoding="utf-8") as f:
    conf_cfg = yaml.safe_load(f)

det_keys = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
model_names_map = {"yolo11s_640": "yolo11s", "yolov8s_640": "yolov8s", "yolov5su_640": "yolov5su"}

detectors_data = {}
for m in det_keys:
    short_m = model_names_map[m]
    conf_thr = conf_cfg["detectors"][short_m]["conf_threshold"]

    ckpt_path = root / f"runs/detector/{m}/weights/last.pt"
    ckpt_sha = sha256_file(ckpt_path) if ckpt_path.exists() else None

    res_dir = root / "runs/residual" / m
    with open(res_dir / "manifest.json", "r", encoding="utf-8") as f:
        mani = json.load(f)

    calib_p = res_dir / "conformal_calib_C.json"

    models_dict = {}
    for k, v in mani.get("models", {}).items():
        sub_p = res_dir / v["file"]
        if sub_p.exists():
            models_dict[k] = {
                "path": f"runs/residual/{m}/{v['file']}",
                "sha256": sha256_file(sub_p),
            }

    detectors_data[m] = {
        "conf_threshold": float(conf_thr),
        "checkpoint": {
            "path": f"runs/detector/{m}/weights/last.pt",
            "sha256": ckpt_sha,
        },
        "best_params_f": mani.get("best_params_f"),
        "best_alpha_f0": mani.get("best_alpha_f0"),
        "models": models_dict,
        "conformal_calib_C": {
            "path": f"runs/residual/{m}/conformal_calib_C.json",
            "sha256": sha256_file(calib_p),
        },
    }

# Key configs
configs_shas = {}
for cpath in [
    "configs/geometry_params.yaml",
    "configs/detector/checkpoints.yaml",
    "configs/detector/conf_thresholds.yaml",
    "configs/residual/residual_prereg_v1.yaml",
    "configs/residual/coverage_prereg_v1.yaml",
]:
    p = root / cpath
    configs_shas[cpath] = sha256_file(p)

pipeline_frozen = {
    "metadata": {
        "name": "KITTI Distance Estimation Frozen Pipeline",
        "schema_version": "1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_tag": "final-config-v1.1",
        "decision_references": ["D25", "D26", "D27", "D49", "D56", "D59", "D60", "D61", "D63", "D64", "D65", "D71", "D72"],
    },
    "splits": splits,
    "detectors": detectors_data,
    "pipeline_parameters": {
        "dontcare_mode": "iou",
        "dontcare_threshold": 0.5,
        "iou_nms": 0.7,
        "iou_match": 0.5,
        "conf_min": 0.05,
        "border_eps": 2.0,
        "alpha": 0.1,
        "seed": 42,
        "seed_quantile": 42,
    },
    "config_files_sha256": configs_shas,
}

out_yaml = root / "configs/pipeline_frozen_v1.yaml"
with open(out_yaml, "w", encoding="utf-8") as f:
    yaml.dump(pipeline_frozen, f, sort_keys=False, indent=2, allow_unicode=True)

print("Generated pipeline_frozen_v1.yaml (v1.1) successfully.")
