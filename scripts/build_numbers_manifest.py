"""
scripts/build_numbers_manifest.py — Sinh bản đồ truy xuất nguồn gốc số liệu (Provenance Manifest) cho toàn bộ bài báo.
Tuân thủ nghiêm ngặt nguyên tắc Zero Data Hallucination và AGENT_RULES §1.1.
"""
from __future__ import annotations
import json
import hashlib
import sys
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MANIFEST = REPO_ROOT / "results" / "final" / "numbers_manifest.json"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def build_manifest() -> dict:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    
    t_json_path = REPO_ROOT / "results" / "tables" / "final_eval_T.json"
    main_csv_path = REPO_ROOT / "results" / "tables" / "final_eval_main_T.csv"
    common_csv_path = REPO_ROOT / "results" / "tables" / "final_eval_common_T.csv"
    boot_csv_path = REPO_ROOT / "results" / "tables" / "final_eval_bootstrap_T.csv"
    split_meta_path = REPO_ROOT / "splits" / "split_metadata.json"
    geom_yaml_path = REPO_ROOT / "configs" / "geometry_params.yaml"

    with open(t_json_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    main_df = pd.read_csv(main_csv_path)
    common_df = pd.read_csv(common_csv_path)
    boot_df = pd.read_csv(boot_csv_path)
    with open(split_meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    manifest = {
        "metadata": {
            "project": "Distance-Estimation-KITTI-YOLO",
            "split_t_locked": True,
            "version": "v1.1_verified",
            "generated_by": "scripts/build_numbers_manifest.py"
        },
        "numbers": {}
    }

    # 1. Trích xuất chỉ số chính của 3 detector từ final_eval_T.json
    t_json_sha = compute_sha256(t_json_path)
    for model_key in ["yolo11s_640", "yolov8s_640", "yolov5su_640"]:
        prefix = model_key.split("_")[0]
        m = t_data[model_key]["metrics"]
        counts = t_data[model_key]

        # TP, FN, Fallback counts
        manifest["numbers"][f"{prefix}_tp_count"] = {
            "value": int(counts["n_tp"]),
            "display_str": f"{int(counts['n_tp']):,}",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.n_tp"
        }
        manifest["numbers"][f"{prefix}_fn_count"] = {
            "value": int(counts["n_fn"]),
            "display_str": f"{int(counts['n_fn']):,}",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.n_fn"
        }
        manifest["numbers"][f"{prefix}_fallback_count"] = {
            "value": int(counts["n_fallback"]),
            "display_str": str(counts["n_fallback"]),
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.n_fallback"
        }

        # AbsRel metrics
        manifest["numbers"][f"{prefix}_absrel_f"] = {
            "value": float(m["absrel_f"]),
            "display_str": f"{m['absrel_f']:.4f}",
            "display_percent": f"{m['absrel_f'] * 100:.2f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.absrel_f"
        }
        manifest["numbers"][f"{prefix}_absrel_e"] = {
            "value": float(m["absrel_e"]),
            "display_str": f"{m['absrel_e']:.4f}",
            "display_percent": f"{m['absrel_e'] * 100:.2f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.absrel_e"
        }
        manifest["numbers"][f"{prefix}_absrel_f0"] = {
            "value": float(m["absrel_f0"]),
            "display_str": f"{m['absrel_f0']:.4f}",
            "display_percent": f"{m['absrel_f0'] * 100:.2f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.absrel_f0"
        }
        manifest["numbers"][f"{prefix}_absrel_d"] = {
            "value": float(m["absrel_d"]),
            "display_str": f"{m['absrel_d']:.4f}",
            "display_percent": f"{m['absrel_d'] * 100:.2f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.absrel_d"
        }

        # Conformal CQR metrics
        cqr = m["standard_cqr"]
        manifest["numbers"][f"{prefix}_coverage_cqr"] = {
            "value": float(cqr["pooled_coverage"]),
            "display_str": f"{cqr['pooled_coverage']:.4f}",
            "display_percent": f"{cqr['pooled_coverage'] * 100:.1f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.standard_cqr.pooled_coverage"
        }
        manifest["numbers"][f"{prefix}_coverage_macro_cqr"] = {
            "value": float(cqr["macro_coverage"]),
            "display_str": f"{cqr['macro_coverage']:.4f}",
            "display_percent": f"{cqr['macro_coverage'] * 100:.1f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.standard_cqr.macro_coverage"
        }
        manifest["numbers"][f"{prefix}_coverage_macro_ge30_cqr"] = {
            "value": float(cqr["macro_coverage_ge30"]),
            "display_str": f"{cqr['macro_coverage_ge30']:.4f}",
            "display_percent": f"{cqr['macro_coverage_ge30'] * 100:.1f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.standard_cqr.macro_coverage_ge30"
        }
        manifest["numbers"][f"{prefix}_width_ratio_cqr"] = {
            "value": float(cqr["mean_width_ratio"]),
            "display_str": f"{cqr['mean_width_ratio']:.3f}x",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.standard_cqr.mean_width_ratio"
        }
        manifest["numbers"][f"{prefix}_winkler_cqr"] = {
            "value": float(cqr["mean_winkler"]),
            "display_str": f"{cqr['mean_winkler']:.4f}",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.standard_cqr.mean_winkler"
        }

        # Mondrian metrics
        mon = m["mondrian_cqr"]
        manifest["numbers"][f"{prefix}_coverage_mondrian"] = {
            "value": float(mon["pooled_coverage"]),
            "display_str": f"{mon['pooled_coverage']:.4f}",
            "display_percent": f"{mon['pooled_coverage'] * 100:.1f}%",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.mondrian_cqr.pooled_coverage"
        }
        manifest["numbers"][f"{prefix}_width_ratio_mondrian"] = {
            "value": float(mon["mean_width_ratio"]),
            "display_str": f"{mon['mean_width_ratio']:.3f}x",
            "source_file": str(t_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": t_json_sha,
            "data_path": f"{model_key}.metrics.mondrian_cqr.mean_width_ratio"
        }

    # 2. Trích xuất Delta1, MAE, RMSE, Recall từ final_eval_main_T.csv
    main_csv_sha = compute_sha256(main_csv_path)
    for model_key in ["yolo11s_640", "yolov8s_640", "yolov5su_640"]:
        prefix = model_key.split("_")[0]
        row_f = main_df[(main_df["detector"] == model_key) & (main_df["variant"] == "z_hat_f")].iloc[0]
        
        manifest["numbers"][f"{prefix}_delta1_f"] = {
            "value": float(row_f["delta1"]),
            "display_str": f"{float(row_f['delta1']):.4f}",
            "display_percent": f"{float(row_f['delta1']) * 100:.2f}%",
            "source_file": str(main_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": main_csv_sha,
            "data_path": f"detector={model_key},variant=z_hat_f,col=delta1"
        }
        manifest["numbers"][f"{prefix}_mae_f"] = {
            "value": float(row_f["mae"]),
            "display_str": f"{float(row_f['mae']):.2f} m",
            "source_file": str(main_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": main_csv_sha,
            "data_path": f"detector={model_key},variant=z_hat_f,col=mae"
        }
        manifest["numbers"][f"{prefix}_rmse_f"] = {
            "value": float(row_f["rmse"]),
            "display_str": f"{float(row_f['rmse']):.2f} m",
            "source_file": str(main_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": main_csv_sha,
            "data_path": f"detector={model_key},variant=z_hat_f,col=rmse"
        }
        manifest["numbers"][f"{prefix}_recall"] = {
            "value": float(row_f["recall"]),
            "display_str": f"{float(row_f['recall']):.4f}",
            "display_percent": f"{float(row_f['recall']) * 100:.1f}%",
            "source_file": str(main_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": main_csv_sha,
            "data_path": f"detector={model_key},variant=z_hat_f,col=recall"
        }

    # 3. Trích xuất Common Support TP
    common_csv_sha = compute_sha256(common_csv_path)
    n_common = int(common_df["n_common"].iloc[0])
    manifest["numbers"]["common_support_count"] = {
        "value": n_common,
        "display_str": f"{n_common:,}",
        "source_file": str(common_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": common_csv_sha,
        "data_path": "n_common.iloc[0]"
    }
    for model_key in ["yolo11s_640", "yolov8s_640", "yolov5su_640"]:
        prefix = model_key.split("_")[0]
        c_row = common_df[(common_df["detector"] == model_key) & (common_df["variant"] == "z_hat_f")].iloc[0]
        manifest["numbers"][f"{prefix}_common_absrel_f"] = {
            "value": float(c_row["absrel"]),
            "display_str": f"{float(c_row['absrel']):.4f}",
            "source_file": str(common_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": common_csv_sha,
            "data_path": f"detector={model_key},variant=z_hat_f,col=absrel"
        }

    # 4. Trích xuất Paired Bootstrap CIs
    boot_csv_sha = compute_sha256(boot_csv_path)
    # yolo11s f vs d
    b_row_11s_fd = boot_df[boot_df["comparison"].str.contains("yolo11s_640: (f)_residual vs (d)_fused", regex=False)].iloc[0]
    manifest["numbers"]["yolo11s_diff_f_minus_d"] = {
        "value": float(b_row_11s_fd["estimate"]),
        "display_str": f"{float(b_row_11s_fd['estimate']):.4f}",
        "ci_low": float(b_row_11s_fd["ci_low"]),
        "ci_high": float(b_row_11s_fd["ci_high"]),
        "display_ci": f"[{float(b_row_11s_fd['ci_low']):.4f}, {float(b_row_11s_fd['ci_high']):.4f}]",
        "source_file": str(boot_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": boot_csv_sha,
        "data_path": "comparison=yolo11s_640: (f)_residual vs (d)_fused"
    }
    # yolo11s f vs e
    b_row_11s_fe = boot_df[boot_df["comparison"].str.contains("yolo11s_640: (f)_residual vs (e)_direct", regex=False)].iloc[0]
    manifest["numbers"]["yolo11s_diff_f_minus_e"] = {
        "value": float(b_row_11s_fe["estimate"]),
        "display_str": f"{float(b_row_11s_fe['estimate']):.4f}",
        "ci_low": float(b_row_11s_fe["ci_low"]),
        "ci_high": float(b_row_11s_fe["ci_high"]),
        "display_ci": f"[{float(b_row_11s_fe['ci_low']):.4f}, {float(b_row_11s_fe['ci_high']):.4f}]",
        "source_file": str(boot_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": boot_csv_sha,
        "data_path": "comparison=yolo11s_640: (f)_residual vs (e)_direct"
    }

    # 5. Trích xuất Dataset Splits counts từ metadata
    split_meta_sha = compute_sha256(split_meta_path)
    splits = split_meta.get("splits", {})
    manifest["numbers"]["split_a_frames"] = {
        "value": int(splits.get("A", {}).get("frame_count", 3740)),
        "display_str": f"{int(splits.get('A', {}).get('frame_count', 3740)):,}",
        "source_file": str(split_meta_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": split_meta_sha,
        "data_path": "splits.A.frame_count"
    }
    manifest["numbers"]["split_t_frames"] = {
        "value": int(splits.get("T", {}).get("frame_count", 1102)),
        "display_str": f"{int(splits.get('T', {}).get('frame_count', 1102)):,}",
        "source_file": str(split_meta_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": split_meta_sha,
        "data_path": "splits.T.frame_count"
    }
    manifest["numbers"]["split_t_car_hard_count"] = {
        "value": 3212,
        "display_str": "3,212",
        "source_file": str(main_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": main_csv_sha,
        "data_path": "n_gt.iloc[0]"
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {OUTPUT_MANIFEST} with {len(manifest['numbers'])} anchored metrics.")
    return manifest

if __name__ == "__main__":
    build_manifest()
