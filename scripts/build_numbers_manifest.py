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
        "value": int(splits["A"]["n_frames"]),
        "display_str": f"{int(splits['A']['n_frames']):,}",
        "source_file": str(split_meta_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": split_meta_sha,
        "data_path": "splits.A.n_frames"
    }
    manifest["numbers"]["split_t_frames"] = {
        "value": int(splits["T"]["n_frames"]),
        "display_str": f"{int(splits['T']['n_frames']):,}",
        "source_file": str(split_meta_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": split_meta_sha,
        "data_path": "splits.T.n_frames"
    }
    manifest["numbers"]["split_t_car_hard_count"] = {
        "value": int(main_df["n_gt"].iloc[0]),
        "display_str": f"{int(main_df['n_gt'].iloc[0]):,}",
        "source_file": str(main_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": main_csv_sha,
        "data_path": "n_gt.iloc[0]"
    }

    # 6. Trích xuất Tier 1 Latency Benchmark metrics từ latency_tier1.json (D94, D98, D100, D104)
    latency_json_path = REPO_ROOT / "results" / "tables" / "latency_tier1.json"
    if latency_json_path.exists():
        latency_json_sha = compute_sha256(latency_json_path)
        with open(latency_json_path, "r", encoding="utf-8") as f:
            lat_data = json.load(f)
        detectors_lat = lat_data.get("detectors", {})
        for model_key in ["yolo11s_640", "yolov8s_640", "yolov5su_640"]:
            prefix = model_key.split("_")[0]
            if model_key in detectors_lat:
                lat = detectors_lat[model_key].get("latency", {})
                gpu_tot = lat.get("total_pipeline_gpu", {})
                cpu_tot = lat.get("total_pipeline_cpu", {})
                manifest["numbers"][f"{prefix}_latency_gpu_median"] = {
                    "value": float(gpu_tot.get("median_ms", 0.0)),
                    "display_str": f"{float(gpu_tot.get('median_ms', 0.0)):.2f} ms",
                    "source_file": str(latency_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": latency_json_sha,
                    "data_path": f"detectors.{model_key}.latency.total_pipeline_gpu.median_ms"
                }
                manifest["numbers"][f"{prefix}_latency_gpu_p95"] = {
                    "value": float(gpu_tot.get("p95_ms", 0.0)),
                    "display_str": f"{float(gpu_tot.get('p95_ms', 0.0)):.2f} ms",
                    "source_file": str(latency_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": latency_json_sha,
                    "data_path": f"detectors.{model_key}.latency.total_pipeline_gpu.p95_ms"
                }
                manifest["numbers"][f"{prefix}_latency_gpu_fps"] = {
                    "value": float(gpu_tot.get("fps_median", 1000.0 / gpu_tot.get("median_ms", 1.0))),
                    "display_str": f"{float(gpu_tot.get('fps_median', 1000.0 / gpu_tot.get('median_ms', 1.0))):.1f} FPS",
                    "source_file": str(latency_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": latency_json_sha,
                    "data_path": f"detectors.{model_key}.latency.total_pipeline_gpu.fps_median"
                }
                manifest["numbers"][f"{prefix}_latency_cpu_median"] = {
                    "value": float(cpu_tot.get("median_ms", 0.0)),
                    "display_str": f"{float(cpu_tot.get('median_ms', 0.0)):.2f} ms",
                    "source_file": str(latency_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": latency_json_sha,
                    "data_path": f"detectors.{model_key}.latency.total_pipeline_cpu.median_ms"
                }
                manifest["numbers"][f"{prefix}_latency_cpu_p95"] = {
                    "value": float(cpu_tot.get("p95_ms", 0.0)),
                    "display_str": f"{float(cpu_tot.get('p95_ms', 0.0)):.2f} ms",
                    "source_file": str(latency_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": latency_json_sha,
                    "data_path": f"detectors.{model_key}.latency.total_pipeline_cpu.p95_ms"
                }
                manifest["numbers"][f"{prefix}_latency_cpu_fps"] = {
                    "value": float(cpu_tot.get("fps_median", 1000.0 / cpu_tot.get("median_ms", 1.0))),
                    "display_str": f"{float(cpu_tot.get('fps_median', 1000.0 / cpu_tot.get('median_ms', 1.0))):.1f} FPS",
                    "source_file": str(latency_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": latency_json_sha,
                    "data_path": f"detectors.{model_key}.latency.total_pipeline_cpu.fps_median"
                }

    # 7. Single cue & fused baseline metrics on Split B OOF
    geom_b_path = REPO_ROOT / "results" / "tables" / "geometry_on_detector_bbox.json"
    res_b_path = REPO_ROOT / "results" / "tables" / "residual_oof_b.json"
    if geom_b_path.exists() and res_b_path.exists():
        geom_b_sha = compute_sha256(geom_b_path)
        res_b_sha = compute_sha256(res_b_path)
        with open(geom_b_path, "r", encoding="utf-8") as f:
            geom_b = json.load(f)
        with open(res_b_path, "r", encoding="utf-8") as f:
            res_b = json.load(f)
            
        det_b = geom_b["detectors"]["yolo11s_640"]["eval_split_B"]
        cs_b = res_b["yolo11s_640"]["common_support"]

        manifest["numbers"]["yolo11s_b_absrel_zw"] = {
            "value": float(det_b["a_zw_det"]["pooled_absrel"]),
            "display_str": f"{float(det_b['a_zw_det']['pooled_absrel']):.4f}",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.a_zw_det.pooled_absrel"
        }
        manifest["numbers"]["yolo11s_b_mae_zw"] = {
            "value": float(det_b["a_zw_det"]["pooled_mae"]),
            "display_str": f"{float(det_b['a_zw_det']['pooled_mae']):.2f} m",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.a_zw_det.pooled_mae"
        }
        manifest["numbers"]["yolo11s_b_delta1_zw"] = {
            "value": float(det_b["a_zw_det"]["pooled_delta1"]),
            "display_str": f"{float(det_b['a_zw_det']['pooled_delta1']) * 100:.1f}%",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.a_zw_det.pooled_delta1"
        }

        manifest["numbers"]["yolo11s_b_absrel_zg"] = {
            "value": float(det_b["c_zg_det"]["pooled_absrel"]),
            "display_str": f"{float(det_b['c_zg_det']['pooled_absrel']):.4f}",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.c_zg_det.pooled_absrel"
        }
        manifest["numbers"]["yolo11s_b_mae_zg"] = {
            "value": float(det_b["c_zg_det"]["pooled_mae"]),
            "display_str": f"{float(det_b['c_zg_det']['pooled_mae']):.2f} m",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.c_zg_det.pooled_mae"
        }
        manifest["numbers"]["yolo11s_b_delta1_zg"] = {
            "value": float(det_b["c_zg_det"]["pooled_delta1"]),
            "display_str": f"{float(det_b['c_zg_det']['pooled_delta1']) * 100:.1f}%",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.c_zg_det.pooled_delta1"
        }

        manifest["numbers"]["yolo11s_b_absrel_zh"] = {
            "value": float(det_b["b_zh_det"]["pooled_absrel"]),
            "display_str": f"{float(det_b['b_zh_det']['pooled_absrel']):.4f}",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.b_zh_det.pooled_absrel"
        }
        manifest["numbers"]["yolo11s_b_mae_zh"] = {
            "value": float(det_b["b_zh_det"]["pooled_mae"]),
            "display_str": f"{float(det_b['b_zh_det']['pooled_mae']):.2f} m",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.b_zh_det.pooled_mae"
        }
        manifest["numbers"]["yolo11s_b_delta1_zh"] = {
            "value": float(det_b["b_zh_det"]["pooled_delta1"]),
            "display_str": f"{float(det_b['b_zh_det']['pooled_delta1']) * 100:.1f}%",
            "source_file": str(geom_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": geom_b_sha,
            "data_path": "detectors.yolo11s_640.eval_split_B.b_zh_det.pooled_delta1"
        }

        manifest["numbers"]["yolo11s_b_absrel_zd"] = {
            "value": float(cs_b["d"]["pooled"]["absrel"]),
            "display_str": f"{float(cs_b['d']['pooled']['absrel']):.4f}",
            "source_file": str(res_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": res_b_sha,
            "data_path": "yolo11s_640.common_support.d.pooled.absrel"
        }
        manifest["numbers"]["yolo11s_b_mae_zd"] = {
            "value": float(cs_b["d"]["pooled"]["mae"]),
            "display_str": f"{float(cs_b['d']['pooled']['mae']):.2f} m",
            "source_file": str(res_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": res_b_sha,
            "data_path": "yolo11s_640.common_support.d.pooled.mae"
        }

    # 8. Ablation study metrics from ablation_oof_b.json
    abl_b_path = REPO_ROOT / "results" / "tables" / "ablation_oof_b.json"
    if abl_b_path.exists():
        abl_b_sha = compute_sha256(abl_b_path)
        with open(abl_b_path, "r", encoding="utf-8") as f:
            abl_b = json.load(f)
        abl_dict = {r["ablation_key"]: r for r in abl_b["records"] if r["model_key"] == "yolo11s_640"}
        
        if "drop_z_h" in abl_dict:
            r_zh = abl_dict["drop_z_h"]
            manifest["numbers"]["yolo11s_abl_drop_zh_delta_pooled"] = {
                "value": float(r_zh["delta_pooled"]),
                "display_str": f"{float(r_zh['delta_pooled']):.4f}",
                "ci_low": float(r_zh["ci_lo"]),
                "ci_high": float(r_zh["ci_hi"]),
                "display_ci": f"[{float(r_zh['ci_lo']):.4f}, {float(r_zh['ci_hi']):.4f}]",
                "source_file": str(abl_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_sha256": abl_b_sha,
                "data_path": "yolo11s_640.drop_z_h"
            }
        if "drop_group_bbox_geometry" in abl_dict:
            r_bbox = abl_dict["drop_group_bbox_geometry"]
            manifest["numbers"]["yolo11s_abl_drop_bbox_delta_macro"] = {
                "value": float(r_bbox["delta_macro"]),
                "display_str": f"{float(r_bbox['delta_macro']):.4f}",
                "ci_low": float(r_bbox["ci_lo"]),
                "ci_high": float(r_bbox["ci_hi"]),
                "display_ci": f"[{float(r_bbox['ci_lo']):.4f}, {float(r_bbox['ci_hi']):.4f}]",
                "source_file": str(abl_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_sha256": abl_b_sha,
                "data_path": "yolo11s_640.drop_group_bbox_geometry"
            }
        if "drop_group_validity_flags" in abl_dict:
            r_val = abl_dict["drop_group_validity_flags"]
            manifest["numbers"]["yolo11s_abl_drop_validity_delta_pooled"] = {
                "value": float(r_val["delta_pooled"]),
                "display_str": f"{float(r_val['delta_pooled']):.4f}",
                "ci_low": float(r_val["ci_lo"]),
                "ci_high": float(r_val["ci_hi"]),
                "display_ci": f"[{float(r_val['ci_lo']):.4f}, {float(r_val['ci_hi']):.4f}]",
                "source_file": str(abl_b_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_sha256": abl_b_sha,
                "data_path": "yolo11s_640.drop_group_validity_flags"
            }

    # 9. 20-Resplit Conformal Stability metrics from coverage_stability_20resplits.json
    resplit_path = REPO_ROOT / "results" / "tables" / "coverage_stability_20resplits.json"
    if resplit_path.exists():
        resplit_sha = compute_sha256(resplit_path)
        with open(resplit_path, "r", encoding="utf-8") as f:
            resplit_data = json.load(f)
        resplit_dict = {item["model_key"]: item for item in resplit_data}
        
        for model_key in ["yolo11s_640", "yolov8s_640", "yolov5su_640"]:
            prefix = model_key.split("_")[0]
            if model_key in resplit_dict:
                cqr_sum = resplit_dict[model_key]["summary"]["cqr"]
                m_cov = float(cqr_sum["mean_pooled_coverage"])
                s_cov = float(cqr_sum["std_pooled_coverage"])
                min_cov = float(cqr_sum["min_pooled_coverage"])
                max_cov = float(cqr_sum["max_pooled_coverage"])
                
                manifest["numbers"][f"{prefix}_resplit20_mean_cov"] = {
                    "value": m_cov,
                    "display_str": f"{m_cov * 100:.2f}%",
                    "source_file": str(resplit_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": resplit_sha,
                    "data_path": f"{model_key}.summary.cqr.mean_pooled_coverage"
                }
                manifest["numbers"][f"{prefix}_resplit20_std_cov"] = {
                    "value": s_cov,
                    "display_str": f"{s_cov * 100:.2f}%",
                    "source_file": str(resplit_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": resplit_sha,
                    "data_path": f"{model_key}.summary.cqr.std_pooled_coverage"
                }
                manifest["numbers"][f"{prefix}_resplit20_min_cov"] = {
                    "value": min_cov,
                    "display_str": f"{min_cov * 100:.2f}%",
                    "source_file": str(resplit_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": resplit_sha,
                    "data_path": f"{model_key}.summary.cqr.min_pooled_coverage"
                }
                manifest["numbers"][f"{prefix}_resplit20_max_cov"] = {
                    "value": max_cov,
                    "display_str": f"{max_cov * 100:.2f}%",
                    "source_file": str(resplit_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": resplit_sha,
                    "data_path": f"{model_key}.summary.cqr.max_pooled_coverage"
                }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {OUTPUT_MANIFEST} with {len(manifest['numbers'])} anchored metrics.")
    return manifest

if __name__ == "__main__":
    build_manifest()
