"""
Capture Golden File on Split C before refactoring (Decision D59).
Records exact baseline predictions and metrics on Split C using frozen models.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline.apply_frozen import apply_frozen_pipeline
from src.residual.models import load_model_f, predict_f
from src.uncertainty.cqr import (
    load_quantile_model,
    predict_interval,
    winkler_score,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]


def capture_golden_c():
    runs_dir = PROJECT_ROOT / "runs" / "residual"
    golden_results = {}

    for model_key in DETECTORS:
        print(f"Capturing golden baseline for {model_key} on Split C...")
        res = apply_frozen_pipeline(model_key, "C", verify_stored_zd=True)

        n_samples = res["n_samples"]
        eval_df = res["eval_df"]
        z_base = res["z_base"]
        z_gt = eval_df["z_gt"].to_numpy(dtype=float)
        drives = eval_df["drive"].to_numpy(dtype=str)
        unique_drives = np.unique(drives)
        feat_mat_f = res["feat_mats"]["f"]

        # 1. Point prediction (f)
        f_path = runs_dir / model_key / "model_f.json"
        model_f = load_model_f(f_path)
        z_hat_f, r_hat_f = predict_f(model_f, feat_mat_f, z_base)
        absrel_f = float(np.mean(np.abs(z_hat_f - z_gt) / z_gt))

        # 2. Conformal calibration from C
        calib_path = runs_dir / model_key / "cqr_calib_C.json"
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_c = json.load(f)
        q_hat = float(calib_c["q_hat_full_c"])

        # 3. Predict interval
        q05_path = runs_dir / model_key / "model_q05.json"
        q95_path = runs_dir / model_key / "model_q95.json"
        model_q05 = load_quantile_model(q05_path)
        model_q95 = load_quantile_model(q95_path)

        X_mat = feat_mat_f.to_numpy(dtype=float)
        q05_pred = model_q05.predict(X_mat)
        q95_pred = model_q95.predict(X_mat)

        z_lo, z_hi, r_lo, r_hi, n_crossings = predict_interval(z_base, q05_pred, q95_pred, q_hat)

        # 4. Metrics
        r_actual = res["r_actual"]
        covered = (z_lo <= z_gt) & (z_gt <= z_hi)
        pooled_cov = float(np.mean(covered))
        macro_cov = float(np.mean([np.mean(covered[drives == d]) for d in unique_drives]))
        mean_width = float(np.mean(z_hi / z_lo))
        mean_winkler = float(np.mean(winkler_score(r_lo, r_hi, r_actual, alpha=0.1)))

        golden_results[model_key] = {
            "n_samples": n_samples,
            "q_hat": q_hat,
            "absrel_f": absrel_f,
            "pooled_coverage": pooled_cov,
            "macro_coverage": macro_cov,
            "mean_width_ratio": mean_width,
            "mean_winkler": mean_winkler,
            "n_crossings": int(n_crossings),
            "z_base_summary": {
                "min": float(np.min(z_base)),
                "mean": float(np.mean(z_base)),
                "max": float(np.max(z_base)),
            },
            "z_hat_f_summary": {
                "min": float(np.min(z_hat_f)),
                "mean": float(np.mean(z_hat_f)),
                "max": float(np.max(z_hat_f)),
            },
            "sample_predictions_first_10": [
                {
                    "idx": i,
                    "drive": drives[i],
                    "z_gt": float(z_gt[i]),
                    "z_base": float(z_base[i]),
                    "z_hat_f": float(z_hat_f[i]),
                    "z_lo": float(z_lo[i]),
                    "z_hi": float(z_hi[i]),
                }
                for i in range(min(10, n_samples))
            ],
        }
        print(f"  {model_key}: n={n_samples}, Cov={pooled_cov*100:.2f}%, Q_hat={q_hat:.5f}, AbsRel={absrel_f:.4f}")

    out_path = PROJECT_ROOT / "runs" / "dryrun_golden_C.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden_results, f, indent=2)

    print(f"\n[Saved] Golden File captured successfully: {out_path}")


if __name__ == "__main__":
    capture_golden_c()
