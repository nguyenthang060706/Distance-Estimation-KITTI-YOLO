"""
scripts/calibrate_conformal_c.py: Full Calibration across 3 Methods on Split C (Decision D56).

Calibrates:
1. Standard CQR: Q_hat
2. Split Conformal: Q_hat
3. Mondrian CQR: 4 distance bins [0, 10), [10, 20), [20, 30), [30, inf) m with fallback bin merging (n < 50)

Serializes results into runs/residual/{model_key}/conformal_calib_C.json before code freeze.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

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
    compute_nonconformity_scores,
    compute_split_conformal_scores,
    conformalize,
    MondrianBinning,
    conformalize_mondrian,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]
ALPHA = 0.1
MONDRIAN_EDGES = [0.0, 10.0, 20.0, 30.0]
MIN_BIN_SAMPLES = 50


def calibrate_split_c_for_detector(model_key: str) -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"Calibrating Conformal Methods on Split C for: {model_key}")
    print(f"==================================================================")

    runs_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    res = apply_frozen_pipeline(model_key, "C", verify_stored_zd=True)

    n_samples = res["n_samples"]
    feat_mat_f = res["feat_mats"]["f"]
    z_base = res["z_base"]
    r_actual = res["r_actual"]
    assert r_actual is not None, "Ground truth residuals required on Split C for calibration!"

    # 1. Point prediction (f)
    f_path = runs_dir / "model_f.json"
    model_f = load_model_f(f_path)
    z_hat_f, r_hat_f = predict_f(model_f, feat_mat_f, z_base)

    # 2. Quantile predictions
    q05_path = runs_dir / "model_q05.json"
    q95_path = runs_dir / "model_q95.json"
    model_q05 = load_quantile_model(q05_path)
    model_q95 = load_quantile_model(q95_path)

    X_mat = feat_mat_f.to_numpy(dtype=float)
    q05_pred = model_q05.predict(X_mat)
    q95_pred = model_q95.predict(X_mat)

    # 3. Method 1: Standard CQR (Primary method per Decision D55, v4 §5.4)
    scores_cqr = compute_nonconformity_scores(q05_pred, q95_pred, r_actual)
    q_hat_cqr = conformalize(scores_cqr, alpha=ALPHA)

    # Update cqr_calib_C.json
    old_calib_path = runs_dir / "cqr_calib_C.json"
    cqr_calib_data = {
        "model_key": model_key,
        "split": "C",
        "alpha": ALPHA,
        "nominal_coverage": 1.0 - ALPHA,
        "q_hat_full_c": float(q_hat_cqr),
        "n_calib": n_samples,
        "updated_posthoc": "D71: updated after fixing base_score in Model e fallback",
    }
    with open(old_calib_path, "w", encoding="utf-8") as f:
        json.dump(cqr_calib_data, f, indent=2)
    print(f"  [Updated] Standard CQR Q_hat = {q_hat_cqr:.6f} saved to cqr_calib_C.json")

    # 4. Method 2: Split Conformal (Descriptive baseline)
    scores_sc = compute_split_conformal_scores(r_actual, r_hat_f)
    q_hat_sc = conformalize(scores_sc, alpha=ALPHA)
    print(f"  [Calibrated] Split Conformal Q_hat = {q_hat_sc:.6f}")

    # 5. Method 3: Mondrian CQR (Descriptive baseline)
    binning = MondrianBinning(calib_z_hat=z_hat_f, base_edges=MONDRIAN_EDGES, min_samples=MIN_BIN_SAMPLES)
    bin_indices = binning.assign_bins(z_hat_f)
    q_hat_mondrian = conformalize_mondrian(scores=scores_cqr, bin_indices=bin_indices, n_bins=binning.n_bins, alpha=ALPHA)
    print(f"  [Calibrated] Mondrian CQR Bins: {binning.labels}")
    for b_idx, b_lbl in enumerate(binning.labels):
        print(f"    Bin {b_idx} ({b_lbl}): Q_hat = {q_hat_mondrian[b_idx]:.6f}")

    calibration_artifact = {
        "model_key": model_key,
        "calibration_split": "C",
        "n_samples": n_samples,
        "alpha": ALPHA,
        "nominal_coverage": 1.0 - ALPHA,
        "standard_cqr": {
            "is_primary": True,
            "q_hat": float(q_hat_cqr),
        },
        "split_conformal": {
            "is_primary": False,
            "q_hat": float(q_hat_sc),
        },
        "mondrian_cqr": {
            "is_primary": False,
            "bin_edges": binning.edges,
            "bin_labels": binning.labels,
            "n_bins": binning.n_bins,
            "min_samples_per_bin": MIN_BIN_SAMPLES,
            "q_hat_per_bin": {int(k): float(v) for k, v in q_hat_mondrian.items()},
        },
    }

    out_file = runs_dir / "conformal_calib_C.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibration_artifact, f, indent=2)
    print(f"  [Saved] Serialized calibration artifact: {out_file.name}")

    return calibration_artifact


def main():
    print("==================================================================")
    print("CALIBRATING ALL CONFORMAL METHODS ON SPLIT C (DECISION D56)")
    print("==================================================================")
    for m in DETECTORS:
        calibrate_split_c_for_detector(m)
    print("\n✓ Full Split C conformal calibration completed successfully for all 3 detectors!")


if __name__ == "__main__":
    main()
