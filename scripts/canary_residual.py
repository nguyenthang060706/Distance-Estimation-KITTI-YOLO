"""
scripts/canary_residual.py: Standalone Canary Test on Real Data (Decision D25 / AGENT_RULES §3.2).

Permutes training residual targets across drives in outer LODO on real Split B detections.
Verifies that Model (f) trained on scrambled data fails to beat baseline Z_d / Z_base on out-of-fold evaluations.
Ensures zero spurious lookahead or structural data leakage in the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import depth_metrics, macro_by_cluster
from src.pipeline.oof import run_nested_lodo_b


def main():
    parser = argparse.ArgumentParser(description="Canary test on real Split B data.")
    parser.add_argument("--model", type=str, default="yolo11s_640", help="Detector model key")
    args = parser.parse_args()

    model_key = args.model
    print("=" * 70)
    print(f"CANARY TEST: Scrambled Training Target on Real Data ({model_key})")
    print("=" * 70)

    feat_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_features.parquet"
    eval_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_eval.parquet"
    cues_path = PROJECT_ROOT / "results" / "datasets" / f"{model_key}_B_cues.parquet"

    for p in [feat_path, eval_path, cues_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset: {p}")

    feats = pd.read_parquet(feat_path)
    evals = pd.read_parquet(eval_path)
    cues = pd.read_parquet(cues_path)

    print(f"Loaded {len(feats)} samples across {cues['drive'].nunique()} drives.")
    print("Running nested LODO with scramble_train_targets=True...")

    oof_df, _ = run_nested_lodo_b(
        features_df=feats,
        eval_df=evals,
        cues_df=cues,
        min_train_drives=3,
        random_state=42,
        n_jobs=1,
        verbose=True,
        scramble_train_targets=True,
    )

    # Evaluate on common support (where d is valid)
    common_mask = ~oof_df["fallback_flag"].to_numpy(dtype=bool)
    eval_common = evals.loc[common_mask].copy()
    oof_common = oof_df.loc[common_mask].copy()

    z_gt = eval_common["z_gt"].to_numpy(dtype=float)
    z_d = oof_common["z_d"].to_numpy(dtype=float)
    z_hat_f = oof_common["z_hat_f"].to_numpy(dtype=float)

    metrics_d = depth_metrics(z_gt, z_d)
    metrics_f = depth_metrics(z_gt, z_hat_f)

    macro_d = macro_by_cluster(eval_common.assign(z_pred=z_d), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive")
    macro_f = macro_by_cluster(eval_common.assign(z_pred=z_hat_f), metric="absrel", gt_col="z_gt", pred_col="z_pred", cluster_col="drive")

    print("\n" + "-" * 70)
    print("CANARY EVALUATION RESULTS (Common Support, N = %d):" % len(eval_common))
    print(f"Baseline (d)  - Pooled AbsRel: {metrics_d['absrel']:.5f} | Macro AbsRel: {macro_d:.5f}")
    print(f"Scrambled (f) - Pooled AbsRel: {metrics_f['absrel']:.5f} | Macro AbsRel: {macro_f:.5f}")
    print("-" * 70)

    # Verification: On scrambled training target, (f) must NOT beat (d)
    passed_pooled = (metrics_f["absrel"] >= metrics_d["absrel"] - 1e-4)
    passed_macro = (macro_f >= macro_d - 1e-4)

    if passed_pooled and passed_macro:
        print("[CANARY PASSED] Scrambled residual model did NOT beat baseline (d).")
        print("Conclusion: No label leakage or artificial shortcuts detected in pipeline.")
    else:
        print("[CANARY FAILED] Scrambled residual model unexpectedly beat baseline (d)!")
        sys.exit(1)


if __name__ == "__main__":
    main()
