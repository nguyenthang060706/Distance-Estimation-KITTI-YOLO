"""
scripts/refit_full_b.py: Refit final models on Split B using OOF Z_base and serialize full_fw.

Decisions D16b, D29, D35, D43:
- Fixes protocol error where final models (f0) and (f) in runs/residual/{model}/ were previously
  fit on in-sample Z_base rather than out-of-fold Z_base (LODO Z_d and OOF Z_e).
- Reads existing validated OOF predictions from results/datasets/{model}_B_oof.parquet without
  re-running nested LODO cross-validation.
- Fits Model (f0) and Model (f) against r = ln(Z_gt) - ln(Z_base_oof) and derived features.
- Fits Model (e) and full geometric fusion weights (full_fw) on all Split B drives for inference on Split C and T.
- Serializes model_f0.joblib, model_f.json, model_e.json, and full_fw.json into runs/residual/{model}/.
- Records updated SHA-256 hashes and metadata into manifest.json and runs/pipeline_log.jsonl.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval import append_jsonl, make_log_record
from src.geometry.fusion import save_fusion_weights
from src.pipeline.oof import fit_full_b_models
from src.residual.models import (
    save_model_e,
    save_model_f,
    save_model_f0,
)

DETECTORS = ["yolo11s_640", "yolov8s_640", "yolov5su_640"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def refit_for_detector(model_key: str, seed: int = 42, n_jobs: int = 1) -> dict:
    print(f"\n================================================================================")
    print(f"  REFITTING FULL SPLIT B MODELS FOR: {model_key}")
    print(f"================================================================================")

    data_dir = PROJECT_ROOT / "results" / "datasets"
    model_dir = PROJECT_ROOT / "runs" / "residual" / model_key
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing existing manifest at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    best_params_f = manifest_data["best_params_f"]
    best_alpha_f0 = float(manifest_data["best_alpha_f0"])
    split_b_hash = manifest_data.get("split_b_hash", "")
    git_commit = manifest_data.get("git_commit", "")
    git_tag = manifest_data.get("git_tag", "")

    print(f"Loaded existing CV consensus hyperparams:")
    print(f"  (f) XGBoost: {best_params_f}")
    print(f"  (f0) Ridge:  alpha={best_alpha_f0}")

    # Load Parquet artifacts
    feat_path = data_dir / f"{model_key}_B_features.parquet"
    eval_path = data_dir / f"{model_key}_B_eval.parquet"
    cues_path = data_dir / f"{model_key}_B_cues.parquet"
    oof_path = data_dir / f"{model_key}_B_oof.parquet"

    for p in [feat_path, eval_path, cues_path, oof_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    features_df = pd.read_parquet(feat_path)
    eval_df = pd.read_parquet(eval_path)
    cues_df = pd.read_parquet(cues_path)
    oof_df = pd.read_parquet(oof_path)

    n_samples = len(features_df)
    assert len(eval_df) == n_samples, f"eval_df length mismatch ({len(eval_df)} != {n_samples})"
    assert len(cues_df) == n_samples, f"cues_df length mismatch ({len(cues_df)} != {n_samples})"
    assert len(oof_df) == n_samples, f"oof_df length mismatch ({len(oof_df)} != {n_samples})"

    z_base_oof = oof_df["z_base"].to_numpy(dtype=float)
    if np.any(np.isnan(z_base_oof)) or np.any(z_base_oof <= 0):
        raise ValueError("z_base in oof_df must be strictly positive and finite")

    print(f"Verified {n_samples} samples with valid OOF z_base from {oof_path.name}")
    print(f">>> Fitting final models with OOF target r = ln(Z_gt) - ln(Z_base_oof)...")

    model_f0, model_f, model_e, full_fw = fit_full_b_models(
        features_df=features_df,
        eval_df=eval_df,
        cues_df=cues_df,
        best_params_f=best_params_f,
        best_alpha_f0=best_alpha_f0,
        z_base_oof=z_base_oof,
        random_state=seed,
        n_jobs=n_jobs,
    )

    # Save artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    f0_path = model_dir / "model_f0.joblib"
    f_path = model_dir / "model_f.json"
    e_path = model_dir / "model_e.json"
    fw_path = model_dir / "full_fw.json"

    save_model_f0(model_f0, f0_path)
    save_model_f(model_f, f_path)
    save_model_e(model_e, e_path)
    save_fusion_weights(full_fw, fw_path)

    sha_f0 = sha256_file(f0_path)
    sha_f = sha256_file(f_path)
    sha_e = sha256_file(e_path)
    sha_fw = sha256_file(fw_path)

    print(f"[Saved] model_f0: {f0_path.name} (SHA: {sha_f0[:16]}...)")
    print(f"[Saved] model_f:  {f_path.name} (SHA: {sha_f[:16]}...)")
    print(f"[Saved] model_e:  {e_path.name} (SHA: {sha_e[:16]}...)")
    print(f"[Saved] full_fw:  {fw_path.name} (SHA: {sha_fw[:16]}...)")
    print(f"        full_fw weights: {full_fw.weights.round(4).tolist()}")

    manifest_data["updated_at"] = datetime.now().isoformat()
    manifest_data["protocol_fix"] = "D43: final models trained on OOF Z_base (LODO Z_d and OOF Z_e)"
    manifest_data["z_base_source"] = "B_oof.parquet"
    manifest_data["models"] = {
        "model_f0": {"file": "model_f0.joblib", "sha256": sha_f0},
        "model_f": {"file": "model_f.json", "sha256": sha_f},
        "model_e": {"file": "model_e.json", "sha256": sha_e},
        "full_fw": {"file": "full_fw.json", "sha256": sha_fw},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[Updated] Manifest file: {manifest_path}")

    # Record pipeline log
    log_rec = make_log_record(
        split="B",
        split_hash=split_b_hash,
        seed=seed,
        n_boot=0,
        tag="D43-Refit-Full-B",
        extra={
            "detector": model_key,
            "git_commit": git_commit,
            "git_tag": git_tag,
            "sha_f0": sha_f0,
            "sha_f": sha_f,
            "sha_e": sha_e,
            "sha_fw": sha_fw,
            "full_fw_weights": full_fw.weights.tolist(),
        },
    )
    append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_rec)
    print(f"[Logged] Recorded run in runs/pipeline_log.jsonl")

    return {
        "model_key": model_key,
        "sha_f0": sha_f0,
        "sha_f": sha_f,
        "sha_e": sha_e,
        "sha_fw": sha_fw,
    }


def main():
    parser = argparse.ArgumentParser(description="Refit final Split B models on OOF Z_base (D43)")
    parser.add_argument("--detector", choices=["all", *DETECTORS], default="all",
                        help="Detector to refit (default: all per Decision D35)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_jobs", type=int, default=1, help="Thread count for reproducible XGBoost")
    args = parser.parse_args()

    target_detectors = DETECTORS if args.detector == "all" else [args.detector]
    results = []
    for det in target_detectors:
        res = refit_for_detector(det, seed=args.seed, n_jobs=args.n_jobs)
        results.append(res)

    print("\n" + "=" * 78)
    print("  REFIT COMPLETE FOR ALL SPECIFIED DETECTORS (DECISION D43)")
    print("=" * 78)
    for r in results:
        print(f"  {r['model_key']}: full_fw SHA={r['sha_fw'][:16]}..., model_f SHA={r['sha_f'][:16]}...")


if __name__ == "__main__":
    main()
