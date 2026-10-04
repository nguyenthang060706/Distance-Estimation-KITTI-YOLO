"""
scripts/build_datasets.py: Build ranging datasets from predictions (Task T01, §5 Data Contract).

Usage:
    python scripts/build_datasets.py --model all --split B C --population pass_thr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

# Ensure utf-8 stdout on Windows (AGENT_RULES.md §4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.build_dataset import (
    build_population,
    compute_file_sha256,
    compute_population_summary,
    load_artifacts,
)
from src.evaluation.eval import append_jsonl, make_log_record
from src.utils.split_builder import compute_split_hash, load_split

# Canonical model map
MODEL_MAP = {
    "yolov8s": "yolov8s_640",
    "yolov8s_640": "yolov8s_640",
    "yolo11s": "yolo11s_640",
    "yolo11s_640": "yolo11s_640",
    "yolov5su": "yolov5su_640",
    "yolov5su_640": "yolov5su_640",
}

ALL_MODELS = ["yolov8s_640", "yolo11s_640", "yolov5su_640"]

# Ground-truth TP counts from detector_eval_b_c.json (Task T01 spec)
EXPECTED_TP_PASS_THR = {
    "B": {
        "yolov8s_640": 3427,
        "yolo11s_640": 3523,
        "yolov5su_640": 3496,
    },
    "C": {
        "yolov8s_640": 1445,
        "yolo11s_640": 1489,
        "yolov5su_640": 1430,
    },
}

EXPECTED_GT_COUNTS = {
    "B": 4776,
    "C": 1826,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ranging dataset (features, eval, fn) from prediction artifacts."
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help="Model name(s) or 'all' for all 3 detectors.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["B", "C"],
        help="Split name(s), e.g. B C. Split T is forbidden.",
    )
    parser.add_argument(
        "--population",
        choices=["pass_thr", "floor"],
        default="pass_thr",
        help="Ranging population selection (default: pass_thr).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/datasets",
        help="Directory to save generated datasets (default: results/datasets).",
    )
    parser.add_argument(
        "--predictions-dir",
        default="results/predictions",
        help="Directory containing predictions parquet files (default: results/predictions).",
    )
    parser.add_argument(
        "--splits-dir",
        default="splits",
        help="Directory containing split frame lists (default: splits).",
    )
    return parser.parse_args()


def get_split_hash(splits_dir: str | Path, split: str) -> str:
    """Retrieve or compute split hash."""
    meta_path = Path(splits_dir) / "split_metadata.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "splits" in meta and split in meta["splits"]:
            return meta["splits"][split]["hash"]

    # Fallback to computing from split file
    frame_ids = load_split(str(splits_dir), split, allow_test=False)
    return compute_split_hash(frame_ids)


def main() -> None:
    args = parse_args()

    # Resolve models
    if "all" in args.model:
        models = ALL_MODELS
    else:
        models = []
        for m in args.model:
            if m not in MODEL_MAP:
                raise ValueError(f"Unknown model: '{m}'. Allowed: {list(MODEL_MAP.keys())} or 'all'")
            resolved = MODEL_MAP[m]
            if resolved not in models:
                models.append(resolved)

    splits = [s.upper() for s in args.split]
    for s in splits:
        if s == "T":
            raise PermissionError(
                "Access to Split T is strictly forbidden during development (Decision D4, AGENT_RULES.md §1.1)!"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = Path(args.predictions_dir)

    print("=" * 80)
    print("BUILDING RANGING DATASETS (Task T01)")
    print(f"Models:      {models}")
    print(f"Splits:      {splits}")
    print(f"Population:  {args.population}")
    print(f"Output Dir:  {out_dir}")
    print("=" * 80)

    for split in splits:
        split_hash = get_split_hash(args.splits_dir, split)
        print(f"\n--- Split {split} (hash: {split_hash[:12]}...) ---")

        for model_key in models:
            print(f"\nProcessing [{model_key}] on Split [{split}]...")

            # 1. Source file SHAs
            det_file = preds_dir / f"{model_key}_{split}_detections.parquet"
            match_file = preds_dir / f"{model_key}_{split}_matches.parquet"
            gt_file = preds_dir / f"{model_key}_{split}_gt.parquet"

            sha_dict = {
                "detections": compute_file_sha256(det_file),
                "matches": compute_file_sha256(match_file),
                "gt": compute_file_sha256(gt_file),
            }

            # 2. Build population
            features_df, eval_df, fn_df = build_population(
                model_key=model_key,
                split=split,
                population=args.population,
                predictions_dir=preds_dir,
            )

            # 3. Load raw artifacts for summary counts
            raw_dets, raw_matches, raw_gt = load_artifacts(
                model_key=model_key, split=split, predictions_dir=preds_dir
            )
            summary = compute_population_summary(
                df_dets=raw_dets,
                df_matches=raw_matches,
                df_gt=raw_gt,
                model_key=model_key,
                split=split,
                population=args.population,
                split_hash=split_hash,
                source_files_sha=sha_dict,
            )

            # 4. Strict assertions against detector_eval_b_c.json (Task T01 spec)
            tp_count = len(features_df)
            fn_count = len(fn_df)
            gt_count = len(raw_gt)

            print(f"  n_gt: {gt_count}, n_TP: {tp_count}, n_FN: {fn_count}, n_FP: {summary['n_FP']}, n_ignored: {summary['n_ignored']}")

            if split in EXPECTED_GT_COUNTS:
                expected_gt = EXPECTED_GT_COUNTS[split]
                assert gt_count == expected_gt, (
                    f"[{model_key} Split {split}] Total GT mismatch: expected {expected_gt}, got {gt_count}"
                )

            if args.population == "pass_thr" and split in EXPECTED_TP_PASS_THR:
                expected_tp = EXPECTED_TP_PASS_THR[split][model_key]
                if tp_count != expected_tp:
                    raise AssertionError(
                        f"[{model_key} Split {split}] TP count mismatch!\n"
                        f"  Expected from detector_eval_b_c.json: {expected_tp}\n"
                        f"  Actual TP in built dataset:           {tp_count}"
                    )
                assert fn_count == gt_count - expected_tp, (
                    f"[{model_key} Split {split}] FN count mismatch: expected {gt_count - expected_tp}, got {fn_count}"
                )
                print(f"  ✓ Verified: TP matches detector_eval_b_c.json exactly ({tp_count})")

            # 5. Save parquet files and summary json
            feat_out = out_dir / f"{model_key}_{split}_features.parquet"
            eval_out = out_dir / f"{model_key}_{split}_eval.parquet"
            fn_out = out_dir / f"{model_key}_{split}_fn.parquet"
            summary_out = out_dir / f"{model_key}_{split}_summary.json"

            features_df.to_parquet(feat_out, index=False)
            eval_df.to_parquet(eval_out, index=False)
            fn_df.to_parquet(fn_out, index=False)

            with open(summary_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            print(f"  Saved features: {feat_out} ({len(features_df)} rows)")
            print(f"  Saved eval:     {eval_out} ({len(eval_df)} rows)")
            print(f"  Saved fn:       {fn_out} ({len(fn_df)} rows)")
            print(f"  Saved summary:  {summary_out}")

            # 6. Log reproducible record to runs/pipeline_log.jsonl (AGENT_RULES.md §5)
            log_record = make_log_record(
                split=split,
                split_hash=split_hash,
                seed=42,
                n_boot=0,
                tag="build_dataset_T01",
                extra={
                    "model_key": model_key,
                    "population": args.population,
                    "n_gt": gt_count,
                    "n_TP": tp_count,
                    "n_FN": fn_count,
                    "n_FP": summary["n_FP"],
                    "n_ignored": summary["n_ignored"],
                    "output_features": str(feat_out),
                    "output_eval": str(eval_out),
                    "output_fn": str(fn_out),
                    "source_sha": sha_dict,
                },
            )
            append_jsonl(PROJECT_ROOT / "runs" / "pipeline_log.jsonl", log_record)

    print("\n" + "=" * 80)
    print("✓ All ranging datasets built, verified, and logged successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
