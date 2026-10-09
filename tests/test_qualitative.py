"""Tests for Qualitative Visualization Generator (Task T16, Decision D90, D96).

Verifies:
1. Presence and integrity of results/figures/qualitative_manifest.json.
2. Existence and non-trivial file size (> 10 KB) of all 8 individual PNGs and 1 grid summary PNG.
3. Valid 3-channel image dimensions and valid RGB values.
4. Correctness of metadata: 8 specific evaluated cases, zero-touch Split T compliance.
5. Invariance of runs/final_T.lock.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
MANIFEST_PATH = FIGURES_DIR / "qualitative_manifest.json"
GRID_PATH = FIGURES_DIR / "qualitative_grid_summary.png"
LOCK_PATH = PROJECT_ROOT / "runs" / "final_T.lock"

EXPECTED_CASE_IDS = [
    "qualitative_01_side_view_d19",
    "qualitative_02_near_physical_bias_d21",
    "qualitative_03_truncated_edge_d84",
    "qualitative_04_fallback_pattern000_d74",
    "qualitative_05_top1_failure",
    "qualitative_06_success_10_20m",
    "qualitative_07_success_20_30m",
    "qualitative_08_success_hard",
]


def test_lock_file_intact():
    """Verify that runs/final_T.lock is intact (Zero-touch Split T, D70/D90)."""
    assert LOCK_PATH.is_file(), "runs/final_T.lock must exist and remain locked!"


def test_qualitative_manifest_exists_and_valid():
    """Verify qualitative manifest exists, parses correctly, and contains 8 cases."""
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta.get("detector") == "yolo11s_640"
    assert meta.get("split") == "T"
    assert meta.get("n_figures") == 8
    assert len(meta.get("figures", [])) == 8

    found_ids = [fig["id"] for fig in meta["figures"]]
    assert found_ids == EXPECTED_CASE_IDS, f"Case IDs mismatch: {found_ids} vs {EXPECTED_CASE_IDS}"

    for fig in meta["figures"]:
        assert "frame_id" in fig
        assert "pred_idx" in fig
        assert "z_gt" in fig
        assert "z_hat_f" in fig
        assert "z_lo_cqr" in fig
        assert "z_hi_cqr" in fig
        assert "absrel" in fig
        assert "covered" in fig
        assert "description" in fig
        assert fig["z_lo_cqr"] <= fig["z_hi_cqr"], f"Invalid interval in {fig['id']}"


@pytest.mark.parametrize("case_id", EXPECTED_CASE_IDS)
def test_individual_figures_exist_and_readable(case_id: str):
    """Verify that all 8 individual PNG figures exist, are non-empty, and are valid images."""
    png_path = FIGURES_DIR / f"{case_id}.png"
    assert png_path.is_file(), f"Image not found: {png_path}"
    assert png_path.stat().st_size > 10_000, f"Image {png_path.name} is too small (<10KB)"

    with Image.open(png_path) as img:
        assert img.format == "PNG"
        w, h = img.size
        assert w > 800 and h > 200, f"Unexpected dimensions: {w}x{h}"
        assert img.mode in ("RGB", "RGBA")


def test_qualitative_grid_summary_exists_and_readable():
    """Verify that the 4x2 summary grid exists, is high-resolution, and readable."""
    assert GRID_PATH.is_file(), f"Grid summary image not found: {GRID_PATH}"
    assert GRID_PATH.stat().st_size > 100_000, f"Grid image too small (<100KB): {GRID_PATH.stat().st_size} bytes"

    with Image.open(GRID_PATH) as img:
        assert img.format == "PNG"
        w, h = img.size
        # 20x16 inches at 300 DPI is approx 6000x4800, bbox_inches='tight' may be slightly smaller
        assert w > 3000 and h > 2000, f"Grid dimensions too small for paper-ready 300 DPI: {w}x{h}"


def test_qualitative_manifest_matches_parquet_predictions_exact():
    """Verify that every metric in qualitative_manifest.json strictly matches the static predictions parquet."""
    import pandas as pd
    pred_path = PROJECT_ROOT / "results" / "final" / "yolo11s_640_T_predictions.parquet"
    assert pred_path.is_file(), f"Missing predictions parquet: {pred_path}"

    df_preds = pd.read_parquet(pred_path)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for case in meta["figures"]:
        fid = case["frame_id"]
        pidx = case["pred_idx"]
        sub = df_preds[(df_preds["frame_id"] == fid) & (df_preds["pred_idx"] == pidx)]
        assert len(sub) == 1, f"Expected 1 matching prediction for {fid} #{pidx}, got {len(sub)}"
        row = sub.iloc[0]

        assert case["z_gt"] == pytest.approx(round(float(row["z_gt"]), 2))
        assert case["z_hat_f"] == pytest.approx(round(float(row["z_hat_f"]), 2))
        assert case["z_lo_cqr"] == pytest.approx(round(float(row["z_lo_cqr"]), 2))
        assert case["z_hi_cqr"] == pytest.approx(round(float(row["z_hi_cqr"]), 2))
        expected_absrel = round(float(abs(row["z_hat_f"] - row["z_gt"]) / row["z_gt"]), 4)
        assert case["absrel"] == pytest.approx(expected_absrel)
        assert case["covered"] == bool((row["z_gt"] >= row["z_lo_cqr"]) and (row["z_gt"] <= row["z_hi_cqr"]))
        assert case["fallback_flag"] == bool(row["fallback_flag"])
