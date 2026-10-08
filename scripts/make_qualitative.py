"""
scripts/make_qualitative.py: Qualitative visualization of vehicle distance estimation and conformal prediction.
(Kế hoạch v4 §6, Decisions D19, D21, D74, D84, D90, D92, D96 - Task T16).

Zero-Touch Guarantee for Split T (Decision D90):
- Reads raw PNG images strictly read-only from data/kitti/image_2/{frame_id}.png.
- Reads prediction numbers strictly from results/final/top_failures_manifest.json and
  results/final/yolo11s_640_T_predictions.parquet.
- Zero model inference, zero refitting, zero raw annotation loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "kitti" / "image_2"
PRED_PATH = PROJECT_ROOT / "results" / "final" / "yolo11s_640_T_predictions.parquet"
MANIFEST_PATH = PROJECT_ROOT / "results" / "final" / "top_failures_manifest.json"
OUT_DIR = PROJECT_ROOT / "results" / "figures"


# Definition of the 8 canonical qualitative cases
QUALITATIVE_CASES = [
    {
        "id": "qualitative_01_side_view_d19",
        "title": "(a) Side View (theta = 8.6 deg, D19)",
        "frame_id": "000006",
        "pred_idx": 0,
        "category": "Viewing Angle Degradation & Compensation",
        "description": "Side aspect dilates bbox width to vehicle length (~4.5m), causing Zw to degenerate to 9.8m. Zh (18.4m) and Residual model (20.6m) accurately recover Z_gt (19.7m), covered by CQR [17.4, 23.6]m.",
        "box_color": (0, 220, 255),  # Yellow/Gold BGR
    },
    {
        "id": "qualitative_02_near_physical_bias_d21",
        "title": "(b) Near Physical 3D Bias (Pattern 111, D21/D92)",
        "frame_id": "000385",
        "pred_idx": 0,
        "category": "Near Distance Physical Offset",
        "description": "Uncut vehicle at 7.9m exhibits systematic negative bias (-11.2%) in Zd (7.0m) due to surface vs center offset. Residual model eliminates bias (7.99m, AbsRel 0.96%), covered by CQR [6.8, 9.0]m.",
        "box_color": (255, 180, 0),  # Cyan/Blue BGR
    },
    {
        "id": "qualitative_03_truncated_edge_d84",
        "title": "(c) Boundary Truncation & Adaptive CQR (D84/D91)",
        "frame_id": "000152",
        "pred_idx": 0,
        "category": "Border Truncation Handling",
        "description": "Vehicle touching bottom edge (truncation 0.35) loses height and ground cues. CQR interval adaptively widens to [4.08, 7.07]m to conservatively contain Z_gt (6.37m) with z_hat (6.14m).",
        "box_color": (0, 165, 255),  # Orange BGR
    },
    {
        "id": "qualitative_04_fallback_pattern000_d74",
        "title": "(d) Successful Fallback Pattern 000 (D74)",
        "frame_id": "000211",
        "pred_idx": 0,
        "category": "Fallback Model (e) Robustness",
        "description": "All 3 geometric cues invalid (cut on multiple borders). Direct model (e) estimates Ze = 7.75m; residual & CQR [7.22, 10.33]m reliably cover Z_gt = 7.91m (AbsRel 4.0%).",
        "box_color": (180, 105, 255), # Purple/Pink BGR
    },
    {
        "id": "qualitative_05_top1_failure",
        "title": "(e) Top-1 Outlier Failure (Rank 1 Manifest)",
        "frame_id": "001414",
        "pred_idx": 0,
        "category": "Failure Analysis & Limitations",
        "description": "Extreme perspective distortion at image corner (x1=1.3, y2=373.3, truncated). Predicted 8.46m vs Z_gt 5.92m (AbsRel 42.92%). Demonstrates edge limitation documented in Section 6.",
        "box_color": (0, 0, 255),    # Red BGR
    },
    {
        "id": "qualitative_06_success_10_20m",
        "title": "(f) Sharp Success: 10-20m Range",
        "frame_id": "003811",
        "pred_idx": 0,
        "category": "High-Precision Estimation",
        "description": "Representative success from frozen manifest (seed=42). Ground truth Z_gt = 11.19m, predicted z_hat = 11.19m (AbsRel 0.04%). Tight CQR interval [9.73, 12.87]m.",
        "box_color": (0, 255, 128),  # Green BGR
    },
    {
        "id": "qualitative_07_success_20_30m",
        "title": "(g) Sharp Success: 20-30m Range",
        "frame_id": "004233",
        "pred_idx": 0,
        "category": "Mid-Range Estimation",
        "description": "Representative success from frozen manifest. Ground truth Z_gt = 19.81m, predicted z_hat = 20.14m (AbsRel 1.69%). CQR interval [17.84, 23.40]m snugly bounds target.",
        "box_color": (0, 255, 128),  # Green BGR
    },
    {
        "id": "qualitative_08_success_hard",
        "title": "(h) Hard-Filtered Object Success",
        "frame_id": "004401",
        "pred_idx": 0,
        "category": "Challenging Occluded Target",
        "description": "Car under KITTI Hard difficulty criteria. Ground truth Z_gt = 20.19m, predicted z_hat = 19.67m (AbsRel 2.59%). CQR interval [17.26, 22.85]m covers object safely.",
        "box_color": (0, 255, 128),  # Green BGR
    },
]


def render_qualitative_image(
    img_bgr: np.ndarray,
    row: pd.Series,
    spec: dict[str, Any],
) -> np.ndarray:
    """
    Renders high-quality bounding box, HUD telemetry badge, and scientific title on an image.
    """
    h_img, w_img = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    x1 = int(round(row["bbox_x1"]))
    y1 = int(round(row["bbox_y1"]))
    x2 = int(round(row["bbox_x2"]))
    y2 = int(round(row["bbox_y2"]))
    box_color = spec["box_color"]

    # 1. Draw 2D Bounding Box with corner accents
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
    corner_len = min(20, (x2 - x1) // 3, (y2 - y1) // 3)
    # Top-left corner
    cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), box_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), box_color, 4, cv2.LINE_AA)
    # Bottom-left corner
    cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), box_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), box_color, 4, cv2.LINE_AA)
    # Top-right corner
    cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), box_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), box_color, 4, cv2.LINE_AA)
    # Bottom-right corner
    cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), box_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), box_color, 4, cv2.LINE_AA)

    # 2. Draw Semi-Transparent Top Banner with Case Title
    overlay = canvas.copy()
    banner_h = 36
    cv2.rectangle(overlay, (0, 0), (w_img, banner_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

    cv2.putText(
        canvas,
        spec["title"],
        (12, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Frame: {spec['frame_id']} | Split T (Zero-Touch)",
        (w_img - 340, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 200, 220),
        1,
        cv2.LINE_AA,
    )

    # 3. Construct Telemetry Badge
    z_hat = float(row["z_hat_f"])
    z_gt = float(row["z_gt"])
    z_lo = float(row["z_lo_cqr"])
    z_hi = float(row["z_hi_cqr"])
    absrel_pct = abs(z_hat - z_gt) / z_gt * 100.0
    covered = (z_gt >= z_lo) and (z_gt <= z_hi)

    cov_str = "[OK COVERED]" if covered else "[MISCOVERED]"
    line1 = f"Pred Z: {z_hat:.2f} m   90% CQR: [{z_lo:.2f}, {z_hi:.2f}] m  {cov_str}"
    line2 = f"True Z: {z_gt:.2f} m   AbsRel: {absrel_pct:.2f}%"

    if row["fallback_flag"]:
        mode_str = "Mode: Fallback Pattern 000 (Model e)"
    else:
        zd_val = float(row["z_d"]) if pd.notna(row["z_d"]) else 0.0
        mode_str = f"Mode: Fused Geometry (Zd={zd_val:.2f} m) -> Residual f"
    line3 = mode_str

    # Position telemetry badge near box (above if space permits, else below or side)
    badge_w = 480
    badge_h = 72
    badge_x = max(10, min(x1, w_img - badge_w - 10))
    if y1 - badge_h - 10 > banner_h:
        badge_y = y1 - badge_h - 8
    elif y2 + badge_h + 10 < h_img:
        badge_y = y2 + 8
    else:
        badge_y = banner_h + 10

    overlay = canvas.copy()
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (15, 18, 24), -1)
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), box_color, 1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    cv2.putText(canvas, line1, (badge_x + 10, badge_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 240, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (badge_x + 10, badge_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (badge_x + 10, badge_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 230), 1, cv2.LINE_AA)

    # 4. Draw Subtitle / Description Footer Banner
    footer_h = 32
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h_img - footer_h), (w_img, h_img), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

    desc_short = spec["description"]
    if len(desc_short) > 130:
        desc_short = desc_short[:127] + "..."
    cv2.putText(
        canvas,
        desc_short,
        (12, h_img - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 225, 240),
        1,
        cv2.LINE_AA,
    )

    return canvas


def build_qualitative_grid(
    rendered_images: list[np.ndarray],
    specs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Assembles the 8 qualitative figures into a pristine 4x2 publication-ready comparison grid.
    """
    n_images = len(rendered_images)
    assert n_images == 8, f"Expected exactly 8 images for 4x2 grid, got {n_images}"

    fig, axes = plt.subplots(4, 2, figsize=(20, 16), dpi=300)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03, hspace=0.10, wspace=0.04)

    fig.suptitle(
        "Qualitative Evaluation & Uncertainty Conformal Prediction on KITTI Split T (Unseen Held-Out Drives)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for idx, (img_bgr, spec) in enumerate(zip(rendered_images, specs)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")
        ax.set_title(f"{spec['title']} — {spec['category']}", fontsize=10, fontweight="semibold", pad=4)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Qualitative Grid Figure: {output_path.name}")


def cv2_imread_unicode(file_path: Path) -> np.ndarray | None:
    """Read image safely from Windows path with non-ASCII characters."""
    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def cv2_imwrite_unicode(file_path: Path, img: np.ndarray) -> bool:
    """Write image safely to Windows path with non-ASCII characters."""
    try:
        ext = file_path.suffix if file_path.suffix else ".png"
        success, enc = cv2.imencode(ext, img)
        if success:
            enc.tofile(str(file_path))
            return True
        return False
    except Exception:
        return False


def main():
    print("=" * 78)
    print("  QUALITATIVE VISUALIZATION GENERATOR (T16) - STARTING")
    print("=" * 78)

    assert PRED_PATH.is_file(), f"Predictions parquet not found at: {PRED_PATH}"
    assert MANIFEST_PATH.is_file(), f"Manifest JSON not found at: {MANIFEST_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_preds = pd.read_parquet(PRED_PATH)
    print(f"Loaded {len(df_preds)} predictions from {PRED_PATH.name}")

    rendered_images: list[np.ndarray] = []
    manifest_records: list[dict[str, Any]] = []

    for spec in QUALITATIVE_CASES:
        fid = spec["frame_id"]
        pidx = spec["pred_idx"]
        img_file = IMAGE_DIR / f"{fid}.png"
        assert img_file.is_file(), f"Missing KITTI image: {img_file}"

        sub = df_preds[(df_preds["frame_id"] == fid) & (df_preds["pred_idx"] == pidx)]
        assert len(sub) == 1, f"Found {len(sub)} rows for frame {fid}, pred_idx {pidx}"
        row = sub.iloc[0]

        img_bgr = cv2_imread_unicode(img_file)
        assert img_bgr is not None, f"Failed to read image {img_file}"

        rendered = render_qualitative_image(img_bgr, row, spec)
        rendered_images.append(rendered)

        out_single_path = OUT_DIR / f"{spec['id']}.png"
        success = cv2_imwrite_unicode(out_single_path, rendered)
        assert success, f"Failed to save image {out_single_path}"
        print(f"[Saved] Individual Figure: {out_single_path.name}")

        manifest_records.append({
            "id": spec["id"],
            "title": spec["title"],
            "category": spec["category"],
            "frame_id": fid,
            "pred_idx": int(pidx),
            "file": str(out_single_path.relative_to(PROJECT_ROOT)),
            "z_gt": round(float(row["z_gt"]), 2),
            "z_hat_f": round(float(row["z_hat_f"]), 2),
            "z_lo_cqr": round(float(row["z_lo_cqr"]), 2),
            "z_hi_cqr": round(float(row["z_hi_cqr"]), 2),
            "absrel": round(float(abs(row["z_hat_f"] - row["z_gt"]) / row["z_gt"]), 4),
            "covered": bool((row["z_gt"] >= row["z_lo_cqr"]) and (row["z_gt"] <= row["z_hi_cqr"])),
            "fallback_flag": bool(row["fallback_flag"]),
            "description": spec["description"],
        })

    # Assemble 4x2 Grid
    grid_path = OUT_DIR / "qualitative_grid_summary.png"
    build_qualitative_grid(rendered_images, QUALITATIVE_CASES, grid_path)

    # Save Manifest JSON (Decision D90, D96)
    meta_json_path = OUT_DIR / "qualitative_manifest.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Qualitative Figures Metadata for KITTI Split T (Task T16)",
            "detector": "yolo11s_640",
            "split": "T",
            "zero_touch_verification": "Strictly read-only images and static predictions parquet (Decision D90)",
            "n_figures": len(manifest_records),
            "grid_file": str(grid_path.relative_to(PROJECT_ROOT)),
            "figures": manifest_records,
        }, f, indent=2)
    print(f"[Saved] Qualitative Manifest JSON: {meta_json_path.name}")
    print("=" * 78)
    print("  QUALITATIVE VISUALIZATION GENERATOR - COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
