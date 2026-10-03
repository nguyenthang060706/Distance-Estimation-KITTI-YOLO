"""
src/detection/matching.py: Bounding box matching for detector evaluation and depth estimation.

Implements Decision D8 and Decision D15:
- Match statuses:
    - TP: True Positive (matches an unmatched GT Hard box with IoU >= iou_threshold).
    - FP: False Positive (unmatched detection, or duplicate detection on already-matched GT).
    - IGNORED_NONHARD: Detection matches a GT car that does not pass the Hard filter.
    - IGNORED_DONTCARE: Detection overlaps a KITTI DontCare region.
- Matching priority per detection (sorted by confidence descending):
    1. Unmatched GT Hard with highest IoU >= iou_threshold -> TP.
    2. Already-matched GT Hard with IoU >= iou_threshold -> FP (duplicate detection).
    3. GT non-Hard with IoU >= iou_threshold -> IGNORED_NONHARD.
    4. DontCare region with overlap >= dontcare_threshold -> IGNORED_DONTCARE.
    5. None of the above -> FP.
- Configurable DontCare overlap criterion:
    - 'iou': standard intersection-over-union (reproduces Decision D6 / find_conf_thresholds.py).
    - 'area_pred': intersection / area(prediction) (standard KITTI official benchmark devkit).
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Sequence, Optional, Union
import numpy as np


class MatchStatus(str, Enum):
    """Detection evaluation status."""
    TP = "TP"
    FP = "FP"
    IGNORED_NONHARD = "IGNORED_NONHARD"
    IGNORED_DONTCARE = "IGNORED_DONTCARE"


@dataclass
class MatchedDetection:
    """Result of matching a single predicted bounding box."""
    pred_idx: int                       # 0-indexed index in original predictions
    bbox: np.ndarray                    # (4,) [x1, y1, x2, y2]
    score: float                        # confidence score
    status: MatchStatus                 # TP, FP, IGNORED_NONHARD, IGNORED_DONTCARE
    matched_gt_idx: Optional[int] = None  # Index in gt_hard_boxes if TP, else None
    matched_iou: float = 0.0            # IoU with matched GT Hard box (or highest overlap)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two 2D boxes [x1, y1, x2, y2].
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0

    areaA = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    areaB = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def box_intersection_over_pred_area(pred_box: Sequence[float], target_box: Sequence[float]) -> float:
    """
    Compute intersection(pred_box, target_box) / area(pred_box).
    Used in KITTI official benchmark for DontCare evaluation.
    """
    xA = max(pred_box[0], target_box[0])
    yA = max(pred_box[1], target_box[1])
    xB = min(pred_box[2], target_box[2])
    yB = min(pred_box[3], target_box[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0

    pred_area = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
    return inter / pred_area if pred_area > 0 else 0.0


def match_detections_frame(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_hard_boxes: np.ndarray,
    gt_non_hard_boxes: Optional[np.ndarray] = None,
    dontcare_boxes: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
    dontcare_mode: str = "iou",
    dontcare_threshold: float = 0.5,
) -> list[MatchedDetection]:
    """
    Match predictions for a single frame against ground truth according to KITTI protocol.

    Args:
        pred_boxes: (N, 4) array of [x1, y1, x2, y2]
        pred_scores: (N,) array of confidence scores
        gt_hard_boxes: (M_hard, 4) array of GT Car boxes passing Hard filter
        gt_non_hard_boxes: (M_nh, 4) array of GT Car boxes failing Hard filter (optional)
        dontcare_boxes: (M_dc, 4) array of DontCare boxes (optional)
        iou_threshold: IoU threshold for matching GT (default 0.5 per v4 §5.1 and Decision D6)
        dontcare_mode: 'iou' or 'area_pred' (default 'iou')
        dontcare_threshold: threshold for DontCare overlap (default 0.5)

    Returns:
        List of MatchedDetection objects, in the original order of pred_boxes.
    """
    n_preds = len(pred_boxes)
    if n_preds == 0:
        return []

    if gt_non_hard_boxes is None:
        gt_non_hard_boxes = np.empty((0, 4))
    if dontcare_boxes is None:
        dontcare_boxes = np.empty((0, 4))

    # Sort prediction indices by confidence descending
    sorted_order = np.argsort(-pred_scores)

    matched_gt_hard = set()
    results: dict[int, MatchedDetection] = {}

    for orig_idx in sorted_order:
        p_box = pred_boxes[orig_idx]
        p_score = float(pred_scores[orig_idx])

        # Step 1: Match with unmatched GT Hard (highest IoU >= iou_threshold)
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_hard_boxes):
            if g_idx in matched_gt_hard:
                continue
            iou = box_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold:
            matched_gt_hard.add(best_gt_idx)
            results[orig_idx] = MatchedDetection(
                pred_idx=orig_idx,
                bbox=p_box,
                score=p_score,
                status=MatchStatus.TP,
                matched_gt_idx=best_gt_idx,
                matched_iou=best_iou,
            )
            continue

        # Step 2: Check if this prediction matches an ALREADY matched GT Hard box
        # (duplicate detection on the same ground truth -> False Positive)
        duplicate_matched = False
        for g_idx in matched_gt_hard:
            if box_iou(p_box, gt_hard_boxes[g_idx]) >= iou_threshold:
                duplicate_matched = True
                break

        if duplicate_matched:
            results[orig_idx] = MatchedDetection(
                pred_idx=orig_idx,
                bbox=p_box,
                score=p_score,
                status=MatchStatus.FP,
                matched_gt_idx=None,
                matched_iou=0.0,
            )
            continue

        # Step 3: Check match with GT non-Hard cars (Ignored: neither TP nor FP)
        matches_non_hard = False
        for nh_box in gt_non_hard_boxes:
            if box_iou(p_box, nh_box) >= iou_threshold:
                matches_non_hard = True
                break

        if matches_non_hard:
            results[orig_idx] = MatchedDetection(
                pred_idx=orig_idx,
                bbox=p_box,
                score=p_score,
                status=MatchStatus.IGNORED_NONHARD,
                matched_gt_idx=None,
                matched_iou=0.0,
            )
            continue

        # Step 4: Check match with DontCare (Ignored: neither TP nor FP)
        matches_dontcare = False
        for dc_box in dontcare_boxes:
            if dontcare_mode == "area_pred":
                overlap = box_intersection_over_pred_area(p_box, dc_box)
            else:
                overlap = box_iou(p_box, dc_box)

            if overlap >= dontcare_threshold:
                matches_dontcare = True
                break

        if matches_dontcare:
            results[orig_idx] = MatchedDetection(
                pred_idx=orig_idx,
                bbox=p_box,
                score=p_score,
                status=MatchStatus.IGNORED_DONTCARE,
                matched_gt_idx=None,
                matched_iou=0.0,
            )
            continue

        # Step 5: Unmatched detection -> False Positive
        results[orig_idx] = MatchedDetection(
            pred_idx=orig_idx,
            bbox=p_box,
            score=p_score,
            status=MatchStatus.FP,
            matched_gt_idx=None,
            matched_iou=0.0,
        )

    # Return in original order of pred_boxes
    return [results[i] for i in range(n_preds)]
