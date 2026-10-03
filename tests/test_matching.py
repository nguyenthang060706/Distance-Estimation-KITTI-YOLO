"""
tests/test_matching.py: Unit tests for src/detection/matching.py (Decision D8, D15).
"""

import numpy as np
import pytest
from src.detection.matching import (
    MatchStatus,
    MatchedDetection,
    box_iou,
    box_intersection_over_pred_area,
    match_detections_frame,
)


def test_box_iou_computation():
    box1 = [0, 0, 10, 10]
    box2 = [5, 0, 15, 10]
    # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50 / 150 = 1/3
    assert abs(box_iou(box1, box2) - 1.0 / 3.0) < 1e-6

    # Disjoint boxes
    box3 = [20, 20, 30, 30]
    assert box_iou(box1, box3) == 0.0


def test_box_intersection_over_pred_area():
    pred_box = [0, 0, 10, 10]      # area = 100
    target_box = [5, 0, 25, 10]    # target box larger, intersection = 5 * 10 = 50
    # inter / area(pred) = 50 / 100 = 0.5
    assert abs(box_intersection_over_pred_area(pred_box, target_box) - 0.5) < 1e-6


def test_priority_and_single_gt_two_detections():
    """
    Mandatory test:
    If 1 GT is matched by 2 detections, the higher-confidence detection is TP,
    and the second detection MUST be FP (not ignored, not TP).
    """
    gt_hard = np.array([[10, 10, 50, 50]])  # 1 GT Hard box
    pred_boxes = np.array([
        [10, 10, 50, 50],   # pred 0: perfect match, conf 0.9
        [12, 12, 48, 48],   # pred 1: also high IoU, conf 0.8
    ])
    pred_scores = np.array([0.9, 0.8])

    matches = match_detections_frame(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        gt_hard_boxes=gt_hard,
        iou_threshold=0.7,
    )

    assert len(matches) == 2
    assert matches[0].status == MatchStatus.TP
    assert matches[0].matched_gt_idx == 0
    assert matches[1].status == MatchStatus.FP
    assert matches[1].matched_gt_idx is None


def test_gt_non_hard_ignored():
    """
    Mandatory test:
    Matching a non-Hard GT vehicle must result in IGNORED_NONHARD.
    """
    gt_hard = np.array([[100, 100, 150, 150]])
    gt_non_hard = np.array([[10, 10, 50, 50]])  # Non-hard GT
    pred_boxes = np.array([
        [11, 11, 49, 49],   # matches non-hard
    ])
    pred_scores = np.array([0.85])

    matches = match_detections_frame(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        gt_hard_boxes=gt_hard,
        gt_non_hard_boxes=gt_non_hard,
        iou_threshold=0.7,
    )

    assert len(matches) == 1
    assert matches[0].status == MatchStatus.IGNORED_NONHARD


def test_dontcare_ignored_both_modes():
    """
    Mandatory test:
    Detection matching DontCare must result in IGNORED_DONTCARE in both iou and area_pred modes.
    """
    gt_hard = np.empty((0, 4))
    dontcares = np.array([[10, 10, 100, 100]])  # large DontCare region
    # Small detection completely inside DontCare:
    # area(pred) = 20 * 20 = 400. intersection = 400.
    # area_pred = 400 / 400 = 1.0 >= 0.5.
    # IoU = 400 / (8100 + 400 - 400) = 400 / 8100 ≈ 0.049 (< 0.5 for IoU mode!)
    small_pred = np.array([[20, 20, 40, 40]])
    scores = np.array([0.8])

    # Mode area_pred: should ignore!
    matches_area = match_detections_frame(
        pred_boxes=small_pred,
        pred_scores=scores,
        gt_hard_boxes=gt_hard,
        dontcare_boxes=dontcares,
        dontcare_mode="area_pred",
        dontcare_threshold=0.5,
    )
    assert matches_area[0].status == MatchStatus.IGNORED_DONTCARE

    # High IoU pred with DontCare in iou mode:
    high_iou_pred = np.array([[12, 12, 98, 98]])
    matches_iou = match_detections_frame(
        pred_boxes=high_iou_pred,
        pred_scores=scores,
        gt_hard_boxes=gt_hard,
        dontcare_boxes=dontcares,
        dontcare_mode="iou",
        dontcare_threshold=0.5,
    )
    assert matches_iou[0].status == MatchStatus.IGNORED_DONTCARE


def test_priority_ordering():
    """
    Verify full priority hierarchy:
    GT Hard > GT non-Hard > DontCare > FP.
    """
    # Box overlapping both a non-hard GT and a DontCare
    box = np.array([10, 10, 50, 50])
    gt_nh = np.array([[10, 10, 50, 50]])
    dc = np.array([[10, 10, 50, 50]])

    matches = match_detections_frame(
        pred_boxes=np.array([box]),
        pred_scores=np.array([0.75]),
        gt_hard_boxes=np.empty((0, 4)),
        gt_non_hard_boxes=gt_nh,
        dontcare_boxes=dc,
        iou_threshold=0.7,
    )
    # Non-hard takes precedence over DontCare
    assert matches[0].status == MatchStatus.IGNORED_NONHARD

    # If completely unmatched
    unmatched_box = np.array([300, 300, 350, 350])
    matches_unmatched = match_detections_frame(
        pred_boxes=np.array([unmatched_box]),
        pred_scores=np.array([0.75]),
        gt_hard_boxes=np.empty((0, 4)),
        gt_non_hard_boxes=gt_nh,
        dontcare_boxes=dc,
        iou_threshold=0.7,
    )
    assert matches_unmatched[0].status == MatchStatus.FP
