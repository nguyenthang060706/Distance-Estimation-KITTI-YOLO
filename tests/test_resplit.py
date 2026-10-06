"""
tests/test_resplit.py: Unit tests for B∪C drive-level resplit generator and validator (Decision D26).
"""

from __future__ import annotations

import pytest
from src.uncertainty.resplit import (
    DRIVES_22,
    get_bc_car_hard_counts,
    generate_resplit,
    validate_resplit,
)


def test_get_bc_car_hard_counts():
    """Verify ground-truth Car Hard counts across all 22 drives in B∪C."""
    counts = get_bc_car_hard_counts()
    assert len(counts) == 22
    assert sorted(counts.keys()) == sorted(DRIVES_22)
    assert all(c > 0 for c in counts.values())
    total_cars = sum(counts.values())
    assert total_cars == 6602  # 4776 (B) + 1826 (C)


def test_generate_resplit_disjoint():
    """Verify partitions are mutually disjoint and cover all 22 drives."""
    res = generate_resplit(seed=42)
    fit_d = set(res["fit_drives"])
    calib_d = set(res["calib_drives"])
    eval_d = set(res["eval_drives"])

    assert len(fit_d & calib_d) == 0
    assert len(fit_d & eval_d) == 0
    assert len(calib_d & eval_d) == 0
    assert len(fit_d | calib_d | eval_d) == 22


def test_generate_resplit_constraints():
    """Verify minimum drive count (>= 4) and max top1 share (<= 0.50)."""
    res = generate_resplit(seed=0)
    assert res["n_drives"]["fit"] >= 4
    assert res["n_drives"]["calib"] >= 4
    assert res["n_drives"]["eval"] >= 4

    assert res["top1_shares"]["fit"] <= 0.50
    assert res["top1_shares"]["calib"] <= 0.50
    assert res["top1_shares"]["eval"] <= 0.50

    # Fractions approximately 50% / 25% / 25%
    assert 0.35 <= res["part_shares"]["fit"] <= 0.65
    assert 0.15 <= res["part_shares"]["calib"] <= 0.35
    assert 0.15 <= res["part_shares"]["eval"] <= 0.35

    assert validate_resplit(res) is True


def test_generate_resplit_determinism():
    """Verify that same seed produces identical partition."""
    res1 = generate_resplit(seed=123)
    res2 = generate_resplit(seed=123)
    assert res1["fit_drives"] == res2["fit_drives"]
    assert res1["calib_drives"] == res2["calib_drives"]
    assert res1["eval_drives"] == res2["eval_drives"]
    assert res1["n_redraws"] == res2["n_redraws"]


def test_all_20_pre_registered_seeds():
    """Verify all 20 pre-registered seeds (0–19) produce valid partitions."""
    counts = get_bc_car_hard_counts()
    for seed in range(20):
        res = generate_resplit(seed=seed, car_counts=counts)
        assert validate_resplit(res) is True, f"Seed {seed} failed validation"
        assert res["n_redraws"] >= 0
        assert res["n_drives"]["fit"] >= 4
        assert res["n_drives"]["calib"] >= 4
        assert res["n_drives"]["eval"] >= 4
        assert res["top1_shares"]["fit"] <= 0.50
        assert res["top1_shares"]["calib"] <= 0.50
        assert res["top1_shares"]["eval"] <= 0.50


def test_validate_resplit_catches_invalid():
    """Verify validator catches overlapping drives or excessive top1 share."""
    valid_res = generate_resplit(seed=0)

    # Overlapping drive
    invalid_overlap = dict(valid_res)
    invalid_overlap["fit_drives"] = valid_res["fit_drives"] + [valid_res["calib_drives"][0]]
    assert validate_resplit(invalid_overlap) is False

    # Too few drives
    invalid_too_few = dict(valid_res)
    invalid_too_few["fit_drives"] = valid_res["fit_drives"][:3]
    assert validate_resplit(invalid_too_few) is False

    # Top1 share exceeded
    invalid_top1 = dict(valid_res)
    invalid_top1["top1_shares"] = {"fit": 0.55, "calib": 0.25, "eval": 0.25}
    assert validate_resplit(invalid_top1) is False
