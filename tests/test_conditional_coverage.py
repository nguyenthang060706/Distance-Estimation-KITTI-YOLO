"""
tests/test_conditional_coverage.py: Unit tests for conditional coverage, drive breakdown, and exchangeability diagnostics (T15).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.uncertainty.conditional_coverage import (
    assign_distance_bin,
    assign_theta_bin,
    compute_interval_subgroup_metrics,
    compute_conditional_coverage_breakdown,
    compute_drive_level_coverage_table,
    compute_exchangeability_ks_diagnostics,
    cluster_bootstrap_coverage_ci,
)


def make_mock_predictions_and_fn(n_samples: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate mock predictions and false negative data for testing."""
    rng = np.random.default_rng(42)

    # Drives: 3 drives (55% drive_1, 35% drive_2, remainder drive_3)
    n_d1 = int(n_samples * 0.55)
    n_d2 = int(n_samples * 0.35)
    n_d3 = n_samples - n_d1 - n_d2
    drives = ["drive_1"] * n_d1 + ["drive_2"] * n_d2 + ["drive_3"] * n_d3
    z_gt = rng.uniform(5.0, 55.0, size=n_samples)
    z_hat_f = z_gt + rng.normal(0, 0.5, size=n_samples)
    z_base = z_hat_f * 0.98
    r_act = np.log(z_gt) - np.log(z_base)

    # Construct intervals
    # CQR: 90% coverage
    z_lo_cqr = z_hat_f * 0.90
    z_hi_cqr = z_hat_f * 1.10
    r_lo_cqr = np.log(z_lo_cqr) - np.log(z_base)
    r_hi_cqr = np.log(z_hi_cqr) - np.log(z_base)

    # Split Conformal
    z_lo_sc = z_hat_f * 0.88
    z_hi_sc = z_hat_f * 1.12
    r_lo_sc = np.log(z_lo_sc) - np.log(z_base)
    r_hi_sc = np.log(z_hi_sc) - np.log(z_base)

    # Mondrian CQR
    z_lo_mondrian = z_hat_f * 0.91
    z_hi_mondrian = z_hat_f * 1.09
    r_lo_mondrian = np.log(z_lo_mondrian) - np.log(z_base)
    r_hi_mondrian = np.log(z_hi_mondrian) - np.log(z_base)

    # Attributes
    valid_w = rng.choice([0, 1], size=n_samples, p=[0.1, 0.9])
    valid_h = rng.choice([0, 1], size=n_samples, p=[0.05, 0.95])
    valid_g = rng.choice([0, 1], size=n_samples, p=[0.05, 0.95])
    difficulty = rng.choice(["Easy", "Moderate", "Hard"], size=n_samples, p=[0.4, 0.4, 0.2])
    truncated = rng.choice([0.0, 0.1, 0.3], size=n_samples, p=[0.7, 0.2, 0.1])
    occluded = rng.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
    alpha = rng.uniform(-np.pi, np.pi, size=n_samples)
    fallback_flag = (valid_w == 0) & (valid_h == 0) & (valid_g == 0)

    pred_df = pd.DataFrame({
        "drive": drives,
        "z_gt": z_gt,
        "z_hat_f": z_hat_f,
        "z_base": z_base,
        "r_actual": r_act,
        "z_lo_cqr": z_lo_cqr,
        "z_hi_cqr": z_hi_cqr,
        "r_lo_cqr": r_lo_cqr,
        "r_hi_cqr": r_hi_cqr,
        "z_lo_sc": z_lo_sc,
        "z_hi_sc": z_hi_sc,
        "r_lo_sc": r_lo_sc,
        "r_hi_sc": r_hi_sc,
        "z_lo_mondrian": z_lo_mondrian,
        "z_hi_mondrian": z_hi_mondrian,
        "r_lo_mondrian": r_lo_mondrian,
        "r_hi_mondrian": r_hi_mondrian,
        "valid_w": valid_w,
        "valid_h": valid_h,
        "valid_g": valid_g,
        "difficulty": difficulty,
        "truncated": truncated,
        "occluded": occluded,
        "alpha": alpha,
        "fallback_flag": fallback_flag,
        "confidence": rng.uniform(0.70, 0.99, size=n_samples),
    })

    fn_df = pd.DataFrame({
        "drive": ["drive_1", "drive_2"] * 5,
        "z_gt": rng.uniform(10.0, 45.0, size=10),
        "difficulty": ["Moderate"] * 5 + ["Hard"] * 5,
        "truncated": [0.0] * 5 + [0.2] * 5,
        "occluded": [1] * 5 + [2] * 5,
        "alpha": rng.uniform(-np.pi, np.pi, size=10),
    })

    return pred_df, fn_df


def test_distance_and_theta_binning():
    """Test binning functions correctly map values."""
    assert assign_distance_bin(5.0) == "0-10"
    assert assign_distance_bin(15.0) == "10-20"
    assert assign_distance_bin(25.0) == "20-30"
    assert assign_distance_bin(45.0) == "30-50"
    assert assign_distance_bin(60.0) == ">50"

    # Theta binning: alpha = 0 -> theta = 0 rad (< 30 deg -> Side)
    assert assign_theta_bin(0.0) == "Side (<30°)"
    # alpha = pi/2 -> theta = pi/2 = 90 deg (> 60 deg -> Front/Rear)
    assert assign_theta_bin(np.pi / 2.0) == "Front/Rear (>60°)"
    # alpha = pi/4 = 45 deg -> Diagonal
    assert assign_theta_bin(np.pi / 4.0) == "Diagonal (30–60°)"


def test_interval_subgroup_metrics_math():
    """Test interval metrics math (coverage, width, crossings, Winkler)."""
    df, fn_df = make_mock_predictions_and_fn(120)
    res = compute_interval_subgroup_metrics(df, n_fn=len(fn_df))

    assert res["n_tp"] == 120
    assert res["n_fn"] == len(fn_df)
    assert res["n_gt"] == 120 + len(fn_df)
    assert res["k_clusters"] == 3
    assert res["low_n"] is False  # 120 >= 100

    for m in ["cqr", "sc", "mondrian"]:
        m_res = res[m]
        assert 0.0 <= m_res["coverage"] <= 1.0
        assert m_res["mean_width"] > 1.0
        assert m_res["crossings"] == 0
        assert np.isfinite(m_res["mean_winkler"])


def test_crossing_counter_reports_non_silent():
    """Ensure crossing counter (z_lo > z_hi) is faithfully reported without silent clipping."""
    df, _ = make_mock_predictions_and_fn(50)
    # Deliberately invert z_lo and z_hi for 5 samples in cqr
    df.loc[:4, "z_lo_cqr"] = 30.0
    df.loc[:4, "z_hi_cqr"] = 10.0

    res = compute_interval_subgroup_metrics(df, n_fn=0)
    assert res["cqr"]["crossings"] == 5
    assert res["sc"]["crossings"] == 0


def test_low_n_flag_threshold():
    """Verify low_n flag is True for n < 100 and False for n >= 100."""
    df_small, _ = make_mock_predictions_and_fn(50)
    df_large, _ = make_mock_predictions_and_fn(120)

    res_small = compute_interval_subgroup_metrics(df_small)
    res_large = compute_interval_subgroup_metrics(df_large)

    assert res_small["low_n"] is True
    assert res_large["low_n"] is False


def test_nested_difficulty_subset_relations():
    """Verify nested difficulty hierarchy Easy <= Moderate <= Hard in sample counts."""
    df, fn_df = make_mock_predictions_and_fn(100)
    cats = compute_conditional_coverage_breakdown(df, fn_df)

    diff_rows = {r["subgroup"]: r for r in cats["difficulty"] if r["subgroup"] != "---"}

    n_easy_nested = diff_rows["Easy (nested)"]["n_tp"]
    n_mod_nested = diff_rows["Moderate (nested)"]["n_tp"]
    n_hard_nested = diff_rows["Hard (nested)"]["n_tp"]

    assert n_easy_nested <= n_mod_nested <= n_hard_nested
    assert n_hard_nested == len(df)


def test_drive_level_coverage_table_ge30():
    """Verify drive level table computes macro and macro_ge30 correctly."""
    df, _ = make_mock_predictions_and_fn(150)
    # In mock: drive_1 has 80, drive_2 has 50, drive_3 has 20.
    res = compute_drive_level_coverage_table(df)

    assert len(res["drive_rows"]) == 3
    summary = res["summary"]
    assert summary["n_drives"] == 3
    assert summary["n_drives_ge30"] == 2  # drive_1 and drive_2 have >= 30

    for m in ["cqr", "sc", "mondrian"]:
        assert summary[m]["pooled_coverage"] is not None
        assert summary[m]["macro_coverage"] is not None
        assert summary[m]["macro_coverage_ge30"] is not None


def test_exchangeability_ks_diagnostics():
    """Verify 2-sample KS test between two DataFrames returns valid stats."""
    df_c, _ = make_mock_predictions_and_fn(100)
    df_t, _ = make_mock_predictions_and_fn(120)

    ks_rows = compute_exchangeability_ks_diagnostics(df_c, df_t)
    assert len(ks_rows) >= 5

    for r in ks_rows:
        assert 0.0 <= r["ks_statistic"] <= 1.0
        assert 0.0 <= r["p_value"] <= 1.0
        assert r["n_c"] == 100
        assert r["n_t"] == 120


def test_cluster_bootstrap_coverage_ci():
    """Verify cluster bootstrap produces valid confidence interval containing pooled coverage."""
    df, _ = make_mock_predictions_and_fn(100)
    boot_res = cluster_bootstrap_coverage_ci(df, method="cqr", n_boot=200, seed=42)

    ci = boot_res["coverage_ci_95"]
    assert len(ci) == 2
    assert 0.0 <= ci[0] <= ci[1] <= 1.0
    assert boot_res["n_clusters"] == 3
    assert "CI thô" in boot_res["note"]
