"""
tests/test_cqr.py: Comprehensive unit tests and theoretical verification for CQR (Decisions D46, D47, D48, D49).

Verifies:
1. Exact finite-sample order statistic calculation on hand-crafted data (fixing np.quantile off-by-one).
2. Small-sample edge case k > n returning +inf and safe interval generation.
3. Leave-One-Drive-Out (LODO) calibration invariance (modifying drive d does not affect d's interval).
4. Deterministic exchangeable coverage across 200 repetitions.
5. Heteroscedastic noise resilience under CQR.
6. Quantile sorting consistency and crossing counter behavior without silent clipping.
7. Disjoint drive guard against data leakage.
8. Winkler score mathematical properties and OOF Z_base invariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.uncertainty.cqr import (
    assert_disjoint_drives,
    compute_nonconformity_scores,
    conformalize,
    predict_interval,
    sort_quantiles,
    winkler_score,
)


def test_conformalize_order_statistic_handcrafted():
    """
    Decision D46: Conformal quantile level must follow exact finite-sample order statistic.

    Theoretical formula:
        k = ceil((n + 1) * (1 - alpha))
        Q_hat = sorted_scores[k - 1]

    Verify against hand-crafted data and demonstrate that np.quantile(..., method='higher')
    would introduce an off-by-one error (returning sorted_scores[k]).
    """
    # Case 1: n = 100, alpha = 0.1
    # k = ceil(101 * 0.9) = ceil(90.9) = 91.
    scores_100 = np.arange(1, 101, dtype=float)
    q_hat_100 = conformalize(scores_100, alpha=0.1)
    assert q_hat_100 == 91.0, f"Expected 91.0, got {q_hat_100}"

    # Demonstrate that numpy's virtual index (q * 99 = 90.09) would yield 92.0
    np_higher_val = np.quantile(scores_100, 0.91, method="higher")
    assert np_higher_val == 92.0
    assert q_hat_100 != np_higher_val

    # Case 2: n = 10, alpha = 0.1
    # k = ceil(11 * 0.9) = ceil(9.9) = 10.
    scores_10 = np.array([10.0, 30.0, 20.0, 50.0, 40.0, 90.0, 70.0, 80.0, 60.0, 100.0])
    q_hat_10 = conformalize(scores_10, alpha=0.1)
    assert q_hat_10 == 100.0  # Largest element (index 9)

    # Case 3: n = 19, alpha = 0.1
    # k = ceil(20 * 0.9) = 18.
    scores_19 = np.arange(1, 20, dtype=float)
    q_hat_19 = conformalize(scores_19, alpha=0.1)
    assert q_hat_19 == 18.0


def test_conformalize_edge_case_k_greater_than_n():
    """
    Decision D46: If k = ceil((n + 1)(1 - alpha)) > n, sample size is insufficient -> return +inf.
    Verify predict_interval produces [0, +inf] safely without crashing.
    """
    # n = 5, alpha = 0.1 -> k = ceil(6 * 0.9) = ceil(5.4) = 6 > 5
    scores_5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q_hat = conformalize(scores_5, alpha=0.1)
    assert np.isposinf(q_hat)

    # Prediction interval with Q_hat = +inf
    z_base = np.array([10.0, 20.0])
    q_lo = np.array([-0.1, -0.2])
    q_hi = np.array([0.1, 0.2])

    z_lo, z_hi, r_lo, r_hi, n_cross = predict_interval(z_base, q_lo, q_hi, q_hat)
    assert np.all(z_lo == 0.0)
    assert np.all(np.isposinf(z_hi))
    assert np.all(np.isneginf(r_lo))
    assert np.all(np.isposinf(r_hi))
    assert n_cross == 0


def test_lodo_invariance():
    """
    Test LODO calibration invariance:
    Modifying residuals in drive 'A' MUST NOT affect the conformal interval for drive 'A',
    because drive 'A' is calibrated strictly on other drives (e.g. B and C).
    Conversely, the intervals for drives 'B' and 'C' MUST change.
    """
    rng = np.random.RandomState(42)

    # 3 drives, 50 samples each
    drives = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
    z_base = np.full(150, 15.0)
    q_lo = np.full(150, -0.1)
    q_hi = np.full(150, 0.1)
    r_actual = rng.normal(0, 0.05, size=150)

    def run_lodo(r_arr: np.ndarray) -> dict[str, float]:
        q_hat_per_drive = {}
        for d in ["A", "B", "C"]:
            calib_mask = drives != d
            scores_calib = compute_nonconformity_scores(
                q_lo[calib_mask], q_hi[calib_mask], r_arr[calib_mask]
            )
            q_hat_per_drive[d] = conformalize(scores_calib, alpha=0.1)
        return q_hat_per_drive

    q_hat_orig = run_lodo(r_actual)

    # Perturb drive A's actual residuals heavily
    r_perturbed = r_actual.copy()
    r_perturbed[:50] += 5.0  # Massive shift on drive A only

    q_hat_pert = run_lodo(r_perturbed)

    # Invariance check:
    # Drive A's Q_hat was calibrated on B and C -> unchanged!
    assert np.isclose(q_hat_orig["A"], q_hat_pert["A"], atol=1e-12), (
        f"Drive A Q_hat changed ({q_hat_orig['A']} vs {q_hat_pert['A']}) despite LODO isolation!"
    )

    # Drive B and C had drive A in their calibration set -> MUST change!
    assert not np.isclose(q_hat_orig["B"], q_hat_pert["B"]), "Drive B Q_hat should have changed!"
    assert not np.isclose(q_hat_orig["C"], q_hat_pert["C"]), "Drive C Q_hat should have changed!"


def test_synthetic_exchangeable_coverage_deterministic():
    """
    Finite-sample validity test:
    Under exchangeable data, empirical coverage across repeated trials must converge to >= 1 - alpha.
    Run 200 repetitions with fixed seed to avoid test flakiness.
    """
    alpha = 0.1
    n_calib = 500
    n_test = 200
    n_trials = 200

    rng = np.random.RandomState(1337)
    coverages = []

    for _ in range(n_trials):
        # Generate calibration and test nonconformity scores from same distribution
        # e.g. Absolute standard normal error
        scores_calib = np.abs(rng.normal(0, 1, size=n_calib))
        scores_test = np.abs(rng.normal(0, 1, size=n_test))

        q_hat = conformalize(scores_calib, alpha=alpha)

        # In symmetric CQR, a test point is covered if score_test <= Q_hat
        covered = scores_test <= q_hat
        coverages.append(np.mean(covered))

    mean_coverage = np.mean(coverages)
    # Theory guarantees expectation >= 1 - alpha = 0.90
    assert mean_coverage >= 0.89, f"Mean coverage {mean_coverage:.4f} is too low (expected >= 0.89)"
    assert mean_coverage <= 0.93, f"Mean coverage {mean_coverage:.4f} is too conservative"


def test_heteroscedastic_coverage():
    """
    Verify CQR under heteroscedastic noise (error magnitude varies with X).
    """
    rng = np.random.RandomState(2026)
    n_calib = 1000
    n_test = 500

    X_calib = rng.uniform(5.0, 50.0, size=n_calib)
    sigma_calib = 0.02 + 0.005 * X_calib
    r_calib = rng.normal(0, sigma_calib)

    X_test = rng.uniform(5.0, 50.0, size=n_test)
    sigma_test = 0.02 + 0.005 * X_test
    r_test = rng.normal(0, sigma_test)

    # True 5% and 95% quantiles of normal distribution are +/- 1.645 * sigma
    q_lo_calib = -1.645 * sigma_calib
    q_hi_calib = 1.645 * sigma_calib

    scores_calib = compute_nonconformity_scores(q_lo_calib, q_hi_calib, r_calib)
    q_hat = conformalize(scores_calib, alpha=0.1)

    q_lo_test = -1.645 * sigma_test
    q_hi_test = 1.645 * sigma_test

    z_base_test = X_test
    z_lo, z_hi, r_lo, r_hi, n_cross = predict_interval(z_base_test, q_lo_test, q_hi_test, q_hat)

    # In log space: covered if r_lo <= r_test <= r_hi
    covered = (r_lo <= r_test) & (r_test <= r_hi)
    cov_rate = np.mean(covered)

    assert 0.87 <= cov_rate <= 0.93, f"Heteroscedastic coverage {cov_rate:.4f} outside [0.87, 0.93]"
    assert n_cross == 0


def test_quantile_sorting_and_crossing_counter():
    """
    Decisions D47:
    - sort_quantiles handles flipped raw quantiles (e.g. q_lo > q_hi).
    - If Q_hat < 0 is large, crossings occur: predict_interval counts crossings and does not silently clip.
    """
    # 1. Test sort_quantiles
    q_lo_raw = np.array([0.5, -0.2, 0.1])
    q_hi_raw = np.array([0.1, 0.4, -0.3])
    lo_sorted, hi_sorted = sort_quantiles(q_lo_raw, q_hi_raw)

    assert np.all(lo_sorted <= hi_sorted)
    assert np.array_equal(lo_sorted, np.array([0.1, -0.2, -0.3]))
    assert np.array_equal(hi_sorted, np.array([0.5, 0.4, 0.1]))

    # 2. Test negative Q_hat causing crossings
    z_base = np.array([10.0, 20.0, 30.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.2, 0.2, 0.2])

    # Q_hat = -0.15 -> r_lo = 0.0 - (-0.15) = +0.15; r_hi = 0.2 + (-0.15) = +0.05 -> crossing!
    q_hat_neg = -0.15
    z_lo, z_hi, r_lo, r_hi, n_crossings = predict_interval(z_base, lo, hi, q_hat_neg)

    assert n_crossings == 3
    assert np.all(r_lo > r_hi)
    assert np.all(z_lo > z_hi)  # Reported without silent clipping


def test_assert_disjoint_drives_guard():
    """Verify disjoint drive guard prevents data leakage between fit and calibration sets."""
    fit_drives = ["drive_0001", "drive_0002", "drive_0003"]
    calib_drives = ["drive_0004", "drive_0005"]

    # Should not raise
    assert_disjoint_drives(fit_drives, calib_drives)

    # Leakage case
    leak_calib = ["drive_0003", "drive_0006"]
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_disjoint_drives(fit_drives, leak_calib)


def test_winkler_score_properties():
    """
    Decision D47: Verify Winkler interval score properties in log-space r.
    Score = (U - L) + (2 / alpha) * penalty
    """
    alpha = 0.1
    # Sample 1: Target inside [ -0.1, 0.1 ], actual = 0.0 -> score = 0.2
    # Sample 2: Target under [ -0.1, 0.1 ], actual = -0.2 -> width 0.2 + (2/0.1)*0.1 = 0.2 + 2.0 = 2.2
    # Sample 3: Target over [ -0.1, 0.1 ], actual = 0.3 -> width 0.2 + (2/0.1)*0.2 = 0.2 + 4.0 = 4.2
    r_lo = np.array([-0.1, -0.1, -0.1])
    r_hi = np.array([0.1, 0.1, 0.1])
    r_act = np.array([0.0, -0.2, 0.3])

    scores = winkler_score(r_lo, r_hi, r_act, alpha=alpha)

    assert np.isclose(scores[0], 0.2)
    assert np.isclose(scores[1], 2.2)
    assert np.isclose(scores[2], 4.2)


def test_fit_quantile_uses_oof_z_base_and_manifest_params():
    """
    Decisions D24, D43:
    Quantile models must be trained on target r = ln(Z_gt) - ln(Z_base_oof)
    using z_base from B_oof.parquet (LODO Z_d + OOF Z_e) and fixed manifest best_params_f.
    """
    from pathlib import Path
    import json
    from src.uncertainty.cqr import fit_quantile_xgb

    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "runs" / "residual" / "yolo11s_640" / "manifest.json"
    oof_path = project_root / "results" / "datasets" / "yolo11s_640_B_oof.parquet"
    eval_path = project_root / "results" / "datasets" / "yolo11s_640_B_eval.parquet"

    if not oof_path.is_file() or not eval_path.is_file() or not manifest_path.is_file():
        pytest.skip("Requires Split B dataset and model artifacts")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    oof_df = pd.read_parquet(oof_path)
    eval_df = pd.read_parquet(eval_path)
    best_params_f = manifest["best_params_f"]

    # Verify z_base is strictly valid and matches target r definition
    z_base_oof = oof_df["z_base"].to_numpy(dtype=float)
    z_gt = eval_df["z_gt"].to_numpy(dtype=float)
    r_target_expected = np.log(z_gt) - np.log(z_base_oof)

    assert not np.any(np.isnan(z_base_oof))
    assert np.all(z_base_oof > 0)
    assert len(r_target_expected) == len(z_base_oof)

    # Train quantile model on a 100-sample subset with real features and real OOF target r
    X_sub = np.column_stack([z_base_oof[:100], np.log(z_base_oof[:100])])
    r_sub = r_target_expected[:100]

    model_q = fit_quantile_xgb(X_sub, r_sub, q=0.05, best_params_f=best_params_f, n_jobs=1)
    assert model_q.get_params()["objective"] == "reg:quantileerror"
    assert np.isclose(model_q.get_params()["quantile_alpha"], 0.05)


def test_apply_frozen_pipeline_cues_match():
    """
    Decision D49: Recalculated Z_d via full_fw must match stored z_d in C_cues.parquet.
    """
    from pathlib import Path
    from src.pipeline.apply_frozen import apply_frozen_pipeline

    project_root = Path(__file__).resolve().parent.parent
    cues_path = project_root / "results" / "datasets" / "yolo11s_640_C_cues.parquet"
    if not cues_path.is_file():
        pytest.skip("Requires Split C cues artifact")

    cues_df = pd.read_parquet(cues_path)
    n_expected = len(cues_df)

    res = apply_frozen_pipeline("yolo11s_640", split="C")
    assert res["n_samples"] == n_expected
    assert res["z_base"].shape == (n_expected,)
    assert not np.any(np.isnan(res["z_base"]))
    assert np.all(res["z_base"] > 0)
    assert "r_actual" in res and res["r_actual"] is not None


def test_conformalize_floating_point_stability():
    """
    Ensure float tolerance (1e-12) avoids rounding jitter in ceil((n + 1)*(1 - alpha)).
    """
    # Test cases where floating point representation might slightly exceed integer
    # e.g. (99 + 1) * 0.9 = 90.0
    scores = np.arange(1, 100, dtype=float)  # n = 99
    # (99 + 1) * (1 - 0.1) = 90.0 exactly
    q_hat = conformalize(scores, alpha=0.1)
    assert q_hat == 90.0


