"""tests/test_eval.py: unit tests for src/evaluation/eval.py (v4 §6)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.evaluation.eval import (
    MIN_N,
    append_jsonl,
    band_masks,
    cluster_bootstrap,
    cluster_row_groups,
    depth_metrics,
    difficulty_masks,
    evaluate_report,
    iter_cluster_resamples,
    make_log_record,
    paired_cluster_bootstrap,
)


def synthetic(
    n_drives: int,
    rows_per_drive: int = 30,
    seed: int = 0,
    drive_sigma: float = 0.15,
    noise: float = 0.02,
) -> pd.DataFrame:
    """Each drive has its own multiplicative bias (shared by all of its frames)."""
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(0.0, drive_sigma, n_drives))
    drive = np.repeat([f"d{i:03d}" for i in range(n_drives)], rows_per_drive)
    z_gt = rng.uniform(10, 50, n_drives * rows_per_drive)
    z_pred = z_gt * np.repeat(scale, rows_per_drive) * np.exp(rng.normal(0, noise, z_gt.size))
    return pd.DataFrame({"z_gt": z_gt, "z_pred": z_pred, "drive": drive})


# ---------------------------------------------------------------- point metrics
def test_depth_metrics_known_values() -> None:
    m = depth_metrics(np.array([10.0, 20.0, 40.0]), np.array([11.0, 18.0, 50.0]))
    assert m["n"] == 3 and m["n_valid"] == 3
    assert m["absrel"] == pytest.approx((0.1 + 0.1 + 0.25) / 3)
    assert m["mae"] == pytest.approx((1 + 2 + 10) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 4 + 100) / 3))
    assert m["sqrel"] == pytest.approx((1 / 10 + 4 / 20 + 100 / 40) / 3)
    # ratios are 1.10, 1.111, 1.25 -> the threshold is strict, so 2 of 3 pass
    assert m["delta1"] == pytest.approx(2 / 3)


def test_perfect_prediction_is_zero_error() -> None:
    z = np.array([5.0, 15.0, 35.0])
    m = depth_metrics(z, z.copy())
    assert m["absrel"] == 0 and m["rmse"] == 0 and m["rmse_log"] == 0 and m["delta1"] == 1


def test_missing_predictions_are_excluded_but_counted() -> None:
    gt = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([10.0, np.nan, -3.0, 40.0])
    m = depth_metrics(gt, pred)
    assert m["n"] == 4 and m["n_valid"] == 2
    assert m["absrel"] == 0


def test_all_missing_gives_nan_not_error() -> None:
    m = depth_metrics(np.array([10.0, 20.0]), np.array([np.nan, np.nan]))
    assert m["n_valid"] == 0 and np.isnan(m["absrel"])


def test_bad_ground_truth_raises() -> None:
    with pytest.raises(ValueError):
        depth_metrics(np.array([10.0, 0.0]), np.array([10.0, 5.0]))
    with pytest.raises(ValueError):
        depth_metrics(np.array([10.0, np.nan]), np.array([10.0, 5.0]))
    with pytest.raises(ValueError):
        depth_metrics(np.array([10.0, 20.0]), np.array([10.0]))


# --------------------------------------------------------------------- grouping
def test_band_edges_are_right_open_and_far_row_is_z_ge_30() -> None:
    z = np.array([9.99, 10.0, 29.99, 30.0, 49.99, 50.0, 80.0])
    m = band_masks(z)
    assert list(z[m["0-10"]]) == [9.99]
    assert list(z[m["10-20"]]) == [10.0]
    assert list(z[m["30-50"]]) == [30.0, 49.99]
    assert list(z[m[">50"]]) == [50.0, 80.0]
    assert list(z[m[">30"]]) == [30.0, 49.99, 50.0, 80.0]
    # the five bands partition the data
    total = sum(m[k].astype(int) for k in ("0-10", "10-20", "20-30", "30-50", ">50"))
    assert np.all(total == 1)


def test_difficulty_subsets_are_nested() -> None:
    d = pd.Series(["Easy", "Moderate", "Hard", "Excluded", "Easy"])
    m = difficulty_masks(d)
    assert np.all(m["Easy"] <= m["Moderate"]) and np.all(m["Moderate"] <= m["Hard"])
    assert m["Easy"].sum() == 2 and m["Moderate"].sum() == 3 and m["Hard"].sum() == 4
    assert not m["Hard"][3]  # Excluded is in no subset


def test_report_flags_low_n_and_reports_valid_frac() -> None:
    df = synthetic(n_drives=10, rows_per_drive=30)  # 300 rows, ~75 per band at most
    df.loc[df.index[:50], "z_pred"] = np.nan
    df["cls"] = "Car"
    rep = evaluate_report(df).set_index(["group_type", "group"])
    overall = rep.loc[("overall", "all")]
    assert overall["n"] == 300 and overall["n_valid"] == 250
    assert overall["valid_frac"] == pytest.approx(250 / 300)
    assert not overall["low_n"]  # 250 >= MIN_N
    assert rep.loc[("band", "0-10"), "low_n"]  # no GT below 10 m -> n_valid = 0 < MIN_N
    assert MIN_N == 100


def test_empty_group_does_not_crash() -> None:
    df = synthetic(n_drives=4)
    rep = evaluate_report(df).set_index(["group_type", "group"])
    assert rep.loc[("band", "0-10"), "n"] == 0
    assert np.isnan(rep.loc[("band", "0-10"), "absrel"])


# -------------------------------------------------------------------- bootstrap
def test_resamples_are_made_of_whole_drives() -> None:
    df = synthetic(n_drives=6, rows_per_drive=7)
    groups = cluster_row_groups(df["drive"].to_numpy())
    assert len(groups) == 6 and all(len(g) == 7 for g in groups)
    rng = np.random.default_rng(1)
    for idx in iter_cluster_resamples(groups, n_boot=50, rng=rng):
        counts = df["drive"].iloc[idx].value_counts()
        assert (counts % 7 == 0).all()  # a drive is drawn k times as a block, never partially
        assert len(idx) == 6 * 7


def test_same_seed_same_ci_and_different_seed_differs() -> None:
    df = synthetic(n_drives=25)
    a = cluster_bootstrap(df, "absrel", seed=7, n_boot=300)
    b = cluster_bootstrap(df, "absrel", seed=7, n_boot=300)
    c = cluster_bootstrap(df, "absrel", seed=8, n_boot=300)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)
    assert a.seed == 7 and a.n_clusters == 25 and a.n_boot == 300
    assert a.ci_low <= a.estimate <= a.ci_high


def test_cluster_ci_is_wider_than_row_ci_when_drives_share_a_bias() -> None:
    df = synthetic(n_drives=25, rows_per_drive=40, drive_sigma=0.15)
    clustered = cluster_bootstrap(df, "absrel", seed=0, n_boot=400)
    df_rows = df.assign(drive=np.arange(len(df)))  # every row its own cluster = naive row bootstrap
    naive = cluster_bootstrap(df_rows, "absrel", seed=0, n_boot=400)
    width_c, width_n = clustered.ci_high - clustered.ci_low, naive.ci_high - naive.ci_low
    assert width_c > 2.0 * width_n


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ci_narrows_with_more_clusters() -> None:
    small = cluster_bootstrap(synthetic(n_drives=10, seed=3), "absrel", seed=0, n_boot=400)
    large = cluster_bootstrap(synthetic(n_drives=160, seed=3), "absrel", seed=0, n_boot=400)
    assert (large.ci_high - large.ci_low) < 0.6 * (small.ci_high - small.ci_low)


def test_few_clusters_warns_and_one_cluster_raises() -> None:
    with pytest.warns(UserWarning, match="clusters"):
        cluster_bootstrap(synthetic(n_drives=8), "absrel", seed=0, n_boot=50)
    with pytest.raises(ValueError):
        cluster_bootstrap(synthetic(n_drives=1), "absrel", seed=0, n_boot=50)


def test_paired_identical_methods_give_zero_and_not_significant() -> None:
    df = synthetic(n_drives=25).assign(z_alt=lambda d: d["z_pred"])
    r = paired_cluster_bootstrap(df, "z_pred", "z_alt", "absrel", seed=0, n_boot=200)
    assert r.estimate == 0 and r.ci_low == 0 and r.ci_high == 0
    assert not r.excludes_zero


def test_paired_detects_a_clearly_better_method() -> None:
    rng = np.random.default_rng(5)
    base = synthetic(n_drives=30, drive_sigma=0.0, noise=0.0)
    base["z_good"] = base["z_gt"] * np.exp(rng.normal(0, 0.03, len(base)))
    base["z_bad"] = base["z_gt"] * np.exp(rng.normal(0, 0.15, len(base)))
    r = paired_cluster_bootstrap(base, "z_good", "z_bad", "absrel", seed=0, n_boot=300)
    assert r.estimate < 0 and r.ci_high < 0 and r.excludes_zero  # A has lower error


def test_paired_uses_common_support_only() -> None:
    df = synthetic(n_drives=25).assign(z_alt=lambda d: d["z_pred"] * 1.1)
    df.loc[df.index[:100], "z_alt"] = np.nan  # B has no estimate on the first 100 rows
    r = paired_cluster_bootstrap(df, "z_pred", "z_alt", "absrel", seed=0, n_boot=100)
    assert r.n_rows == len(df) - 100


# ---------------------------------------------------------------------- logging
def test_log_record_has_seed_and_split_hash(tmp_path) -> None:
    rec = make_log_record(split="B", split_hash="abc123", seed=42, n_boot=1000, tag="day5",
                          extra={"metric": "absrel"})
    path = tmp_path / "logs" / "eval_log.jsonl"
    append_jsonl(path, rec)
    append_jsonl(path, rec)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    loaded = json.loads(lines[0])
    assert loaded["seed"] == 42 and loaded["split_hash"] == "abc123" and loaded["metric"] == "absrel"


def test_no_warning_with_enough_clusters() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cluster_bootstrap(synthetic(n_drives=25), "absrel", seed=0, n_boot=50)
