"""tests/test_repartition_bct.py: unit tests for scripts/repartition_bct.py (Decision D14)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.repartition_bct import (
    Criteria,
    DriveStat,
    SeedResult,
    SplitMetrics,
    anneal,
    assemble_splits,
    build_pool,
    criterion_flags,
    criterion_pass_rates,
    evaluate,
    n_eff,
    run_search,
    select_best,
    target_fractions,
    write_splits_v2,
)


def make_pool(n_car_drives: int = 45, n_empty: int = 35, seed: int = 0):
    rng = np.random.default_rng(seed)
    stats = []
    for i in range(n_car_drives):
        n = int(np.clip(rng.lognormal(5.0, 0.7), 20, 700))
        stats.append(DriveStat(f"car_{i:03d}", int(rng.integers(20, 90)), rng.gamma(5.0, 5.0, n)))
    for i in range(n_empty):
        stats.append(DriveStat(f"empty_{i:03d}", int(rng.integers(5, 60)), np.empty(0)))
    return build_pool(stats)


def fake_result(seed: int, passed: bool, n_eff_vals: tuple[float, float, float], ks: float) -> SeedResult:
    m = SplitMetrics((0.4, 0.2, 0.3), (0.4, 0.2, 0.3), (0.3, 0.3, 0.3), n_eff_vals, (10, 10, 10), (ks, ks, ks))
    flags = {"frames": passed, "cars": True, "top1": True, "car_drives": True, "ks": True}
    return SeedResult(seed, passed, flags, m, np.zeros(3, dtype=np.intp))


# ------------------------------------------------------------------ metrics
def test_n_eff_equal_shares_and_single_drive() -> None:
    assert n_eff([10, 10, 10, 10]) == pytest.approx(4.0)
    assert n_eff([100]) == pytest.approx(1.0)
    assert n_eff([0, 0, 50, 50]) == pytest.approx(2.0)  # empty drives do not count
    assert n_eff([]) == 0.0


def test_n_eff_reproduces_audit_value_for_split_c() -> None:
    # top5 of split C from audit_drive_concentration.py + 6 small drives summing to 131
    counts = [1185, 344, 277, 253, 141] + [131 / 6] * 6
    assert n_eff(counts) == pytest.approx(3.2, abs=0.15)


def test_histogram_ks_close_to_exact_ks() -> None:
    pool = make_pool(seed=1)
    rng = np.random.default_rng(3)
    assign = rng.integers(0, 3, len(pool.drives)).astype(np.intp)
    approx = evaluate(pool, assign, exact=False).ks
    exact = evaluate(pool, assign, exact=True).ks
    assert np.allclose(approx, exact, atol=0.02)


def test_top1_share_and_drive_counts() -> None:
    stats = [
        DriveStat("a", 10, np.full(60, 20.0)),
        DriveStat("b", 10, np.full(40, 20.0)),
        DriveStat("c", 10, np.empty(0)),
    ]
    pool = build_pool(stats)  # sorted: a, b, c
    m = evaluate(pool, np.array([0, 0, 1], dtype=np.intp))
    assert m.top1_share[0] == pytest.approx(0.6)
    assert m.n_car_drives == (2, 0, 0)
    assert m.frame_frac[0] == pytest.approx(20 / 30)


# ------------------------------------------------------------------- search
def test_anneal_is_deterministic_per_seed() -> None:
    pool = make_pool()
    a1 = anneal(pool, Criteria(), seed=5, n_iter=500)
    a2 = anneal(pool, Criteria(), seed=5, n_iter=500)
    a3 = anneal(pool, Criteria(), seed=6, n_iter=500)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)


def test_search_finds_feasible_partition_when_one_exists() -> None:
    pool = make_pool()
    results = run_search(pool, Criteria(), range(6), n_iter=2500)
    best = select_best(results)
    assert best is not None, f"pass rates: {criterion_pass_rates(results)}"
    m = best.metrics
    assert max(m.top1_share) <= 0.35 + 1e-9
    assert min(m.n_car_drives) >= 10
    assert max(m.ks) <= 0.07 + 1e-9
    tf = target_fractions()
    assert all(abs(m.car_share[k] - tf[k]) * 100 <= 5.0 + 1e-9 for k in range(3))
    assert all(abs(m.frame_frac[k] - tf[k]) * 100 <= 3.0 + 1e-9 for k in range(3))


def test_impossible_criteria_select_nothing() -> None:
    pool = make_pool()
    crit = Criteria(min_car_drives=1000)
    results = run_search(pool, crit, range(3), n_iter=300)
    assert select_best(results) is None
    assert criterion_pass_rates(results)["car_drives"] == 0.0


def test_criterion_flags_detect_each_violation() -> None:
    tf = target_fractions()
    ok = SplitMetrics(tf, tf, (0.3, 0.3, 0.3), (6, 6, 6), (10, 10, 10), (0.05, 0.05, 0.05))
    assert all(criterion_flags(ok, Criteria(), tf).values())
    bad_top1 = SplitMetrics(tf, tf, (0.3, 0.51, 0.3), (6, 6, 6), (10, 10, 10), (0.05, 0.05, 0.05))
    assert not criterion_flags(bad_top1, Criteria(), tf)["top1"]
    bad_ks = SplitMetrics(tf, tf, (0.3, 0.3, 0.3), (6, 6, 6), (10, 10, 10), (0.05, 0.0723, 0.05))
    assert not criterion_flags(bad_ks, Criteria(), tf)["ks"]
    bad_drives = SplitMetrics(tf, tf, (0.3, 0.3, 0.3), (6, 6, 6), (10, 9, 10), (0.05, 0.05, 0.05))
    assert not criterion_flags(bad_drives, Criteria(), tf)["car_drives"]
    bad_frames = SplitMetrics((0.5, 0.2, 0.3), tf, (0.3, 0.3, 0.3), (6, 6, 6), (10, 10, 10), (0.05,) * 3)
    assert not criterion_flags(bad_frames, Criteria(), tf)["frames"]
    starved_c = SplitMetrics(tf, (0.55, 0.09, 0.36), (0.3, 0.3, 0.3), (6, 6, 6), (10, 10, 10), (0.05,) * 3)
    assert not criterion_flags(starved_c, Criteria(), tf)["cars"]  # C with 9% of cars must fail


def test_select_best_rule() -> None:
    results = [
        fake_result(0, False, (9.0, 9.0, 9.0), 0.01),   # fails: ignored even though n_eff is highest
        fake_result(1, True, (5.0, 4.0, 6.0), 0.05),    # min n_eff 4.0
        fake_result(2, True, (5.0, 6.0, 6.0), 0.06),    # min n_eff 5.0  <- winner
        fake_result(3, True, (5.0, 6.0, 6.0), 0.04),    # same min n_eff, lower KS -> beats seed 2
    ]
    assert select_best(results).seed == 3  # type: ignore[union-attr]
    tie = [fake_result(7, True, (5.0, 5.0, 5.0), 0.05), fake_result(4, True, (5.0, 5.0, 5.0), 0.05)]
    assert select_best(tie).seed == 4  # type: ignore[union-attr]
    assert select_best([fake_result(0, False, (5.0, 5.0, 5.0), 0.05)]) is None


# ---------------------------------------------------------- splits assembly
def _toy_world():
    drive_frames = {
        "d1": ["000001", "000002"], "d2": ["000003"], "d3": ["000004", "000005"],
        "d4": ["000006"], "dA": ["000007", "000008"], "dV": ["000009"],
    }
    old = {
        "A": ["000007", "000008"], "V": ["000009"],
        "B": ["000001", "000002", "000003"], "C": ["000004", "000005"], "T": ["000006"],
    }
    return old, drive_frames


def test_assemble_keeps_A_V_and_is_an_exact_partition() -> None:
    old, drive_frames = _toy_world()
    mapping = {"d1": "T", "d2": "C", "d3": "B", "d4": "B"}
    new = assemble_splits(old, {d: drive_frames[d] for d in mapping}, mapping)
    assert new["A"] == old["A"] and new["V"] == old["V"]
    assert new["B"] == ["000004", "000005", "000006"]
    assert new["C"] == ["000003"] and new["T"] == ["000001", "000002"]
    assert sum(len(v) for v in new.values()) == 9


def test_assemble_rejects_drive_that_touches_frozen_split() -> None:
    old, drive_frames = _toy_world()
    leaky = dict(drive_frames, d1=["000001", "000002", "000007"])  # drive spans B and A
    mapping = {"d1": "T", "d2": "C", "d3": "B", "d4": "B"}
    with pytest.raises(ValueError, match="outside B/C/T"):
        assemble_splits(old, {d: leaky[d] for d in mapping}, mapping)


def test_assemble_rejects_incomplete_partition() -> None:
    old, drive_frames = _toy_world()
    mapping = {"d1": "T", "d2": "C", "d3": "B"}  # d4 forgotten
    with pytest.raises(ValueError, match="exact partition"):
        assemble_splits(old, {d: drive_frames[d] for d in mapping}, mapping)


# ----------------------------------------------------------------- writing
def _sha(ids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def test_write_splits_v2_backs_up_and_preserves_A_V(tmp_path: Path) -> None:
    old, drive_frames = _toy_world()
    sd = tmp_path / "splits"
    sd.mkdir()
    drive_of = {f: d for d, fs in drive_frames.items() for f in fs}
    meta = {"seed": 42, "splits": {
        k: {"n_frames": len(v), "hash": _sha(v), "n_drives": 1, "drives": ["x"]} for k, v in old.items()
    }}
    for k, v in old.items():
        (sd / f"{k}.txt").write_text("\n".join(v) + "\n", encoding="utf-8")
    (sd / "split_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    mapping = {"d1": "T", "d2": "C", "d3": "B", "d4": "B"}
    new = assemble_splits(old, {d: drive_frames[d] for d in mapping}, mapping)
    new_meta = write_splits_v2(sd, new, drive_of, {"selected_seed": 17}, _sha)

    assert (sd / "superseded_v1" / "B.txt").read_text(encoding="utf-8").split() == old["B"]
    assert new_meta["splits"]["A"] == meta["splits"]["A"] and new_meta["splits"]["V"] == meta["splits"]["V"]
    assert (sd / "A.txt").read_text(encoding="utf-8").split() == old["A"]
    for name in ("B", "C", "T"):
        ids = (sd / f"{name}.txt").read_text(encoding="utf-8").split()
        assert ids == new[name]
        assert new_meta["splits"][name]["hash"] == _sha(ids)
    assert new_meta["version"] == "v2"
    assert new_meta["bct_repartition"]["selected_seed"] == 17
    assert new_meta["bct_repartition"]["parent_hashes"]["B"] == meta["splits"]["B"]["hash"]

    with pytest.raises(FileExistsError):
        write_splits_v2(sd, new, drive_of, {}, _sha)  # never overwrite a previous repartition
