"""
scripts/repartition_bct.py: Re-partition DRIVES among B / C / T (Decision D14).

Why
---
audit_drive_concentration.py: n_eff(C)=3.2 (drive 0059 holds 50.8% of Car Hard in C),
n_eff(B)=5.4, n_eff(T)=5.6. A and V are FROZEN (hash unchanged, detector queue untouched).
Only drives currently in B u C u T move.

Pre-registered protocol (write it into NHAT_KY_QUYET_DINH.md BEFORE running)
----------------------------------------------------------------------------
- Only LABELS are read (Car Hard count and depth Z per drive). No model output, no inference.
- <= 200 independent restarts of simulated annealing; the restart id is the seed.
- Hard criteria (ALL must hold): top1_share <= top1_max in B, C, T; >= min_car_drives drives
  with Car Hard in each set; KS STATISTIC (not p-value) of Car Hard depth <= ks_max for
  B-C, B-T, C-T; |frame share - target| <= frame_tol_pts for 20:10:15; and
  |Car Hard share - target| <= car_tol_pts (otherwise n_eff can be "won" by starving C of cars).
- Selection among seeds that pass: maximise min(n_eff over B, C, T); tie -> lowest max KS;
  tie -> lowest seed. Never "first seed that passes".
- Every seed (passing or not) is written to results/tables/repartition_bct_report.json.
- Default is a DRY-RUN. --write freezes splits-v2 and moves old B/C/T to splits/superseded_v1/.
- If no seed passes, nothing is written. Relaxation happens only through explicit CLI flags
  following the ladder fixed in the decision log, and is recorded in the metadata.

Usage
-----
    python scripts/repartition_bct.py                 # dry-run, prints report
    python scripts/repartition_bct.py --write         # freeze splits-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

SPLIT_NAMES: tuple[str, str, str] = ("B", "C", "T")
FROZEN_SPLITS: tuple[str, str] = ("A", "V")
PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
PAIR_LABELS: tuple[str, ...] = ("B-C", "B-T", "C-T")
TARGET_FRAME_WEIGHTS: tuple[float, float, float] = (20.0, 10.0, 15.0)

MAX_SEEDS = 200
N_ITER = 4000
BIN_WIDTH = 0.25      # metres; histogram-CDF KS used inside the search, exact KS for the verdict
DEPTH_MAX = 100.0
EPS = 1e-12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criteria:
    top1_max: float = 0.35
    min_car_drives: int = 10
    ks_max: float = 0.07
    frame_tol_pts: float = 3.0
    car_tol_pts: float = 5.0


@dataclass(frozen=True, eq=False)
class DriveStat:
    drive: str
    n_frames: int
    car_depths: FloatArray  # depth Z of every Car Hard object in this drive


@dataclass(frozen=True, eq=False)
class Pool:
    drives: tuple[str, ...]
    frames: IntArray
    cars: IntArray
    hist: FloatArray
    depths: tuple[FloatArray, ...]


@dataclass(frozen=True)
class SplitMetrics:
    frame_frac: tuple[float, ...]   # per split in SPLIT_NAMES order
    car_share: tuple[float, ...]
    top1_share: tuple[float, ...]
    n_eff: tuple[float, ...]
    n_car_drives: tuple[int, ...]
    ks: tuple[float, ...]           # per pair in PAIR_LABELS order


@dataclass(frozen=True, eq=False)
class SeedResult:
    seed: int
    passed: bool
    flags: Mapping[str, bool]
    metrics: SplitMetrics
    assign: IntArray

    @property
    def min_n_eff(self) -> float:
        return min(self.metrics.n_eff)

    @property
    def max_ks(self) -> float:
        return max(self.metrics.ks)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def n_eff(counts: Sequence[float] | FloatArray) -> float:
    """Kish effective number of clusters: 1 / sum(share^2) over drives with count > 0."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    share = c / c.sum()
    return float(1.0 / np.sum(share**2))


def target_fractions() -> tuple[float, float, float]:
    total = sum(TARGET_FRAME_WEIGHTS)
    a, b, c = (w / total for w in TARGET_FRAME_WEIGHTS)
    return a, b, c


def build_pool(drive_stats: Sequence[DriveStat]) -> Pool:
    ordered = sorted(drive_stats, key=lambda s: s.drive)
    edges = np.arange(0.0, DEPTH_MAX + BIN_WIDTH, BIN_WIDTH)
    hist = np.zeros((len(ordered), len(edges) - 1))
    for i, s in enumerate(ordered):
        d = np.clip(np.asarray(s.car_depths, dtype=float), 0.0, DEPTH_MAX - 1e-9)
        hist[i] = np.histogram(d, bins=edges)[0]
    return Pool(
        drives=tuple(s.drive for s in ordered),
        frames=np.array([s.n_frames for s in ordered], dtype=np.intp),
        cars=np.array([len(s.car_depths) for s in ordered], dtype=np.intp),
        hist=hist,
        depths=tuple(np.asarray(s.car_depths, dtype=float) for s in ordered),
    )


def _pairwise_ks(pool: Pool, assign: IntArray, exact: bool) -> tuple[float, ...]:
    out: list[float] = []
    if exact:
        pooled = [
            np.concatenate([pool.depths[i] for i in np.flatnonzero(assign == k)] or [np.empty(0)])
            for k in range(3)
        ]
        for a, b in PAIRS:
            if pooled[a].size == 0 or pooled[b].size == 0:
                out.append(1.0)
            else:
                out.append(float(stats.ks_2samp(pooled[a], pooled[b]).statistic))
    else:
        cdfs: list[FloatArray | None] = []
        for k in range(3):
            h = pool.hist[assign == k].sum(axis=0)
            s = h.sum()
            cdfs.append(np.cumsum(h) / s if s > 0 else None)
        for a, b in PAIRS:
            ca, cb = cdfs[a], cdfs[b]
            out.append(1.0 if ca is None or cb is None else float(np.max(np.abs(ca - cb))))
    return tuple(out)


def evaluate(pool: Pool, assign: IntArray, exact: bool = False) -> SplitMetrics:
    frames_total = float(pool.frames.sum())
    cars_total = float(pool.cars.sum())
    frame_frac: list[float] = []
    car_share: list[float] = []
    top1: list[float] = []
    neff: list[float] = []
    ncar: list[int] = []
    for k in range(3):
        m = assign == k
        c = pool.cars[m]
        tot = float(c.sum())
        frame_frac.append(float(pool.frames[m].sum()) / frames_total)
        car_share.append(tot / cars_total if cars_total > 0 else 0.0)
        top1.append(float(c.max()) / tot if tot > 0 else 1.0)
        neff.append(n_eff(c))
        ncar.append(int((c > 0).sum()))
    return SplitMetrics(
        tuple(frame_frac), tuple(car_share), tuple(top1), tuple(neff), tuple(ncar),
        _pairwise_ks(pool, assign, exact),
    )


def criterion_flags(m: SplitMetrics, crit: Criteria, target: Sequence[float]) -> dict[str, bool]:
    return {
        "frames": all(abs(m.frame_frac[k] - target[k]) * 100.0 <= crit.frame_tol_pts + EPS for k in range(3)),
        "cars": all(abs(m.car_share[k] - target[k]) * 100.0 <= crit.car_tol_pts + EPS for k in range(3)),
        "top1": all(t <= crit.top1_max + EPS for t in m.top1_share),
        "car_drives": all(n >= crit.min_car_drives for n in m.n_car_drives),
        "ks": all(v <= crit.ks_max + EPS for v in m.ks),
    }


def _energy(m: SplitMetrics, crit: Criteria, target: Sequence[float]) -> float:
    """Lower is better. Violations dominate; among feasible states prefer larger min n_eff."""
    pen = 0.0
    for k in range(3):
        dev = abs(m.frame_frac[k] - target[k]) * 100.0
        pen += max(0.0, dev - crit.frame_tol_pts) / crit.frame_tol_pts
        cdev = abs(m.car_share[k] - target[k]) * 100.0
        pen += max(0.0, cdev - crit.car_tol_pts) / crit.car_tol_pts
        pen += max(0.0, m.top1_share[k] - crit.top1_max) / crit.top1_max * 10.0
        pen += max(0.0, crit.min_car_drives - m.n_car_drives[k])
    for v in m.ks:
        pen += max(0.0, v - crit.ks_max) / crit.ks_max * 5.0
    car_dev = max(abs(m.car_share[k] - target[k]) for k in range(3))  # soft pull toward the target
    return 10.0 * pen - min(m.n_eff) + 5.0 * car_dev


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def anneal(pool: Pool, crit: Criteria, seed: int, n_iter: int = N_ITER) -> IntArray:
    """One restart: simulated annealing over drive -> {B, C, T}. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    target = target_fractions()
    n_drives = len(pool.drives)
    cur = rng.choice(3, size=n_drives, p=np.asarray(target)).astype(np.intp)
    cur_e = _energy(evaluate(pool, cur), crit, target)
    best, best_e = cur.copy(), cur_e
    for t in np.geomspace(2.0, 0.02, n_iter):
        cand = cur.copy()
        if rng.random() < 0.5:
            i = int(rng.integers(n_drives))
            cand[i] = (cand[i] + int(rng.integers(1, 3))) % 3
        else:
            i, j = (int(x) for x in rng.integers(n_drives, size=2))
            if cur[i] == cur[j]:
                continue
            cand[i], cand[j] = cur[j], cur[i]
        e = _energy(evaluate(pool, cand), crit, target)
        d = e - cur_e
        if d <= 0 or rng.random() < np.exp(-d / t):
            cur, cur_e = cand, e
            if e < best_e:
                best, best_e = cand.copy(), e
    return best


def run_search(
    pool: Pool,
    crit: Criteria,
    seeds: Sequence[int],
    n_iter: int = N_ITER,
    on_seed: Callable[[SeedResult], None] | None = None,
) -> list[SeedResult]:
    target = target_fractions()
    results: list[SeedResult] = []
    for s in seeds:
        assign = anneal(pool, crit, s, n_iter)
        m = evaluate(pool, assign, exact=True)  # verdict uses the exact KS
        flags = criterion_flags(m, crit, target)
        r = SeedResult(s, all(flags.values()), flags, m, assign)
        results.append(r)
        if on_seed is not None:
            on_seed(r)
    return results


def select_best(results: Sequence[SeedResult]) -> SeedResult | None:
    """Pre-registered rule: max min n_eff, then lowest max KS, then lowest seed."""
    passers = [r for r in results if r.passed]
    if not passers:
        return None
    return min(passers, key=lambda r: (-round(r.min_n_eff, 3), round(r.max_ks, 4), r.seed))


def criterion_pass_rates(results: Sequence[SeedResult]) -> dict[str, float]:
    if not results:
        return {}
    names = list(results[0].flags)
    return {n: sum(r.flags[n] for r in results) / len(results) for n in names}


# ---------------------------------------------------------------------------
# Splits assembly / writing
# ---------------------------------------------------------------------------
def assemble_splits(
    old_splits: Mapping[str, Sequence[str]],
    drive_frames: Mapping[str, Sequence[str]],
    drive_to_split: Mapping[str, str],
) -> dict[str, list[str]]:
    """A and V are copied verbatim; B/C/T are rebuilt from whole drives."""
    pool_ids: set[str] = set()
    for k in SPLIT_NAMES:
        pool_ids.update(old_splits[k])
    new: dict[str, list[str]] = {k: sorted(old_splits[k]) for k in FROZEN_SPLITS}
    bct: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
    for drive, split in drive_to_split.items():
        frames = list(drive_frames[drive])
        outside = [f for f in frames if f not in pool_ids]
        if outside:
            raise ValueError(f"Drive {drive} has frames outside B/C/T (e.g. {outside[0]}): would touch A or V")
        bct[split].extend(frames)
    moved = [f for k in SPLIT_NAMES for f in bct[k]]
    if len(moved) != len(set(moved)) or set(moved) != pool_ids:
        raise ValueError("Repartition is not an exact partition of the old B u C u T pool")
    new.update({k: sorted(v) for k, v in bct.items()})
    return new


def write_splits_v2(
    splits_dir: str | Path,
    new_splits: Mapping[str, Sequence[str]],
    drive_of: Mapping[str, str],
    selection: Mapping[str, object],
    hash_fn: Callable[[list[str]], str],
) -> dict:
    """Back up old B/C/T + metadata, write new B/C/T, keep A/V entries byte-for-byte in metadata."""
    sd = Path(splits_dir)
    backup = sd / "superseded_v1"
    if backup.exists():
        raise FileExistsError(f"{backup} already exists: splits were already repartitioned")
    meta_path = sd / "split_metadata.json"
    old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backup.mkdir()
    for name in SPLIT_NAMES:
        shutil.copy2(sd / f"{name}.txt", backup / f"{name}.txt")
    shutil.copy2(meta_path, backup / "split_metadata.json")

    meta = json.loads(json.dumps(old_meta))
    meta["version"] = "v2"
    meta["bct_repartition"] = {
        "created_at": datetime.now().isoformat(),
        "parent_hashes": {n: old_meta["splits"][n]["hash"] for n in SPLIT_NAMES},
        **selection,
    }
    for name in SPLIT_NAMES:
        ids = list(new_splits[name])
        (sd / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        drives = sorted({drive_of[f] for f in ids})
        meta["splits"][name] = {
            "n_frames": len(ids),
            "hash": hash_fn(ids),
            "n_drives": len(drives),
            "drives": drives,
        }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def result_to_dict(r: SeedResult) -> dict:
    return {
        "seed": r.seed,
        "passed": r.passed,
        "flags": dict(r.flags),
        "min_n_eff": r.min_n_eff,
        "max_ks": r.max_ks,
        "metrics": asdict(r.metrics),
    }


# ---------------------------------------------------------------------------
# Loader glue (project imports are lazy so the pure logic is testable stand-alone)
# ---------------------------------------------------------------------------
def collect_drive_stats(loader, frame_ids: Sequence[str]) -> list[DriveStat]:  # type: ignore[no-untyped-def]
    n_frames: dict[str, int] = {}
    depths: dict[str, list[float]] = {}
    for fid in frame_ids:
        fr = loader.load_frame(fid)
        n_frames[fr.drive] = n_frames.get(fr.drive, 0) + 1
        bucket = depths.setdefault(fr.drive, [])
        bucket.extend(
            o.depth for o in fr.objects if o.obj_class == "Car" and o.passes_hard_filter() and o.depth > 0
        )
    return [DriveStat(d, n_frames[d], np.asarray(depths[d], dtype=float)) for d in n_frames]


def print_report(pool: Pool, r: SeedResult, crit: Criteria) -> None:
    m = r.metrics
    tf = target_fractions()
    print(f"\nSeed {r.seed}: passed={r.passed}  flags={dict(r.flags)}")
    print(f"{'Split':<5} {'frames':>7} {'frac%':>6} {'tgt%':>5} {'cars':>6} {'car%':>6} "
          f"{'drives':>6} {'top1':>5} {'n_eff':>6}")
    for k, name in enumerate(SPLIT_NAMES):
        mk = r.assign == k
        print(f"{name:<5} {int(pool.frames[mk].sum()):>7d} {100 * m.frame_frac[k]:>6.1f} {100 * tf[k]:>5.1f} "
              f"{int(pool.cars[mk].sum()):>6d} {100 * m.car_share[k]:>6.1f} {m.n_car_drives[k]:>6d} "
              f"{m.top1_share[k]:>5.2f} {m.n_eff[k]:>6.2f}")
    print("KS statistic: " + "  ".join(f"{p}={v:.4f}" for p, v in zip(PAIR_LABELS, m.ks))
          + f"   (max allowed {crit.ks_max})")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="D14: repartition drives among B/C/T (A, V frozen).")
    ap.add_argument("--data-root", default="data/kitti")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--report", default="results/tables/repartition_bct_report.json")
    ap.add_argument("--n-seeds", type=int, default=MAX_SEEDS, help=f"restarts, hard cap {MAX_SEEDS}")
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--top1-max", type=float, default=Criteria.top1_max)
    ap.add_argument("--min-car-drives", type=int, default=Criteria.min_car_drives)
    ap.add_argument("--ks-max", type=float, default=Criteria.ks_max)
    ap.add_argument("--frame-tol-pts", type=float, default=Criteria.frame_tol_pts)
    ap.add_argument("--car-tol-pts", type=float, default=Criteria.car_tol_pts)
    ap.add_argument("--write", action="store_true", help="freeze splits-v2 (default: dry-run)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.n_seeds <= MAX_SEEDS:
        raise SystemExit(f"--n-seeds must be in [1, {MAX_SEEDS}] (pre-registered cap)")
    crit = Criteria(args.top1_max, args.min_car_drives, args.ks_max, args.frame_tol_pts, args.car_tol_pts)

    sys.path.insert(0, ".")
    from src.evaluation.eval import append_jsonl, make_log_record
    from src.utils.kitti_loader import KITTILoader
    from src.utils.split_builder import assert_split_disjoint_by_drive, compute_split_hash, load_splits

    loader = KITTILoader(args.data_root)
    old = load_splits(args.splits_dir)
    pool_ids = [f for k in SPLIT_NAMES for f in old[k]]
    print(f"Pool B u C u T: {len(pool_ids)} frames. A ({len(old['A'])}) and V ({len(old['V'])}) frozen.")
    print("Reading LABELS only (Car Hard depth per drive); no inference, no model output.")
    pool = build_pool(collect_drive_stats(loader, pool_ids))
    print(f"Drives in pool: {len(pool.drives)} ({int((pool.cars > 0).sum())} with Car Hard, "
          f"{int(pool.cars.sum())} Car Hard objects)")
    print(f"Criteria: {crit}")

    results = run_search(
        pool, crit, range(args.n_seeds), args.n_iter,
        on_seed=lambda r: print(f"  seed {r.seed:>3d}: passed={r.passed!s:<5} min_n_eff={r.min_n_eff:5.2f} "
                                f"max_ks={r.max_ks:.4f} flags={sum(r.flags.values())}/5"),
    )
    best = select_best(results)
    rates = criterion_pass_rates(results)
    print(f"\nSeeds passing ALL criteria: {sum(r.passed for r in results)}/{len(results)}")
    print("Per-criterion pass rate: " + "  ".join(f"{k}={v:.0%}" for k, v in rates.items()))

    report = {
        "created_at": datetime.now().isoformat(),
        "criteria": asdict(crit),
        "n_seeds": len(results),
        "n_iter": args.n_iter,
        "pass_rates": rates,
        "selected_seed": None if best is None else best.seed,
        "written": False,
        "seeds": [result_to_dict(r) for r in results],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if best is None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No seed satisfies all pre-registered criteria. Nothing written.")
        print("Look at the per-criterion pass rates, then apply the relaxation ladder from the decision log.")
        return 2

    print_report(pool, best, crit)
    if not args.write:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nDRY-RUN: report in {report_path}. Re-run with --write to freeze splits-v2.")
        return 0

    drive_to_split = {pool.drives[i]: SPLIT_NAMES[int(a)] for i, a in enumerate(best.assign)}
    all_drives = loader.get_all_drives()
    drive_frames = {d: all_drives[d] for d in drive_to_split}
    new_splits = assemble_splits(old, drive_frames, drive_to_split)
    assert_split_disjoint_by_drive(new_splits, loader._drive_mapping)
    assert sum(len(v) for v in new_splits.values()) == loader.num_frames
    for k in FROZEN_SPLITS:
        assert new_splits[k] == sorted(old[k]), f"Split {k} must not change"

    drive_of = {f: loader.get_drive(f) for k in SPLIT_NAMES for f in new_splits[k]}
    selection = {
        "selected_seed": best.seed,
        "n_seeds_tried": len(results),
        "criteria": asdict(crit),
        "metrics": asdict(best.metrics),
        "labels_only": True,
    }
    meta = write_splits_v2(args.splits_dir, new_splits, drive_of, selection, compute_split_hash)
    report["written"] = True
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    hashes = {n: meta["splits"][n]["hash"] for n in SPLIT_NAMES}
    combined = hashlib.sha256("|".join(hashes[n] for n in SPLIT_NAMES).encode()).hexdigest()
    append_jsonl(
        "runs/splits_log.jsonl",
        make_log_record(split="BCT", split_hash=combined, seed=best.seed, n_boot=0, tag="D14_repartition",
                        extra={"hashes": hashes, "n_seeds_tried": len(results)}),
    )
    print("\nWROTE splits-v2:", {n: (meta["splits"][n]["n_frames"], hashes[n][:8]) for n in SPLIT_NAMES})
    print("Next: git tag splits-v2; update tests/test_splits.py frame counts; refit geometry (geometry-v2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
