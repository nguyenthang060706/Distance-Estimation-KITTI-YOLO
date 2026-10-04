"""
src/evaluation/eval.py: metrics, grouped reports and cluster bootstrap (KE_HOACH_V4 §6).

Frozen conventions (see NHAT_KY_QUYET_DINH.md):
- Target is depth Z (KITTI location_z), never Euclidean distance.
- Bands are right-open: [0,10) [10,20) [20,30) [30,50) [50,inf), plus the merged
  row ">30" defined as Z >= 30 (D3).
- A row whose prediction is NaN or <= 0 is "no estimate". It is excluded from the
  metric values but is always reported through n, n_valid and valid_frac, so that
  an estimator cannot look better by silently dropping hard cases (v4 §5.1).
- The bootstrap resamples whole drives (clusters), never rows or frames.
- Ground truth columns (z_gt, cls, difficulty) are used here for EVALUATION and
  GROUPING only. They must never reach the feature extractor (D11).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

BAND_EDGES: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 50.0, np.inf)
BAND_LABELS: tuple[str, ...] = ("0-10", "10-20", "20-30", "30-50", ">50")
FAR_MERGED_LABEL: str = ">30"
FAR_MERGED_START: float = 30.0
MIN_N: int = 100                 # rows below this get low_n=True (the '*' flag)
MIN_CLUSTERS_WARN: int = 20      # rule of thumb for a stable cluster bootstrap
DELTA_THRESHOLD: float = 1.25


# ---------------------------------------------------------------------------
# Point metrics (operate on arrays that are already valid and positive)
# ---------------------------------------------------------------------------
def _mae(g: FloatArray, p: FloatArray) -> float:
    return float(np.mean(np.abs(p - g)))


def _rmse(g: FloatArray, p: FloatArray) -> float:
    return float(np.sqrt(np.mean((p - g) ** 2)))


def _absrel(g: FloatArray, p: FloatArray) -> float:
    return float(np.mean(np.abs(p - g) / g))


def _sqrel(g: FloatArray, p: FloatArray) -> float:
    return float(np.mean((p - g) ** 2 / g))


def _rmse_log(g: FloatArray, p: FloatArray) -> float:
    return float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))


def _delta1(g: FloatArray, p: FloatArray) -> float:
    return float(np.mean(np.maximum(p / g, g / p) < DELTA_THRESHOLD))


METRICS: dict[str, Callable[[FloatArray, FloatArray], float]] = {
    "mae": _mae,
    "rmse": _rmse,
    "absrel": _absrel,
    "sqrel": _sqrel,
    "rmse_log": _rmse_log,
    "delta1": _delta1,
}
HIGHER_IS_BETTER: frozenset[str] = frozenset({"delta1"})


def _check_gt(z_gt: FloatArray) -> None:
    if not np.all(np.isfinite(z_gt)) or np.any(z_gt <= 0):
        raise ValueError("z_gt must be finite and > 0 (data bug, not a missing estimate)")


def valid_prediction_mask(z_pred: FloatArray) -> BoolArray:
    """A prediction is usable iff it is finite and strictly positive."""
    return np.isfinite(z_pred) & (z_pred > 0)


def depth_metrics(z_gt: FloatArray, z_pred: FloatArray) -> dict[str, float]:
    """
    Metrics over rows with a valid prediction.

    Returns n (all rows), n_valid (rows with an estimate) and every metric in
    METRICS. With no valid row the metrics are NaN (never an exception).
    """
    g_all = np.asarray(z_gt, dtype=float)
    p_all = np.asarray(z_pred, dtype=float)
    if g_all.ndim != 1 or g_all.shape != p_all.shape:
        raise ValueError(f"z_gt and z_pred must be 1-D with equal shape, got {g_all.shape} vs {p_all.shape}")
    _check_gt(g_all)
    valid = valid_prediction_mask(p_all)
    g, p = g_all[valid], p_all[valid]
    out: dict[str, float] = {"n": int(g_all.size), "n_valid": int(valid.sum())}
    for name, fn in METRICS.items():
        out[name] = fn(g, p) if g.size else float("nan")
    return out


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def band_masks(z_gt: FloatArray) -> dict[str, BoolArray]:
    """Five right-open distance bands plus the merged '>30' row (Z >= 30)."""
    z = np.asarray(z_gt, dtype=float)
    masks: dict[str, BoolArray] = {
        label: (z >= lo) & (z < hi)
        for label, lo, hi in zip(BAND_LABELS, BAND_EDGES[:-1], BAND_EDGES[1:])
    }
    masks[FAR_MERGED_LABEL] = z >= FAR_MERGED_START
    return masks


def difficulty_masks(difficulty: pd.Series) -> dict[str, BoolArray]:
    """
    Cumulative KITTI subsets: Easy is a subset of Moderate, which is a subset of Hard.
    'Excluded' belongs to none of them.
    """
    d = difficulty.to_numpy()
    easy = d == "Easy"
    moderate = easy | (d == "Moderate")
    hard = moderate | (d == "Hard")
    return {"Easy": easy, "Moderate": moderate, "Hard": hard}


def class_masks(cls: pd.Series) -> dict[str, BoolArray]:
    values = cls.to_numpy()
    return {str(c): values == c for c in sorted(pd.unique(values))}


def evaluate_groups(
    df: pd.DataFrame,
    masks: Mapping[str, Mapping[str, BoolArray]],
    *,
    gt_col: str = "z_gt",
    pred_col: str = "z_pred",
) -> pd.DataFrame:
    """One row per (group_type, group): n, n_valid, valid_frac, low_n, then all metrics."""
    z_gt = df[gt_col].to_numpy(dtype=float)
    z_pred = df[pred_col].to_numpy(dtype=float)
    _check_gt(z_gt)
    rows: list[dict[str, Any]] = []
    for group_type, groups in masks.items():
        for name, m in groups.items():
            met = depth_metrics(z_gt[m], z_pred[m])
            n, n_valid = met.pop("n"), met.pop("n_valid")
            rows.append({
                "group_type": group_type,
                "group": name,
                "n": n,
                "n_valid": n_valid,
                "valid_frac": n_valid / n if n else float("nan"),
                "low_n": n_valid < MIN_N,
                **met,
            })
    return pd.DataFrame(rows)


def evaluate_report(
    df: pd.DataFrame,
    *,
    gt_col: str = "z_gt",
    pred_col: str = "z_pred",
    cls_col: str = "cls",
    difficulty_col: str = "difficulty",
) -> pd.DataFrame:
    """Overall + distance bands (+ '>30') + class + difficulty (the last two if present)."""
    z = df[gt_col].to_numpy(dtype=float)
    masks: dict[str, dict[str, BoolArray]] = {
        "overall": {"all": np.ones(len(df), dtype=bool)},
        "band": band_masks(z),
    }
    if cls_col in df.columns:
        masks["class"] = class_masks(df[cls_col])
    if difficulty_col in df.columns:
        masks["difficulty"] = difficulty_masks(df[difficulty_col])
    return evaluate_groups(df, masks, gt_col=gt_col, pred_col=pred_col)


# ---------------------------------------------------------------------------
# Cluster bootstrap (resample drives, not rows)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    n_rows: int
    n_clusters: int
    n_boot: int
    seed: int
    alpha: float

    @property
    def excludes_zero(self) -> bool:
        """For paired differences: 'statistically significant' only if the CI excludes 0 (v4 §6)."""
        return self.ci_low > 0 or self.ci_high < 0


def cluster_row_groups(clusters: NDArray[Any]) -> list[IntArray]:
    """Row indices of each cluster, in a deterministic (sorted-cluster) order."""
    order = np.argsort(clusters, kind="stable")
    _, starts = np.unique(clusters[order], return_index=True)
    return np.split(order, starts[1:])


def iter_cluster_resamples(
    groups: list[IntArray], n_boot: int, rng: np.random.Generator
) -> Iterator[IntArray]:
    """Yield n_boot row-index arrays, each built from whole clusters drawn with replacement."""
    k = len(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, k, size=k)
        yield np.concatenate([groups[i] for i in pick])


def _prepare(
    df: pd.DataFrame, gt_col: str, pred_cols: list[str], cluster_col: str
) -> tuple[FloatArray, list[FloatArray], NDArray[Any]]:
    """Keep rows where every listed prediction is valid (common support)."""
    g = df[gt_col].to_numpy(dtype=float)
    _check_gt(g)
    preds = [df[c].to_numpy(dtype=float) for c in pred_cols]
    keep = np.ones(len(df), dtype=bool)
    for p in preds:
        keep &= valid_prediction_mask(p)
    clusters = df[cluster_col].to_numpy()[keep]
    return g[keep], [p[keep] for p in preds], clusters


def _run_bootstrap(
    stat: Callable[[IntArray], float],
    point: float,
    clusters: NDArray[Any],
    *,
    metric: str,
    n_boot: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    groups = cluster_row_groups(clusters)
    if len(groups) < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters (drives)")
    if len(groups) < MIN_CLUSTERS_WARN:
        warnings.warn(
            f"Only {len(groups)} clusters: the bootstrap CI is coarse and likely too narrow.",
            UserWarning,
            stacklevel=3,
        )
    rng = np.random.default_rng(seed)
    stats = np.array([stat(idx) for idx in iter_cluster_resamples(groups, n_boot, rng)])
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(metric, float(point), float(lo), float(hi), int(clusters.size),
                           len(groups), n_boot, seed, alpha)


def cluster_bootstrap(
    df: pd.DataFrame,
    metric: str = "absrel",
    *,
    seed: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    gt_col: str = "z_gt",
    pred_col: str = "z_pred",
    cluster_col: str = "drive",
) -> BootstrapResult:
    """
    CI of one metric. Filter df to the subgroup of interest BEFORE calling.
    Rows without a valid prediction are dropped first (report n_valid separately).
    """
    fn = METRICS[metric]
    g, (p,), clusters = _prepare(df, gt_col, [pred_col], cluster_col)
    return _run_bootstrap(lambda idx: fn(g[idx], p[idx]), fn(g, p), clusters,
                          metric=metric, n_boot=n_boot, seed=seed, alpha=alpha)


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    pred_col_a: str,
    pred_col_b: str,
    metric: str = "absrel",
    *,
    seed: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    gt_col: str = "z_gt",
    cluster_col: str = "drive",
) -> BootstrapResult:
    """
    CI of metric(A) - metric(B) on the common support (rows where both A and B
    have an estimate), with the SAME resampled drives for both methods.
    For error metrics a negative difference means A is better; for delta1 the sign flips.
    """
    fn = METRICS[metric]
    g, (pa, pb), clusters = _prepare(df, gt_col, [pred_col_a, pred_col_b], cluster_col)
    return _run_bootstrap(lambda idx: fn(g[idx], pa[idx]) - fn(g[idx], pb[idx]),
                          fn(g, pa) - fn(g, pb), clusters,
                          metric=f"{metric}[{pred_col_a} - {pred_col_b}]",
                          n_boot=n_boot, seed=seed, alpha=alpha)


# ---------------------------------------------------------------------------
# Logging (seed + split hash on every evaluation run)
# ---------------------------------------------------------------------------
def make_log_record(
    *,
    split: str,
    split_hash: str,
    seed: int,
    n_boot: int,
    tag: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSONL record. split_hash comes from splits/split_metadata.json."""
    record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "split": split,
        "split_hash": split_hash,
        "seed": seed,
        "n_boot": n_boot,
        "tag": tag,
    }
    if extra:
        record.update(extra)
    return record


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Sign test (exact binomial test, one-sided)
# ---------------------------------------------------------------------------
def sign_test_one_sided(wins: int, losses: int) -> float:
    """
    One-sided sign test (H1: wins > losses) ignoring ties, using exact binomial test:
    scipy.stats.binomtest(wins, wins + losses, 0.5, alternative="greater").
    """
    if wins < 0 or losses < 0:
        raise ValueError(f"wins and losses must be non-negative, got wins={wins}, losses={losses}")
    n = wins + losses
    if n == 0:
        return 1.0
    from scipy.stats import binomtest

    res = binomtest(wins, n, p=0.5, alternative="greater")
    return float(res.pvalue)


# ---------------------------------------------------------------------------
# Macro average across clusters / drives (D18, D20)
# ---------------------------------------------------------------------------
def macro_by_cluster(
    df: pd.DataFrame,
    metric: str = "absrel",
    *,
    gt_col: str = "z_gt",
    pred_col: str = "z_pred",
    cluster_col: str = "drive",
    min_valid: int = 1,
) -> float:
    """
    Unweighted macro average of a metric across unique clusters (drives).

    Only clusters having at least `min_valid` valid predictions are included.
    Returns NaN if no cluster has at least `min_valid` valid predictions.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Expected one of {list(METRICS.keys())}")

    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found in dataframe.")

    fn = METRICS[metric]
    cluster_values = []

    for _, group in df.groupby(cluster_col, sort=False):
        z_gt = group[gt_col].to_numpy(dtype=float)
        z_pred = group[pred_col].to_numpy(dtype=float)

        valid = valid_prediction_mask(z_pred)
        if valid.sum() >= min_valid:
            _check_gt(z_gt[valid])
            cluster_values.append(fn(z_gt[valid], z_pred[valid]))

    if not cluster_values:
        return float("nan")

    return float(np.mean(cluster_values))

