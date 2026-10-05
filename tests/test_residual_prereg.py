"""Unit tests for residual pre-registration configuration (D25, D33, D34).

Verifies the integrity and constraints of configs/residual/residual_prereg_v1.yaml:
- Exact feature counts for (f), (f0), (e)
- Exclusion of class_id (constant 0 for Car)
- Exclusion of fallback_flag (metadata only, D28)
- Model (e) feature isolation (no cues, no validity flags, no ln_z_base)
- XGBoost grid search size <= 24 configs and max_depth <= 4
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "residual" / "residual_prereg_v1.yaml"
OLD_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "residual" / "residual_config.yaml"


@pytest.fixture(scope="module")
def prereg_cfg() -> dict:
    assert CONFIG_PATH.exists(), f"Configuration file not found: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "Parsed config must be a dictionary"
    return cfg


def test_residual_prereg_feature_counts(prereg_cfg: dict):
    """Verify feature counts match pre-registration specification."""
    f_features = prereg_cfg["model_f"]["features"]
    f0_features = prereg_cfg["model_f0"]["features"]
    e_features = prereg_cfg["model_e"]["features"]

    assert isinstance(f_features, list), "model_f.features must be a list"
    assert isinstance(f0_features, list), "model_f0.features must be a list"
    assert isinstance(e_features, list), "model_e.features must be a list"

    assert len(f_features) == 17, f"Expected 17 features for model (f), got {len(f_features)}: {f_features}"
    assert len(f0_features) == 5, f"Expected 5 features for model (f0), got {len(f0_features)}: {f0_features}"
    assert len(e_features) == 10, f"Expected 10 features for model (e), got {len(e_features)}: {e_features}"


def test_class_id_excluded_from_all_models(prereg_cfg: dict):
    """class_id is constant (0) for Car-only Hard population and must be excluded."""
    for model_name in ["model_f", "model_f0", "model_e"]:
        feats = prereg_cfg[model_name]["features"]
        assert "class_id" not in feats, f"class_id found in {model_name}.features: {feats}"


def test_fallback_flag_excluded_from_features(prereg_cfg: dict):
    """fallback_flag is metadata only (D28) and must never be in any feature matrix."""
    for model_name in ["model_f", "model_f0", "model_e"]:
        feats = prereg_cfg[model_name]["features"]
        assert "fallback_flag" not in feats, f"fallback_flag found in {model_name}.features: {feats}"


def test_model_e_feature_isolation(prereg_cfg: dict):
    """Model (e) direct regression must NOT see any cues, validity flags, or ln_z_base."""
    forbidden_for_e = {
        "ln_z_w", "ln_z_h", "ln_z_g",
        "valid_w", "valid_h", "valid_g",
        "ln_z_base",
    }
    e_feats = set(prereg_cfg["model_e"]["features"])
    overlap = e_feats.intersection(forbidden_for_e)
    assert not overlap, f"Model (e) contains forbidden cue/validity/base features: {overlap}"


def test_grid_search_constraints(prereg_cfg: dict):
    """XGBoost grid search must be small (<= 24 configs) and shallow (depth <= 4) (D25)."""
    grid = prereg_cfg["model_f"]["grid"]
    assert isinstance(grid, dict), "model_f.grid must be a dict"

    # Calculate total grid combinations
    total_configs = math.prod(len(v) if isinstance(v, list) else 1 for v in grid.values())
    assert total_configs <= 24, f"Grid has {total_configs} configs, exceeding max 24 allowed (D25)"
    assert total_configs == 12, f"Expected exactly 12 configs for prereg-residual-v1, got {total_configs}"

    # Check max depth constraints
    depths = grid.get("max_depth", [])
    if isinstance(depths, list):
        for d in depths:
            assert d <= 4, f"Grid max_depth {d} exceeds limit 4 (D25)"
    elif isinstance(depths, int):
        assert depths <= 4

    e_depth = prereg_cfg["model_e"].get("max_depth", 0)
    assert e_depth <= 4, f"Model (e) max_depth {e_depth} exceeds limit 4"


def test_f0_ridge_hyperparameters(prereg_cfg: dict):
    """Model (f0) baseline linear Ridge hyperparameters specification."""
    f0 = prereg_cfg["model_f0"]
    assert f0["type"] == "ridge"
    assert f0.get("preprocessing") == "StandardScaler"
    assert isinstance(f0.get("alpha_grid"), list) and len(f0["alpha_grid"]) > 0
    assert "ln_z_base" in f0["features"], "f0 must include ln_z_base (D30)"


def test_inner_cv_specification(prereg_cfg: dict):
    """Inner CV must use GroupKFold grouped by drive (D24, D25)."""
    inner_cv = prereg_cfg["inner_cv"]
    assert inner_cv["method"] == "GroupKFold"
    assert inner_cv["group_col"] == "drive"
    assert inner_cv["n_splits"] <= 5


def test_old_config_marked_superseded():
    """Verify residual_config.yaml is marked SUPERSEDED."""
    assert OLD_CONFIG_PATH.exists()
    with open(OLD_CONFIG_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert "SUPERSEDED" in first_line, f"Old config first line lacks SUPERSEDED notice: {first_line}"
