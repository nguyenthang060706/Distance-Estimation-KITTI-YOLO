"""
tests/test_geometry.py: Unit tests for geometric cues (§5.2) and log-space fusion.

Tests:
    1. Roundtrip: create bbox from known Z, recover Z from cue — must match.
    2. Boundary masking: correct cues masked for each border.
    3. Weights sum to 1.
    4. NaN does not propagate to valid cues.
    5. Fusion with known covariance recovers expected weighted average.
    6. NNLS fallback when covariance induces negative weights.
    7. Single-cue and two-cue partial fusion.
    8. Sub-matrix consistency: fusing subset uses correct Σ_S.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.geometric_cues import (
    compute_Z_w, compute_Z_h, compute_Z_g,
    compute_cues_batch,
    GeometricPriors, CameraIntrinsics, CueResult,
    BORDER_EPS,
    load_geometry_v2, load_priors_from_yaml,
)
from src.geometry.fusion import (
    fit_fusion_weights, fuse_depths, fuse_depths_vectorised,
    fuse_depths_lodo,
    FusionWeights, _optimal_weights, _nnls_weights,
    _ledoit_wolf_shrinkage,
    estimate_covariance_grouped_cv,
)


# ---------------------------------------------------------------------------
# Fixtures: typical KITTI-like camera parameters
# ---------------------------------------------------------------------------
FX = 721.5
FY = 721.5
CX = 609.5
CY = 172.8
IMG_W = 1242
IMG_H = 375
W_EFF = 2.618   # Car median width from Split A
H_OBJ = 1.680   # Car median height from Split A
H_CAM = 1.886   # Camera height from Split A
Y_HORIZON = CY   # horizon ≈ cy


@pytest.fixture
def intrinsics():
    return CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY)


@pytest.fixture
def priors():
    return GeometricPriors(W_eff=W_EFF, H_obj=H_OBJ, H_cam=H_CAM, y_horizon=Y_HORIZON)


# ---------------------------------------------------------------------------
# 1. Roundtrip tests: known Z → bbox → cue → recovered Z must match
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Create bboxes from a known depth, then verify cue recovery."""

    def test_Z_w_roundtrip(self):
        """Z_w = fx * W_eff / w ⟹ w = fx * W_eff / Z_true."""
        Z_true = 30.0
        w = FX * W_EFF / Z_true
        # Place bbox centred, away from borders
        cx_img = IMG_W / 2
        x1 = cx_img - w / 2
        x2 = cx_img + w / 2
        bbox = np.array([x1, 50.0, x2, 200.0])  # y values don't matter for Z_w

        Z_est, valid = compute_Z_w(bbox, FX, W_EFF, IMG_W)
        assert valid
        np.testing.assert_allclose(Z_est, Z_true, rtol=1e-10)

    def test_Z_h_roundtrip(self):
        """Z_h = fy * H_obj / h ⟹ h = fy * H_obj / Z_true."""
        Z_true = 25.0
        h = FY * H_OBJ / Z_true
        # Place bbox centred vertically, away from borders
        y_mid = IMG_H / 2
        y1 = y_mid - h / 2
        y2 = y_mid + h / 2
        bbox = np.array([300.0, y1, 400.0, y2])

        Z_est, valid = compute_Z_h(bbox, FY, H_OBJ, IMG_H)
        assert valid
        np.testing.assert_allclose(Z_est, Z_true, rtol=1e-10)

    def test_Z_g_roundtrip(self):
        """Z_g = fy * H_cam / (y_bottom - y_horizon)."""
        Z_true = 40.0
        y_bottom = Y_HORIZON + FY * H_CAM / Z_true
        # Ensure bbox is well inside image
        assert y_bottom < IMG_H - BORDER_EPS - 1, "y_bottom outside image for this Z_true"
        y1 = y_bottom - 50  # some height
        assert y1 > BORDER_EPS, "y1 at border"
        bbox = np.array([300.0, y1, 400.0, y_bottom])

        Z_est, valid = compute_Z_g(bbox, FY, H_CAM, Y_HORIZON, IMG_H)
        assert valid
        np.testing.assert_allclose(Z_est, Z_true, rtol=1e-10)

    def test_batch_roundtrip(self, intrinsics, priors):
        """Batch computation matches individual cues."""
        Z_true = 20.0
        w = FX * W_EFF / Z_true
        h = FY * H_OBJ / Z_true
        y_bottom = Y_HORIZON + FY * H_CAM / Z_true

        # Construct bbox that satisfies all three cues
        x_mid = IMG_W / 2
        bbox = np.array([[x_mid - w / 2, y_bottom - h, x_mid + w / 2, y_bottom]])

        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert result.valid_w[0]
        assert result.valid_h[0]
        assert result.valid_g[0]
        np.testing.assert_allclose(result.Z_w[0], Z_true, rtol=1e-10)
        np.testing.assert_allclose(result.Z_h[0], Z_true, rtol=1e-10)
        np.testing.assert_allclose(result.Z_g[0], Z_true, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. Boundary masking tests
# ---------------------------------------------------------------------------

class TestBoundaryMask:
    """Verify that boundary touches invalidate the correct cues."""

    def test_left_border_masks_Z_w(self, intrinsics, priors):
        """Bbox touching left border → Z_w invalid, Z_h/Z_g may be valid."""
        bbox = np.array([[0.0, 100.0, 100.0, 250.0]])  # x1=0 touches left
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_w[0], "Z_w should be invalid (left border)"
        assert np.isnan(result.Z_w[0])
        # Z_h and Z_g should be valid (no top/bottom border touch)
        assert result.valid_h[0], "Z_h should still be valid"
        assert result.valid_g[0], "Z_g should still be valid"

    def test_right_border_masks_Z_w(self, intrinsics, priors):
        """Bbox touching right border → Z_w invalid."""
        bbox = np.array([[1100.0, 100.0, IMG_W - 1.0, 250.0]])
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_w[0], "Z_w should be invalid (right border)"
        assert np.isnan(result.Z_w[0])

    def test_top_border_masks_Z_h_and_Z_g(self, intrinsics, priors):
        """Bbox touching top border → Z_h, Z_g invalid; Z_w may be valid."""
        bbox = np.array([[300.0, 0.0, 500.0, 250.0]])  # y1=0 touches top
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_h[0], "Z_h should be invalid (top border)"
        assert not result.valid_g[0], "Z_g should be invalid (top border)"
        assert np.isnan(result.Z_h[0])
        assert np.isnan(result.Z_g[0])
        # Z_w should be valid (no left/right border touch)
        assert result.valid_w[0], "Z_w should still be valid"

    def test_bottom_border_masks_Z_h_and_Z_g(self, intrinsics, priors):
        """Bbox touching bottom border → Z_h, Z_g invalid."""
        bbox = np.array([[300.0, 100.0, 500.0, IMG_H - 1.0]])
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_h[0], "Z_h should be invalid (bottom border)"
        assert not result.valid_g[0], "Z_g should be invalid (bottom border)"
        assert result.valid_w[0], "Z_w should still be valid"

    def test_y_bottom_at_horizon_masks_Z_g(self, intrinsics, priors):
        """y_bottom ≤ y_horizon + ε → Z_g invalid, but Z_h valid."""
        y_bot = Y_HORIZON + BORDER_EPS  # exactly at threshold
        bbox = np.array([[300.0, 50.0, 500.0, y_bot]])
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_g[0], "Z_g should be invalid (at horizon)"
        assert np.isnan(result.Z_g[0])
        # Z_h should be valid (y values don't touch border)
        assert result.valid_h[0], "Z_h should still be valid"

    def test_y_bottom_above_horizon_masks_Z_g(self, intrinsics, priors):
        """y_bottom < y_horizon → Z_g invalid."""
        y_bot = Y_HORIZON - 10  # above horizon
        bbox = np.array([[300.0, 50.0, 500.0, y_bot]])
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_g[0]

    def test_all_borders_touched_all_invalid(self, intrinsics, priors):
        """Bbox touching all borders → all cues invalid."""
        bbox = np.array([[0.0, 0.0, IMG_W - 1.0, IMG_H - 1.0]])
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert not result.valid_w[0]
        assert not result.valid_h[0]
        assert not result.valid_g[0]
        assert np.all(np.isnan([result.Z_w[0], result.Z_h[0], result.Z_g[0]]))

    def test_non_square_resolution_1224_370(self, intrinsics, priors):
        """Verify boundary masking behaves correctly on 1224x370 KITTI images."""
        w_img, h_img = 1224, 370
        # 1. Bbox ending at x2=1222 touches right border of 1224 (1224 - 1 - 2 = 1221)
        bbox_touch_right = np.array([[1000.0, 50.0, 1222.0, 200.0]])
        res_right = compute_cues_batch(bbox_touch_right, intrinsics, priors, w_img, h_img)
        assert not res_right.valid_w[0], "x2=1222 must be masked on 1224-wide image"

        # On a 1242-wide image, x2=1222 is well within bounds (1242 - 1 - 2 = 1239)
        res_wide = compute_cues_batch(bbox_touch_right, intrinsics, priors, 1242, 375)
        assert res_wide.valid_w[0], "x2=1222 must be valid on 1242-wide image"

        # 2. Bbox ending at y2=368 touches bottom border of 370 (370 - 1 - 2 = 367)
        bbox_touch_bottom = np.array([[500.0, 100.0, 700.0, 368.0]])
        res_bottom = compute_cues_batch(bbox_touch_bottom, intrinsics, priors, w_img, h_img)
        assert not res_bottom.valid_h[0], "y2=368 must be masked on 370-tall image"
        assert not res_bottom.valid_g[0], "y2=368 must mask Z_g on 370-tall image"

        # On a 375-tall image, y2=368 is valid (375 - 1 - 2 = 372)
        res_tall = compute_cues_batch(bbox_touch_bottom, intrinsics, priors, 1242, 375)
        assert res_tall.valid_h[0], "y2=368 must be valid on 375-tall image"
        assert res_tall.valid_g[0], "y2=368 must be valid on 375-tall image"

    def test_non_square_resolution_1242_375(self, intrinsics, priors):
        """Verify boundary masking behaves correctly on 1242x375 KITTI images."""
        w_img, h_img = 1242, 375
        bbox_border = np.array([[1000.0, 50.0, 1240.5, 373.5]])
        res = compute_cues_batch(bbox_border, intrinsics, priors, w_img, h_img)
        assert not res.valid_w[0], "Touches right border on 1242"
        assert not res.valid_h[0], "Touches bottom border on 375"
        assert not res.valid_g[0], "Touches bottom border on 375"


# ---------------------------------------------------------------------------
# 3. Fusion weight tests
# ---------------------------------------------------------------------------

class TestFusionWeights:
    """Test optimal weight computation."""

    def test_weights_sum_to_one(self):
        """Optimal weights from any PD covariance sum to 1."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            # Generate random PD matrix
            A = rng.standard_normal((3, 3))
            sigma = A @ A.T + 0.1 * np.eye(3)
            w = _optimal_weights(sigma)
            np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-10)

    def test_equal_variance_equal_weights(self):
        """When all cues have equal variance and zero correlation: w_k = 1/K."""
        sigma = 2.0 * np.eye(3)
        w = _optimal_weights(sigma)
        np.testing.assert_allclose(w, np.array([1/3, 1/3, 1/3]), atol=1e-10)

    def test_unequal_variance_weights(self):
        """Cue with lower variance gets higher weight."""
        sigma = np.diag([1.0, 4.0, 9.0])
        w = _optimal_weights(sigma)
        # w_k ∝ 1/σ²_k for diagonal case
        expected = np.array([1/1, 1/4, 1/9])
        expected = expected / expected.sum()
        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_nnls_no_negative(self):
        """NNLS ensures all weights ≥ 0."""
        # Create a covariance with strong positive correlation
        # that might produce negative weights
        sigma = np.array([
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 10.0],
        ])
        w = _nnls_weights(sigma)
        assert np.all(w >= 0), f"Negative weights found: {w}"
        np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. NaN isolation tests
# ---------------------------------------------------------------------------

class TestNaNIsolation:
    """NaN in one cue must not contaminate others."""

    def test_nan_does_not_spread(self, intrinsics, priors):
        """One NaN cue should not make other cues NaN."""
        # Bbox touches left border (Z_w=NaN) but not top/bottom (Z_h, Z_g valid)
        bbox = np.array([[0.0, 100.0, 100.0, 250.0]])
        result = compute_cues_batch(bbox, intrinsics, priors, IMG_W, IMG_H)

        assert np.isnan(result.Z_w[0])
        assert not np.isnan(result.Z_h[0])
        assert not np.isnan(result.Z_g[0])

    def test_fusion_with_partial_nan(self):
        """Fusion with one NaN cue uses only valid cues."""
        # Set up a simple fusion weight
        sigma = np.eye(3)
        fw = FusionWeights(
            weights=np.array([1/3, 1/3, 1/3]),
            cov_matrix=sigma,
            cov_shrunk=sigma,
            shrinkage_alpha=0.0,
            cue_names=["Z_w", "Z_h", "Z_g"],
            constrained=False,
            n_samples=100,
            n_drives=10,
        )

        # Object 0: all valid; Object 1: Z_w = NaN
        Z_cues = np.array([
            [20.0, 22.0, 21.0],
            [np.nan, 22.0, 21.0],
        ])
        valid_mask = np.array([
            [True, True, True],
            [False, True, True],
        ])

        Z_d = fuse_depths(Z_cues, valid_mask, fw)

        # Object 0: geometric mean of [20, 22, 21]
        expected_0 = np.exp(np.mean(np.log([20.0, 22.0, 21.0])))
        np.testing.assert_allclose(Z_d[0], expected_0, rtol=1e-10)

        # Object 1: geometric mean of [22, 21] (equal weights for 2x2 identity sub-matrix)
        expected_1 = np.exp(np.mean(np.log([22.0, 21.0])))
        np.testing.assert_allclose(Z_d[1], expected_1, rtol=1e-10)

        # Neither should be NaN
        assert not np.isnan(Z_d[0])
        assert not np.isnan(Z_d[1])

    def test_all_nan_gives_nan(self):
        """When all cues are NaN, fused depth is NaN."""
        sigma = np.eye(3)
        fw = FusionWeights(
            weights=np.array([1/3, 1/3, 1/3]),
            cov_matrix=sigma,
            cov_shrunk=sigma,
            shrinkage_alpha=0.0,
            cue_names=["Z_w", "Z_h", "Z_g"],
            constrained=False,
            n_samples=100,
            n_drives=10,
        )

        Z_cues = np.array([[np.nan, np.nan, np.nan]])
        valid_mask = np.array([[False, False, False]])
        Z_d = fuse_depths(Z_cues, valid_mask, fw)
        assert np.isnan(Z_d[0])


# ---------------------------------------------------------------------------
# 5. Fusion integration test
# ---------------------------------------------------------------------------

class TestFusionIntegration:
    """Integration test: fit weights on synthetic data, then fuse."""

    def test_fit_and_fuse(self):
        """Fit fusion weights on synthetic data with known covariance."""
        rng = np.random.default_rng(42)
        N = 500
        n_drives = 10

        # Ground truth depths
        Z_gt = rng.uniform(10, 60, N)

        # True error covariance in log space
        true_sigma = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.03],
        ])

        # Generate log errors
        log_errors = rng.multivariate_normal(np.zeros(3), true_sigma, N)

        # Cue depths
        Z_cues = np.column_stack([
            Z_gt * np.exp(log_errors[:, 0]),
            Z_gt * np.exp(log_errors[:, 1]),
            Z_gt * np.exp(log_errors[:, 2]),
        ])

        valid_mask = np.ones((N, 3), dtype=bool)
        drive_ids = np.array([f"drive_{i % n_drives}" for i in range(N)])

        fw = fit_fusion_weights(Z_cues, Z_gt, valid_mask, drive_ids)

        # Weights should sum to 1
        np.testing.assert_allclose(fw.weights.sum(), 1.0, atol=1e-10)

        # All weights should be positive for this well-behaved case
        assert np.all(fw.weights >= 0), f"Negative weight: {fw.weights}"

        # Fuse
        Z_d = fuse_depths_vectorised(Z_cues, valid_mask, fw)

        # Fused depth should be closer to GT than any single cue
        abs_rel_fused = np.mean(np.abs(Z_d - Z_gt) / Z_gt)
        for k in range(3):
            abs_rel_k = np.mean(np.abs(Z_cues[:, k] - Z_gt) / Z_gt)
            # Fused should be no worse (allow small tolerance for sampling)
            assert abs_rel_fused <= abs_rel_k + 0.02, (
                f"Fused AbsRel {abs_rel_fused:.4f} worse than cue {k}: {abs_rel_k:.4f}"
            )


# ---------------------------------------------------------------------------
# 6. Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------

class TestShrinkage:
    """Test Ledoit-Wolf shrinkage properties."""

    def test_shrinkage_returns_pd(self):
        """Shrunk matrix should be positive definite."""
        S = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0],
        ])
        S_shrunk, alpha = _ledoit_wolf_shrinkage(S, 50)
        eigenvalues = np.linalg.eigvalsh(S_shrunk)
        assert np.all(eigenvalues > 0), f"Non-PD: eigenvalues = {eigenvalues}"
        assert 0 <= alpha <= 1

    def test_identity_no_shrinkage(self):
        """Identity matrix should not be changed much by shrinkage."""
        S = np.eye(3)
        S_shrunk, alpha = _ledoit_wolf_shrinkage(S, 1000)
        np.testing.assert_allclose(S_shrunk, S, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Single-cue fusion
# ---------------------------------------------------------------------------

class TestSingleCue:
    """When only one cue is valid, fusion returns that cue's value."""

    def test_single_valid_cue(self):
        sigma = np.eye(3)
        fw = FusionWeights(
            weights=np.array([1/3, 1/3, 1/3]),
            cov_matrix=sigma,
            cov_shrunk=sigma,
            shrinkage_alpha=0.0,
            cue_names=["Z_w", "Z_h", "Z_g"],
            constrained=False,
            n_samples=100,
            n_drives=10,
        )

        Z_cues = np.array([[np.nan, 25.0, np.nan]])
        valid_mask = np.array([[False, True, False]])
        Z_d = fuse_depths(Z_cues, valid_mask, fw)
        np.testing.assert_allclose(Z_d[0], 25.0)


# ---------------------------------------------------------------------------
# 8. Vectorised vs loop consistency
# ---------------------------------------------------------------------------

class TestVectorisedConsistency:
    """fuse_depths and fuse_depths_vectorised should give identical results."""

    def test_same_output(self):
        rng = np.random.default_rng(123)
        N = 50

        sigma = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.03],
        ])
        w = _optimal_weights(sigma)
        fw = FusionWeights(
            weights=w,
            cov_matrix=sigma,
            cov_shrunk=sigma,
            shrinkage_alpha=0.0,
            cue_names=["Z_w", "Z_h", "Z_g"],
            constrained=False,
            n_samples=100,
            n_drives=10,
        )

        Z_cues = rng.uniform(10, 60, (N, 3))
        # Randomly invalidate some cues
        valid_mask = rng.random((N, 3)) > 0.2
        Z_cues[~valid_mask] = np.nan

        Z_d_loop = fuse_depths(Z_cues, valid_mask, fw)
        Z_d_vec = fuse_depths_vectorised(Z_cues, valid_mask, fw)

        # Where both are valid, they should match
        both_valid = ~np.isnan(Z_d_loop) & ~np.isnan(Z_d_vec)
        if np.any(both_valid):
            np.testing.assert_allclose(
                Z_d_loop[both_valid], Z_d_vec[both_valid], rtol=1e-10
            )

        # Where both are NaN, they should agree
        np.testing.assert_array_equal(np.isnan(Z_d_loop), np.isnan(Z_d_vec))


class TestGeometryLoaders:
    """Tests for geometry configuration loaders (Decision D10/D14)."""

    def test_load_geometry_v2(self):
        """load_geometry_v2 must load calibrated effective parameters from geometry_params.yaml."""
        priors, cfg = load_geometry_v2("configs/geometry_params.yaml")
        assert cfg["metadata"]["tag"] == "geometry-v2"
        assert priors.W_eff == pytest.approx(2.6184, rel=1e-3)
        assert priors.H_obj == pytest.approx(1.6797, rel=1e-3)
        assert priors.H_cam == pytest.approx(2.0422, rel=1e-3)
        assert priors.delta_horizon == pytest.approx(-4.6782, rel=1e-3)
        assert "fusion_split_B" in cfg
        assert cfg["fusion_split_B"]["weights"] == [0.0807, 0.6634, 0.2560]

    def test_load_priors_from_yaml_deprecated_warning(self):
        """load_priors_from_yaml should trigger a DeprecationWarning."""
        with pytest.deprecated_call():
            priors, meta = load_priors_from_yaml("configs/geometry_priors.yaml")
            assert priors.H_cam == pytest.approx(1.886, rel=1e-3)


class TestLODOFusion:
    """Tests for Leave-One-Drive-Out (LODO) fusion (Decision D16c)."""

    def test_lodo_invariance(self):
        """
        Modifying Z_gt of drive d alone MUST NOT change Z_d of drive d,
        while Z_d of all other drives MUST change.
        """
        rng = np.random.default_rng(42)
        n_per_drive = 20
        drives = np.repeat([f"drive_{i:02d}" for i in range(5)], n_per_drive)
        N = len(drives)
        Z_gt = rng.uniform(10, 50, N)
        # 3 cues with correlated noise
        cues = np.column_stack([
            Z_gt * np.exp(rng.normal(0, 0.1, N)),
            Z_gt * np.exp(rng.normal(0, 0.05, N)),
            Z_gt * np.exp(rng.normal(0, 0.08, N)),
        ])
        valid_mask = np.ones((N, 3), dtype=bool)

        Z_d_orig, fits_orig = fuse_depths_lodo(cues, Z_gt, valid_mask, drives, min_train_drives=3)

        # Now perturb Z_gt of drive_00 only
        target_drive = "drive_00"
        mask_target = (drives == target_drive)
        Z_gt_perturbed = Z_gt.copy()
        Z_gt_perturbed[mask_target] *= 1.5

        Z_d_pert, fits_pert = fuse_depths_lodo(cues, Z_gt_perturbed, valid_mask, drives, min_train_drives=3)

        # Z_d for target_drive must be IDENTICAL (its weights were trained on ~mask_target where Z_gt was unchanged)
        np.testing.assert_allclose(Z_d_pert[mask_target], Z_d_orig[mask_target], rtol=1e-12)

        # Z_d for other drives must DIFFER (their training sets included target_drive whose Z_gt changed)
        diff_other = np.abs(Z_d_pert[~mask_target] - Z_d_orig[~mask_target])
        assert np.any(diff_other > 1e-4), "Z_d for other drives should have changed when drive_00 Z_gt was perturbed"

    def test_lodo_too_few_drives_raises(self):
        """When total drives < min_train_drives + 1, must raise ValueError."""
        drives = np.array(["d1", "d1", "d2", "d2"])
        cues = np.ones((4, 3)) * 20.0
        gt = np.ones(4) * 20.0
        mask = np.ones((4, 3), dtype=bool)
        with pytest.raises(ValueError, match="Too few training drives"):
            fuse_depths_lodo(cues, gt, mask, drives, min_train_drives=3)

    def test_fit_fusion_weights_insufficient_complete_cases_raises(self):
        """fit_fusion_weights must raise ValueError when complete cases < K+1."""
        cues = np.ones((2, 3)) * 20.0
        gt = np.ones(2) * 20.0
        mask = np.ones((2, 3), dtype=bool)
        drives = np.array(["d1", "d2"])
        with pytest.raises(ValueError, match="Insufficient complete cases"):
            fit_fusion_weights(cues, gt, mask, drives)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

