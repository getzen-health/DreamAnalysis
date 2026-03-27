"""Tests for CORAL domain adaptation (processing/domain_adaptation.py).

Covers: fit/transform shape, covariance alignment, save/load round-trip,
error handling (transform-before-fit, dimension mismatch), single-feature
edge case, and the identity case (source == target).

GitHub issue: #541
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from processing.domain_adaptation import CORALAdapter


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_source(n_samples: int = 200, n_features: int = 41, seed: int = 0) -> np.ndarray:
    """Simulate research EEG features (e.g. DEAP, 32-channel gel electrodes)."""
    rng = np.random.RandomState(seed)
    mean = rng.randn(n_features) * 3.0
    cov = rng.randn(n_features, n_features)
    cov = cov @ cov.T / n_features + np.eye(n_features) * 0.1
    return rng.multivariate_normal(mean, cov, size=n_samples)


def _make_target(n_samples: int = 100, n_features: int = 41, seed: int = 42) -> np.ndarray:
    """Simulate consumer Muse 2 features (shifted + scaled vs source)."""
    rng = np.random.RandomState(seed)
    mean = rng.randn(n_features) * 2.0 + 5.0  # shifted
    cov = rng.randn(n_features, n_features)
    cov = cov @ cov.T / n_features + np.eye(n_features) * 0.2
    return rng.multivariate_normal(mean, cov, size=n_samples)


def _cov_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm between two covariance matrices."""
    return float(np.linalg.norm(np.cov(A.T) - np.cov(B.T), "fro"))


# ── Tests ────────────────────────────────────────────────────────────────────


class TestFitTransformShape:
    """fit + transform should preserve array dimensions."""

    def test_batch_shape_preserved(self):
        source = _make_source()
        target = _make_target()
        adapter = CORALAdapter().fit(source, target)
        aligned = adapter.transform(source)
        assert aligned.shape == source.shape

    def test_single_sample_shape(self):
        source = _make_source()
        target = _make_target()
        adapter = CORALAdapter().fit(source, target)
        single = source[0]  # (n_features,)
        aligned = adapter.transform(single)
        assert aligned.shape == single.shape

    def test_small_batch(self):
        source = _make_source(n_samples=5)
        target = _make_target(n_samples=5)
        adapter = CORALAdapter().fit(source, target)
        aligned = adapter.transform(source[:3])
        assert aligned.shape == (3, source.shape[1])


class TestCovarianceAlignment:
    """After CORAL, transformed source covariance should be closer to target."""

    def test_covariance_closer_to_target(self):
        source = _make_source(n_samples=300)
        target = _make_target(n_samples=300)
        adapter = CORALAdapter().fit(source, target)
        aligned = adapter.transform(source)

        dist_before = _cov_distance(source, target)
        dist_after = _cov_distance(aligned, target)
        assert dist_after < dist_before, (
            f"CORAL should bring source closer to target covariance: "
            f"before={dist_before:.2f}, after={dist_after:.2f}"
        )

    def test_mean_closer_to_target(self):
        source = _make_source(n_samples=300)
        target = _make_target(n_samples=300)
        adapter = CORALAdapter().fit(source, target)
        aligned = adapter.transform(source)

        mean_dist_before = np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))
        mean_dist_after = np.linalg.norm(aligned.mean(axis=0) - target.mean(axis=0))
        assert mean_dist_after < mean_dist_before


class TestSaveLoadRoundTrip:
    """save() then load() should reproduce identical transform results."""

    def test_roundtrip(self):
        source = _make_source()
        target = _make_target()
        adapter = CORALAdapter().fit(source, target)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "coral_adapter.npz")
            adapter.save(path)
            loaded = CORALAdapter.load(path)

            assert loaded.fitted is True
            np.testing.assert_array_almost_equal(
                loaded.source_mean, adapter.source_mean
            )
            np.testing.assert_array_almost_equal(
                loaded.source_cov_whitening, adapter.source_cov_whitening
            )
            np.testing.assert_array_almost_equal(
                loaded.target_mean, adapter.target_mean
            )
            np.testing.assert_array_almost_equal(
                loaded.target_cov_coloring, adapter.target_cov_coloring
            )

            # Transform output matches
            original = adapter.transform(source[:5])
            roundtripped = loaded.transform(source[:5])
            np.testing.assert_array_almost_equal(original, roundtripped)

    def test_save_unfitted_raises(self):
        adapter = CORALAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bad.npz")
            with pytest.raises(ValueError, match="unfitted"):
                adapter.save(path)


class TestErrorHandling:
    """Proper errors for misuse."""

    def test_transform_before_fit_raises(self):
        adapter = CORALAdapter()
        with pytest.raises(ValueError, match="fit"):
            adapter.transform(np.random.randn(10, 5))

    def test_dimension_mismatch_raises(self):
        source = _make_source(n_features=10)
        target = _make_target(n_features=20)
        adapter = CORALAdapter()
        with pytest.raises(ValueError, match="mismatch"):
            adapter.fit(source, target)

    def test_empty_source_raises(self):
        adapter = CORALAdapter()
        with pytest.raises(ValueError, match="non-empty"):
            adapter.fit(np.empty((0, 10)), _make_target(n_features=10))

    def test_empty_target_raises(self):
        adapter = CORALAdapter()
        with pytest.raises(ValueError, match="non-empty"):
            adapter.fit(_make_source(n_features=10), np.empty((0, 10)))


class TestSingleFeature:
    """Edge case: 1-dimensional features (n_features=1)."""

    def test_single_feature_fit_transform(self):
        rng = np.random.RandomState(99)
        source = rng.randn(50, 1) * 2.0 + 3.0
        target = rng.randn(50, 1) * 0.5 + 1.0
        adapter = CORALAdapter().fit(source, target)
        aligned = adapter.transform(source)
        assert aligned.shape == source.shape

        # Mean should shift toward target
        mean_dist_before = abs(source.mean() - target.mean())
        mean_dist_after = abs(aligned.mean() - target.mean())
        assert mean_dist_after < mean_dist_before


class TestIdentityCase:
    """When source == target, transform should be near-identity."""

    def test_same_distribution_near_identity(self):
        data = _make_source(n_samples=500, seed=7)
        adapter = CORALAdapter().fit(data, data)
        aligned = adapter.transform(data)
        # Should be very close to original
        np.testing.assert_allclose(aligned, data, atol=1e-6)

    def test_same_data_exact_near_identity(self):
        """Identical data in source and target => transform ~ identity."""
        data = _make_source(n_samples=200, n_features=10, seed=3)
        adapter = CORALAdapter().fit(data, data)
        aligned = adapter.transform(data[:10])
        np.testing.assert_allclose(aligned, data[:10], atol=1e-6)


class TestRegularization:
    """Custom regularization parameter."""

    def test_custom_reg(self):
        source = _make_source(n_features=5)
        target = _make_target(n_features=5)
        adapter = CORALAdapter(reg=1e-3).fit(source, target)
        aligned = adapter.transform(source)
        assert aligned.shape == source.shape
        assert adapter.reg == 1e-3
