"""Tests for cross-subject domain adaptation module."""
import numpy as np
import pytest

from models.domain_adapter import DomainAdapter


@pytest.fixture
def adapter():
    return DomainAdapter()


def _source_stats(n_features=20):
    """Population-average source domain statistics."""
    np.random.seed(0)
    mean = np.random.randn(n_features) * 2.0
    std = np.abs(np.random.randn(n_features)) + 0.5
    return mean, std


def _target_features(n_samples=50, n_features=20, shift=3.0, scale=2.0, seed=42):
    """Simulate a target user whose features are shifted/scaled from source."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features) * scale + shift


class TestInitialization:
    def test_not_calibrated_at_start(self, adapter):
        assert adapter.is_calibrated() is False

    def test_get_stats_empty(self, adapter):
        stats = adapter.get_stats()
        assert stats["source_set"] is False
        assert stats["target_calibrated"] is False
        assert stats["n_target_samples"] == 0

    def test_default_min_samples(self, adapter):
        stats = adapter.get_stats()
        assert stats["min_samples"] == 10

    def test_custom_min_samples(self):
        adapter = DomainAdapter(min_calibration_samples=5)
        stats = adapter.get_stats()
        assert stats["min_samples"] == 5


class TestSetSourceStats:
    def test_set_source_stats(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        stats = adapter.get_stats()
        assert stats["source_set"] is True
        assert stats["n_features"] == 20

    def test_set_source_from_lists(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean.tolist(), std.tolist())
        stats = adapter.get_stats()
        assert stats["source_set"] is True

    def test_set_source_mismatched_lengths(self, adapter):
        with pytest.raises(ValueError, match="same length"):
            adapter.set_source_stats(np.zeros(10), np.ones(20))

    def test_set_source_zero_std_clipped(self, adapter):
        """Zero std values should be clipped to a small epsilon."""
        mean = np.zeros(5)
        std = np.array([1.0, 0.0, 0.0, 2.0, 0.0])
        adapter.set_source_stats(mean, std)
        stats = adapter.get_stats()
        assert stats["source_set"] is True
        # Should not raise on adapt later

    def test_set_source_negative_std_clipped(self, adapter):
        """Negative std values should be handled gracefully."""
        mean = np.zeros(5)
        std = np.array([1.0, -0.5, 0.3, 2.0, -1.0])
        adapter.set_source_stats(mean, std)
        stats = adapter.get_stats()
        assert stats["source_set"] is True

    def test_set_source_1d_arrays(self, adapter):
        adapter.set_source_stats(np.zeros(10), np.ones(10))
        stats = adapter.get_stats()
        assert stats["n_features"] == 10


class TestCalibrate:
    def test_calibrate_with_enough_samples(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        result = adapter.calibrate(target)
        assert result["calibrated"] is True
        assert result["n_samples"] == 15
        assert adapter.is_calibrated() is True

    def test_calibrate_insufficient_samples(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=5)
        result = adapter.calibrate(target)
        assert result["calibrated"] is False
        assert adapter.is_calibrated() is False

    def test_calibrate_without_source_raises(self, adapter):
        with pytest.raises(ValueError, match="source"):
            adapter.calibrate(_target_features())

    def test_calibrate_wrong_feature_count(self, adapter):
        mean, std = _source_stats(n_features=20)
        adapter.set_source_stats(mean, std)
        with pytest.raises(ValueError, match="features"):
            adapter.calibrate(_target_features(n_features=10))

    def test_calibrate_single_sample_1d(self, adapter):
        """1D input should be treated as a single sample."""
        adapter_low = DomainAdapter(min_calibration_samples=1)
        mean, std = _source_stats(n_features=20)
        adapter_low.set_source_stats(mean, std)
        single = np.random.randn(20)
        result = adapter_low.calibrate(single)
        assert result["n_samples"] == 1
        assert result["calibrated"] is True

    def test_calibrate_returns_alignment_before(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        result = adapter.calibrate(target)
        assert "alignment_before" in result
        assert 0.0 <= result["alignment_before"] <= 1.0

    def test_calibrate_returns_alignment_after(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        result = adapter.calibrate(target)
        assert "alignment_after" in result
        assert 0.0 <= result["alignment_after"] <= 1.0

    def test_alignment_improves_after_calibration(self, adapter):
        """After calibration, alignment should improve (higher score)."""
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=50, shift=5.0, scale=3.0)
        result = adapter.calibrate(target)
        assert result["alignment_after"] >= result["alignment_before"]

    def test_recalibrate_replaces_stats(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target1 = _target_features(n_samples=15, shift=2.0, seed=10)
        adapter.calibrate(target1)
        target2 = _target_features(n_samples=15, shift=5.0, seed=20)
        result = adapter.calibrate(target2)
        assert result["calibrated"] is True
        assert result["n_samples"] == 15


class TestAdapt:
    def test_adapt_returns_dict(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        features = np.random.randn(20)
        result = adapter.adapt(features)
        assert "adapted_features" in result
        assert "alignment_score" in result
        assert "adaptation_applied" in result

    def test_adapt_applied_when_calibrated(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        features = np.random.randn(20) * 2 + 3
        result = adapter.adapt(features)
        assert result["adaptation_applied"] is True

    def test_adapt_returns_numpy_array(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        features = np.random.randn(20)
        result = adapter.adapt(features)
        assert isinstance(result["adapted_features"], np.ndarray)
        assert result["adapted_features"].shape == (20,)

    def test_adapt_not_applied_without_calibration(self, adapter):
        """Without calibration, adapt returns original features."""
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        features = np.random.randn(20)
        result = adapter.adapt(features)
        assert result["adaptation_applied"] is False
        np.testing.assert_array_equal(result["adapted_features"], features)

    def test_adapt_not_applied_without_source(self, adapter):
        features = np.random.randn(20)
        result = adapter.adapt(features)
        assert result["adaptation_applied"] is False
        np.testing.assert_array_equal(result["adapted_features"], features)

    def test_adapt_alignment_score_range(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        features = np.random.randn(20)
        result = adapter.adapt(features)
        assert 0.0 <= result["alignment_score"] <= 1.0

    def test_adapt_wrong_feature_count(self, adapter):
        mean, std = _source_stats(n_features=20)
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15, n_features=20)
        adapter.calibrate(target)
        with pytest.raises(ValueError, match="features"):
            adapter.adapt(np.random.randn(10))

    def test_adapt_2d_input(self, adapter):
        """adapt() should handle 2D input (batch of samples)."""
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        batch = np.random.randn(5, 20)
        result = adapter.adapt(batch)
        assert result["adapted_features"].shape == (5, 20)
        assert result["adaptation_applied"] is True

    def test_adapt_reduces_distribution_distance(self, adapter):
        """Adapted features should be closer to source than raw target features."""
        np.random.seed(99)
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=50, shift=5.0, scale=3.0)
        adapter.calibrate(target)

        # Generate new target sample from same distribution
        new_sample = np.random.randn(20) * 3.0 + 5.0
        result = adapter.adapt(new_sample)

        # Distance from source mean should be smaller after adaptation
        dist_before = np.mean((new_sample - mean) ** 2)
        dist_after = np.mean((result["adapted_features"] - mean) ** 2)
        assert dist_after < dist_before


class TestGetAlignmentScore:
    def test_score_without_calibration(self, adapter):
        score = adapter.get_alignment_score()
        assert score == 0.0

    def test_score_after_calibration(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        score = adapter.get_alignment_score()
        assert 0.0 <= score <= 1.0

    def test_identical_distributions_high_score(self):
        """When source and target have same distribution, score should be high."""
        adapter = DomainAdapter()
        np.random.seed(42)
        mean = np.zeros(20)
        std = np.ones(20)
        adapter.set_source_stats(mean, std)
        # Target drawn from N(0, 1) — same as source
        target = np.random.randn(100, 20)
        adapter.calibrate(target)
        score = adapter.get_alignment_score()
        assert score > 0.7

    def test_very_different_distributions_low_score_before(self):
        """When source and target are very different, pre-adapt score is low."""
        adapter = DomainAdapter()
        mean = np.zeros(20)
        std = np.ones(20)
        adapter.set_source_stats(mean, std)
        # Target from very different distribution
        target = np.random.randn(100, 20) * 10.0 + 50.0
        result = adapter.calibrate(target)
        assert result["alignment_before"] < 0.5


class TestReset:
    def test_reset_clears_everything(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        assert adapter.is_calibrated() is True

        adapter.reset()
        assert adapter.is_calibrated() is False
        stats = adapter.get_stats()
        assert stats["source_set"] is False
        assert stats["target_calibrated"] is False
        assert stats["n_target_samples"] == 0

    def test_reset_allows_recalibration(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        adapter.reset()

        # Re-setup and recalibrate
        adapter.set_source_stats(mean, std)
        target2 = _target_features(n_samples=15, seed=99)
        result = adapter.calibrate(target2)
        assert result["calibrated"] is True
        assert adapter.is_calibrated() is True


class TestGetStats:
    def test_stats_keys(self, adapter):
        stats = adapter.get_stats()
        expected_keys = {
            "source_set", "target_calibrated", "n_target_samples",
            "n_features", "min_samples", "alignment_score",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_stats_after_full_setup(self, adapter):
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=15)
        adapter.calibrate(target)
        stats = adapter.get_stats()
        assert stats["source_set"] is True
        assert stats["target_calibrated"] is True
        assert stats["n_target_samples"] == 15
        assert stats["n_features"] == 20
        assert 0.0 <= stats["alignment_score"] <= 1.0


class TestEdgeCases:
    def test_constant_features(self, adapter):
        """All features have zero variance in target domain."""
        mean, std = _source_stats(n_features=5)
        adapter.set_source_stats(mean, std)
        # Target with zero variance (constant features)
        target = np.ones((15, 5)) * 3.0
        result = adapter.calibrate(target)
        assert result["calibrated"] is True
        # Should not produce NaN or inf
        adapted = adapter.adapt(np.ones(5) * 3.0)
        assert not np.any(np.isnan(adapted["adapted_features"]))
        assert not np.any(np.isinf(adapted["adapted_features"]))

    def test_very_large_features(self, adapter):
        mean, std = _source_stats(n_features=10)
        adapter.set_source_stats(mean, std)
        target = np.random.randn(15, 10) * 1e6
        adapter.calibrate(target)
        result = adapter.adapt(np.random.randn(10) * 1e6)
        assert not np.any(np.isnan(result["adapted_features"]))
        assert result["adaptation_applied"] is True

    def test_very_small_features(self, adapter):
        mean, std = _source_stats(n_features=10)
        adapter.set_source_stats(mean, std)
        target = np.random.randn(15, 10) * 1e-10
        adapter.calibrate(target)
        result = adapter.adapt(np.random.randn(10) * 1e-10)
        assert not np.any(np.isnan(result["adapted_features"]))

    def test_single_feature(self):
        """Works with a single feature dimension."""
        adapter = DomainAdapter(min_calibration_samples=5)
        adapter.set_source_stats(np.array([0.0]), np.array([1.0]))
        target = np.random.randn(10, 1) * 2.0 + 5.0
        result = adapter.calibrate(target)
        assert result["calibrated"] is True
        adapted = adapter.adapt(np.array([5.0]))
        assert adapted["adaptation_applied"] is True
        assert adapted["adapted_features"].shape == (1,)

    def test_many_features(self):
        """Works with high-dimensional feature vectors."""
        adapter = DomainAdapter()
        n = 200
        adapter.set_source_stats(np.zeros(n), np.ones(n))
        target = np.random.randn(50, n) * 3.0 + 2.0
        result = adapter.calibrate(target)
        assert result["calibrated"] is True
        adapted = adapter.adapt(np.random.randn(n) * 3.0 + 2.0)
        assert adapted["adapted_features"].shape == (n,)

    def test_nan_in_features_handled(self, adapter):
        """NaN in input features should be handled gracefully."""
        mean, std = _source_stats(n_features=10)
        adapter.set_source_stats(mean, std)
        target = np.random.randn(15, 10)
        adapter.calibrate(target)
        features = np.random.randn(10)
        features[3] = np.nan
        result = adapter.adapt(features)
        # NaN feature should remain NaN (not crash)
        assert result["adaptation_applied"] is True


class TestMinCalibrationSamples:
    def test_boundary_exact_min(self):
        adapter = DomainAdapter(min_calibration_samples=10)
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=10)
        result = adapter.calibrate(target)
        assert result["calibrated"] is True

    def test_boundary_one_below_min(self):
        adapter = DomainAdapter(min_calibration_samples=10)
        mean, std = _source_stats()
        adapter.set_source_stats(mean, std)
        target = _target_features(n_samples=9)
        result = adapter.calibrate(target)
        assert result["calibrated"] is False

    def test_min_samples_one(self):
        adapter = DomainAdapter(min_calibration_samples=1)
        adapter.set_source_stats(np.zeros(5), np.ones(5))
        result = adapter.calibrate(np.random.randn(1, 5))
        assert result["calibrated"] is True
