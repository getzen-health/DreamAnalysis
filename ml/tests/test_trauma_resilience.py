"""Tests for hyperarousal detector and resilience tracker.

Covers:
  - HyperarousalDetector: single-channel, multi-channel, output structure
  - ResilienceTracker: baseline, modulation, trend, reset, no-baseline edge case
  - compute_shannon_entropy: normalization, edge cases
"""

import numpy as np
import pytest

from models.hyperarousal_detector import HyperarousalDetector, compute_shannon_entropy
from models.resilience_tracker import ResilienceTracker


# ── Shannon entropy ───────────────────────────────────────────────────────────

class TestShannonEntropy:
    def test_returns_float_in_unit_range(self, sample_eeg):
        result = compute_shannon_entropy(sample_eeg)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_uniform_signal_has_lower_entropy_than_random(self):
        """A flat signal concentrates in one bin -> low entropy."""
        flat = np.ones(1024)
        random_sig = np.random.randn(1024) * 20
        assert compute_shannon_entropy(flat) < compute_shannon_entropy(random_sig)

    def test_empty_signal_returns_neutral(self):
        """Signal with <2 samples returns 0.5 (neutral)."""
        assert compute_shannon_entropy(np.array([1.0])) == 0.5

    def test_constant_signal_low_entropy(self):
        """A constant signal should have very low entropy."""
        constant = np.full(1024, 5.0)
        assert compute_shannon_entropy(constant) < 0.2


# ── HyperarousalDetector ─────────────────────────────────────────────────────

class TestHyperarousalDetector:
    @pytest.fixture
    def detector(self):
        return HyperarousalDetector()

    def test_output_structure_multichannel(self, detector, multichannel_eeg, fs):
        result = detector.predict(multichannel_eeg, fs)

        # Top-level keys
        assert "hyperarousal_index" in result
        assert "risk_level" in result
        assert "components" in result
        assert "channel_entropies" in result

        # Index range
        assert 0.0 <= result["hyperarousal_index"] <= 1.0

        # Risk level is one of the expected values
        assert result["risk_level"] in ("low", "moderate", "high")

        # Components exist
        components = result["components"]
        for key in ("frontal_entropy", "af8_alpha_power", "spectral_entropy",
                     "entropy_component", "alpha_component", "spectral_component"):
            assert key in components

        # Channel entropies for all 4 channels
        assert len(result["channel_entropies"]) == 4
        for name in ("TP9", "AF7", "AF8", "TP10"):
            assert name in result["channel_entropies"]

    def test_output_structure_single_channel(self, detector, sample_eeg, fs):
        result = detector.predict(sample_eeg, fs)

        assert 0.0 <= result["hyperarousal_index"] <= 1.0
        assert result["risk_level"] in ("low", "moderate", "high")
        # Single channel -> only TP9 in channel_entropies
        assert "TP9" in result["channel_entropies"]

    def test_flat_signal_high_hyperarousal(self, detector, fs):
        """A flat signal has very low entropy -> should produce high hyperarousal."""
        flat = np.full((4, 1024), 0.001)
        result = detector.predict(flat, fs)
        # Low entropy + low alpha should push index toward high
        assert result["components"]["entropy_component"] > 0.5

    def test_risk_level_thresholds(self, detector, fs):
        """Verify risk level categories correspond to index value."""
        # We can't easily control exact index, but we can verify
        # that the mapping is consistent with the returned index
        for _ in range(10):
            signals = np.random.randn(4, 1024) * 20
            result = detector.predict(signals, fs)
            idx = result["hyperarousal_index"]
            if idx >= 0.7:
                assert result["risk_level"] == "high"
            elif idx >= 0.4:
                assert result["risk_level"] == "moderate"
            else:
                assert result["risk_level"] == "low"

    def test_components_are_bounded(self, detector, multichannel_eeg, fs):
        """All component scores should be in [0, 1]."""
        result = detector.predict(multichannel_eeg, fs)
        for key, val in result["components"].items():
            if key in ("entropy_component", "alpha_component", "spectral_component"):
                assert 0.0 <= val <= 1.0, f"{key} out of bounds: {val}"


# ── ResilienceTracker ─────────────────────────────────────────────────────────

class TestResilienceTracker:
    @pytest.fixture
    def tracker(self):
        return ResilienceTracker()

    def test_no_baseline_returns_zero_score(self, tracker, multichannel_eeg, fs):
        """Without baseline, resilience score should be 0 and has_baseline=False."""
        result = tracker.compute_modulation(multichannel_eeg, fs)
        assert result["resilience_score"] == 0.0
        assert result["has_baseline"] is False
        assert result["direction"] == "unknown"

    def test_set_baseline_enables_modulation(self, tracker, fs):
        """After setting baseline, modulation should return has_baseline=True."""
        baseline = np.random.randn(4, 1024) * 15
        tracker.set_baseline(baseline, fs)
        assert tracker.has_baseline is True

        task = np.random.randn(4, 1024) * 30  # different amplitude
        result = tracker.compute_modulation(task, fs)
        assert result["has_baseline"] is True
        assert result["resilience_score"] >= 0.0

    def test_modulation_output_structure(self, tracker, fs):
        baseline = np.random.randn(4, 1024) * 20
        tracker.set_baseline(baseline, fs)
        task = np.random.randn(4, 1024) * 20
        result = tracker.compute_modulation(task, fs)

        for key in ("resilience_score", "entropy_modulation", "baseline_entropy",
                     "task_entropy", "direction", "has_baseline"):
            assert key in result

        assert result["direction"] in ("increase", "decrease")
        assert 0.0 <= result["resilience_score"] <= 1.0

    def test_identical_signals_low_modulation(self, tracker, fs):
        """Same signal for baseline and task -> near-zero modulation."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024) * 20
        tracker.set_baseline(signals.copy(), fs)
        result = tracker.compute_modulation(signals.copy(), fs)
        # Same data -> modulation should be very small (not exactly 0 due to processing)
        assert result["entropy_modulation"] < 0.1

    def test_trend_insufficient_data(self, tracker):
        """No measurements -> trend should be insufficient_data."""
        trend = tracker.get_trend()
        assert trend["trend"] == "insufficient_data"
        assert trend["sessions"] == 0

    def test_trend_accumulates_scores(self, tracker, fs):
        """Multiple modulation calls should build up session history."""
        baseline = np.random.randn(4, 1024) * 20
        tracker.set_baseline(baseline, fs)

        for _ in range(5):
            task = np.random.randn(4, 1024) * 20
            tracker.compute_modulation(task, fs)

        trend = tracker.get_trend()
        assert trend["sessions"] == 5
        assert "mean_score" in trend
        assert "latest_score" in trend

    def test_reset_clears_everything(self, tracker, fs):
        """Reset should clear baseline and session history."""
        baseline = np.random.randn(4, 1024) * 20
        tracker.set_baseline(baseline, fs)
        task = np.random.randn(4, 1024) * 20
        tracker.compute_modulation(task, fs)

        tracker.reset()
        assert tracker.has_baseline is False
        trend = tracker.get_trend()
        assert trend["sessions"] == 0

    def test_single_channel_input(self, tracker, sample_eeg, fs):
        """Single-channel input (1-D array) should work without error."""
        tracker.set_baseline(sample_eeg, fs)
        result = tracker.compute_modulation(sample_eeg, fs)
        assert result["has_baseline"] is True
