"""Tests for tinnitus severity assessor."""
import numpy as np
import pytest

from models.tinnitus_assessor import TinnitusAssessor, MEDICAL_DISCLAIMER


@pytest.fixture
def assessor():
    return TinnitusAssessor()


def _make_signal(fs=256, duration=4, n_channels=4, alpha_amp=1.0, gamma_amp=0.1):
    """Create synthetic EEG signal with controlled alpha and gamma content."""
    t = np.arange(int(fs * duration)) / fs
    alpha_wave = alpha_amp * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    gamma_wave = gamma_amp * np.sin(2 * np.pi * 35 * t)  # 35 Hz gamma
    signal = alpha_wave + gamma_wave + 0.05 * np.random.randn(len(t))
    if n_channels == 1:
        return signal
    return np.tile(signal, (n_channels, 1))


class TestBasicAssessment:
    def test_assess_without_baseline(self, assessor):
        """Should work with population-average baseline."""
        signals = _make_signal()
        result = assessor.assess(signals)
        assert "severity" in result
        assert result["using_personal_baseline"] is False

    def test_assess_with_baseline(self, assessor):
        """Should use personal baseline when set."""
        baseline_signal = _make_signal(alpha_amp=1.0)
        assessor.set_baseline(baseline_signal)
        result = assessor.assess(_make_signal(alpha_amp=0.5))
        assert result["using_personal_baseline"] is True

    def test_severity_index_range(self, assessor):
        """Severity index should be between 0 and 1."""
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            result = assessor.assess(_make_signal(alpha_amp=alpha))
            assert 0 <= result["severity_index"] <= 1


class TestSeverityLevels:
    def test_low_alpha_reduction_none_detected(self, assessor):
        """Similar alpha to baseline → none_detected."""
        baseline = _make_signal(alpha_amp=1.0)
        assessor.set_baseline(baseline)
        result = assessor.assess(_make_signal(alpha_amp=1.0))
        assert result["severity"] == "none_detected"

    def test_reduced_alpha_indicates_severity(self, assessor):
        """Significantly reduced alpha should indicate tinnitus markers."""
        baseline = _make_signal(alpha_amp=2.0)
        assessor.set_baseline(baseline)
        result = assessor.assess(_make_signal(alpha_amp=0.3, gamma_amp=0.5))
        assert result["severity"] in ("mild_indicators", "moderate_indicators", "elevated_indicators")
        assert result["alpha_reduction"] > 0

    def test_elevated_gamma_increases_severity(self, assessor):
        """Elevated gamma should increase severity index."""
        baseline = _make_signal(alpha_amp=1.0, gamma_amp=0.1)
        assessor.set_baseline(baseline)
        result = assessor.assess(_make_signal(alpha_amp=1.0, gamma_amp=1.0))
        assert result["gamma_elevation"] > 0


class TestInputShapes:
    def test_single_channel(self, assessor):
        """Should handle 1D single-channel input."""
        signal = _make_signal(n_channels=1)
        result = assessor.assess(signal)
        assert "severity" in result

    def test_multichannel_4(self, assessor):
        """Should handle standard 4-channel Muse 2 input."""
        signals = _make_signal(n_channels=4)
        result = assessor.assess(signals)
        assert "severity" in result

    def test_multichannel_2(self, assessor):
        """Should handle 2-channel input."""
        signals = _make_signal(n_channels=2)
        result = assessor.assess(signals)
        assert "severity" in result


class TestBaseline:
    def test_set_baseline_returns_confirmation(self, assessor):
        """set_baseline should return status dict."""
        result = assessor.set_baseline(_make_signal())
        assert result["status"] == "baseline_set"
        assert "baseline_alpha" in result

    def test_has_baseline(self, assessor):
        """has_baseline should reflect state."""
        assert assessor.has_baseline() is False
        assessor.set_baseline(_make_signal())
        assert assessor.has_baseline() is True

    def test_reset_baseline(self, assessor):
        """reset_baseline should clear stored baseline."""
        assessor.set_baseline(_make_signal())
        result = assessor.reset_baseline()
        assert result["had_baseline"] is True
        assert assessor.has_baseline() is False

    def test_reset_nonexistent(self, assessor):
        """Reset when no baseline should not crash."""
        result = assessor.reset_baseline("nobody")
        assert result["had_baseline"] is False

    def test_per_user_baselines(self, assessor):
        """Different users should have independent baselines."""
        assessor.set_baseline(_make_signal(alpha_amp=1.0), user_id="alice")
        assessor.set_baseline(_make_signal(alpha_amp=2.0), user_id="bob")
        assert assessor.has_baseline("alice")
        assert assessor.has_baseline("bob")
        assessor.reset_baseline("alice")
        assert not assessor.has_baseline("alice")
        assert assessor.has_baseline("bob")


class TestDisclaimer:
    def test_disclaimer_in_assess(self, assessor):
        """Assessment should always include medical disclaimer."""
        result = assessor.assess(_make_signal())
        assert result["disclaimer"] == MEDICAL_DISCLAIMER

    def test_disclaimer_in_baseline(self, assessor):
        """Baseline setting should include medical disclaimer."""
        result = assessor.set_baseline(_make_signal())
        assert result["disclaimer"] == MEDICAL_DISCLAIMER

    def test_gamma_emg_caveat(self, assessor):
        """Assessment should warn about EMG contamination."""
        result = assessor.assess(_make_signal())
        assert "EMG" in result["gamma_emg_caveat"] or "emg" in result["gamma_emg_caveat"].lower()


class TestEdgeCases:
    def test_flat_signal(self, assessor):
        """Flat (DC) signal should not crash."""
        flat = np.ones((4, 1024))
        result = assessor.assess(flat)
        assert "severity" in result

    def test_very_short_signal(self, assessor):
        """Very short signal should still work."""
        short = _make_signal(duration=0.5)
        result = assessor.assess(short)
        assert "severity" in result

    def test_output_keys(self, assessor):
        """All expected keys should be present."""
        result = assessor.assess(_make_signal())
        expected = {"severity", "severity_index", "alpha_reduction", "gamma_elevation",
                    "current_alpha", "current_gamma", "baseline_alpha", "baseline_gamma",
                    "using_personal_baseline", "gamma_emg_caveat", "disclaimer"}
        assert expected.issubset(set(result.keys()))
