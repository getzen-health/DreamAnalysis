"""Tests for SeizureDetector — 25+ tests covering init, predict, alarm, singleton."""
import numpy as np
import pytest

from models.seizure_detector import SeizureDetector, get_seizure_detector, MEDICAL_DISCLAIMER

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def detector():
    """Fresh detector for each test."""
    return SeizureDetector(alarm_threshold=0.7, alarm_trigger_count=3, fs=256.0)


# ── Signal generators ─────────────────────────────────────────────────────────


def _make_normal_eeg(fs=256, duration=4, n_channels=4, seed=42):
    """Normal resting EEG: dominant alpha + some theta/beta, low amplitude."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = 15.0 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        theta = 5.0 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        beta = 3.0 * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = 2.0 * rng.standard_normal(len(t))
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


def _make_high_ll_eeg(fs=256, duration=4, n_channels=4, seed=42):
    """High line-length EEG: large rapid oscillations simulating seizure morphology."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        # Large-amplitude, fast-changing signal → high line length
        sig = 100.0 * np.sin(2 * np.pi * 25 * t) + 50.0 * np.sin(2 * np.pi * 15 * t)
        noise = 5.0 * rng.standard_normal(len(t))
        signals.append(sig + noise)
    return np.array(signals)


def _make_low_ll_eeg(fs=256, duration=4, n_channels=4, seed=42):
    """Low line-length EEG: slow, smooth waveform."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        sig = 3.0 * np.sin(2 * np.pi * 1.0 * t)
        noise = 0.5 * rng.standard_normal(len(t))
        signals.append(sig + noise)
    return np.array(signals)


# ── Init / status tests ────────────────────────────────────────────────────────


class TestInit:
    def test_status_has_alarm_active_key(self, detector):
        status = detector.get_status()
        assert "alarm_active" in status

    def test_status_has_consecutive_ictal_key(self, detector):
        status = detector.get_status()
        assert "consecutive_ictal" in status

    def test_status_has_threshold_key(self, detector):
        status = detector.get_status()
        assert "threshold" in status

    def test_default_alarm_active_is_false(self, detector):
        assert detector.get_status()["alarm_active"] is False

    def test_default_consecutive_ictal_is_zero(self, detector):
        assert detector.get_status()["consecutive_ictal"] == 0

    def test_threshold_reflects_constructor(self):
        det = SeizureDetector(alarm_threshold=0.85)
        assert det.get_status()["threshold"] == pytest.approx(0.85)


# ── Predict output structure tests ────────────────────────────────────────────


class TestPredictKeys:
    def test_returns_detection_key(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert "detection" in result

    def test_returns_probability_key(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert "probability" in result

    def test_returns_seizure_probability_key(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert "seizure_probability" in result

    def test_returns_alert_level_key(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert "alert_level" in result

    def test_returns_features_key(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert "features" in result

    def test_returns_confidence_key(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert "confidence" in result


# ── Predict value range / type tests ─────────────────────────────────────────


class TestPredictValues:
    def test_detection_is_ictal_or_interictal(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert result["detection"] in ("ictal", "interictal")

    def test_probability_in_range(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert 0.0 <= result["probability"] <= 1.0

    def test_seizure_probability_in_range(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert 0.0 <= result["seizure_probability"] <= 1.0

    def test_probability_equals_seizure_probability(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert result["probability"] == result["seizure_probability"]

    def test_alert_level_is_valid(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert result["alert_level"] in ("none", "warning", "alert", "critical")

    def test_features_is_dict(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert isinstance(result["features"], dict)

    def test_confidence_in_range(self, detector):
        result = detector.predict(_make_normal_eeg())
        assert 0.0 <= result["confidence"] <= 1.0


# ── Input shape tests ─────────────────────────────────────────────────────────


class TestInputShapes:
    def test_1d_input_works(self, detector):
        eeg_1d = np.random.default_rng(0).standard_normal(256 * 4)
        result = detector.predict(eeg_1d)
        assert result["detection"] in ("ictal", "interictal")

    def test_4_channel_input_works(self, detector):
        eeg_4ch = _make_normal_eeg(n_channels=4)
        result = detector.predict(eeg_4ch)
        assert result["detection"] in ("ictal", "interictal")

    def test_short_signal_handled_gracefully(self, detector):
        short = np.random.default_rng(0).standard_normal((4, 32))
        result = detector.predict(short)
        assert "detection" in result
        assert result["detection"] in ("ictal", "interictal")


# ── Alarm buffer tests ─────────────────────────────────────────────────────────


class TestAlarmBuffer:
    def test_consecutive_ictal_increments_on_ictal(self):
        """With very low threshold, repeated predictions should increment counter."""
        det = SeizureDetector(alarm_threshold=0.01, alarm_trigger_count=5, fs=256.0)
        eeg = _make_high_ll_eeg()
        det.predict(eeg)
        det.predict(eeg)
        assert det.get_status()["consecutive_ictal"] >= 2

    def test_reset_alarm_resets_counter(self):
        det = SeizureDetector(alarm_threshold=0.01, alarm_trigger_count=5, fs=256.0)
        eeg = _make_high_ll_eeg()
        det.predict(eeg)
        det.predict(eeg)
        det.reset_alarm()
        assert det.get_status()["consecutive_ictal"] == 0

    def test_reset_alarm_clears_alarm_active(self):
        det = SeizureDetector(alarm_threshold=0.01, alarm_trigger_count=1, fs=256.0)
        eeg = _make_high_ll_eeg()
        det.predict(eeg)
        det.reset_alarm()
        assert det.get_status()["alarm_active"] is False

    def test_non_ictal_prediction_resets_counter(self):
        """After an ictal detection, normal EEG should reset consecutive count."""
        det = SeizureDetector(alarm_threshold=0.01, alarm_trigger_count=10, fs=256.0)
        high_ll = _make_high_ll_eeg()
        det.predict(high_ll)
        assert det._consecutive_ictal >= 1

        # Now send a very flat / low-ll signal with high threshold so it's interictal
        det2 = SeizureDetector(alarm_threshold=0.99, alarm_trigger_count=10, fs=256.0)
        # Manually set consecutive > 0 to verify it resets
        det2._consecutive_ictal = 5
        low_ll = _make_low_ll_eeg()
        det2.predict(low_ll)
        # prob < 0.99 threshold → counter should be reset to 0
        assert det2._consecutive_ictal == 0


# ── Singleton tests ─────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_seizure_detector_returns_same_instance_twice(self):
        inst1 = get_seizure_detector()
        inst2 = get_seizure_detector()
        assert inst1 is inst2

    def test_singleton_is_seizure_detector(self):
        inst = get_seizure_detector()
        assert isinstance(inst, SeizureDetector)


# ── Detection sensitivity tests ───────────────────────────────────────────────


class TestDetectionSensitivity:
    def test_high_line_length_gives_higher_seizure_probability(self, detector):
        high_ll = _make_high_ll_eeg()
        low_ll = _make_low_ll_eeg()
        result_high = detector.predict(high_ll)
        # Fresh detector to avoid alarm state bleed
        det2 = SeizureDetector(alarm_threshold=0.7, alarm_trigger_count=3, fs=256.0)
        result_low = det2.predict(low_ll)
        assert result_high["seizure_probability"] > result_low["seizure_probability"]

    def test_low_line_length_gives_lower_seizure_probability(self, detector):
        low_ll = _make_low_ll_eeg()
        result = detector.predict(low_ll)
        # Should be well below threshold for a smooth, small-amplitude signal
        assert result["seizure_probability"] < 0.7
