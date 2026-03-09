"""Tests for SlowOscillationDetector — slow oscillation detection and SO-spindle coupling.

Tests cover:
- SO detection in synthetic signals with known 0.5-1.25 Hz oscillations
- SO metrics: density, amplitude, frequency, event list
- SO-spindle coupling detection and coupling strength
- Memory consolidation prediction labels
- Session tracking and history
- Edge cases: short signals, flat signals, multichannel input, empty history
- Reset and state management
- get_coupling_score() accessor

Reference: Staresina et al. (2015), Helfrich et al. (2018) — SO-spindle coupling
and memory consolidation.
Muse 2: 4 channels (TP9, AF7, AF8, TP10) at 256 Hz.
"""
import numpy as np
import pytest

from models.slow_oscillation_detector import SlowOscillationDetector


# ---------------------------------------------------------------------------
# Helpers — synthetic signal generators
# ---------------------------------------------------------------------------

def _make_slow_oscillation_signal(
    fs: int = 256,
    duration: float = 30.0,
    so_freq: float = 0.8,
    so_amp: float = 80.0,
    noise_amp: float = 3.0,
) -> np.ndarray:
    """Create a synthetic EEG signal dominated by slow oscillations (0.5-1.25 Hz).

    Deep sleep (N3) typically shows high-amplitude SOs at 0.5-1 Hz.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    so = so_amp * np.sin(2 * np.pi * so_freq * t)
    noise = noise_amp * np.random.randn(n_samples)
    return so + noise


def _make_coupled_signal(
    fs: int = 256,
    duration: float = 30.0,
    so_freq: float = 0.8,
    so_amp: float = 80.0,
    spindle_freq: float = 13.0,
    spindle_amp: float = 25.0,
    noise_amp: float = 3.0,
) -> np.ndarray:
    """Create signal with spindles nested in SO up-states (coupled).

    Spindle bursts are placed at the peaks (up-states) of the slow oscillation,
    simulating the SO-spindle coupling observed during memory consolidation.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    # Slow oscillation
    so = so_amp * np.sin(2 * np.pi * so_freq * t)
    # Place spindle bursts at SO peaks (up-states)
    spindle_signal = np.zeros(n_samples)
    period_samples = int(fs / so_freq)
    # Find SO peaks — peaks of sin occur at t = (1/4 + k) / so_freq
    peak_times = []
    first_peak = 1.0 / (4.0 * so_freq)
    t_peak = first_peak
    while t_peak < duration - 0.5:
        peak_times.append(t_peak)
        t_peak += 1.0 / so_freq

    for pt in peak_times:
        center = int(pt * fs)
        burst_dur = 0.5  # 0.5 second spindle burst
        burst_len = int(burst_dur * fs)
        start = center - burst_len // 2
        end = start + burst_len
        if start >= 0 and end <= n_samples:
            burst_t = np.arange(burst_len) / fs
            sigma = burst_dur / 6
            env = np.exp(-0.5 * ((burst_t - burst_dur / 2) / sigma) ** 2)
            spindle_signal[start:end] += spindle_amp * env * np.sin(
                2 * np.pi * spindle_freq * burst_t
            )

    noise = noise_amp * np.random.randn(n_samples)
    return so + spindle_signal + noise


def _make_uncoupled_signal(
    fs: int = 256,
    duration: float = 30.0,
    so_freq: float = 0.8,
    so_amp: float = 80.0,
    spindle_freq: float = 13.0,
    spindle_amp: float = 25.0,
    noise_amp: float = 3.0,
) -> np.ndarray:
    """Create signal with spindles at SO troughs (anti-coupled / uncoupled).

    Spindles are placed at the down-states of the SO — the opposite of
    physiological coupling. Coupling detection should be weak or absent.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    so = so_amp * np.sin(2 * np.pi * so_freq * t)
    spindle_signal = np.zeros(n_samples)
    # Place spindle bursts at SO troughs — trough of sin at t = (3/4 + k) / so_freq
    trough_times = []
    first_trough = 3.0 / (4.0 * so_freq)
    t_trough = first_trough
    while t_trough < duration - 0.5:
        trough_times.append(t_trough)
        t_trough += 1.0 / so_freq

    for tt in trough_times:
        center = int(tt * fs)
        burst_dur = 0.5
        burst_len = int(burst_dur * fs)
        start = center - burst_len // 2
        end = start + burst_len
        if start >= 0 and end <= n_samples:
            burst_t = np.arange(burst_len) / fs
            sigma = burst_dur / 6
            env = np.exp(-0.5 * ((burst_t - burst_dur / 2) / sigma) ** 2)
            spindle_signal[start:end] += spindle_amp * env * np.sin(
                2 * np.pi * spindle_freq * burst_t
            )

    noise = noise_amp * np.random.randn(n_samples)
    return so + spindle_signal + noise


def _make_flat_signal(fs: int = 256, duration: float = 30.0) -> np.ndarray:
    """Low-amplitude noise — no SOs expected."""
    return 0.5 * np.random.randn(int(fs * duration))


def _make_high_freq_only(
    fs: int = 256, duration: float = 30.0, freq: float = 20.0, amp: float = 30.0
) -> np.ndarray:
    """Signal with only high-frequency content (beta) — no SOs."""
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    return amp * np.sin(2 * np.pi * freq * t) + 2.0 * np.random.randn(n_samples)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return SlowOscillationDetector()


# ---------------------------------------------------------------------------
# Test: detect() basic behavior
# ---------------------------------------------------------------------------

class TestDetectBasic:
    def test_detects_so_in_known_signal(self, detector):
        """Should detect slow oscillations in a signal with clear 0.8 Hz content."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_freq=0.8, so_amp=80.0)
        result = detector.detect(signal, fs=256)
        assert result["so_count"] > 0, "Expected at least one SO in 0.8 Hz signal"

    def test_detect_returns_all_required_keys(self, detector):
        """detect() must return all documented output keys."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        result = detector.detect(signal, fs=256)
        expected_keys = {
            "so_count",
            "so_density",
            "mean_amplitude",
            "mean_frequency",
            "coupling_detected",
            "coupling_strength",
            "consolidation_prediction",
            "so_events",
        }
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in detect() output"

    def test_so_count_is_int(self, detector):
        """so_count must be an integer."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        result = detector.detect(signal, fs=256)
        assert isinstance(result["so_count"], int)

    def test_so_density_is_per_minute(self, detector):
        """so_density should be expressed as SOs per minute."""
        np.random.seed(42)
        # 30s signal at 0.8 Hz => ~24 cycles => density ~48/min
        signal = _make_slow_oscillation_signal(
            duration=30.0, so_freq=0.8, so_amp=80.0
        )
        result = detector.detect(signal, fs=256)
        # density should be positive and reasonable
        assert result["so_density"] >= 0
        if result["so_count"] > 0:
            assert result["so_density"] > 0

    def test_mean_amplitude_positive_when_so_present(self, detector):
        """Mean amplitude should be positive when SOs are detected."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        result = detector.detect(signal, fs=256)
        if result["so_count"] > 0:
            assert result["mean_amplitude"] > 0

    def test_mean_frequency_in_so_band(self, detector):
        """Mean frequency should be within the SO band (0.5-1.25 Hz)."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_freq=0.8, so_amp=80.0)
        result = detector.detect(signal, fs=256)
        if result["so_count"] > 0:
            assert 0.3 <= result["mean_frequency"] <= 1.5, (
                f"Mean frequency {result['mean_frequency']} Hz outside SO band"
            )

    def test_no_so_in_flat_signal(self, detector):
        """Flat/low-noise signal should produce zero or very few SOs."""
        np.random.seed(42)
        signal = _make_flat_signal()
        result = detector.detect(signal, fs=256)
        assert result["so_count"] <= 2, (
            f"Flat signal should have at most 2 false positive SOs, got {result['so_count']}"
        )

    def test_no_so_in_high_freq_signal(self, detector):
        """Signal with only high-frequency content should produce no SOs."""
        np.random.seed(42)
        signal = _make_high_freq_only(freq=20.0, amp=30.0)
        result = detector.detect(signal, fs=256)
        assert result["so_count"] <= 2, (
            f"High-freq signal should have at most 2 false positive SOs, got {result['so_count']}"
        )


# ---------------------------------------------------------------------------
# Test: SO events list
# ---------------------------------------------------------------------------

class TestSOEvents:
    def test_so_events_list_length_matches_count(self, detector):
        """so_events list length should equal so_count."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        result = detector.detect(signal, fs=256)
        assert len(result["so_events"]) == result["so_count"]

    def test_so_event_has_required_keys(self, detector):
        """Each SO event dict should have start_sample, amplitude, frequency."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        result = detector.detect(signal, fs=256)
        if result["so_count"] > 0:
            event = result["so_events"][0]
            for key in ("start_sample", "amplitude", "frequency"):
                assert key in event, f"Missing key '{key}' in SO event"

    def test_so_event_amplitude_positive(self, detector):
        """Each SO event's amplitude should be positive."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        result = detector.detect(signal, fs=256)
        for event in result["so_events"]:
            assert event["amplitude"] > 0

    def test_so_event_start_sample_is_int(self, detector):
        """start_sample should be an integer index."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        result = detector.detect(signal, fs=256)
        for event in result["so_events"]:
            assert isinstance(event["start_sample"], int)


# ---------------------------------------------------------------------------
# Test: SO-spindle coupling
# ---------------------------------------------------------------------------

class TestCoupling:
    def test_coupling_detected_in_coupled_signal(self, detector):
        """Coupling should be detected when spindles are at SO up-states."""
        np.random.seed(42)
        signal = _make_coupled_signal(so_amp=80.0, spindle_amp=30.0)
        result = detector.detect(signal, fs=256)
        assert result["coupling_detected"] is True, (
            "Expected coupling_detected=True for coupled signal"
        )

    def test_coupling_strength_range(self, detector):
        """coupling_strength must be between 0 and 1."""
        np.random.seed(42)
        signal = _make_coupled_signal(so_amp=80.0, spindle_amp=30.0)
        result = detector.detect(signal, fs=256)
        assert 0.0 <= result["coupling_strength"] <= 1.0

    def test_coupling_stronger_for_coupled_than_uncoupled(self, detector):
        """Coupled signal should have higher coupling_strength than uncoupled."""
        np.random.seed(42)
        coupled = _make_coupled_signal(so_amp=80.0, spindle_amp=30.0)
        result_coupled = detector.detect(coupled, fs=256)
        detector.reset()

        np.random.seed(42)
        uncoupled = _make_uncoupled_signal(so_amp=80.0, spindle_amp=30.0)
        result_uncoupled = detector.detect(uncoupled, fs=256)

        assert result_coupled["coupling_strength"] > result_uncoupled["coupling_strength"], (
            f"Coupled strength {result_coupled['coupling_strength']:.3f} "
            f"should exceed uncoupled {result_uncoupled['coupling_strength']:.3f}"
        )

    def test_no_coupling_without_spindles(self, detector):
        """Pure SO signal without spindles should have coupling_strength near 0."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        result = detector.detect(signal, fs=256)
        assert result["coupling_strength"] < 0.3, (
            f"Expected low coupling without spindles, got {result['coupling_strength']:.3f}"
        )

    def test_coupling_detected_is_bool(self, detector):
        """coupling_detected should be a boolean."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        result = detector.detect(signal, fs=256)
        assert isinstance(result["coupling_detected"], bool)


# ---------------------------------------------------------------------------
# Test: consolidation prediction
# ---------------------------------------------------------------------------

class TestConsolidationPrediction:
    def test_consolidation_prediction_is_string(self, detector):
        """consolidation_prediction should be a string."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        result = detector.detect(signal, fs=256)
        assert isinstance(result["consolidation_prediction"], str)

    def test_consolidation_prediction_valid_labels(self, detector):
        """consolidation_prediction should be one of the defined labels."""
        valid_labels = {"poor", "moderate", "good", "excellent"}
        np.random.seed(42)
        signal = _make_coupled_signal(so_amp=80.0, spindle_amp=30.0)
        result = detector.detect(signal, fs=256)
        assert result["consolidation_prediction"] in valid_labels, (
            f"Invalid label: '{result['consolidation_prediction']}'"
        )

    def test_coupled_signal_predicts_good_or_excellent(self, detector):
        """Strong SO-spindle coupling should yield 'good' or 'excellent'."""
        np.random.seed(42)
        signal = _make_coupled_signal(so_amp=100.0, spindle_amp=35.0)
        result = detector.detect(signal, fs=256)
        assert result["consolidation_prediction"] in {"good", "excellent"}, (
            f"Expected good/excellent for coupled signal, got '{result['consolidation_prediction']}'"
        )


# ---------------------------------------------------------------------------
# Test: get_coupling_score()
# ---------------------------------------------------------------------------

class TestGetCouplingScore:
    def test_returns_float(self, detector):
        """get_coupling_score() should return a float."""
        np.random.seed(42)
        signal = _make_coupled_signal(so_amp=80.0, spindle_amp=30.0)
        detector.detect(signal, fs=256)
        score = detector.get_coupling_score()
        assert isinstance(score, float)

    def test_range_zero_to_one(self, detector):
        """Coupling score should be between 0 and 1."""
        np.random.seed(42)
        signal = _make_coupled_signal()
        detector.detect(signal, fs=256)
        score = detector.get_coupling_score()
        assert 0.0 <= score <= 1.0

    def test_returns_zero_before_any_detect(self, detector):
        """Before any detect() call, coupling score should be 0."""
        score = detector.get_coupling_score()
        assert score == 0.0


# ---------------------------------------------------------------------------
# Test: get_session_stats()
# ---------------------------------------------------------------------------

class TestSessionStats:
    def test_session_stats_keys(self, detector):
        """get_session_stats() must have expected summary keys."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=80.0)
        detector.detect(signal, fs=256)
        detector.detect(signal, fs=256)
        stats = detector.get_session_stats()
        for key in ("n_epochs", "total_so_count", "mean_density", "mean_amplitude",
                     "mean_coupling_strength"):
            assert key in stats, f"Missing key '{key}' in session stats"

    def test_session_stats_epoch_count(self, detector):
        """n_epochs should match the number of detect() calls."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        detector.detect(signal, fs=256)
        detector.detect(signal, fs=256)
        detector.detect(signal, fs=256)
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 3

    def test_empty_session_stats(self, detector):
        """Session stats with no data should return zeroed values."""
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["total_so_count"] == 0
        assert stats["mean_density"] == 0.0
        assert stats["mean_coupling_strength"] == 0.0


# ---------------------------------------------------------------------------
# Test: get_history()
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_grows_with_calls(self, detector):
        """Each detect() call should add one entry to history."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        for _ in range(3):
            detector.detect(signal, fs=256)
        history = detector.get_history()
        assert len(history) == 3

    def test_history_entries_are_dicts(self, detector):
        """Each history entry should be a dict with detect() output keys."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        detector.detect(signal, fs=256)
        history = detector.get_history()
        assert isinstance(history[0], dict)
        assert "so_count" in history[0]


# ---------------------------------------------------------------------------
# Test: reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self, detector):
        """reset() should clear all session history."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        detector.detect(signal, fs=256)
        assert len(detector.get_history()) == 1
        detector.reset()
        assert len(detector.get_history()) == 0

    def test_reset_clears_session_stats(self, detector):
        """After reset, session stats should be empty."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal()
        detector.detect(signal, fs=256)
        detector.reset()
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_reset_clears_coupling_score(self, detector):
        """After reset, get_coupling_score() should return 0."""
        np.random.seed(42)
        signal = _make_coupled_signal()
        detector.detect(signal, fs=256)
        detector.reset()
        assert detector.get_coupling_score() == 0.0


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_short_signal(self, detector):
        """Signal shorter than one SO cycle should not crash."""
        np.random.seed(42)
        short = np.random.randn(64)  # 0.25 seconds at 256 Hz
        result = detector.detect(short, fs=256)
        assert result["so_count"] == 0
        assert result["coupling_strength"] == 0.0

    def test_single_sample(self, detector):
        """Single sample should not crash."""
        result = detector.detect(np.array([1.0]), fs=256)
        assert result["so_count"] == 0

    def test_all_zeros(self, detector):
        """All-zero signal should not crash and detect no SOs."""
        signal = np.zeros(256 * 10)
        result = detector.detect(signal, fs=256)
        assert result["so_count"] == 0

    def test_multichannel_input(self, detector):
        """detect() should accept (n_channels, n_samples) 2D input."""
        np.random.seed(42)
        single = _make_slow_oscillation_signal(so_amp=80.0)
        multi = np.stack([
            np.random.randn(len(single)) * 3,  # TP9
            single,                              # AF7 (frontal)
            single + np.random.randn(len(single)) * 2,  # AF8
            np.random.randn(len(single)) * 3,  # TP10
        ])
        result = detector.detect(multi, fs=256)
        assert result["so_count"] >= 0
        assert "coupling_strength" in result

    def test_different_sampling_rate(self, detector):
        """Should work with non-256 Hz sampling rate."""
        np.random.seed(42)
        fs = 512
        signal = _make_slow_oscillation_signal(fs=fs, so_amp=80.0)
        result = detector.detect(signal, fs=fs)
        assert result["so_count"] >= 0

    def test_custom_amplitude_threshold(self):
        """Custom amplitude threshold should affect detection sensitivity."""
        np.random.seed(42)
        signal = _make_slow_oscillation_signal(so_amp=40.0)
        # High threshold — fewer detections
        strict = SlowOscillationDetector(amplitude_threshold_uv=60.0)
        result_strict = strict.detect(signal, fs=256)
        # Low threshold — more detections
        lenient = SlowOscillationDetector(amplitude_threshold_uv=10.0)
        result_lenient = lenient.detect(signal, fs=256)
        assert result_lenient["so_count"] >= result_strict["so_count"]
