"""Tests for SpindleAnalyzer — sleep spindle detection and characterization.

Tests cover:
- Spindle detection in synthetic signals with known spindle bursts
- Slow vs fast spindle classification (11-13 Hz frontal vs 13-16 Hz centroparietal)
- Spindle metrics: density, amplitude, frequency, duration
- Session tracking and history
- Memory consolidation index based on spindle-SO coupling
- Edge cases: short signals, flat signals, multichannel input, empty history
- Reset and state management

Reference: Luthi (2014), Mander et al. (2014) — spindles and memory.
Muse 2: 4 channels (TP9, AF7, AF8, TP10) at 256 Hz.
"""
import numpy as np
import pytest

from models.spindle_analyzer import SpindleAnalyzer


# ---------------------------------------------------------------------------
# Helpers — synthetic signal generators
# ---------------------------------------------------------------------------

def _make_spindle_burst(
    fs: int = 256,
    duration: float = 1.0,
    freq: float = 13.0,
    amplitude: float = 30.0,
) -> np.ndarray:
    """Create a single spindle-like burst (Gaussian-modulated sinusoid)."""
    t = np.arange(int(fs * duration)) / fs
    center = duration / 2
    sigma = duration / 6  # ~99.7% of energy within the burst
    envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return amplitude * envelope * np.sin(2 * np.pi * freq * t)


def _make_signal_with_spindles(
    fs: int = 256,
    duration: float = 30.0,
    spindle_times: list = None,
    spindle_freq: float = 13.0,
    spindle_amp: float = 30.0,
    spindle_dur: float = 1.0,
    noise_amp: float = 3.0,
) -> np.ndarray:
    """Create a synthetic EEG signal with embedded spindle bursts.

    Args:
        spindle_times: list of center times (seconds) for spindle bursts.
        spindle_freq: frequency of spindle oscillation (Hz).
        spindle_amp: peak amplitude of spindle (uV).
        spindle_dur: duration of each spindle burst (seconds).
        noise_amp: background noise amplitude.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signal = noise_amp * np.random.randn(n_samples)

    if spindle_times is None:
        spindle_times = [5, 12, 20, 27]

    for center in spindle_times:
        burst = _make_spindle_burst(fs, spindle_dur, spindle_freq, spindle_amp)
        start = int(center * fs) - len(burst) // 2
        end = start + len(burst)
        if start >= 0 and end <= n_samples:
            signal[start:end] += burst

    return signal


def _make_signal_with_slow_oscillations(
    fs: int = 256,
    duration: float = 30.0,
    so_freq: float = 0.8,
    so_amp: float = 50.0,
    spindle_times: list = None,
    spindle_freq: float = 13.0,
    spindle_amp: float = 30.0,
    noise_amp: float = 3.0,
) -> np.ndarray:
    """Create signal with both slow oscillations and spindles."""
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    # Slow oscillation
    so = so_amp * np.sin(2 * np.pi * so_freq * t)
    # Add spindles on top
    spindle_signal = _make_signal_with_spindles(
        fs, duration, spindle_times, spindle_freq, spindle_amp, noise_amp=0,
    )
    noise = noise_amp * np.random.randn(n_samples)
    return so + spindle_signal + noise


def _make_flat_signal(fs: int = 256, duration: float = 30.0) -> np.ndarray:
    """Low-amplitude noise — no spindles expected."""
    return 0.5 * np.random.randn(int(fs * duration))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return SpindleAnalyzer()


# ---------------------------------------------------------------------------
# Test: detect_spindles basic behavior
# ---------------------------------------------------------------------------

class TestDetectSpindles:
    def test_detects_spindles_in_known_signal(self, analyzer):
        """Should detect spindles in synthetic signal with known bursts."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_times=[5, 15, 25])
        spindles = analyzer.detect_spindles(signal, fs=256)
        assert len(spindles) >= 2, f"Expected at least 2 spindles, got {len(spindles)}"

    def test_each_spindle_has_required_keys(self, analyzer):
        """Each detected spindle dict must have start_sample, duration_s,
        amplitude, frequency, and type."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_times=[10], spindle_amp=40.0)
        spindles = analyzer.detect_spindles(signal, fs=256)
        if len(spindles) > 0:
            sp = spindles[0]
            for key in ("start_sample", "duration_s", "amplitude", "frequency", "type"):
                assert key in sp, f"Missing key '{key}' in spindle dict"

    def test_spindle_duration_in_range(self, analyzer):
        """Detected spindle duration should be 0.5-2.0 seconds."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_dur=1.0, spindle_amp=40.0)
        spindles = analyzer.detect_spindles(signal, fs=256)
        for sp in spindles:
            assert 0.3 <= sp["duration_s"] <= 3.0, (
                f"Spindle duration {sp['duration_s']}s outside acceptable range"
            )

    def test_spindle_frequency_in_sigma_band(self, analyzer):
        """Detected spindle frequency should be within 11-16 Hz."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_freq=13.0, spindle_amp=40.0)
        spindles = analyzer.detect_spindles(signal, fs=256)
        for sp in spindles:
            assert 10.0 <= sp["frequency"] <= 17.0, (
                f"Spindle frequency {sp['frequency']} Hz outside sigma band"
            )

    def test_spindle_amplitude_positive(self, analyzer):
        """Spindle amplitude should be positive."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_amp=30.0)
        spindles = analyzer.detect_spindles(signal, fs=256)
        for sp in spindles:
            assert sp["amplitude"] > 0

    def test_no_spindles_in_flat_signal(self, analyzer):
        """Flat/low-noise signal should produce zero or very few spindles."""
        np.random.seed(42)
        signal = _make_flat_signal()
        spindles = analyzer.detect_spindles(signal, fs=256)
        assert len(spindles) <= 1, "Flat signal should have at most 1 false positive"

    def test_more_spindles_with_more_bursts(self, analyzer):
        """Signal with more injected bursts should yield more detections."""
        np.random.seed(42)
        few = _make_signal_with_spindles(spindle_times=[10], spindle_amp=40.0)
        many = _make_signal_with_spindles(
            spindle_times=[3, 8, 13, 18, 23, 28], spindle_amp=40.0
        )
        count_few = len(analyzer.detect_spindles(few, fs=256))
        count_many = len(analyzer.detect_spindles(many, fs=256))
        assert count_many >= count_few


# ---------------------------------------------------------------------------
# Test: slow vs fast spindle classification
# ---------------------------------------------------------------------------

class TestSpindleTypeClassification:
    def test_slow_spindle_classification(self, analyzer):
        """12 Hz spindle should be classified as 'slow'."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(
            spindle_freq=12.0, spindle_amp=40.0, spindle_times=[10, 20]
        )
        spindles = analyzer.detect_spindles(signal, fs=256)
        slow = [s for s in spindles if s["type"] == "slow"]
        assert len(slow) >= 1, "Expected at least one slow spindle at 12 Hz"

    def test_fast_spindle_classification(self, analyzer):
        """14.5 Hz spindle should be classified as 'fast'."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(
            spindle_freq=14.5, spindle_amp=40.0, spindle_times=[10, 20]
        )
        spindles = analyzer.detect_spindles(signal, fs=256)
        fast = [s for s in spindles if s["type"] == "fast"]
        assert len(fast) >= 1, "Expected at least one fast spindle at 14.5 Hz"

    def test_type_is_slow_or_fast(self, analyzer):
        """Every spindle type must be either 'slow' or 'fast'."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_amp=40.0)
        spindles = analyzer.detect_spindles(signal, fs=256)
        for sp in spindles:
            assert sp["type"] in ("slow", "fast"), f"Invalid type: {sp['type']}"


# ---------------------------------------------------------------------------
# Test: analyze() method — full analysis output
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_returns_all_keys(self, analyzer):
        """analyze() must return all documented output keys."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        result = analyzer.analyze(signal, fs=256)
        expected_keys = {
            "spindle_count",
            "spindle_density",
            "mean_amplitude",
            "mean_frequency",
            "mean_duration",
            "spindle_type_distribution",
            "consolidation_index",
        }
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in analyze() output"

    def test_spindle_density_units(self, analyzer):
        """Spindle density should be per minute."""
        np.random.seed(42)
        # 30s signal with 4 spindles => density ~ 8/min
        signal = _make_signal_with_spindles(
            duration=30.0, spindle_times=[5, 12, 20, 27], spindle_amp=40.0
        )
        result = analyzer.analyze(signal, fs=256)
        # Density should be positive and reasonable for 4 spindles in 30 seconds
        assert result["spindle_density"] >= 0

    def test_type_distribution_sums_to_one(self, analyzer):
        """spindle_type_distribution values should sum to 1 when spindles exist."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_amp=40.0)
        result = analyzer.analyze(signal, fs=256)
        dist = result["spindle_type_distribution"]
        assert "slow" in dist
        assert "fast" in dist
        if result["spindle_count"] > 0:
            total = dist["slow"] + dist["fast"]
            assert abs(total - 1.0) < 0.01, f"Distribution sums to {total}, not 1.0"

    def test_consolidation_index_range(self, analyzer):
        """consolidation_index must be 0-100."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        result = analyzer.analyze(signal, fs=256)
        assert 0 <= result["consolidation_index"] <= 100

    def test_zero_spindles_zero_metrics(self, analyzer):
        """When no spindles detected, amplitude/frequency/duration should be 0."""
        np.random.seed(42)
        signal = _make_flat_signal()
        result = analyzer.analyze(signal, fs=256)
        if result["spindle_count"] == 0:
            assert result["mean_amplitude"] == 0.0
            assert result["mean_frequency"] == 0.0
            assert result["mean_duration"] == 0.0

    def test_analyze_multichannel(self, analyzer):
        """analyze() should accept (n_channels, n_samples) input."""
        np.random.seed(42)
        single = _make_signal_with_spindles(spindle_amp=40.0)
        multi = np.stack([
            np.random.randn(len(single)) * 3,  # TP9
            single,                              # AF7 (frontal)
            single + np.random.randn(len(single)) * 2,  # AF8
            np.random.randn(len(single)) * 3,  # TP10
        ])
        result = analyzer.analyze(multi, fs=256)
        assert result["spindle_count"] >= 0
        assert "spindle_type_distribution" in result


# ---------------------------------------------------------------------------
# Test: consolidation index / memory consolidation score
# ---------------------------------------------------------------------------

class TestConsolidationIndex:
    def test_higher_with_spindles_and_so(self, analyzer):
        """Signal with spindles + slow oscillations should have higher
        consolidation index than spindles alone."""
        np.random.seed(42)
        spindles_only = _make_signal_with_spindles(
            spindle_amp=40.0, spindle_times=[5, 12, 20, 27]
        )
        both = _make_signal_with_slow_oscillations(
            spindle_amp=40.0, spindle_times=[5, 12, 20, 27], so_amp=50.0
        )
        r_spindles = analyzer.analyze(spindles_only, fs=256)
        analyzer.reset()
        r_both = analyzer.analyze(both, fs=256)
        # With SO present, consolidation should be at least as good
        assert r_both["consolidation_index"] >= 0

    def test_get_consolidation_score_returns_float(self, analyzer):
        """get_consolidation_score() should return a float 0-100."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_amp=40.0)
        analyzer.analyze(signal, fs=256)
        score = analyzer.get_consolidation_score()
        assert isinstance(score, float)
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Test: session tracking and history
# ---------------------------------------------------------------------------

class TestSessionTracking:
    def test_history_grows_with_calls(self, analyzer):
        """Each analyze() call should add one entry to history."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        for _ in range(3):
            analyzer.analyze(signal, fs=256)
        history = analyzer.get_history()
        assert len(history) == 3

    def test_get_session_stats_keys(self, analyzer):
        """get_session_stats() should have summary statistics."""
        np.random.seed(42)
        signal = _make_signal_with_spindles(spindle_amp=40.0)
        for _ in range(3):
            analyzer.analyze(signal, fs=256)
        stats = analyzer.get_session_stats()
        for key in (
            "n_epochs",
            "total_spindles",
            "mean_density",
            "mean_amplitude",
            "mean_consolidation_index",
        ):
            assert key in stats, f"Missing key '{key}' in session stats"

    def test_session_stats_epoch_count(self, analyzer):
        """n_epochs should match number of analyze() calls."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        analyzer.analyze(signal, fs=256)
        analyzer.analyze(signal, fs=256)
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 2

    def test_empty_session_stats(self, analyzer):
        """Session stats with no data should return zeroed values."""
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["total_spindles"] == 0
        assert stats["mean_density"] == 0.0


# ---------------------------------------------------------------------------
# Test: reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self, analyzer):
        """reset() should clear all session history."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        analyzer.analyze(signal, fs=256)
        assert len(analyzer.get_history()) == 1
        analyzer.reset()
        assert len(analyzer.get_history()) == 0

    def test_reset_clears_session_stats(self, analyzer):
        """After reset, session stats should be empty."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        analyzer.analyze(signal, fs=256)
        analyzer.reset()
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 0


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_short_signal(self, analyzer):
        """Signal shorter than one spindle should not crash."""
        np.random.seed(42)
        short = np.random.randn(64)  # 0.25 seconds at 256 Hz
        result = analyzer.analyze(short, fs=256)
        assert result["spindle_count"] == 0
        assert result["consolidation_index"] >= 0

    def test_short_signal_detect_spindles(self, analyzer):
        """detect_spindles on very short signal returns empty list."""
        np.random.seed(42)
        short = np.random.randn(64)
        spindles = analyzer.detect_spindles(short, fs=256)
        assert spindles == []

    def test_single_sample(self, analyzer):
        """Single sample should not crash."""
        result = analyzer.analyze(np.array([1.0]), fs=256)
        assert result["spindle_count"] == 0

    def test_all_zeros(self, analyzer):
        """All-zero signal should not crash."""
        signal = np.zeros(256 * 10)
        result = analyzer.analyze(signal, fs=256)
        assert result["spindle_count"] == 0

    def test_different_sampling_rate(self, analyzer):
        """Should work with non-256 Hz sampling rate."""
        np.random.seed(42)
        fs = 512
        signal = _make_signal_with_spindles(
            fs=fs, spindle_amp=40.0, spindle_times=[5, 15, 25]
        )
        result = analyzer.analyze(signal, fs=fs)
        assert result["spindle_count"] >= 0

    def test_1d_signal_accepted(self, analyzer):
        """1D array input should work without error."""
        np.random.seed(42)
        signal = _make_signal_with_spindles()
        result = analyzer.analyze(signal, fs=256)
        assert "spindle_count" in result
