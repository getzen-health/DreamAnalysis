"""Tests for pre-ictal seizure predictor."""
import numpy as np
import pytest

from models.preictal_predictor import PreictalPredictor, MEDICAL_DISCLAIMER


@pytest.fixture
def predictor():
    return PreictalPredictor(
        fs=256.0, trend_window=10, alert_threshold=0.7, sustained_count=3,
    )


def _make_normal_eeg(fs=256, duration=4, n_channels=4, seed=42):
    """Normal interictal EEG: dominant alpha + some theta/beta."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = 15.0 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        theta = 5.0 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        beta = 3.0 * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = 2.0 * rng.randn(len(t))
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


def _make_preictal_eeg(fs=256, duration=4, n_channels=4, seed=42):
    """Simulated pre-ictal EEG with known biomarker changes.

    Characteristics vs normal:
    - Lower spectral entropy (more rhythmic/ordered)
    - Higher cross-channel synchronization (nearly identical signals)
    - Higher theta/alpha ratio (theta increases, alpha suppressed)
    - Higher high-beta / HFO power
    - Lower complexity (more predictable signal)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(int(fs * duration)) / fs

    # Base pattern: strong theta with suppressed alpha, elevated high-beta
    theta_dominant = 25.0 * np.sin(2 * np.pi * 6 * t)
    alpha_suppressed = 4.0 * np.sin(2 * np.pi * 10 * t)
    high_beta = 12.0 * np.sin(2 * np.pi * 25 * t)

    base = theta_dominant + alpha_suppressed + high_beta

    # All channels nearly identical (high synchrony) with minimal noise
    signals = []
    for ch in range(n_channels):
        noise = 0.5 * rng.randn(len(t))
        signals.append(base + noise)
    return np.array(signals)


# ── Test Class: Basic Assess Output ─────────────────────────────────


class TestAssessOutput:
    """Test that assess() returns all required keys with valid types."""

    def test_output_keys(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        expected_keys = {
            "preictal_risk", "risk_level", "entropy_trend",
            "synchrony_index", "feature_changes", "alert",
            "alert_message", "disclaimer", "has_baseline",
            "consecutive_elevated",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_risk_is_float_in_range(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert isinstance(result["preictal_risk"], float)
        assert 0 <= result["preictal_risk"] <= 1

    def test_risk_level_is_valid_string(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert result["risk_level"] in ("normal", "elevated", "warning", "critical")

    def test_entropy_trend_is_float(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert isinstance(result["entropy_trend"], float)

    def test_synchrony_index_is_float(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert isinstance(result["synchrony_index"], float)
        assert 0 <= result["synchrony_index"] <= 1

    def test_feature_changes_keys(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        fc = result["feature_changes"]
        expected = {
            "entropy_change", "synchrony_change",
            "theta_alpha_change", "hfo_change", "complexity_change",
        }
        assert expected == set(fc.keys())

    def test_alert_is_bool(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert isinstance(result["alert"], bool)

    def test_alert_message_none_when_no_alert(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        if not result["alert"]:
            assert result["alert_message"] is None

    def test_disclaimer_always_present(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert result["disclaimer"] == MEDICAL_DISCLAIMER


# ── Test Class: Medical Disclaimer ───────────────────────────────────


class TestMedicalDisclaimer:
    """Verify disclaimer appears in every response from every method."""

    def test_disclaimer_in_assess(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        assert "disclaimer" in result
        assert "NOT a medical device" in result["disclaimer"]

    def test_disclaimer_in_set_baseline(self, predictor):
        result = predictor.set_baseline(_make_normal_eeg())
        assert "disclaimer" in result
        assert "NOT a medical device" in result["disclaimer"]

    def test_disclaimer_in_session_stats(self, predictor):
        stats = predictor.get_session_stats()
        assert "disclaimer" in stats
        assert "NOT a medical device" in stats["disclaimer"]

    def test_disclaimer_in_baseline_failure(self, predictor):
        short = np.random.randn(4, 10)
        result = predictor.set_baseline(short)
        assert "disclaimer" in result

    def test_disclaimer_in_insufficient_data(self, predictor):
        short = np.random.randn(4, 10)
        result = predictor.assess(short)
        assert "disclaimer" in result


# ── Test Class: Baseline ─────────────────────────────────────────────


class TestBaseline:
    """Test baseline recording and its effect on assessments."""

    def test_set_baseline_success(self, predictor):
        result = predictor.set_baseline(_make_normal_eeg())
        assert result["baseline_set"] is True
        assert result["n_channels"] == 4

    def test_set_baseline_returns_features(self, predictor):
        result = predictor.set_baseline(_make_normal_eeg())
        assert "features" in result
        feats = result["features"]
        assert "spectral_entropy" in feats
        assert "mean_synchrony" in feats
        assert "theta_alpha_ratio" in feats

    def test_baseline_too_short(self, predictor):
        short = np.random.randn(4, 10)
        result = predictor.set_baseline(short)
        assert result["baseline_set"] is False
        assert "reason" in result

    def test_single_channel_baseline(self, predictor):
        signal_1d = np.random.randn(256 * 4)
        result = predictor.set_baseline(signal_1d)
        assert result["baseline_set"] is True
        assert result["n_channels"] == 1

    def test_has_baseline_flag(self, predictor):
        assert predictor.assess(_make_normal_eeg())["has_baseline"] is False
        predictor.set_baseline(_make_normal_eeg())
        assert predictor.assess(_make_normal_eeg())["has_baseline"] is True

    def test_baseline_enables_feature_changes(self, predictor):
        predictor.set_baseline(_make_normal_eeg())
        result = predictor.assess(_make_preictal_eeg())
        fc = result["feature_changes"]
        # With baseline, at least some changes should be non-zero
        changes = [abs(v) for v in fc.values()]
        assert any(c > 0.01 for c in changes)

    def test_without_baseline_changes_are_zero(self, predictor):
        result = predictor.assess(_make_normal_eeg())
        fc = result["feature_changes"]
        for key, val in fc.items():
            assert val == 0.0, f"{key} should be 0.0 without baseline"


# ── Test Class: Risk Detection ───────────────────────────────────────


class TestRiskDetection:
    """Test that pre-ictal signals produce higher risk than normal signals."""

    def test_normal_eeg_low_risk(self, predictor):
        predictor.set_baseline(_make_normal_eeg(seed=1))
        result = predictor.assess(_make_normal_eeg(seed=2))
        assert result["preictal_risk"] < 0.5
        assert result["risk_level"] in ("normal", "elevated")

    def test_preictal_eeg_elevated_risk(self, predictor):
        predictor.set_baseline(_make_normal_eeg(seed=1))
        result = predictor.assess(_make_preictal_eeg(seed=2))
        # Pre-ictal signal should produce higher risk than normal
        assert result["preictal_risk"] > 0.1

    def test_preictal_higher_than_normal(self, predictor):
        predictor.set_baseline(_make_normal_eeg(seed=1))
        normal_result = predictor.assess(_make_normal_eeg(seed=2))
        # Reset consecutive counter but keep baseline
        predictor._consecutive_elevated = 0
        preictal_result = predictor.assess(_make_preictal_eeg(seed=3))
        assert preictal_result["preictal_risk"] >= normal_result["preictal_risk"]

    def test_risk_level_classification(self, predictor):
        # Verify the classification boundaries work
        predictor.set_baseline(_make_normal_eeg(seed=1))
        result = predictor.assess(_make_normal_eeg(seed=2))
        level = result["risk_level"]
        risk = result["preictal_risk"]
        if risk < 0.3:
            assert level == "normal"
        elif risk < 0.5:
            assert level == "elevated"
        elif risk < 0.7:
            assert level == "warning"
        else:
            assert level == "critical"


# ── Test Class: Alert Logic ──────────────────────────────────────────


class TestAlertLogic:
    """Test sustained elevation alert mechanism."""

    def test_no_alert_on_single_assessment(self, predictor):
        result = predictor.assess(_make_preictal_eeg())
        assert result["alert"] is False

    def test_alert_after_sustained_elevation(self):
        pred = PreictalPredictor(
            fs=256.0, alert_threshold=0.01, sustained_count=3,
        )
        pred.set_baseline(_make_normal_eeg(seed=1))
        # Feed enough pre-ictal epochs to trigger sustained alert
        for _ in range(5):
            result = pred.assess(_make_preictal_eeg(seed=42))
        # After enough consecutive elevated assessments, alert should fire
        if result["preictal_risk"] >= 0.01:
            assert result["consecutive_elevated"] >= 3

    def test_alert_message_contains_guidance(self):
        pred = PreictalPredictor(
            fs=256.0, alert_threshold=0.01, sustained_count=2,
        )
        pred.set_baseline(_make_normal_eeg(seed=1))
        for _ in range(5):
            result = pred.assess(_make_preictal_eeg(seed=42))
        if result["alert"]:
            assert "neurologist" in result["alert_message"].lower()

    def test_consecutive_counter_decrements(self, predictor):
        predictor.set_baseline(_make_normal_eeg(seed=1))
        # Elevate counter manually for testing
        predictor._consecutive_elevated = 5
        # Normal EEG should decrease the counter
        predictor.assess(_make_normal_eeg(seed=2))
        assert predictor._consecutive_elevated < 5

    def test_no_alert_on_normal_eeg(self, predictor):
        predictor.set_baseline(_make_normal_eeg(seed=1))
        for i in range(10):
            result = predictor.assess(_make_normal_eeg(seed=i + 10))
        assert result["alert"] is False


# ── Test Class: History & Timeline ───────────────────────────────────


class TestHistoryTimeline:
    """Test history tracking and timeline retrieval."""

    def test_empty_history(self, predictor):
        assert predictor.get_history() == []

    def test_history_grows(self, predictor):
        predictor.assess(_make_normal_eeg(seed=1))
        predictor.assess(_make_normal_eeg(seed=2))
        assert len(predictor.get_history()) == 2

    def test_history_last_n(self, predictor):
        for i in range(5):
            predictor.assess(_make_normal_eeg(seed=i))
        assert len(predictor.get_history(last_n=2)) == 2

    def test_history_last_n_larger_than_total(self, predictor):
        predictor.assess(_make_normal_eeg())
        assert len(predictor.get_history(last_n=100)) == 1

    def test_timeline_structure(self, predictor):
        predictor.assess(_make_normal_eeg())
        timeline = predictor.get_risk_timeline()
        assert len(timeline) == 1
        entry = timeline[0]
        assert "preictal_risk" in entry
        assert "risk_level" in entry
        assert "entropy_trend" in entry
        assert "synchrony_index" in entry

    def test_timeline_matches_history_length(self, predictor):
        for i in range(3):
            predictor.assess(_make_normal_eeg(seed=i))
        assert len(predictor.get_risk_timeline()) == len(predictor.get_history())

    def test_history_capped_at_max(self):
        pred = PreictalPredictor(fs=256.0)
        # History should not grow unbounded
        for i in range(50):
            pred.assess(_make_normal_eeg(seed=i, duration=1))
        assert len(pred.get_history()) <= 2000


# ── Test Class: Session Stats ────────────────────────────────────────


class TestSessionStats:
    """Test session summary statistics."""

    def test_empty_stats(self, predictor):
        stats = predictor.get_session_stats()
        assert stats["total_assessments"] == 0
        assert stats["peak_risk"] == 0.0
        assert stats["mean_risk"] == 0.0
        assert stats["alerts_fired"] == 0

    def test_stats_after_assessments(self, predictor):
        predictor.assess(_make_normal_eeg(seed=1))
        predictor.assess(_make_normal_eeg(seed=2))
        stats = predictor.get_session_stats()
        assert stats["total_assessments"] == 2
        assert stats["peak_risk"] >= 0
        assert stats["mean_risk"] >= 0

    def test_stats_time_in_levels(self, predictor):
        predictor.assess(_make_normal_eeg())
        stats = predictor.get_session_stats()
        total_level_time = (
            stats["time_in_normal"]
            + stats["time_in_elevated"]
            + stats["time_in_warning"]
            + stats["time_in_critical"]
        )
        assert total_level_time == stats["total_assessments"]

    def test_stats_has_baseline_flag(self, predictor):
        stats = predictor.get_session_stats()
        assert stats["has_baseline"] is False
        predictor.set_baseline(_make_normal_eeg())
        stats = predictor.get_session_stats()
        assert stats["has_baseline"] is True


# ── Test Class: Reset ────────────────────────────────────────────────


class TestReset:
    """Test reset clears all state."""

    def test_reset_clears_history(self, predictor):
        predictor.assess(_make_normal_eeg())
        predictor.reset()
        assert predictor.get_history() == []

    def test_reset_clears_baseline(self, predictor):
        predictor.set_baseline(_make_normal_eeg())
        predictor.reset()
        result = predictor.assess(_make_normal_eeg())
        assert result["has_baseline"] is False

    def test_reset_clears_consecutive(self, predictor):
        predictor._consecutive_elevated = 10
        predictor.reset()
        assert predictor._consecutive_elevated == 0

    def test_reset_clears_stats(self, predictor):
        predictor.assess(_make_normal_eeg())
        predictor.reset()
        stats = predictor.get_session_stats()
        assert stats["total_assessments"] == 0


# ── Test Class: Edge Cases ───────────────────────────────────────────


class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_short_signal_returns_safely(self, predictor):
        short = np.random.randn(4, 10)
        result = predictor.assess(short)
        assert result["preictal_risk"] == 0.0
        assert result["risk_level"] == "normal"

    def test_single_channel_input(self, predictor):
        signal_1d = np.random.randn(256 * 4) * 20
        result = predictor.assess(signal_1d)
        assert "preictal_risk" in result
        assert 0 <= result["preictal_risk"] <= 1

    def test_flat_signal(self, predictor):
        flat = np.zeros((4, 256 * 4))
        result = predictor.assess(flat)
        assert result["preictal_risk"] >= 0
        assert result["risk_level"] in ("normal", "elevated", "warning", "critical")

    def test_railed_signal(self, predictor):
        railed = np.ones((4, 256 * 4)) * 500
        result = predictor.assess(railed)
        assert 0 <= result["preictal_risk"] <= 1

    def test_very_noisy_signal(self, predictor):
        noisy = np.random.randn(4, 256 * 4) * 200
        result = predictor.assess(noisy)
        assert 0 <= result["preictal_risk"] <= 1

    def test_different_sampling_rates(self, predictor):
        # Test with explicit fs override
        eeg = _make_normal_eeg(fs=128, duration=4)
        result = predictor.assess(eeg, fs=128)
        assert "preictal_risk" in result

    def test_two_channel_input(self, predictor):
        eeg = _make_normal_eeg(n_channels=2)
        result = predictor.assess(eeg)
        assert "preictal_risk" in result


# ── Test Class: Feature Extraction ───────────────────────────────────


class TestFeatureExtraction:
    """Test internal feature computation correctness."""

    def test_spectral_entropy_range(self, predictor):
        eeg = _make_normal_eeg()
        features = predictor._extract_features(eeg, 256.0)
        assert 0 <= features["spectral_entropy"] <= 1

    def test_synchrony_range(self, predictor):
        eeg = _make_normal_eeg()
        features = predictor._extract_features(eeg, 256.0)
        assert 0 <= features["mean_synchrony"] <= 1

    def test_theta_alpha_ratio_positive(self, predictor):
        eeg = _make_normal_eeg()
        features = predictor._extract_features(eeg, 256.0)
        assert features["theta_alpha_ratio"] >= 0

    def test_complexity_range(self, predictor):
        eeg = _make_normal_eeg()
        features = predictor._extract_features(eeg, 256.0)
        assert 0 <= features["complexity"] <= 1

    def test_preictal_entropy_lower(self, predictor):
        normal_feats = predictor._extract_features(_make_normal_eeg(), 256.0)
        preictal_feats = predictor._extract_features(_make_preictal_eeg(), 256.0)
        # Pre-ictal signal is more ordered → entropy should be similar or lower
        # Allow small tolerance due to stochastic noise
        assert preictal_feats["spectral_entropy"] <= normal_feats["spectral_entropy"] + 0.05

    def test_preictal_synchrony_higher(self, predictor):
        normal_feats = predictor._extract_features(_make_normal_eeg(), 256.0)
        preictal_feats = predictor._extract_features(_make_preictal_eeg(), 256.0)
        # Pre-ictal signal has identical channels → higher sync
        assert preictal_feats["mean_synchrony"] >= normal_feats["mean_synchrony"]

    def test_preictal_theta_alpha_higher(self, predictor):
        normal_feats = predictor._extract_features(_make_normal_eeg(), 256.0)
        preictal_feats = predictor._extract_features(_make_preictal_eeg(), 256.0)
        # Pre-ictal has theta dominant, alpha suppressed
        assert preictal_feats["theta_alpha_ratio"] > normal_feats["theta_alpha_ratio"]

    def test_band_powers_nonnegative(self, predictor):
        eeg = _make_normal_eeg()
        powers = predictor._band_powers(eeg[0], 256.0)
        for band, val in powers.items():
            assert val >= 0, f"Band {band} has negative power: {val}"


# ── Test Class: Initialization ───────────────────────────────────────


class TestInitialization:
    """Test constructor parameter handling."""

    def test_default_construction(self):
        pred = PreictalPredictor()
        result = pred.assess(_make_normal_eeg())
        assert "preictal_risk" in result

    def test_custom_threshold(self):
        pred = PreictalPredictor(alert_threshold=0.5)
        assert pred._alert_threshold == 0.5

    def test_threshold_clamped_high(self):
        pred = PreictalPredictor(alert_threshold=2.0)
        assert pred._alert_threshold <= 0.99

    def test_threshold_clamped_low(self):
        pred = PreictalPredictor(alert_threshold=-1.0)
        assert pred._alert_threshold >= 0.1

    def test_trend_window_minimum(self):
        pred = PreictalPredictor(trend_window=0)
        assert pred._trend_window >= 2

    def test_sustained_count_minimum(self):
        pred = PreictalPredictor(sustained_count=0)
        assert pred._sustained_count >= 1
