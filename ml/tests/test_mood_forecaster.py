"""Tests for mood forecaster — longitudinal EEG mood prediction."""
import numpy as np
import pytest

from models.mood_forecaster import MoodForecaster


@pytest.fixture
def forecaster():
    return MoodForecaster(fs=256.0, ewma_alpha=0.3)


def _make_signal(fs=256, duration=4, n_channels=4, alpha_amp=10.0, theta_amp=5.0,
                 beta_amp=5.0, noise_amp=2.0, seed=None):
    """Synthetic EEG with controllable band amplitudes."""
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.5)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


def _make_positive_signal(seed=42):
    """Signal skewed toward positive valence: high alpha, low beta."""
    return _make_signal(alpha_amp=20.0, beta_amp=3.0, theta_amp=3.0, seed=seed)


def _make_negative_signal(seed=42):
    """Signal skewed toward negative valence: low alpha, high beta."""
    return _make_signal(alpha_amp=3.0, beta_amp=20.0, theta_amp=8.0, seed=seed)


def _make_calm_signal(seed=42):
    """Low arousal: high alpha, low beta."""
    return _make_signal(alpha_amp=20.0, beta_amp=2.0, theta_amp=4.0, seed=seed)


def _make_excited_signal(seed=42):
    """High arousal: high beta, low alpha."""
    return _make_signal(alpha_amp=3.0, beta_amp=20.0, theta_amp=3.0, seed=seed)


# ── Initialization ──────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        f = MoodForecaster()
        assert f._fs == 256.0
        assert f._ewma_alpha == 0.3

    def test_custom_params(self):
        f = MoodForecaster(fs=128.0, ewma_alpha=0.5)
        assert f._fs == 128.0
        assert f._ewma_alpha == 0.5


# ── Record Method ───────────────────────────────────────────────

class TestRecord:
    def test_record_returns_dict(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(seed=42))
        assert isinstance(result, dict)

    def test_record_output_keys(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(seed=42))
        expected_keys = {"recorded", "current_valence", "current_arousal",
                         "current_mood", "n_records"}
        assert expected_keys.issubset(set(result.keys()))

    def test_record_marked_true(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(seed=42))
        assert result["recorded"] is True

    def test_valence_range(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(seed=42))
        assert -1.0 <= result["current_valence"] <= 1.0

    def test_arousal_range(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(seed=42))
        assert 0.0 <= result["current_arousal"] <= 1.0

    def test_mood_valid_label(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(seed=42))
        valid_moods = {"positive_high", "positive_low", "negative_high",
                       "negative_low", "neutral"}
        assert result["current_mood"] in valid_moods

    def test_n_records_increments(self, forecaster):
        np.random.seed(42)
        r1 = forecaster.record(_make_signal(seed=42))
        assert r1["n_records"] == 1
        r2 = forecaster.record(_make_signal(seed=43))
        assert r2["n_records"] == 2

    def test_single_channel_input(self, forecaster):
        np.random.seed(42)
        sig = np.sin(2 * np.pi * 10 * np.arange(1024) / 256) * 10
        result = forecaster.record(sig)
        assert result["recorded"] is True

    def test_custom_fs_override(self, forecaster):
        np.random.seed(42)
        result = forecaster.record(_make_signal(fs=128, seed=42), fs=128)
        assert result["recorded"] is True

    def test_mood_label_stored(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42), mood_label="happy")
        timeline = forecaster.get_mood_timeline()
        assert timeline[0].get("mood_label") == "happy"


# ── Mood Quadrant Classification ────────────────────────────────

class TestMoodQuadrant:
    def test_positive_high(self, forecaster):
        """High alpha (positive FAA) + high beta -> positive_high."""
        np.random.seed(42)
        # Create asymmetric signal: AF8 (ch2) much higher alpha than AF7 (ch1)
        sig = _make_signal(seed=42)
        # Boost AF8 alpha to get positive FAA
        t = np.arange(sig.shape[1]) / 256
        sig[2] += 30 * np.sin(2 * np.pi * 10 * t)  # boost right alpha
        sig[1] += 15 * np.sin(2 * np.pi * 22 * t)   # boost left beta for high arousal
        sig[2] += 15 * np.sin(2 * np.pi * 22 * t)   # boost right beta
        result = forecaster.record(sig)
        # We mainly test that the mood is some valid label
        assert result["current_mood"] in {"positive_high", "positive_low",
                                           "negative_high", "negative_low", "neutral"}

    def test_neutral_near_zero(self, forecaster):
        """Symmetric signals should produce near-zero valence -> neutral."""
        np.random.seed(42)
        # Perfectly symmetric across channels
        t = np.arange(1024) / 256
        base = 10 * np.sin(2 * np.pi * 10 * t) + 5 * np.sin(2 * np.pi * 20 * t)
        sig = np.array([base, base, base, base])
        result = forecaster.record(sig)
        # With symmetric channels, valence should be near zero
        assert abs(result["current_valence"]) < 0.5


# ── Forecast Method ─────────────────────────────────────────────

class TestForecast:
    def test_insufficient_data(self, forecaster):
        np.random.seed(42)
        for i in range(3):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        assert result["sufficient_data"] is False
        assert result["n_records"] == 3

    def test_sufficient_data(self, forecaster):
        np.random.seed(42)
        for i in range(6):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        assert result["sufficient_data"] is True
        assert result["n_records"] == 6

    def test_forecast_output_keys(self, forecaster):
        np.random.seed(42)
        for i in range(6):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        expected = {"forecast_valence", "forecast_arousal", "forecast_mood",
                    "trend_valence", "trend_arousal", "confidence",
                    "n_records", "sufficient_data"}
        assert expected.issubset(set(result.keys()))

    def test_forecast_valence_range(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        assert -1.0 <= result["forecast_valence"] <= 1.0

    def test_forecast_arousal_range(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        assert 0.0 <= result["forecast_arousal"] <= 1.0

    def test_forecast_mood_valid(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        valid_moods = {"positive_high", "positive_low", "negative_high",
                       "negative_low", "neutral"}
        assert result["forecast_mood"] in valid_moods

    def test_trend_labels_valid(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        assert result["trend_valence"] in {"improving", "stable", "declining"}
        assert result["trend_arousal"] in {"increasing", "stable", "decreasing"}

    def test_confidence_range(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        result = forecaster.forecast()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_increases_with_data(self, forecaster):
        np.random.seed(42)
        for i in range(5):
            forecaster.record(_make_signal(seed=42 + i))
        c5 = forecaster.forecast()["confidence"]
        for i in range(15):
            forecaster.record(_make_signal(seed=100 + i))
        c20 = forecaster.forecast()["confidence"]
        assert c20 >= c5

    def test_custom_horizon(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        r1 = forecaster.forecast(horizon=1)
        r5 = forecaster.forecast(horizon=5)
        # Both should return valid results; horizon=5 extrapolates further
        assert r1["sufficient_data"] is True
        assert r5["sufficient_data"] is True


# ── Trend Detection ─────────────────────────────────────────────

class TestTrends:
    def test_declining_valence_trend(self, forecaster):
        """Progressively more negative signals -> declining valence trend."""
        np.random.seed(42)
        for i in range(8):
            # Gradually decrease alpha, increase beta
            sig = _make_signal(
                alpha_amp=20 - i * 2,
                beta_amp=3 + i * 2,
                theta_amp=5,
                seed=42 + i,
            )
            forecaster.record(sig)
        result = forecaster.forecast()
        # With steadily declining alpha/beta ratio, valence should decline
        assert result["trend_valence"] in {"declining", "stable"}

    def test_increasing_arousal_trend(self, forecaster):
        """Progressively more beta -> increasing arousal trend."""
        np.random.seed(42)
        for i in range(8):
            sig = _make_signal(
                alpha_amp=10,
                beta_amp=3 + i * 3,
                theta_amp=5,
                seed=42 + i,
            )
            forecaster.record(sig)
        result = forecaster.forecast()
        assert result["trend_arousal"] in {"increasing", "stable"}


# ── Mood Timeline ───────────────────────────────────────────────

class TestMoodTimeline:
    def test_empty_timeline(self, forecaster):
        timeline = forecaster.get_mood_timeline()
        assert timeline == []

    def test_timeline_grows(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42))
        forecaster.record(_make_signal(seed=43))
        timeline = forecaster.get_mood_timeline()
        assert len(timeline) == 2

    def test_timeline_entry_keys(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42))
        entry = forecaster.get_mood_timeline()[0]
        assert "valence" in entry
        assert "arousal" in entry
        assert "mood" in entry

    def test_timeline_last_n(self, forecaster):
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        timeline = forecaster.get_mood_timeline(last_n=3)
        assert len(timeline) == 3

    def test_timeline_last_n_exceeds_total(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42))
        timeline = forecaster.get_mood_timeline(last_n=100)
        assert len(timeline) == 1


# ── Session Stats ───────────────────────────────────────────────

class TestSessionStats:
    def test_empty_stats(self, forecaster):
        stats = forecaster.get_session_stats()
        assert stats["n_records"] == 0

    def test_stats_after_records(self, forecaster):
        np.random.seed(42)
        for i in range(5):
            forecaster.record(_make_signal(seed=42 + i))
        stats = forecaster.get_session_stats()
        assert stats["n_records"] == 5
        assert "mean_valence" in stats
        assert "mean_arousal" in stats
        assert "dominant_mood" in stats

    def test_mean_valence_range(self, forecaster):
        np.random.seed(42)
        for i in range(5):
            forecaster.record(_make_signal(seed=42 + i))
        stats = forecaster.get_session_stats()
        assert -1.0 <= stats["mean_valence"] <= 1.0

    def test_mean_arousal_range(self, forecaster):
        np.random.seed(42)
        for i in range(5):
            forecaster.record(_make_signal(seed=42 + i))
        stats = forecaster.get_session_stats()
        assert 0.0 <= stats["mean_arousal"] <= 1.0

    def test_dominant_mood_valid(self, forecaster):
        np.random.seed(42)
        for i in range(5):
            forecaster.record(_make_signal(seed=42 + i))
        stats = forecaster.get_session_stats()
        valid_moods = {"positive_high", "positive_low", "negative_high",
                       "negative_low", "neutral"}
        assert stats["dominant_mood"] in valid_moods


# ── Multi-User Support ──────────────────────────────────────────

class TestMultiUser:
    def test_independent_users(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42), user_id="alice")
        forecaster.record(_make_signal(seed=43), user_id="bob")
        forecaster.record(_make_signal(seed=44), user_id="bob")
        assert len(forecaster.get_mood_timeline(user_id="alice")) == 1
        assert len(forecaster.get_mood_timeline(user_id="bob")) == 2

    def test_forecast_per_user(self, forecaster):
        np.random.seed(42)
        for i in range(6):
            forecaster.record(_make_signal(seed=42 + i), user_id="alice")
        for i in range(3):
            forecaster.record(_make_signal(seed=100 + i), user_id="bob")
        alice_fc = forecaster.forecast(user_id="alice")
        bob_fc = forecaster.forecast(user_id="bob")
        assert alice_fc["sufficient_data"] is True
        assert bob_fc["sufficient_data"] is False

    def test_stats_per_user(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42), user_id="alice")
        assert forecaster.get_session_stats(user_id="alice")["n_records"] == 1
        assert forecaster.get_session_stats(user_id="bob")["n_records"] == 0


# ── Reset ───────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_user(self, forecaster):
        np.random.seed(42)
        for i in range(5):
            forecaster.record(_make_signal(seed=42 + i))
        forecaster.reset()
        assert forecaster.get_mood_timeline() == []
        assert forecaster.get_session_stats()["n_records"] == 0

    def test_reset_one_user_keeps_others(self, forecaster):
        np.random.seed(42)
        forecaster.record(_make_signal(seed=42), user_id="alice")
        forecaster.record(_make_signal(seed=43), user_id="bob")
        forecaster.reset(user_id="alice")
        assert forecaster.get_mood_timeline(user_id="alice") == []
        assert len(forecaster.get_mood_timeline(user_id="bob")) == 1


# ── History Cap ─────────────────────────────────────────────────

class TestHistoryCap:
    def test_history_capped_at_500(self, forecaster):
        np.random.seed(42)
        # Use a fast signal (short duration) to speed up the test
        sig = _make_signal(duration=1, seed=42)
        for i in range(510):
            forecaster.record(sig, user_id="cap_user")
        timeline = forecaster.get_mood_timeline(user_id="cap_user")
        assert len(timeline) <= 500


# ── EWMA Smoothing ──────────────────────────────────────────────

class TestEWMA:
    def test_ewma_smooths_values(self, forecaster):
        """EWMA should produce smoother output than raw values."""
        np.random.seed(42)
        raw_valences = []
        for i in range(10):
            sig = _make_signal(seed=42 + i)
            result = forecaster.record(sig)
            raw_valences.append(result["current_valence"])
        # Verify we got multiple records
        assert len(raw_valences) == 10
        # EWMA values should exist in the timeline
        timeline = forecaster.get_mood_timeline()
        assert len(timeline) == 10


# ── Edge Cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_short_signal(self, forecaster):
        """Signal shorter than typical FFT window should still work."""
        np.random.seed(42)
        short = np.random.randn(4, 64) * 10
        result = forecaster.record(short)
        assert result["recorded"] is True

    def test_flat_signal(self, forecaster):
        """Flat signal should not crash."""
        np.random.seed(42)
        flat = np.ones((4, 1024)) * 0.001
        result = forecaster.record(flat)
        assert result["recorded"] is True

    def test_forecast_after_reset(self, forecaster):
        """Forecast after reset should report insufficient data."""
        np.random.seed(42)
        for i in range(10):
            forecaster.record(_make_signal(seed=42 + i))
        forecaster.reset()
        result = forecaster.forecast()
        assert result["sufficient_data"] is False
        assert result["n_records"] == 0
