"""Tests for AlexithymiaDetector.

Covers:
  - Construction and defaults
  - set_baseline: output structure, multi-user, single-channel fallback
  - screen: output structure, score bounds, risk levels, component ranges,
    disclaimer, has_baseline flag, FAA flatness, right dominance,
    coherence deficit, emotional modulation, multi-user isolation
  - get_session_stats: empty, after screening, multi-user
  - get_history: empty, populated, last_n slicing, cap at 500
  - reset: clears baseline and history
  - Edge cases: flat signal, single channel, very short signal
"""

import numpy as np
import pytest

from models.alexithymia_detector import (
    AlexithymiaDetector,
    _band_power,
    _compute_faa,
    _interhemispheric_coherence,
    _DISCLAIMER,
    _MAX_HISTORY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return AlexithymiaDetector(fs=256.0)


@pytest.fixture
def eeg_4ch():
    """4-channel x 4 seconds of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(4, 1024) * 20


@pytest.fixture
def eeg_1ch():
    """Single-channel 4 seconds of synthetic EEG."""
    np.random.seed(42)
    return np.random.randn(1024) * 20


# ── Helper functions ──────────────────────────────────────────────────────────

class TestBandPower:
    def test_returns_float(self, eeg_1ch):
        result = _band_power(eeg_1ch, 256.0, 8.0, 12.0)
        assert isinstance(result, float)

    def test_within_zero_one(self, eeg_1ch):
        result = _band_power(eeg_1ch, 256.0, 8.0, 12.0)
        assert 0.0 <= result <= 1.0

    def test_very_short_signal_returns_zero(self):
        short = np.array([1.0, 2.0])
        result = _band_power(short, 256.0, 8.0, 12.0)
        assert result == 0.0


class TestComputeFAA:
    def test_returns_float(self, eeg_4ch):
        result = _compute_faa(eeg_4ch, 256.0)
        assert isinstance(result, float)

    def test_single_channel_returns_zero(self, eeg_1ch):
        sig = eeg_1ch.reshape(1, -1)
        assert _compute_faa(sig, 256.0) == 0.0

    def test_two_channel_returns_zero(self):
        sig = np.random.randn(2, 1024) * 20
        assert _compute_faa(sig, 256.0) == 0.0


class TestInterhemisphericCoherence:
    def test_returns_float(self, eeg_4ch):
        result = _interhemispheric_coherence(
            eeg_4ch[1], eeg_4ch[2], 256.0, 8.0, 12.0
        )
        assert isinstance(result, float)

    def test_bounded_zero_one(self, eeg_4ch):
        result = _interhemispheric_coherence(
            eeg_4ch[1], eeg_4ch[2], 256.0, 8.0, 12.0
        )
        assert 0.0 <= result <= 1.0

    def test_identical_signals_high_coherence(self):
        np.random.seed(99)
        sig = np.random.randn(1024) * 20
        result = _interhemispheric_coherence(sig, sig.copy(), 256.0, 8.0, 12.0)
        assert result > 0.9


# ── Constructor ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_default_fs(self):
        d = AlexithymiaDetector()
        assert d.fs == 256.0

    def test_custom_fs(self):
        d = AlexithymiaDetector(fs=512.0)
        assert d.fs == 512.0

    def test_no_users_at_init(self):
        d = AlexithymiaDetector()
        assert len(d._users) == 0


# ── set_baseline ──────────────────────────────────────────────────────────────

class TestSetBaseline:
    def test_output_keys(self, detector, eeg_4ch):
        result = detector.set_baseline(eeg_4ch)
        assert "baseline_set" in result
        assert "baseline_faa" in result
        assert "baseline_coherence" in result

    def test_baseline_set_true(self, detector, eeg_4ch):
        result = detector.set_baseline(eeg_4ch)
        assert result["baseline_set"] is True

    def test_baseline_faa_is_float(self, detector, eeg_4ch):
        result = detector.set_baseline(eeg_4ch)
        assert isinstance(result["baseline_faa"], float)

    def test_baseline_coherence_bounded(self, detector, eeg_4ch):
        result = detector.set_baseline(eeg_4ch)
        assert 0.0 <= result["baseline_coherence"] <= 1.0

    def test_single_channel_baseline(self, detector, eeg_1ch):
        result = detector.set_baseline(eeg_1ch)
        assert result["baseline_set"] is True
        # Single channel -> FAA is 0
        assert result["baseline_faa"] == 0.0

    def test_multi_user_baselines_independent(self, detector, eeg_4ch):
        detector.set_baseline(eeg_4ch, user_id="alice")
        # bob has no baseline
        stats = detector.get_session_stats(user_id="bob")
        assert stats["has_baseline"] is False
        stats_alice = detector.get_session_stats(user_id="alice")
        assert stats_alice["has_baseline"] is True

    def test_custom_fs_override(self, detector, eeg_4ch):
        result = detector.set_baseline(eeg_4ch, fs=128.0)
        assert result["baseline_set"] is True


# ── screen ────────────────────────────────────────────────────────────────────

class TestScreen:
    def test_output_keys(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        for key in (
            "alexithymia_score", "risk_level", "faa_flatness",
            "right_dominance", "coherence_deficit", "emotional_modulation",
            "biomarkers", "disclaimer", "has_baseline",
        ):
            assert key in result, f"Missing key: {key}"

    def test_score_bounds(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        assert 0.0 <= result["alexithymia_score"] <= 100.0

    def test_risk_level_valid(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        assert result["risk_level"] in ("low", "mild", "moderate", "elevated")

    def test_component_scores_bounded(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        assert 0.0 <= result["faa_flatness"] <= 1.0
        assert 0.0 <= result["right_dominance"] <= 1.0
        assert 0.0 <= result["coherence_deficit"] <= 1.0
        assert 0.0 <= result["emotional_modulation"] <= 1.0

    def test_disclaimer_present(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        assert result["disclaimer"] == _DISCLAIMER
        assert "not" in result["disclaimer"].lower()
        assert "diagnostic" in result["disclaimer"].lower()

    def test_has_baseline_false_without_baseline(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_baseline(self, detector, eeg_4ch):
        detector.set_baseline(eeg_4ch)
        result = detector.screen(eeg_4ch)
        assert result["has_baseline"] is True

    def test_biomarkers_keys(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch)
        bio = result["biomarkers"]
        for key in (
            "current_faa", "mean_coherence",
            "left_theta_alpha_power", "right_theta_alpha_power", "faa_std",
        ):
            assert key in bio, f"Missing biomarker: {key}"

    def test_single_channel_screen(self, detector, eeg_1ch):
        result = detector.screen(eeg_1ch)
        assert 0.0 <= result["alexithymia_score"] <= 100.0
        assert result["risk_level"] in ("low", "mild", "moderate", "elevated")

    def test_risk_level_low(self, detector):
        assert detector._risk_level(10.0) == "low"
        assert detector._risk_level(0.0) == "low"
        assert detector._risk_level(24.9) == "low"

    def test_risk_level_mild(self, detector):
        assert detector._risk_level(25.0) == "mild"
        assert detector._risk_level(35.0) == "mild"
        assert detector._risk_level(44.9) == "mild"

    def test_risk_level_moderate(self, detector):
        assert detector._risk_level(45.0) == "moderate"
        assert detector._risk_level(60.0) == "moderate"
        assert detector._risk_level(69.9) == "moderate"

    def test_risk_level_elevated(self, detector):
        assert detector._risk_level(70.0) == "elevated"
        assert detector._risk_level(85.0) == "elevated"
        assert detector._risk_level(100.0) == "elevated"

    def test_faa_flatness_increases_with_constant_faa(self, detector):
        """When FAA barely changes across epochs, flatness should be high."""
        np.random.seed(10)
        # Use the same signal repeatedly -> FAA won't change -> high flatness
        sig = np.random.randn(4, 1024) * 20
        for _ in range(10):
            result = detector.screen(sig.copy())
        assert result["faa_flatness"] > 0.5

    def test_emotional_modulation_with_baseline(self, detector, eeg_4ch):
        """Modulation should be computable when baseline is set."""
        detector.set_baseline(eeg_4ch)
        # Screen with the same signal -> low modulation
        result = detector.screen(eeg_4ch)
        assert isinstance(result["emotional_modulation"], float)
        assert 0.0 <= result["emotional_modulation"] <= 1.0

    def test_emotional_modulation_low_same_signal(self, detector, eeg_4ch):
        """Same signal as baseline -> modulation should be low."""
        detector.set_baseline(eeg_4ch)
        result = detector.screen(eeg_4ch)
        # Same data -> near zero change
        assert result["emotional_modulation"] < 0.3

    def test_screen_stores_in_history(self, detector, eeg_4ch):
        detector.screen(eeg_4ch)
        history = detector.get_history()
        assert len(history) == 1

    def test_multi_user_screening_isolated(self, detector, eeg_4ch):
        detector.screen(eeg_4ch, user_id="alice")
        detector.screen(eeg_4ch, user_id="alice")
        detector.screen(eeg_4ch, user_id="bob")
        assert len(detector.get_history(user_id="alice")) == 2
        assert len(detector.get_history(user_id="bob")) == 1

    def test_risk_level_consistent_with_score(self, detector):
        """Verify risk level matches the score across many random signals."""
        np.random.seed(77)
        for _ in range(20):
            sig = np.random.randn(4, 1024) * 20
            result = detector.screen(sig, user_id=f"test_{_}")
            score = result["alexithymia_score"]
            level = result["risk_level"]
            if score < 25:
                assert level == "low", f"score={score}, level={level}"
            elif score < 45:
                assert level == "mild", f"score={score}, level={level}"
            elif score < 70:
                assert level == "moderate", f"score={score}, level={level}"
            else:
                assert level == "elevated", f"score={score}, level={level}"


# ── get_session_stats ─────────────────────────────────────────────────────────

class TestGetSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_score"] == 0.0
        assert stats["has_baseline"] is False

    def test_after_screening(self, detector, eeg_4ch):
        detector.screen(eeg_4ch)
        detector.screen(eeg_4ch)
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 2
        assert isinstance(stats["mean_score"], float)

    def test_has_baseline_in_stats(self, detector, eeg_4ch):
        detector.set_baseline(eeg_4ch)
        stats = detector.get_session_stats()
        assert stats["has_baseline"] is True

    def test_multi_user_stats_independent(self, detector, eeg_4ch):
        detector.screen(eeg_4ch, user_id="alice")
        detector.screen(eeg_4ch, user_id="alice")
        detector.screen(eeg_4ch, user_id="bob")
        assert detector.get_session_stats("alice")["n_epochs"] == 2
        assert detector.get_session_stats("bob")["n_epochs"] == 1


# ── get_history ───────────────────────────────────────────────────────────────

class TestGetHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_populated_history(self, detector, eeg_4ch):
        detector.screen(eeg_4ch)
        h = detector.get_history()
        assert len(h) == 1
        assert "alexithymia_score" in h[0]

    def test_last_n_slicing(self, detector, eeg_4ch):
        for _ in range(5):
            detector.screen(eeg_4ch)
        assert len(detector.get_history(last_n=3)) == 3
        assert len(detector.get_history(last_n=10)) == 5

    def test_history_capped_at_max(self, detector):
        """History should not exceed _MAX_HISTORY entries."""
        np.random.seed(88)
        sig = np.random.randn(4, 256) * 20  # 1 second, fast
        for i in range(_MAX_HISTORY + 10):
            detector.screen(sig)
        assert len(detector.get_history()) == _MAX_HISTORY

    def test_history_returns_copy(self, detector, eeg_4ch):
        detector.screen(eeg_4ch)
        h1 = detector.get_history()
        h1.clear()
        assert len(detector.get_history()) == 1  # original unaffected


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self, detector, eeg_4ch):
        detector.screen(eeg_4ch)
        detector.reset()
        assert detector.get_history() == []

    def test_reset_clears_baseline(self, detector, eeg_4ch):
        detector.set_baseline(eeg_4ch)
        detector.reset()
        stats = detector.get_session_stats()
        assert stats["has_baseline"] is False

    def test_reset_one_user_keeps_others(self, detector, eeg_4ch):
        detector.screen(eeg_4ch, user_id="alice")
        detector.screen(eeg_4ch, user_id="bob")
        detector.reset(user_id="alice")
        assert len(detector.get_history(user_id="alice")) == 0
        assert len(detector.get_history(user_id="bob")) == 1

    def test_reset_nonexistent_user_safe(self, detector):
        """Resetting a user that doesn't exist should not error."""
        detector.reset(user_id="nobody")  # should not raise


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_flat_signal(self, detector):
        """Flat signal should not crash."""
        flat = np.full((4, 1024), 0.001)
        result = detector.screen(flat)
        assert 0.0 <= result["alexithymia_score"] <= 100.0

    def test_three_channel_signal(self, detector):
        """3-channel input should work (no TP10)."""
        sig = np.random.randn(3, 1024) * 20
        result = detector.screen(sig)
        assert 0.0 <= result["alexithymia_score"] <= 100.0
        assert result["risk_level"] in ("low", "mild", "moderate", "elevated")

    def test_two_channel_signal(self, detector):
        """2-channel input should fall back gracefully."""
        sig = np.random.randn(2, 1024) * 20
        result = detector.screen(sig)
        assert 0.0 <= result["alexithymia_score"] <= 100.0

    def test_custom_fs_in_screen(self, detector, eeg_4ch):
        result = detector.screen(eeg_4ch, fs=128.0)
        assert 0.0 <= result["alexithymia_score"] <= 100.0

    def test_large_amplitude_signal(self, detector):
        """Very large amplitude should not crash or produce NaN."""
        large = np.random.randn(4, 1024) * 500
        result = detector.screen(large)
        assert not np.isnan(result["alexithymia_score"])
        assert 0.0 <= result["alexithymia_score"] <= 100.0
