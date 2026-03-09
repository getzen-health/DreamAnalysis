"""Tests for VisualAttentionDetector — EEG-based visual attention direction and focus."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.visual_attention import VisualAttentionDetector


# ── Helpers ────────────────────────────────────────────────────────

def _make_eeg(rng, n_samples=1024, n_channels=4, scale=20.0):
    """Multichannel random EEG."""
    return rng.normal(0, scale, (n_channels, n_samples)).astype(np.float64)


def _make_alpha_lateralized(rng, direction="left", fs=256, duration=4.0):
    """Craft a 4-channel signal with alpha lateralised to one hemisphere.

    Attending LEFT suppresses right-hemisphere alpha (AF8/TP10).
    Attending RIGHT suppresses left-hemisphere alpha (AF7/TP9).
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    alpha_freq = 10.0  # Hz
    alpha_signal = 30.0 * np.sin(2 * np.pi * alpha_freq * t)
    noise = rng.normal(0, 5, (4, n))

    signals = noise.copy()
    if direction == "left":
        # Attending left -> suppress right alpha (ch2=AF8, ch3=TP10)
        # Strong alpha on left (ch0=TP9, ch1=AF7)
        signals[0] += alpha_signal
        signals[1] += alpha_signal
        # Minimal alpha on right
    elif direction == "right":
        # Attending right -> suppress left alpha (ch0=TP9, ch1=AF7)
        # Strong alpha on right (ch2=AF8, ch3=TP10)
        signals[2] += alpha_signal
        signals[3] += alpha_signal
    elif direction == "center":
        # Symmetric alpha
        signals[0] += alpha_signal * 0.7
        signals[1] += alpha_signal * 0.7
        signals[2] += alpha_signal * 0.7
        signals[3] += alpha_signal * 0.7
    return signals


def _make_eyes_closed(rng, fs=256, duration=4.0):
    """Strong bilateral alpha — eyes closed resting state."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    alpha = 50.0 * np.sin(2 * np.pi * 10 * t)
    noise = rng.normal(0, 3, (4, n))
    signals = noise.copy()
    for ch in range(4):
        signals[ch] += alpha
    return signals


def _make_low_alpha(rng, fs=256, duration=4.0):
    """Signal with very little alpha — active visual processing or scanning."""
    n = int(fs * duration)
    # Dominated by beta / noise
    beta = 15.0 * np.sin(2 * np.pi * 20 * t) if False else np.zeros(n)
    signals = rng.normal(0, 15, (4, n))
    return signals


# ── Test: Output Structure ─────────────────────────────────────────

class TestOutputStructure:
    def test_assess_returns_all_keys(self):
        rng = np.random.default_rng(42)
        det = VisualAttentionDetector(fs=256.0)
        eeg = _make_eeg(rng)
        result = det.assess(eeg)

        assert "attention_direction" in result
        assert "laterality_index" in result
        assert "visual_focus_score" in result
        assert "alpha_suppression" in result
        assert "attention_state" in result
        assert "has_baseline" in result

    def test_assess_direction_values(self):
        rng = np.random.default_rng(43)
        det = VisualAttentionDetector()
        eeg = _make_eeg(rng)
        result = det.assess(eeg)
        assert result["attention_direction"] in {"left", "right", "center", "diffuse"}

    def test_assess_state_values(self):
        rng = np.random.default_rng(44)
        det = VisualAttentionDetector()
        eeg = _make_eeg(rng)
        result = det.assess(eeg)
        assert result["attention_state"] in {"focused", "scanning", "unfocused", "eyes_closed"}

    def test_has_baseline_false_initially(self):
        rng = np.random.default_rng(45)
        det = VisualAttentionDetector()
        result = det.assess(_make_eeg(rng))
        assert result["has_baseline"] is False


# ── Test: Value Ranges ─────────────────────────────────────────────

class TestValueRanges:
    def test_laterality_index_range(self):
        rng = np.random.default_rng(50)
        det = VisualAttentionDetector()
        for seed in range(50, 55):
            rng2 = np.random.default_rng(seed)
            result = det.assess(_make_eeg(rng2))
            assert -1.0 <= result["laterality_index"] <= 1.0

    def test_visual_focus_score_range(self):
        rng = np.random.default_rng(60)
        det = VisualAttentionDetector()
        for seed in range(60, 65):
            rng2 = np.random.default_rng(seed)
            result = det.assess(_make_eeg(rng2))
            assert 0 <= result["visual_focus_score"] <= 100

    def test_alpha_suppression_range(self):
        rng = np.random.default_rng(70)
        det = VisualAttentionDetector()
        result = det.assess(_make_eeg(rng))
        assert 0.0 <= result["alpha_suppression"] <= 1.0

    def test_visual_focus_score_is_numeric(self):
        rng = np.random.default_rng(71)
        det = VisualAttentionDetector()
        result = det.assess(_make_eeg(rng))
        assert isinstance(result["visual_focus_score"], (int, float))


# ── Test: Laterality Direction ─────────────────────────────────────

class TestLateralityDirection:
    def test_left_attention_negative_laterality(self):
        """Attending left -> right alpha suppressed -> laterality < 0."""
        rng = np.random.default_rng(100)
        det = VisualAttentionDetector()
        # Set baseline with symmetric alpha
        baseline = _make_alpha_lateralized(rng, direction="center")
        det.set_baseline(baseline)
        # Now attending left
        left_signal = _make_alpha_lateralized(rng, direction="left")
        result = det.assess(left_signal)
        assert result["laterality_index"] < 0, (
            f"Attending left should yield negative laterality, got {result['laterality_index']}"
        )

    def test_right_attention_positive_laterality(self):
        """Attending right -> left alpha suppressed -> laterality > 0."""
        rng = np.random.default_rng(101)
        det = VisualAttentionDetector()
        baseline = _make_alpha_lateralized(rng, direction="center")
        det.set_baseline(baseline)
        right_signal = _make_alpha_lateralized(rng, direction="right")
        result = det.assess(right_signal)
        assert result["laterality_index"] > 0, (
            f"Attending right should yield positive laterality, got {result['laterality_index']}"
        )

    def test_center_laterality_near_zero(self):
        """Symmetric alpha -> laterality near zero."""
        rng = np.random.default_rng(102)
        det = VisualAttentionDetector()
        center_signal = _make_alpha_lateralized(rng, direction="center")
        result = det.assess(center_signal)
        assert abs(result["laterality_index"]) < 0.5, (
            f"Center attention should have laterality near zero, got {result['laterality_index']}"
        )


# ── Test: Alpha Suppression ───────────────────────────────────────

class TestAlphaSuppression:
    def test_eyes_closed_low_suppression(self):
        """Eyes closed (high alpha) -> alpha_suppression near 0."""
        rng = np.random.default_rng(110)
        det = VisualAttentionDetector()
        # Set baseline with eyes-closed alpha
        baseline = _make_eyes_closed(rng)
        det.set_baseline(baseline)
        # Assess with same high-alpha signal
        eyes_closed = _make_eyes_closed(rng)
        result = det.assess(eyes_closed)
        # When current alpha matches baseline, suppression should be low
        assert result["alpha_suppression"] < 0.3

    def test_active_visual_high_suppression(self):
        """Active visual processing (low alpha) -> alpha_suppression high."""
        rng = np.random.default_rng(111)
        det = VisualAttentionDetector()
        baseline = _make_eyes_closed(rng)
        det.set_baseline(baseline)
        # Active visual = low alpha
        active = _make_low_alpha(rng)
        result = det.assess(active)
        assert result["alpha_suppression"] > 0.3


# ── Test: Attention States ─────────────────────────────────────────

class TestAttentionStates:
    def test_eyes_closed_detection(self):
        """Very high bilateral alpha = eyes_closed state."""
        rng = np.random.default_rng(120)
        det = VisualAttentionDetector()
        eeg = _make_eyes_closed(rng)
        result = det.assess(eeg)
        assert result["attention_state"] == "eyes_closed"

    def test_low_alpha_not_eyes_closed(self):
        """Low alpha signal should not be classified as eyes_closed."""
        rng = np.random.default_rng(121)
        det = VisualAttentionDetector()
        eeg = _make_low_alpha(rng)
        result = det.assess(eeg)
        assert result["attention_state"] != "eyes_closed"


# ── Test: Baseline ─────────────────────────────────────────────────

class TestBaseline:
    def test_set_baseline_returns_structure(self):
        rng = np.random.default_rng(130)
        det = VisualAttentionDetector()
        result = det.set_baseline(_make_eeg(rng))
        assert result["baseline_set"] is True
        assert "baseline_alpha" in result

    def test_has_baseline_after_set(self):
        rng = np.random.default_rng(131)
        det = VisualAttentionDetector()
        det.set_baseline(_make_eeg(rng))
        result = det.assess(_make_eeg(rng))
        assert result["has_baseline"] is True

    def test_baseline_per_user(self):
        rng = np.random.default_rng(132)
        det = VisualAttentionDetector()
        det.set_baseline(_make_eeg(rng), user_id="alice")
        # alice has baseline
        r1 = det.assess(_make_eeg(rng), user_id="alice")
        assert r1["has_baseline"] is True
        # bob does not
        r2 = det.assess(_make_eeg(rng), user_id="bob")
        assert r2["has_baseline"] is False


# ── Test: Session Stats ────────────────────────────────────────────

class TestSessionStats:
    def test_empty_stats(self):
        det = VisualAttentionDetector()
        stats = det.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_assessments(self):
        rng = np.random.default_rng(140)
        det = VisualAttentionDetector()
        for _ in range(5):
            det.assess(_make_eeg(rng))
        stats = det.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_focus" in stats
        assert "dominant_direction" in stats
        assert "direction_distribution" in stats
        assert 0 <= stats["mean_focus"] <= 100

    def test_stats_direction_distribution_sums(self):
        rng = np.random.default_rng(141)
        det = VisualAttentionDetector()
        for i in range(10):
            det.assess(_make_eeg(np.random.default_rng(200 + i)))
        stats = det.get_session_stats()
        total = sum(stats["direction_distribution"].values())
        assert total == 10


# ── Test: History ──────────────────────────────────────────────────

class TestHistory:
    def test_empty_history(self):
        det = VisualAttentionDetector()
        h = det.get_history()
        assert h == []

    def test_history_grows(self):
        rng = np.random.default_rng(150)
        det = VisualAttentionDetector()
        det.assess(_make_eeg(rng))
        det.assess(_make_eeg(rng))
        h = det.get_history()
        assert len(h) == 2

    def test_history_last_n(self):
        rng = np.random.default_rng(151)
        det = VisualAttentionDetector()
        for _ in range(10):
            det.assess(_make_eeg(rng))
        h = det.get_history(last_n=3)
        assert len(h) == 3

    def test_history_capped_at_500(self):
        rng = np.random.default_rng(152)
        det = VisualAttentionDetector()
        # Push 510 epochs
        eeg = _make_eeg(rng)
        for _ in range(510):
            det.assess(eeg)
        h = det.get_history()
        assert len(h) <= 500


# ── Test: Multi-User ───────────────────────────────────────────────

class TestMultiUser:
    def test_independent_histories(self):
        rng = np.random.default_rng(160)
        det = VisualAttentionDetector()
        det.assess(_make_eeg(rng), user_id="alice")
        det.assess(_make_eeg(rng), user_id="alice")
        det.assess(_make_eeg(rng), user_id="bob")

        assert len(det.get_history(user_id="alice")) == 2
        assert len(det.get_history(user_id="bob")) == 1

    def test_independent_baselines(self):
        rng = np.random.default_rng(161)
        det = VisualAttentionDetector()
        det.set_baseline(_make_eeg(rng), user_id="alice")
        stats_a = det.get_session_stats(user_id="alice")
        stats_b = det.get_session_stats(user_id="bob")
        # alice should show baseline related info but bob should not
        assert stats_a["n_epochs"] == 0  # no assessments yet
        assert stats_b["n_epochs"] == 0


# ── Test: Reset ────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self):
        rng = np.random.default_rng(170)
        det = VisualAttentionDetector()
        det.assess(_make_eeg(rng))
        det.assess(_make_eeg(rng))
        det.reset()
        assert det.get_history() == []

    def test_reset_clears_baseline(self):
        rng = np.random.default_rng(171)
        det = VisualAttentionDetector()
        det.set_baseline(_make_eeg(rng))
        det.reset()
        result = det.assess(_make_eeg(rng))
        assert result["has_baseline"] is False

    def test_reset_user_isolation(self):
        rng = np.random.default_rng(172)
        det = VisualAttentionDetector()
        det.assess(_make_eeg(rng), user_id="alice")
        det.assess(_make_eeg(rng), user_id="bob")
        det.reset(user_id="alice")
        assert len(det.get_history(user_id="alice")) == 0
        assert len(det.get_history(user_id="bob")) == 1


# ── Test: Edge Cases ──────────────────────────────────────────────

class TestEdgeCases:
    def test_single_channel(self):
        """Single-channel (1D) input should work without crashing."""
        rng = np.random.default_rng(180)
        det = VisualAttentionDetector()
        eeg = rng.normal(0, 20, 1024)
        result = det.assess(eeg)
        assert "attention_direction" in result
        assert result["laterality_index"] == 0.0  # no asymmetry with 1 channel

    def test_zero_signal(self):
        """All-zero signal should not raise."""
        det = VisualAttentionDetector()
        eeg = np.zeros((4, 1024))
        result = det.assess(eeg)
        assert isinstance(result["visual_focus_score"], (int, float))

    def test_short_signal(self):
        """Very short signal (< 1 sec) should still return valid output."""
        rng = np.random.default_rng(182)
        det = VisualAttentionDetector()
        eeg = rng.normal(0, 20, (4, 64))  # 0.25 sec at 256 Hz
        result = det.assess(eeg)
        assert 0 <= result["visual_focus_score"] <= 100

    def test_custom_fs(self):
        """Custom sampling rate should work."""
        rng = np.random.default_rng(183)
        det = VisualAttentionDetector(fs=512.0)
        eeg = rng.normal(0, 20, (4, 2048))
        result = det.assess(eeg, fs=512.0)
        assert "attention_direction" in result

    def test_two_channels(self):
        """2-channel input should work (only frontal)."""
        rng = np.random.default_rng(184)
        det = VisualAttentionDetector()
        eeg = rng.normal(0, 20, (2, 1024))
        result = det.assess(eeg)
        assert "attention_direction" in result

    def test_fs_override_in_assess(self):
        """fs parameter in assess() overrides constructor fs."""
        rng = np.random.default_rng(185)
        det = VisualAttentionDetector(fs=128.0)
        eeg = rng.normal(0, 20, (4, 1024))
        result = det.assess(eeg, fs=256.0)
        assert "attention_direction" in result

    def test_fs_override_in_set_baseline(self):
        """fs parameter in set_baseline() overrides constructor fs."""
        rng = np.random.default_rng(186)
        det = VisualAttentionDetector(fs=128.0)
        eeg = rng.normal(0, 20, (4, 1024))
        result = det.set_baseline(eeg, fs=256.0)
        assert result["baseline_set"] is True
