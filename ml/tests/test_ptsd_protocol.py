"""Tests for PTSD neurofeedback protocol."""
import numpy as np
import pytest

from models.ptsd_protocol import PTSDProtocol


@pytest.fixture
def protocol():
    return PTSDProtocol(fs=256.0)


def _make_eeg(
    fs=256,
    duration=4,
    n_channels=4,
    alpha_amp_left=10.0,
    alpha_amp_right=10.0,
    hbeta_amp=5.0,
    theta_amp=5.0,
    noise_amp=2.0,
):
    """Synthetic 4-channel EEG with controllable asymmetry and band powers.

    Channel layout (Muse 2):
      ch0=TP9 (left temporal), ch1=AF7 (left frontal),
      ch2=AF8 (right frontal), ch3=TP10 (right temporal).

    Alpha asymmetry is controlled via alpha_amp_left (AF7) and
    alpha_amp_right (AF8).
    """
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        # Alpha: different amplitudes for frontal channels
        if ch == 1:  # AF7 (left frontal)
            alpha = alpha_amp_left * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        elif ch == 2:  # AF8 (right frontal)
            alpha = alpha_amp_right * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        else:
            alpha = ((alpha_amp_left + alpha_amp_right) / 2) * np.sin(
                2 * np.pi * 10 * t + ch * 0.3
            )
        hbeta = hbeta_amp * np.sin(2 * np.pi * 25 * t + ch * 0.5)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.7)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + hbeta + theta + noise)
    return np.array(signals)


# ── Baseline ───────────────────────────────────────────────────────


class TestBaseline:
    def test_set_baseline_returns_expected_keys(self, protocol):
        np.random.seed(42)
        result = protocol.set_baseline(_make_eeg())
        assert result["baseline_set"] is True
        assert "faa_baseline" in result
        assert "alpha_left" in result
        assert "alpha_right" in result

    def test_baseline_faa_symmetric(self, protocol):
        """Equal left/right alpha should give FAA near zero."""
        np.random.seed(42)
        result = protocol.set_baseline(
            _make_eeg(alpha_amp_left=10, alpha_amp_right=10)
        )
        assert abs(result["faa_baseline"]) < 0.5

    def test_baseline_faa_right_excess(self, protocol):
        """Excess right alpha (PTSD pattern) gives positive FAA."""
        np.random.seed(42)
        result = protocol.set_baseline(
            _make_eeg(alpha_amp_left=5, alpha_amp_right=15)
        )
        # FAA = ln(right_alpha) - ln(left_alpha)  ->  positive when right > left
        assert result["faa_baseline"] > 0

    def test_single_channel_input(self, protocol):
        """1D signal should not crash -- graceful handling."""
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = protocol.set_baseline(sig)
        assert result["baseline_set"] is True


# ── Evaluate: output structure ─────────────────────────────────────


class TestEvaluateOutputStructure:
    def test_evaluate_returns_all_required_keys(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.evaluate(_make_eeg())
        required_keys = {
            "faa_current",
            "faa_baseline",
            "asymmetry_normalized",
            "hyperarousal_detected",
            "dissociation_risk",
            "regulation_score",
            "feedback_message",
            "clinical_disclaimer",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_clinical_disclaimer_always_present(self, protocol):
        np.random.seed(42)
        result = protocol.evaluate(_make_eeg())
        assert "clinical_disclaimer" in result
        assert len(result["clinical_disclaimer"]) > 10
        assert "not a substitute" in result["clinical_disclaimer"].lower() or \
               "clinical" in result["clinical_disclaimer"].lower()

    def test_regulation_score_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.evaluate(_make_eeg())
        assert 0 <= result["regulation_score"] <= 100

    def test_dissociation_risk_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.evaluate(_make_eeg())
        assert 0 <= result["dissociation_risk"] <= 1

    def test_asymmetry_normalized_is_bool(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.evaluate(_make_eeg())
        assert isinstance(result["asymmetry_normalized"], bool)

    def test_hyperarousal_detected_is_bool(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.evaluate(_make_eeg())
        assert isinstance(result["hyperarousal_detected"], bool)


# ── Evaluate: asymmetry normalization ──────────────────────────────


class TestAsymmetryNormalization:
    def test_balanced_asymmetry_is_normalized(self, protocol):
        """Balanced left/right alpha should be considered normalized."""
        np.random.seed(42)
        protocol.set_baseline(
            _make_eeg(alpha_amp_left=10, alpha_amp_right=10)
        )
        result = protocol.evaluate(
            _make_eeg(alpha_amp_left=10, alpha_amp_right=10)
        )
        # Near-zero FAA should be considered normalized
        assert result["asymmetry_normalized"] is True

    def test_right_excess_not_normalized(self, protocol):
        """Large right-alpha excess (PTSD pattern) should NOT be normalized."""
        np.random.seed(42)
        protocol.set_baseline(
            _make_eeg(alpha_amp_left=5, alpha_amp_right=20)
        )
        result = protocol.evaluate(
            _make_eeg(alpha_amp_left=5, alpha_amp_right=20)
        )
        # Strong positive FAA -> not normalized
        assert result["asymmetry_normalized"] is False

    def test_left_dominant_is_normalized(self, protocol):
        """Left-dominant FAA (healthy pattern) should be normalized."""
        np.random.seed(42)
        protocol.set_baseline(
            _make_eeg(alpha_amp_left=15, alpha_amp_right=10)
        )
        result = protocol.evaluate(
            _make_eeg(alpha_amp_left=15, alpha_amp_right=10)
        )
        # Negative FAA (left > right alpha) -> normalized or at least not PTSD pattern
        assert result["asymmetry_normalized"] is True


# ── Hyperarousal detection ─────────────────────────────────────────


class TestHyperarousal:
    def test_detect_hyperarousal_returns_dict(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg(hbeta_amp=5))
        result = protocol.detect_hyperarousal(_make_eeg(hbeta_amp=20), fs=256)
        assert isinstance(result, dict)
        assert "hyperarousal_detected" in result

    def test_high_beta_triggers_hyperarousal(self, protocol):
        """Very high high-beta should trigger hyperarousal detection."""
        np.random.seed(42)
        protocol.set_baseline(_make_eeg(hbeta_amp=5))
        result = protocol.detect_hyperarousal(
            _make_eeg(hbeta_amp=30, alpha_amp_left=3, alpha_amp_right=3), fs=256
        )
        assert result["hyperarousal_detected"] is True

    def test_low_beta_no_hyperarousal(self, protocol):
        """Low high-beta should not trigger hyperarousal."""
        np.random.seed(42)
        protocol.set_baseline(_make_eeg(hbeta_amp=5))
        result = protocol.detect_hyperarousal(
            _make_eeg(hbeta_amp=3, alpha_amp_left=15, alpha_amp_right=15), fs=256
        )
        assert result["hyperarousal_detected"] is False

    def test_hyperarousal_intensity_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg(hbeta_amp=5))
        result = protocol.detect_hyperarousal(_make_eeg(hbeta_amp=15), fs=256)
        assert 0 <= result["hyperarousal_intensity"] <= 1


# ── Dissociation detection ─────────────────────────────────────────


class TestDissociation:
    def test_detect_dissociation_returns_dict(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.detect_dissociation(_make_eeg(), fs=256)
        assert isinstance(result, dict)
        assert "dissociation_risk" in result

    def test_theta_surge_raises_dissociation(self, protocol):
        """Sudden theta dominance should raise dissociation risk."""
        np.random.seed(42)
        # Normal baseline
        protocol.set_baseline(
            _make_eeg(theta_amp=5, alpha_amp_left=10, alpha_amp_right=10)
        )
        # Sudden theta surge with alpha collapse
        result = protocol.detect_dissociation(
            _make_eeg(theta_amp=25, alpha_amp_left=3, alpha_amp_right=3), fs=256
        )
        assert result["dissociation_risk"] > 0.4

    def test_normal_theta_low_dissociation(self, protocol):
        """Normal theta levels should give low dissociation risk."""
        np.random.seed(42)
        protocol.set_baseline(
            _make_eeg(theta_amp=5, alpha_amp_left=10, alpha_amp_right=10)
        )
        result = protocol.detect_dissociation(
            _make_eeg(theta_amp=5, alpha_amp_left=10, alpha_amp_right=10), fs=256
        )
        assert result["dissociation_risk"] < 0.4

    def test_dissociation_risk_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.detect_dissociation(_make_eeg(), fs=256)
        assert 0 <= result["dissociation_risk"] <= 1


# ── Evaluate: feedback messages ────────────────────────────────────


class TestFeedbackMessages:
    def test_feedback_message_is_string(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        result = protocol.evaluate(_make_eeg())
        assert isinstance(result["feedback_message"], str)
        assert len(result["feedback_message"]) > 0

    def test_hyperarousal_feedback_mentions_grounding(self, protocol):
        """When hyperarousal detected, feedback should guide calming."""
        np.random.seed(42)
        protocol.set_baseline(_make_eeg(hbeta_amp=5))
        result = protocol.evaluate(
            _make_eeg(hbeta_amp=30, alpha_amp_left=3, alpha_amp_right=3)
        )
        if result["hyperarousal_detected"]:
            msg = result["feedback_message"].lower()
            assert any(
                word in msg
                for word in ["ground", "breath", "safe", "slow", "calm"]
            )


# ── Session stats ──────────────────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self, protocol):
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_evaluations(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        for _ in range(5):
            protocol.evaluate(_make_eeg())
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_regulation_score" in stats
        assert "hyperarousal_count" in stats
        assert "normalization_rate" in stats

    def test_regulation_trend(self, protocol):
        """With enough data, should compute a trend."""
        np.random.seed(42)
        protocol.set_baseline(
            _make_eeg(alpha_amp_left=5, alpha_amp_right=15, hbeta_amp=10)
        )
        # Simulate improving: gradually normalize asymmetry
        for i in range(20):
            shift = i * 0.5
            protocol.evaluate(
                _make_eeg(
                    alpha_amp_left=5 + shift,
                    alpha_amp_right=15 - shift * 0.5,
                    hbeta_amp=10 - i * 0.2,
                )
            )
        stats = protocol.get_session_stats()
        assert stats["trend"] in ("improving", "stable", "worsening", "insufficient_data")


# ── History ────────────────────────────────────────────────────────


class TestHistory:
    def test_empty_history(self, protocol):
        assert protocol.get_history() == []

    def test_history_grows(self, protocol):
        np.random.seed(42)
        protocol.evaluate(_make_eeg())
        protocol.evaluate(_make_eeg())
        assert len(protocol.get_history()) == 2

    def test_history_last_n(self, protocol):
        np.random.seed(42)
        for _ in range(10):
            protocol.evaluate(_make_eeg())
        assert len(protocol.get_history(last_n=3)) == 3


# ── Reset ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_everything(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        protocol.evaluate(_make_eeg())
        protocol.reset()
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

    def test_reset_specific_user(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_eeg(), user_id="a")
        protocol.evaluate(_make_eeg(), user_id="a")
        protocol.set_baseline(_make_eeg(), user_id="b")
        protocol.evaluate(_make_eeg(), user_id="b")
        protocol.reset(user_id="a")
        assert protocol.get_session_stats("a")["n_epochs"] == 0
        assert protocol.get_session_stats("b")["n_epochs"] == 1


# ── Multi-user ─────────────────────────────────────────────────────


class TestMultiUser:
    def test_independent_baselines(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(
            _make_eeg(alpha_amp_left=5, alpha_amp_right=15), user_id="ptsd"
        )
        protocol.set_baseline(
            _make_eeg(alpha_amp_left=10, alpha_amp_right=10), user_id="control"
        )
        protocol.evaluate(_make_eeg(), user_id="ptsd")
        protocol.evaluate(_make_eeg(), user_id="control")
        assert protocol.get_session_stats("ptsd")["n_epochs"] == 1
        assert protocol.get_session_stats("control")["n_epochs"] == 1


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_evaluate_without_baseline(self, protocol):
        """Should still work without baseline, just with default values."""
        np.random.seed(42)
        result = protocol.evaluate(_make_eeg())
        assert "faa_current" in result
        assert "clinical_disclaimer" in result
        assert result["regulation_score"] >= 0

    def test_very_short_signal(self, protocol):
        """Very short signal should not crash."""
        np.random.seed(42)
        short = np.random.randn(4, 32)  # Only 32 samples
        result = protocol.evaluate(short)
        assert "clinical_disclaimer" in result

    def test_single_channel_evaluate(self, protocol):
        """1D signal should be handled gracefully."""
        np.random.seed(42)
        sig = np.random.randn(1024) * 10
        result = protocol.evaluate(sig)
        assert "clinical_disclaimer" in result

    def test_history_cap_at_1000(self, protocol):
        """Session history should not grow unbounded."""
        np.random.seed(42)
        protocol.set_baseline(_make_eeg())
        for _ in range(1050):
            protocol.evaluate(_make_eeg(duration=1))
        assert len(protocol.get_history()) <= 1000
