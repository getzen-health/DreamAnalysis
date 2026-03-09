"""Tests for NeurofeedbackAudioEngine — EEG-driven audio parameter generation.

Covers: protocol management, baseline setting, audio parameter computation,
per-protocol behavior (alpha, SMR, theta), binaural beats, reward detection,
session stats, history tracking, and reset.
"""
import numpy as np
import pytest

from models.neurofeedback_audio import NeurofeedbackAudioEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256


def _make_eeg(duration_s: float = 4.0, n_channels: int = 4,
              dominant_freq: float = None, amplitude: float = 20.0) -> np.ndarray:
    """Generate synthetic multichannel EEG.

    If *dominant_freq* is given the signal is a pure sine at that frequency
    (useful for driving a specific band high).  Otherwise white noise.
    """
    n_samples = int(duration_s * FS)
    t = np.arange(n_samples) / FS
    if dominant_freq is not None:
        base = amplitude * np.sin(2 * np.pi * dominant_freq * t)
        noise = np.random.randn(n_samples) * (amplitude * 0.1)
        row = base + noise
    else:
        row = np.random.randn(n_samples) * amplitude
    return np.tile(row, (n_channels, 1))


def _alpha_eeg(duration_s: float = 4.0) -> np.ndarray:
    """EEG dominated by 10 Hz alpha."""
    return _make_eeg(duration_s=duration_s, dominant_freq=10.0, amplitude=40.0)


def _smr_eeg(duration_s: float = 4.0) -> np.ndarray:
    """EEG dominated by 13 Hz SMR (low beta)."""
    return _make_eeg(duration_s=duration_s, dominant_freq=13.0, amplitude=40.0)


def _theta_eeg(duration_s: float = 4.0) -> np.ndarray:
    """EEG dominated by 6 Hz theta."""
    return _make_eeg(duration_s=duration_s, dominant_freq=6.0, amplitude=40.0)


def _beta_eeg(duration_s: float = 4.0) -> np.ndarray:
    """EEG dominated by 20 Hz beta (non-target noise)."""
    return _make_eeg(duration_s=duration_s, dominant_freq=20.0, amplitude=40.0)


# ===================================================================
# Protocol management
# ===================================================================

class TestProtocols:
    def test_list_protocols_returns_at_least_three(self):
        engine = NeurofeedbackAudioEngine()
        protocols = engine.get_available_protocols()
        assert isinstance(protocols, list)
        assert len(protocols) >= 3

    def test_list_protocols_contains_expected_names(self):
        engine = NeurofeedbackAudioEngine()
        protocols = engine.get_available_protocols()
        names = [p["name"] for p in protocols]
        assert "alpha_uptraining" in names
        assert "smr_training" in names
        assert "theta_training" in names

    def test_protocol_entries_have_required_keys(self):
        engine = NeurofeedbackAudioEngine()
        for p in engine.get_available_protocols():
            assert "name" in p
            assert "target_band" in p
            assert "description" in p

    def test_set_protocol_valid(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("smr_training")
        assert engine.protocol == "smr_training"

    def test_set_protocol_invalid_raises(self):
        engine = NeurofeedbackAudioEngine()
        with pytest.raises(ValueError, match="Unknown protocol"):
            engine.set_protocol("nonexistent_protocol")

    def test_default_protocol_is_alpha_uptraining(self):
        engine = NeurofeedbackAudioEngine()
        assert engine.protocol == "alpha_uptraining"


# ===================================================================
# Baseline
# ===================================================================

class TestBaseline:
    def test_set_baseline_stores_value(self):
        engine = NeurofeedbackAudioEngine()
        eeg = _make_eeg()
        engine.set_baseline(eeg, FS)
        assert engine.baseline is not None

    def test_set_baseline_per_protocol(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        engine.set_baseline(_alpha_eeg(), FS)
        alpha_bl = engine.baseline

        engine.set_protocol("smr_training")
        engine.set_baseline(_smr_eeg(), FS)
        smr_bl = engine.baseline

        # Baselines are protocol-specific band powers so they should differ
        assert alpha_bl != smr_bl

    def test_baseline_affects_ratio(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_baseline(_make_eeg(), FS)
        params = engine.get_audio_params(_alpha_eeg(), FS)
        assert "baseline_ratio" in params
        assert isinstance(params["baseline_ratio"], float)


# ===================================================================
# Audio parameter output structure
# ===================================================================

class TestAudioParams:
    def test_output_keys(self):
        engine = NeurofeedbackAudioEngine()
        params = engine.get_audio_params(_make_eeg(), FS)
        required = {
            "base_frequency", "volume", "pan", "reward_tone",
            "reward_count", "binaural_beat_freq", "current_band_power",
            "baseline_ratio", "protocol", "feedback_message",
        }
        assert required.issubset(params.keys())

    def test_frequency_in_audible_range(self):
        engine = NeurofeedbackAudioEngine()
        params = engine.get_audio_params(_make_eeg(), FS)
        assert 100.0 <= params["base_frequency"] <= 1000.0

    def test_volume_in_zero_one(self):
        engine = NeurofeedbackAudioEngine()
        params = engine.get_audio_params(_make_eeg(), FS)
        assert 0.0 <= params["volume"] <= 1.0

    def test_pan_in_range(self):
        engine = NeurofeedbackAudioEngine()
        params = engine.get_audio_params(_make_eeg(), FS)
        assert -1.0 <= params["pan"] <= 1.0

    def test_protocol_field_matches_active(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("theta_training")
        params = engine.get_audio_params(_theta_eeg(), FS)
        assert params["protocol"] == "theta_training"


# ===================================================================
# Alpha up-training protocol
# ===================================================================

class TestAlphaProtocol:
    def test_high_alpha_triggers_reward(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        # Use noise as baseline (low alpha) then present high-alpha signal
        engine.set_baseline(_make_eeg(), FS)
        params = engine.get_audio_params(_alpha_eeg(), FS)
        assert params["reward_tone"] is True

    def test_low_alpha_no_reward(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        # Baseline is alpha-rich, live signal is beta-dominant
        engine.set_baseline(_alpha_eeg(), FS)
        params = engine.get_audio_params(_beta_eeg(), FS)
        assert params["reward_tone"] is False

    def test_high_alpha_raises_volume(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        engine.set_baseline(_make_eeg(), FS)
        high = engine.get_audio_params(_alpha_eeg(), FS)
        low = engine.get_audio_params(_beta_eeg(), FS)
        assert high["volume"] > low["volume"]


# ===================================================================
# SMR training protocol
# ===================================================================

class TestSMRProtocol:
    def test_smr_target_band(self):
        engine = NeurofeedbackAudioEngine()
        protos = engine.get_available_protocols()
        smr = [p for p in protos if p["name"] == "smr_training"][0]
        assert smr["target_band"] == "smr"

    def test_smr_reward_on_high_smr(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("smr_training")
        engine.set_baseline(_make_eeg(), FS)
        params = engine.get_audio_params(_smr_eeg(), FS)
        assert params["reward_tone"] is True

    def test_smr_no_reward_on_theta(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("smr_training")
        engine.set_baseline(_make_eeg(), FS)
        params = engine.get_audio_params(_theta_eeg(), FS)
        assert params["reward_tone"] is False


# ===================================================================
# Theta training protocol
# ===================================================================

class TestThetaProtocol:
    def test_theta_reward_on_high_theta(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("theta_training")
        engine.set_baseline(_make_eeg(), FS)
        params = engine.get_audio_params(_theta_eeg(), FS)
        assert params["reward_tone"] is True

    def test_theta_no_reward_on_beta(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("theta_training")
        engine.set_baseline(_make_eeg(), FS)
        params = engine.get_audio_params(_beta_eeg(), FS)
        assert params["reward_tone"] is False


# ===================================================================
# Binaural beats
# ===================================================================

class TestBinauralBeats:
    def test_binaural_freq_present(self):
        engine = NeurofeedbackAudioEngine()
        params = engine.get_audio_params(_make_eeg(), FS)
        assert "binaural_beat_freq" in params
        assert isinstance(params["binaural_beat_freq"], float)

    def test_alpha_binaural_in_alpha_range(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        params = engine.get_audio_params(_make_eeg(), FS)
        # Alpha band: 8-12 Hz — binaural beat should target within this
        assert 8.0 <= params["binaural_beat_freq"] <= 12.0

    def test_smr_binaural_in_smr_range(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("smr_training")
        params = engine.get_audio_params(_make_eeg(), FS)
        # SMR band: 12-15 Hz
        assert 12.0 <= params["binaural_beat_freq"] <= 15.0

    def test_theta_binaural_in_theta_range(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("theta_training")
        params = engine.get_audio_params(_make_eeg(), FS)
        # Theta band: 4-8 Hz
        assert 4.0 <= params["binaural_beat_freq"] <= 8.0


# ===================================================================
# Reward detection and counting
# ===================================================================

class TestReward:
    def test_reward_count_starts_at_zero(self):
        engine = NeurofeedbackAudioEngine()
        params = engine.get_audio_params(_make_eeg(), FS)
        # Without baseline the first call may or may not reward;
        # but reward_count should be a non-negative int
        assert isinstance(params["reward_count"], int)
        assert params["reward_count"] >= 0

    def test_reward_count_increments(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        engine.set_baseline(_make_eeg(), FS)  # low-alpha baseline
        # Send high-alpha signals repeatedly
        for _ in range(5):
            params = engine.get_audio_params(_alpha_eeg(), FS)
        assert params["reward_count"] >= 3  # at least some should reward

    def test_no_reward_no_increment(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        engine.set_baseline(_alpha_eeg(), FS)  # high-alpha baseline
        # Non-alpha signal should not trigger reward
        initial_count = 0
        for _ in range(5):
            params = engine.get_audio_params(_beta_eeg(), FS)
        assert params["reward_count"] == initial_count


# ===================================================================
# Session stats
# ===================================================================

class TestSessionStats:
    def test_empty_stats(self):
        engine = NeurofeedbackAudioEngine()
        stats = engine.get_session_stats()
        assert stats["total_epochs"] == 0
        assert stats["reward_count"] == 0

    def test_stats_after_epochs(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_baseline(_make_eeg(), FS)
        engine.get_audio_params(_alpha_eeg(), FS)
        engine.get_audio_params(_beta_eeg(), FS)
        engine.get_audio_params(_alpha_eeg(), FS)
        stats = engine.get_session_stats()
        assert stats["total_epochs"] == 3
        assert "reward_rate" in stats
        assert 0.0 <= stats["reward_rate"] <= 1.0

    def test_stats_contains_mean_band_power(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_baseline(_make_eeg(), FS)
        engine.get_audio_params(_alpha_eeg(), FS)
        stats = engine.get_session_stats()
        assert "mean_band_power" in stats


# ===================================================================
# History
# ===================================================================

class TestHistory:
    def test_empty_history(self):
        engine = NeurofeedbackAudioEngine()
        assert engine.get_history() == []

    def test_history_grows(self):
        engine = NeurofeedbackAudioEngine()
        engine.get_audio_params(_make_eeg(), FS)
        engine.get_audio_params(_make_eeg(), FS)
        assert len(engine.get_history()) == 2

    def test_history_last_n(self):
        engine = NeurofeedbackAudioEngine()
        for _ in range(10):
            engine.get_audio_params(_make_eeg(), FS)
        last_3 = engine.get_history(last_n=3)
        assert len(last_3) == 3

    def test_history_entries_have_required_fields(self):
        engine = NeurofeedbackAudioEngine()
        engine.get_audio_params(_alpha_eeg(), FS)
        entry = engine.get_history()[0]
        assert "base_frequency" in entry
        assert "reward_tone" in entry
        assert "current_band_power" in entry


# ===================================================================
# Reset
# ===================================================================

class TestReset:
    def test_reset_clears_state(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_baseline(_make_eeg(), FS)
        engine.get_audio_params(_alpha_eeg(), FS)
        engine.get_audio_params(_alpha_eeg(), FS)
        engine.reset()
        assert engine.baseline is None
        assert engine.get_history() == []
        stats = engine.get_session_stats()
        assert stats["total_epochs"] == 0
        assert stats["reward_count"] == 0

    def test_reset_preserves_protocol(self):
        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("theta_training")
        engine.reset()
        assert engine.protocol == "theta_training"
