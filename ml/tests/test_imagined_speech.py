"""Tests for imagined speech command decoder.

Validates the ImaginedSpeechDecoder class which uses spectral template
matching to decode imagined speech commands from 4-channel Muse 2 EEG
(TP9, AF7, AF8, TP10 at 256 Hz).

This is experimental — 55-65% accuracy for 5 classes on consumer EEG
is the realistic ceiling. Tests verify the API contract, not neurological
validity.
"""
import numpy as np
import pytest

from models.imagined_speech import (
    BANDS,
    DISCLAIMER,
    MAX_HISTORY,
    ImaginedSpeechDecoder,
)

FS = 256
DURATION = 4  # seconds
N_SAMPLES = FS * DURATION


def _synth_eeg(n_channels=4, n_samples=N_SAMPLES, fs=FS, seed=42):
    """Generate broadband synthetic EEG (~20 uV RMS) with alpha/theta."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    signals = rng.randn(n_channels, n_samples) * 10
    for ch in range(n_channels):
        signals[ch] += 12 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        signals[ch] += 8 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
    return signals


def _speech_imagery_signal(command, seed=100):
    """Generate synthetic EEG with command-specific spectral signature.

    Different commands get different spectral profiles to test that the
    decoder can distinguish them. This is synthetic — real imagined speech
    differences are far more subtle.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_SAMPLES) / FS
    signals = rng.randn(4, N_SAMPLES) * 5

    # Each command gets a different spectral emphasis
    profiles = {
        "yes": {"theta_amp": 20, "alpha_amp": 5, "beta_amp": 15, "left_bias": 1.5},
        "no": {"theta_amp": 8, "alpha_amp": 18, "beta_amp": 8, "left_bias": 0.7},
        "stop": {"theta_amp": 12, "alpha_amp": 10, "beta_amp": 25, "left_bias": 1.3},
        "go": {"theta_amp": 25, "alpha_amp": 8, "beta_amp": 10, "left_bias": 1.8},
        "help": {"theta_amp": 15, "alpha_amp": 12, "beta_amp": 20, "left_bias": 1.1},
    }

    p = profiles.get(command, profiles["yes"])

    for ch in range(4):
        # Left-hemisphere bias (ch0=TP9, ch1=AF7 are left)
        bias = p["left_bias"] if ch in (0, 1) else 1.0 / p["left_bias"]
        signals[ch] += p["theta_amp"] * bias * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        signals[ch] += p["alpha_amp"] * bias * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        signals[ch] += p["beta_amp"] * bias * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))

    return signals


def _calibrate_decoder(decoder, commands=("yes", "no", "stop"), n_per_class=3, user_id="default"):
    """Helper: calibrate a decoder with synthetic data for multiple commands."""
    for i, cmd in enumerate(commands):
        for j in range(n_per_class):
            sig = _speech_imagery_signal(cmd, seed=i * 100 + j)
            decoder.calibrate(sig, cmd, user_id=user_id)


# ── TestInit ───────────────────────────────────────────────────────


class TestInit:
    def test_default_init(self):
        d = ImaginedSpeechDecoder()
        assert d._fs == 256.0
        assert d._n_classes == 5

    def test_custom_init(self):
        d = ImaginedSpeechDecoder(fs=128.0, n_classes=3)
        assert d._fs == 128.0
        assert d._n_classes == 3


# ── TestCalibrate ──────────────────────────────────────────────────


class TestCalibrate:
    def test_calibrate_returns_dict(self):
        d = ImaginedSpeechDecoder()
        result = d.calibrate(_synth_eeg(), "yes")
        assert isinstance(result, dict)

    def test_calibrate_required_keys(self):
        d = ImaginedSpeechDecoder()
        result = d.calibrate(_synth_eeg(), "yes")
        assert "calibrated" in result
        assert "n_samples" in result
        assert "n_classes_seen" in result
        assert "classes_seen" in result

    def test_single_class_not_calibrated(self):
        d = ImaginedSpeechDecoder()
        result = d.calibrate(_synth_eeg(), "yes")
        assert result["calibrated"] is False
        assert result["n_classes_seen"] == 1
        assert result["n_samples"] == 1

    def test_two_classes_calibrated(self):
        d = ImaginedSpeechDecoder()
        d.calibrate(_synth_eeg(seed=1), "yes")
        result = d.calibrate(_synth_eeg(seed=2), "no")
        assert result["calibrated"] is True
        assert result["n_classes_seen"] == 2
        assert result["n_samples"] == 2

    def test_multiple_samples_per_class(self):
        d = ImaginedSpeechDecoder()
        d.calibrate(_synth_eeg(seed=1), "yes")
        d.calibrate(_synth_eeg(seed=2), "yes")
        result = d.calibrate(_synth_eeg(seed=3), "no")
        assert result["n_samples"] == 3
        assert result["n_classes_seen"] == 2

    def test_classes_seen_sorted(self):
        d = ImaginedSpeechDecoder()
        d.calibrate(_synth_eeg(seed=1), "stop")
        d.calibrate(_synth_eeg(seed=2), "go")
        result = d.calibrate(_synth_eeg(seed=3), "help")
        assert result["classes_seen"] == ["go", "help", "stop"]

    def test_max_classes_enforced(self):
        d = ImaginedSpeechDecoder(n_classes=2)
        d.calibrate(_synth_eeg(seed=1), "yes")
        d.calibrate(_synth_eeg(seed=2), "no")
        result = d.calibrate(_synth_eeg(seed=3), "stop")
        # Should ignore "stop" — already at max 2 classes
        assert result["n_classes_seen"] == 2
        assert "stop" not in result["classes_seen"]

    def test_calibrate_1d_signal(self):
        d = ImaginedSpeechDecoder()
        result = d.calibrate(_synth_eeg()[0], "yes")
        assert result["n_samples"] == 1

    def test_calibrate_custom_fs(self):
        d = ImaginedSpeechDecoder(fs=256.0)
        result = d.calibrate(_synth_eeg(), "yes", fs=128.0)
        assert result["n_samples"] == 1


# ── TestDecode ─────────────────────────────────────────────────────


class TestDecode:
    def test_decode_uncalibrated(self):
        d = ImaginedSpeechDecoder()
        result = d.decode(_synth_eeg())
        assert result["predicted_command"] is None
        assert result["confidence"] == 0.0
        assert result["probabilities"] == {}
        assert result["is_calibrated"] is False

    def test_decode_single_class_not_calibrated(self):
        d = ImaginedSpeechDecoder()
        d.calibrate(_synth_eeg(seed=1), "yes")
        result = d.decode(_synth_eeg(seed=2))
        assert result["predicted_command"] is None
        assert result["is_calibrated"] is False

    def test_decode_returns_required_keys(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        result = d.decode(_synth_eeg())
        required = {
            "predicted_command",
            "confidence",
            "probabilities",
            "is_calibrated",
            "n_calibration_samples",
            "disclaimer",
        }
        assert required.issubset(result.keys())

    def test_decode_predicted_command_is_known_class(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no", "stop"))
        result = d.decode(_synth_eeg())
        assert result["predicted_command"] in ("yes", "no", "stop")

    def test_decode_probabilities_sum_to_one(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no", "stop"))
        result = d.decode(_synth_eeg())
        probs = result["probabilities"]
        assert set(probs.keys()) == {"yes", "no", "stop"}
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_decode_confidence_in_range(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        result = d.decode(_synth_eeg())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_decode_always_has_disclaimer(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        result = d.decode(_synth_eeg())
        assert result["disclaimer"] == DISCLAIMER

    def test_decode_uncalibrated_has_disclaimer(self):
        d = ImaginedSpeechDecoder()
        result = d.decode(_synth_eeg())
        assert result["disclaimer"] == DISCLAIMER

    def test_decode_n_calibration_samples(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"), n_per_class=3)
        result = d.decode(_synth_eeg())
        assert result["n_calibration_samples"] == 6

    def test_decode_1d_signal(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        result = d.decode(_synth_eeg()[0])
        # 1D decode against 4-ch calibration: dimension mismatch in features
        # produces zero similarities and uniform probabilities with zero confidence
        assert "predicted_command" in result
        assert result["predicted_command"] in ("yes", "no")
        assert result["confidence"] == 0.0

    def test_decode_distinguishes_different_commands(self):
        """The decoder should predict the correct class when fed
        a signal that matches the calibration profile for that class."""
        d = ImaginedSpeechDecoder()
        commands = ("yes", "no", "stop", "go", "help")

        # Calibrate with command-specific signals (5 samples each)
        for cmd in commands:
            for j in range(5):
                sig = _speech_imagery_signal(cmd, seed=hash(cmd) % 10000 + j)
                d.calibrate(sig, cmd)

        # Decode with a signal matching "go" profile
        test_sig = _speech_imagery_signal("go", seed=9999)
        result = d.decode(test_sig)
        # The predicted command should be "go" given matching spectral profile
        assert result["predicted_command"] == "go"

    def test_decode_higher_confidence_for_matching_signal(self):
        """Signal matching a calibrated class should have higher confidence
        than a random signal."""
        d = ImaginedSpeechDecoder()
        commands = ("yes", "no")
        for cmd in commands:
            for j in range(5):
                sig = _speech_imagery_signal(cmd, seed=hash(cmd) % 10000 + j)
                d.calibrate(sig, cmd)

        # Matching signal
        matched = d.decode(_speech_imagery_signal("yes", seed=8888))
        # Random noise
        rng = np.random.RandomState(7777)
        random_result = d.decode(rng.randn(4, N_SAMPLES) * 50)
        # Matching should have higher max probability
        max_matched = max(matched["probabilities"].values())
        max_random = max(random_result["probabilities"].values())
        assert max_matched >= max_random


# ── TestSessionStats ───────────────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self):
        d = ImaginedSpeechDecoder()
        stats = d.get_session_stats()
        assert stats["n_decodings"] == 0
        assert stats["n_calibration_samples"] == 0
        assert stats["classes_available"] == []

    def test_stats_after_calibration(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"), n_per_class=2)
        stats = d.get_session_stats()
        assert stats["n_calibration_samples"] == 4
        assert stats["classes_available"] == ["no", "yes"]

    def test_stats_after_decode(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        d.decode(_synth_eeg(seed=10))
        d.decode(_synth_eeg(seed=20))
        stats = d.get_session_stats()
        assert stats["n_decodings"] == 2


# ── TestHistory ────────────────────────────────────────────────────


class TestHistory:
    def test_empty_history(self):
        d = ImaginedSpeechDecoder()
        assert d.get_history() == []

    def test_history_grows(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        d.decode(_synth_eeg(seed=10))
        d.decode(_synth_eeg(seed=20))
        assert len(d.get_history()) == 2

    def test_history_last_n(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        for i in range(7):
            d.decode(_synth_eeg(seed=i))
        last_3 = d.get_history(last_n=3)
        assert len(last_3) == 3

    def test_history_last_n_larger_than_history(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        d.decode(_synth_eeg(seed=1))
        last_10 = d.get_history(last_n=10)
        assert len(last_10) == 1

    def test_history_capped_at_max(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        for i in range(MAX_HISTORY + 50):
            d.decode(_synth_eeg(seed=i))
        assert len(d.get_history()) == MAX_HISTORY

    def test_history_contains_decode_results(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        d.decode(_synth_eeg(seed=10))
        history = d.get_history()
        assert len(history) == 1
        assert "predicted_command" in history[0]
        assert "confidence" in history[0]

    def test_uncalibrated_decode_appears_in_history(self):
        d = ImaginedSpeechDecoder()
        d.decode(_synth_eeg())
        assert len(d.get_history()) == 1
        assert d.get_history()[0]["predicted_command"] is None


# ── TestMultiUser ──────────────────────────────────────────────────


class TestMultiUser:
    def test_independent_users(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"), user_id="alice")
        _calibrate_decoder(d, ("stop", "go", "help"), user_id="bob")
        assert d.get_session_stats("alice")["classes_available"] == ["no", "yes"]
        assert d.get_session_stats("bob")["classes_available"] == ["go", "help", "stop"]

    def test_decode_independent_per_user(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"), user_id="alice")
        _calibrate_decoder(d, ("stop", "go"), user_id="bob")
        r_alice = d.decode(_synth_eeg(seed=1), user_id="alice")
        r_bob = d.decode(_synth_eeg(seed=1), user_id="bob")
        assert r_alice["predicted_command"] in ("yes", "no")
        assert r_bob["predicted_command"] in ("stop", "go")

    def test_history_independent_per_user(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"), user_id="alice")
        _calibrate_decoder(d, ("stop", "go"), user_id="bob")
        d.decode(_synth_eeg(seed=1), user_id="alice")
        d.decode(_synth_eeg(seed=2), user_id="bob")
        d.decode(_synth_eeg(seed=3), user_id="bob")
        assert len(d.get_history(user_id="alice")) == 1
        assert len(d.get_history(user_id="bob")) == 2


# ── TestReset ──────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_state(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        d.decode(_synth_eeg(seed=1))
        d.reset()
        assert d.get_history() == []
        stats = d.get_session_stats()
        assert stats["n_decodings"] == 0
        assert stats["n_calibration_samples"] == 0
        assert stats["classes_available"] == []

    def test_reset_specific_user(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"), user_id="alice")
        _calibrate_decoder(d, ("stop", "go"), user_id="bob")
        d.decode(_synth_eeg(seed=1), user_id="alice")
        d.decode(_synth_eeg(seed=2), user_id="bob")
        d.reset(user_id="alice")
        assert d.get_session_stats("alice")["n_decodings"] == 0
        assert d.get_session_stats("alice")["n_calibration_samples"] == 0
        assert d.get_session_stats("bob")["n_decodings"] == 1
        assert d.get_session_stats("bob")["n_calibration_samples"] == 6

    def test_decode_after_reset_is_uncalibrated(self):
        d = ImaginedSpeechDecoder()
        _calibrate_decoder(d, ("yes", "no"))
        d.reset()
        result = d.decode(_synth_eeg())
        assert result["predicted_command"] is None
        assert result["is_calibrated"] is False


# ── TestFeatureExtraction ──────────────────────────────────────────


class TestFeatureExtraction:
    def test_feature_vector_length(self):
        d = ImaginedSpeechDecoder()
        signals = _synth_eeg(n_channels=4)
        features = d._extract_features(d._ensure_2d(signals), FS)
        # 4 channels x 5 bands = 20
        assert len(features) == 20

    def test_feature_vector_length_1ch(self):
        d = ImaginedSpeechDecoder()
        signals = _synth_eeg(n_channels=1)
        features = d._extract_features(d._ensure_2d(signals), FS)
        # 1 channel x 5 bands = 5
        assert len(features) == 5

    def test_features_nonnegative(self):
        d = ImaginedSpeechDecoder()
        signals = _synth_eeg()
        features = d._extract_features(d._ensure_2d(signals), FS)
        assert np.all(features >= 0.0)

    def test_flat_signal_low_features(self):
        d = ImaginedSpeechDecoder()
        flat = np.ones((4, N_SAMPLES)) * 0.001
        features = d._extract_features(flat, FS)
        # Flat signal should have near-zero band power (tiny DC only)
        assert np.all(features < 0.01)


# ── TestCosineSimilarity ──────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        sim = ImaginedSpeechDecoder._cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = ImaginedSpeechDecoder._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        sim = ImaginedSpeechDecoder._cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        sim = ImaginedSpeechDecoder._cosine_similarity(a, b)
        assert sim == 0.0


# ── TestSoftmax ────────────────────────────────────────────────────


class TestSoftmax:
    def test_probabilities_sum_to_one(self):
        sims = {"yes": 0.9, "no": 0.7, "stop": 0.5}
        probs = ImaginedSpeechDecoder._softmax_from_similarities(sims)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_highest_similarity_gets_highest_prob(self):
        sims = {"yes": 0.95, "no": 0.3, "stop": 0.1}
        probs = ImaginedSpeechDecoder._softmax_from_similarities(sims)
        assert probs["yes"] > probs["no"]
        assert probs["no"] > probs["stop"]

    def test_empty_similarities(self):
        probs = ImaginedSpeechDecoder._softmax_from_similarities({})
        assert probs == {}

    def test_equal_similarities_uniform(self):
        sims = {"a": 0.5, "b": 0.5, "c": 0.5}
        probs = ImaginedSpeechDecoder._softmax_from_similarities(sims)
        for v in probs.values():
            assert abs(v - 1.0 / 3) < 1e-6


# ── TestEdgeCases ──────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_short_signal(self):
        d = ImaginedSpeechDecoder()
        short = np.random.randn(4, 16)
        result = d.calibrate(short, "yes")
        assert result["n_samples"] == 1

    def test_very_short_signal_decode(self):
        d = ImaginedSpeechDecoder()
        # Calibrate with normal length
        _calibrate_decoder(d, ("yes", "no"))
        # Decode with very short signal
        short = np.random.randn(4, 16)
        result = d.decode(short)
        assert "predicted_command" in result

    def test_bands_constant_defined(self):
        assert len(BANDS) == 5
        for name, low, high in BANDS:
            assert isinstance(name, str)
            assert low < high

    def test_disclaimer_present(self):
        assert "experimental" in DISCLAIMER.lower()
        assert "not reliable" in DISCLAIMER.lower()
