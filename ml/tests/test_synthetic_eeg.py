"""Tests for synthetic EEG generation engine (issue #445)."""

import numpy as np
import pytest

from models.synthetic_eeg import (
    BANDS,
    GenerationStats,
    augment_eeg,
    compute_generation_stats,
    generate_emotion_conditioned_eeg,
    generate_synthetic_eeg,
    inject_artifacts,
    stats_to_dict,
    validate_synthetic_quality,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_realistic_eeg(fs=256, duration=4.0, n_channels=4, seed=42):
    """Generate a realistic-looking EEG for testing."""
    return generate_synthetic_eeg(
        duration=duration, fs=fs, n_channels=n_channels, seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  generate_synthetic_eeg
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateSyntheticEEG:
    def test_output_shape_default(self):
        sig = generate_synthetic_eeg(seed=0)
        assert sig.shape == (4, 1024)  # 4 channels, 4s * 256Hz

    def test_output_shape_custom(self):
        sig = generate_synthetic_eeg(
            duration=2.0, fs=128, n_channels=2, seed=0,
        )
        assert sig.shape == (2, 256)  # 2 channels, 2s * 128Hz

    def test_amplitude_reasonable(self):
        sig = generate_synthetic_eeg(amplitude_uv=20.0, seed=1)
        rms = np.sqrt(np.mean(sig ** 2))
        # RMS should be in the ballpark of the requested amplitude
        assert 5.0 < rms < 60.0

    def test_seed_reproducibility(self):
        a = generate_synthetic_eeg(seed=42)
        b = generate_synthetic_eeg(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate_synthetic_eeg(seed=1)
        b = generate_synthetic_eeg(seed=2)
        assert not np.array_equal(a, b)

    def test_custom_band_powers(self):
        # Heavy alpha signal
        sig = generate_synthetic_eeg(
            band_powers={"alpha": 0.8, "beta": 0.1, "theta": 0.1},
            seed=10,
        )
        assert sig.shape[0] == 4
        # Validate that alpha band has reasonable power
        from scipy.signal import welch
        _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        f, pxx = welch(sig[0], fs=256, nperseg=512)
        alpha_mask = (f >= 8) & (f <= 12)
        total_mask = (f >= 0.5) & (f <= 45)
        alpha_power = _trapezoid(pxx[alpha_mask], f[alpha_mask])
        total_power = _trapezoid(pxx[total_mask], f[total_mask])
        # Alpha should be a significant portion
        assert alpha_power / total_power > 0.10


# ══════════════════════════════════════════════════════════════════════════════
#  generate_emotion_conditioned_eeg
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionConditioned:
    def test_valid_emotion(self):
        result = generate_emotion_conditioned_eeg("happy", seed=0)
        assert result["emotion"] == "happy"
        assert result["signals"].shape == (4, 1024)
        assert isinstance(result["band_profile"], dict)
        assert result["n_channels"] == 4

    def test_invalid_emotion_raises(self):
        with pytest.raises(ValueError, match="Unknown emotion"):
            generate_emotion_conditioned_eeg("confused")

    def test_all_emotions_generate(self):
        emotions = ["happy", "sad", "angry", "fear", "neutral",
                     "surprise", "relaxed", "focused"]
        for emo in emotions:
            result = generate_emotion_conditioned_eeg(emo, seed=0)
            assert result["signals"].shape[0] > 0
            assert result["emotion"] == emo

    def test_different_emotions_differ(self):
        happy = generate_emotion_conditioned_eeg("happy", seed=42)
        sad = generate_emotion_conditioned_eeg("sad", seed=42)
        # Same seed but different profiles should produce different signals
        assert not np.array_equal(happy["signals"], sad["signals"])


# ══════════════════════════════════════════════════════════════════════════════
#  inject_artifacts
# ══════════════════════════════════════════════════════════════════════════════

class TestInjectArtifacts:
    def test_no_artifacts_at_zero_rate(self):
        sig = _make_realistic_eeg()
        result = inject_artifacts(
            sig, blink_rate=0, muscle_rate=0, electrode_pop_rate=0, seed=0,
        )
        np.testing.assert_array_equal(result["signals"], sig)
        assert result["n_artifacts"] == 0

    def test_blinks_injected(self):
        sig = _make_realistic_eeg()
        result = inject_artifacts(sig, blink_rate=2.0, seed=0)
        assert result["n_artifacts"] > 0
        blinks = [a for a in result["artifact_log"] if a["type"] == "eye_blink"]
        assert len(blinks) > 0

    def test_original_not_mutated(self):
        sig = _make_realistic_eeg()
        original = sig.copy()
        inject_artifacts(sig, blink_rate=5.0, seed=0)
        np.testing.assert_array_equal(sig, original)

    def test_artifact_log_structure(self):
        sig = _make_realistic_eeg()
        result = inject_artifacts(
            sig, blink_rate=1.0, muscle_rate=1.0, electrode_pop_rate=1.0,
            seed=0,
        )
        for entry in result["artifact_log"]:
            assert "type" in entry
            assert "onset_sample" in entry
            assert "duration_samples" in entry
            assert "amplitude_uv" in entry
            assert "channels" in entry
            assert entry["type"] in ("eye_blink", "muscle", "electrode_pop")


# ══════════════════════════════════════════════════════════════════════════════
#  augment_eeg
# ══════════════════════════════════════════════════════════════════════════════

class TestAugmentEEG:
    def test_correct_number_of_augmentations(self):
        sig = _make_realistic_eeg()
        results = augment_eeg(sig, n_augmentations=3, seed=0)
        assert len(results) == 3

    def test_augmented_shape_matches_original(self):
        sig = _make_realistic_eeg()
        results = augment_eeg(sig, n_augmentations=1, seed=0)
        assert results[0]["signals"].shape == sig.shape

    def test_augmented_differs_from_original(self):
        sig = _make_realistic_eeg()
        results = augment_eeg(sig, n_augmentations=1, seed=0)
        assert not np.array_equal(results[0]["signals"], sig)

    def test_transforms_recorded(self):
        sig = _make_realistic_eeg()
        results = augment_eeg(
            sig, n_augmentations=1, seed=0,
            time_shift=True, amplitude_scale=True,
            additive_noise=True, band_perturbation=True,
        )
        assert len(results[0]["transforms"]) > 0

    def test_single_channel_input(self):
        sig = np.random.randn(1024)  # 1D input
        results = augment_eeg(sig, n_augmentations=2, seed=0)
        assert len(results) == 2
        assert results[0]["signals"].ndim == 2


# ══════════════════════════════════════════════════════════════════════════════
#  validate_synthetic_quality
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_clean_signal_passes(self):
        sig = generate_synthetic_eeg(seed=42)
        stats = validate_synthetic_quality(sig, fs=256)
        # Generated with default profile should usually pass
        assert isinstance(stats, GenerationStats)
        assert stats.n_signals == 4

    def test_flat_signal_fails(self):
        sig = np.zeros((4, 1024))
        stats = validate_synthetic_quality(sig, fs=256)
        # Flat signal has near-zero power -- should fail
        assert stats.is_valid is False
        assert len(stats.failure_reasons) > 0

    def test_band_powers_sum_approximately_one(self):
        sig = generate_synthetic_eeg(seed=42)
        stats = validate_synthetic_quality(sig, fs=256)
        total = sum(stats.mean_band_powers.values())
        assert 0.8 < total < 1.2  # some rounding tolerance

    def test_spectral_entropy_in_range(self):
        sig = generate_synthetic_eeg(seed=42)
        stats = validate_synthetic_quality(sig, fs=256)
        assert 0.0 <= stats.spectral_entropy <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  compute_generation_stats & stats_to_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchStats:
    def test_empty_batch(self):
        stats = compute_generation_stats([])
        assert stats.n_signals == 0

    def test_batch_of_valid_signals(self):
        batch = [generate_synthetic_eeg(seed=i) for i in range(3)]
        stats = compute_generation_stats(batch, fs=256)
        assert stats.n_signals == 3
        assert stats.n_passed + stats.n_failed == 3

    def test_stats_to_dict_keys(self):
        sig = generate_synthetic_eeg(seed=42)
        stats = validate_synthetic_quality(sig, fs=256)
        d = stats_to_dict(stats)
        expected_keys = {
            "n_signals", "n_passed", "n_failed", "mean_band_powers",
            "power_in_range", "total_power", "total_power_valid",
            "spectral_entropy", "is_valid", "failure_reasons",
        }
        assert expected_keys == set(d.keys())

    def test_stats_to_dict_types(self):
        sig = generate_synthetic_eeg(seed=42)
        stats = validate_synthetic_quality(sig, fs=256)
        d = stats_to_dict(stats)
        assert isinstance(d["is_valid"], bool)
        assert isinstance(d["mean_band_powers"], dict)
        assert isinstance(d["failure_reasons"], list)
        assert isinstance(d["total_power"], float)
