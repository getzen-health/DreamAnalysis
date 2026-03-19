"""Tests for voice emotion data augmentation pipeline (issue #384).

15+ tests covering inject_noise, shift_pitch, stretch_time,
apply_spec_augment, augment_voice_sample, create_augmentation_pipeline,
and pipeline_stats_to_dict.
"""

import numpy as np
import pytest

from models.voice_augmentation import (
    PipelineConfig,
    PipelineStats,
    apply_spec_augment,
    augment_voice_sample,
    create_augmentation_pipeline,
    inject_noise,
    pipeline_stats_to_dict,
    shift_pitch,
    stretch_time,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_audio(duration: float = 1.0, sr: int = 16000, freq: float = 440.0,
                seed: int = 42) -> np.ndarray:
    """Generate a simple sine wave audio signal."""
    t = np.arange(int(sr * duration)) / sr
    rng = np.random.default_rng(seed)
    return np.sin(2.0 * np.pi * freq * t) * 0.5 + rng.standard_normal(len(t)) * 0.01


def _make_spectrogram(n_mels: int = 80, n_frames: int = 100, seed: int = 42) -> np.ndarray:
    """Generate a fake mel spectrogram."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal((n_mels, n_frames))) + 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  inject_noise
# ══════════════════════════════════════════════════════════════════════════════

class TestInjectNoise:
    def test_output_shape_matches_input(self):
        audio = _make_audio()
        result = inject_noise(audio, snr_db=20.0, seed=0)
        assert result["audio"].shape == audio.shape

    def test_white_noise_changes_signal(self):
        audio = _make_audio()
        result = inject_noise(audio, noise_type="white", snr_db=10.0, seed=0)
        assert not np.allclose(result["audio"], audio)

    def test_pink_noise_changes_signal(self):
        audio = _make_audio()
        result = inject_noise(audio, noise_type="pink", snr_db=10.0, seed=0)
        assert not np.allclose(result["audio"], audio)

    def test_babble_noise_changes_signal(self):
        audio = _make_audio()
        result = inject_noise(audio, noise_type="babble", snr_db=10.0, seed=0)
        assert not np.allclose(result["audio"], audio)

    def test_high_snr_preserves_signal(self):
        """At very high SNR, noise should be negligible."""
        audio = _make_audio()
        result = inject_noise(audio, snr_db=60.0, seed=0)
        # Correlation should be very high
        corr = np.corrcoef(audio, result["audio"])[0, 1]
        assert corr > 0.99

    def test_invalid_noise_type_raises(self):
        audio = _make_audio()
        with pytest.raises(ValueError, match="Unknown noise_type"):
            inject_noise(audio, noise_type="invalid")

    def test_2d_input_raises(self):
        audio = np.random.randn(2, 1000)
        with pytest.raises(ValueError, match="1-D"):
            inject_noise(audio)

    def test_seed_reproducibility(self):
        audio = _make_audio()
        a = inject_noise(audio, seed=42)
        b = inject_noise(audio, seed=42)
        np.testing.assert_array_equal(a["audio"], b["audio"])

    def test_returns_power_info(self):
        audio = _make_audio()
        result = inject_noise(audio, snr_db=20.0, seed=0)
        assert "signal_power" in result
        assert "noise_power" in result
        assert result["signal_power"] > 0
        assert result["noise_power"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  shift_pitch
# ══════════════════════════════════════════════════════════════════════════════

class TestShiftPitch:
    def test_output_length_matches_input(self):
        audio = _make_audio()
        result = shift_pitch(audio, semitones=1.0)
        assert result["output_length"] == len(audio)

    def test_positive_shift_changes_signal(self):
        audio = _make_audio()
        result = shift_pitch(audio, semitones=2.0)
        assert not np.allclose(result["audio"], audio)

    def test_negative_shift_changes_spectral_centroid(self):
        """Negative pitch shift should lower the spectral centroid."""
        audio = _make_audio(freq=440.0)
        result = shift_pitch(audio, semitones=-2.0)
        # Compute spectral centroid of original vs shifted
        orig_fft = np.abs(np.fft.rfft(audio))
        shifted_fft = np.abs(np.fft.rfft(result["audio"]))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / 16000)
        orig_centroid = np.sum(freqs * orig_fft) / (np.sum(orig_fft) + 1e-10)
        shifted_centroid = np.sum(freqs * shifted_fft) / (np.sum(shifted_fft) + 1e-10)
        assert shifted_centroid < orig_centroid

    def test_zero_shift_preserves_signal(self):
        audio = _make_audio()
        result = shift_pitch(audio, semitones=0.0)
        np.testing.assert_array_equal(result["audio"], audio)

    def test_semitones_clamped(self):
        audio = _make_audio()
        result = shift_pitch(audio, semitones=5.0)
        assert result["semitones"] == 2.0  # clamped to max


# ══════════════════════════════════════════════════════════════════════════════
#  stretch_time
# ══════════════════════════════════════════════════════════════════════════════

class TestStretchTime:
    def test_faster_produces_shorter_output(self):
        audio = _make_audio()
        result = stretch_time(audio, rate=1.1)
        assert result["output_length"] < len(audio)

    def test_slower_produces_longer_output(self):
        audio = _make_audio()
        result = stretch_time(audio, rate=0.9)
        assert result["output_length"] > len(audio)

    def test_rate_1_preserves_signal(self):
        audio = _make_audio()
        result = stretch_time(audio, rate=1.0)
        np.testing.assert_array_equal(result["audio"], audio)

    def test_rate_clamped(self):
        audio = _make_audio()
        result = stretch_time(audio, rate=2.0)
        assert result["rate"] == 1.1  # clamped to max


# ══════════════════════════════════════════════════════════════════════════════
#  apply_spec_augment
# ══════════════════════════════════════════════════════════════════════════════

class TestApplySpecAugment:
    def test_output_shape_matches_input(self):
        spec = _make_spectrogram()
        result = apply_spec_augment(spec, seed=0)
        assert result["spectrogram"].shape == spec.shape

    def test_masking_creates_zeros(self):
        spec = _make_spectrogram() + 1.0  # all positive
        result = apply_spec_augment(spec, n_freq_masks=1, n_time_masks=1, seed=0)
        # Should have some zeros from masking
        assert np.any(result["spectrogram"] == 0.0)

    def test_returns_mask_ranges(self):
        spec = _make_spectrogram()
        result = apply_spec_augment(spec, n_freq_masks=2, n_time_masks=3, seed=0)
        assert len(result["freq_mask_ranges"]) == 2
        assert len(result["time_mask_ranges"]) == 3

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            apply_spec_augment(np.ones(100))


# ══════════════════════════════════════════════════════════════════════════════
#  augment_voice_sample
# ══════════════════════════════════════════════════════════════════════════════

class TestAugmentVoiceSample:
    def test_returns_audio_and_transforms(self):
        audio = _make_audio()
        result = augment_voice_sample(audio, seed=0)
        assert "audio" in result
        assert "transforms_applied" in result
        assert isinstance(result["transforms_applied"], list)

    def test_all_probs_zero_preserves_signal(self):
        audio = _make_audio()
        result = augment_voice_sample(
            audio, noise_prob=0.0, pitch_prob=0.0, stretch_prob=0.0, seed=0
        )
        np.testing.assert_array_equal(result["audio"], audio)
        assert len(result["transforms_applied"]) == 0

    def test_all_probs_one_applies_transforms(self):
        audio = _make_audio()
        result = augment_voice_sample(
            audio, noise_prob=1.0, pitch_prob=1.0, stretch_prob=1.0, seed=42
        )
        assert len(result["transforms_applied"]) == 3

    def test_seed_reproducibility(self):
        audio = _make_audio()
        a = augment_voice_sample(audio, noise_prob=1.0, pitch_prob=1.0, seed=99)
        b = augment_voice_sample(audio, noise_prob=1.0, pitch_prob=1.0, seed=99)
        np.testing.assert_array_equal(a["audio"], b["audio"])


# ══════════════════════════════════════════════════════════════════════════════
#  create_augmentation_pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateAugmentationPipeline:
    def test_returns_config_and_fn(self):
        pipeline = create_augmentation_pipeline()
        assert "config" in pipeline
        assert "augment_fn" in pipeline
        assert "stats" in pipeline
        assert callable(pipeline["augment_fn"])

    def test_augment_fn_produces_output(self):
        pipeline = create_augmentation_pipeline()
        audio = _make_audio()
        result = pipeline["augment_fn"](audio, seed=0)
        assert "audio" in result
        assert len(result["audio"]) > 0

    def test_stats_track_samples(self):
        pipeline = create_augmentation_pipeline()
        audio = _make_audio()
        pipeline["augment_fn"](audio, seed=0)
        pipeline["augment_fn"](audio, seed=1)
        assert pipeline["stats"].n_samples_processed == 2

    def test_custom_config(self):
        config = PipelineConfig(noise_prob=1.0, pitch_prob=0.0, stretch_prob=0.0)
        pipeline = create_augmentation_pipeline(config)
        assert pipeline["config"].noise_prob == 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  pipeline_stats_to_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineStatsToDict:
    def test_returns_dict(self):
        stats = PipelineStats()
        d = pipeline_stats_to_dict(stats)
        assert isinstance(d, dict)

    def test_all_keys_present(self):
        stats = PipelineStats(n_samples_processed=5, n_noise_applied=2)
        d = pipeline_stats_to_dict(stats)
        expected_keys = {
            "n_samples_processed", "n_noise_applied", "n_pitch_applied",
            "n_stretch_applied", "n_spec_augment_applied", "noise_types_used",
            "mean_snr_db", "mean_semitones", "mean_stretch_rate", "config",
        }
        assert expected_keys.issubset(d.keys())

    def test_values_are_native_types(self):
        stats = PipelineStats(n_samples_processed=10)
        d = pipeline_stats_to_dict(stats)
        assert isinstance(d["n_samples_processed"], int)
        assert isinstance(d["mean_snr_db"], float)
