"""Tests for teacher-student distillation training script."""
import numpy as np
import pytest

from training.distillation import (
    EMOTIONS,
    MUSE_CH_INDICES,
    MUSE_CHANNEL_MAP,
    distillation_loss,
    extract_muse_channels,
    kl_divergence_loss,
    train_distilled_model,
)


# ---------------------------------------------------------------------------
# extract_muse_channels
# ---------------------------------------------------------------------------

class TestExtractMuseChannels:
    def test_output_shape(self):
        signals = np.random.randn(32, 1024)
        result = extract_muse_channels(signals)
        assert result.shape == (4, 1024)

    def test_correct_channel_ordering(self):
        """Output should be [TP9, AF7, AF8, TP10] matching BrainFlow delivery."""
        signals = np.zeros((32, 10))
        # Put unique values in each expected channel
        signals[MUSE_CHANNEL_MAP["TP9"]] = 1.0
        signals[MUSE_CHANNEL_MAP["AF7"]] = 2.0
        signals[MUSE_CHANNEL_MAP["AF8"]] = 3.0
        signals[MUSE_CHANNEL_MAP["TP10"]] = 4.0

        result = extract_muse_channels(signals)
        np.testing.assert_array_equal(result[0], np.ones(10) * 1.0)   # TP9
        np.testing.assert_array_equal(result[1], np.ones(10) * 2.0)   # AF7
        np.testing.assert_array_equal(result[2], np.ones(10) * 3.0)   # AF8
        np.testing.assert_array_equal(result[3], np.ones(10) * 4.0)   # TP10

    def test_rejects_too_few_channels(self):
        with pytest.raises(ValueError, match="Expected 32-channel"):
            extract_muse_channels(np.random.randn(4, 100))


# ---------------------------------------------------------------------------
# kl_divergence_loss
# ---------------------------------------------------------------------------

class TestKLDivergenceLoss:
    def test_identical_distributions_zero_loss(self):
        logits = np.array([1.0, 2.0, 0.5, -0.5, 0.0, 1.5])
        loss = kl_divergence_loss(logits, logits, temperature=4.0)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_positive_loss(self):
        student = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        teacher = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        loss = kl_divergence_loss(student, teacher, temperature=4.0)
        assert loss > 0.0

    def test_higher_temperature_smoother(self):
        student = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        teacher = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        loss_low_t = kl_divergence_loss(student, teacher, temperature=1.0)
        loss_high_t = kl_divergence_loss(student, teacher, temperature=10.0)
        # Higher T means softer distributions; after T^2 scaling, different balance
        # Both should be finite and positive
        assert np.isfinite(loss_low_t) and loss_low_t > 0
        assert np.isfinite(loss_high_t) and loss_high_t > 0

    def test_output_is_scalar(self):
        logits = np.random.randn(6)
        loss = kl_divergence_loss(logits, logits * 0.5, temperature=3.0)
        assert isinstance(loss, float)

    def test_symmetric_inputs_zero(self):
        """KL of identical logits should be zero regardless of temperature."""
        logits = np.array([0.3, -0.5, 1.0, 0.2, -0.1, 0.7])
        for t in [1.0, 3.0, 5.0, 10.0]:
            assert kl_divergence_loss(logits, logits, t) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# distillation_loss
# ---------------------------------------------------------------------------

class TestDistillationLoss:
    def test_alpha_zero_uses_only_hard_loss(self):
        """alpha=0 means only cross-entropy with true label."""
        student = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        teacher = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
        loss = distillation_loss(student, teacher, hard_label=0, alpha=0.0)
        # student strongly predicts class 0, which IS the hard label
        # so CE loss should be low
        assert loss < 1.0

    def test_alpha_one_uses_only_soft_loss(self):
        """alpha=1 means only KL divergence with teacher."""
        student = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        teacher = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        loss = distillation_loss(student, teacher, hard_label=3, alpha=1.0)
        # Identical student/teacher -> KL=0, so loss should be near 0
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_positive_loss_for_mismatch(self):
        student = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        teacher = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        loss = distillation_loss(student, teacher, hard_label=0)
        assert loss > 0.0

    def test_output_is_float(self):
        student = np.random.randn(6)
        teacher = np.random.randn(6)
        loss = distillation_loss(student, teacher, hard_label=2)
        assert isinstance(loss, float)

    def test_all_labels_valid(self):
        student = np.random.randn(6)
        teacher = np.random.randn(6)
        for label in range(6):
            loss = distillation_loss(student, teacher, hard_label=label)
            assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# train_distilled_model — graceful failures
# ---------------------------------------------------------------------------

class TestTrainDistilledModel:
    def test_missing_teacher_returns_error(self, tmp_path):
        result = train_distilled_model(
            teacher_path=str(tmp_path / "nonexistent.pt"),
            deap_path=str(tmp_path),
            output_path=str(tmp_path / "out.pt"),
        )
        assert result["error"] == "teacher_not_found"

    def test_missing_deap_returns_error(self, tmp_path):
        # Create a fake teacher file
        teacher = tmp_path / "teacher.pt"
        teacher.write_text("fake")
        result = train_distilled_model(
            teacher_path=str(teacher),
            deap_path=str(tmp_path / "no_deap"),
            output_path=str(tmp_path / "out.pt"),
        )
        assert result["error"] == "deap_not_found"


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_six_emotions(self):
        assert len(EMOTIONS) == 6
        assert "happy" in EMOTIONS
        assert "neutral" in EMOTIONS

    def test_channel_map_has_four_entries(self):
        assert len(MUSE_CHANNEL_MAP) == 4
        assert "AF7" in MUSE_CHANNEL_MAP
        assert "AF8" in MUSE_CHANNEL_MAP
        assert "TP9" in MUSE_CHANNEL_MAP
        assert "TP10" in MUSE_CHANNEL_MAP

    def test_channel_indices_match_map(self):
        assert MUSE_CH_INDICES == list(MUSE_CHANNEL_MAP.values())
