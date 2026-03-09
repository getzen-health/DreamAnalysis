"""Tests for EEGNet-Lite compact on-device EEG emotion classifier."""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ is on the path so model imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.eegnet_lite import EEGNetLite, TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


@pytest.fixture
def model():
    """Fresh EEGNetLite instance (no saved weights)."""
    return EEGNetLite()


@pytest.fixture
def eeg_4ch():
    """4 channels x 1 second of synthetic EEG at 256 Hz (~10 µV RMS)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 256)) * 10.0


@pytest.fixture
def eeg_1ch():
    """Single-channel synthetic EEG (1-D array)."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(256) * 10.0


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_without_error(self):
        """EEGNetLite() must construct without raising."""
        m = EEGNetLite()
        assert m is not None

    def test_instantiates_with_nonexistent_path(self, tmp_path):
        """Passing a missing model_path must not raise; weights just silently skip."""
        m = EEGNetLite(model_path=str(tmp_path / "missing.pt"))
        assert m is not None

    def test_model_attribute_set_when_torch_available(self, model):
        """model.model is a nn.Module when torch is available, else None."""
        if TORCH_AVAILABLE:
            import torch.nn as nn
            assert isinstance(model.model, nn.Module)
        else:
            assert model.model is None


# ---------------------------------------------------------------------------
# get_model_info()
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    REQUIRED_KEYS = {
        "architecture",
        "n_params",
        "model_type",
        "torch_available",
        "n_channels",
        "n_classes",
        "input_samples",
        "emotions",
        "reference",
    }

    def test_returns_dict(self, model):
        info = model.get_model_info()
        assert isinstance(info, dict)

    def test_all_required_keys_present(self, model):
        info = model.get_model_info()
        assert self.REQUIRED_KEYS.issubset(info.keys())

    def test_architecture_name(self, model):
        assert model.get_model_info()["architecture"] == "EEGNet-Lite"

    def test_n_classes_is_6(self, model):
        assert model.get_model_info()["n_classes"] == 6

    def test_n_channels_is_4(self, model):
        assert model.get_model_info()["n_channels"] == 4

    def test_emotions_list_length(self, model):
        assert len(model.get_model_info()["emotions"]) == 6

    def test_emotions_list_matches_class_constant(self, model):
        assert model.get_model_info()["emotions"] == EMOTIONS_6

    def test_n_params_matches_count_params(self, model):
        """n_params in info must equal _count_params()."""
        assert model.get_model_info()["n_params"] == model._count_params()

    def test_torch_available_reflects_import(self, model):
        assert model.get_model_info()["torch_available"] == TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_dict(self, model, eeg_4ch):
        result = model.predict(eeg_4ch)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, model, eeg_4ch):
        result = model.predict(eeg_4ch)
        for key in ("emotion", "probabilities", "confidence", "model_type", "n_params"):
            assert key in result, f"missing key: {key}"

    def test_emotion_is_valid(self, model, eeg_4ch):
        """Returned emotion must be one of the 6 canonical classes."""
        assert model.predict(eeg_4ch)["emotion"] in EMOTIONS_6

    def test_probabilities_keys_match_emotions(self, model, eeg_4ch):
        probs = model.predict(eeg_4ch)["probabilities"]
        assert set(probs.keys()) == set(EMOTIONS_6)

    def test_probabilities_sum_to_one(self, model, eeg_4ch):
        probs = model.predict(eeg_4ch)["probabilities"]
        total = sum(probs.values())
        assert math.isclose(total, 1.0, abs_tol=0.01), f"probs sum={total}"

    def test_all_probabilities_non_negative(self, model, eeg_4ch):
        probs = model.predict(eeg_4ch)["probabilities"]
        for emotion, p in probs.items():
            assert p >= 0.0, f"{emotion} probability is negative: {p}"

    def test_confidence_in_unit_interval(self, model, eeg_4ch):
        conf = model.predict(eeg_4ch)["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_confidence_equals_max_probability(self, model, eeg_4ch):
        result = model.predict(eeg_4ch)
        max_p = max(result["probabilities"].values())
        assert math.isclose(result["confidence"], max_p, abs_tol=0.0001)

    def test_predict_1d_input(self, model, eeg_1ch):
        """Single-channel 1-D input must be accepted without error."""
        result = model.predict(eeg_1ch)
        assert result["emotion"] in EMOTIONS_6

    def test_predict_1d_probabilities_sum(self, model, eeg_1ch):
        probs = model.predict(eeg_1ch)["probabilities"]
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=0.01)

    def test_predict_too_many_channels_trimmed(self, model):
        """Input with 6 channels must not raise; extra channels are trimmed."""
        rng = np.random.default_rng(99)
        eeg_6ch = rng.standard_normal((6, 256)) * 10.0
        result = model.predict(eeg_6ch)
        assert result["emotion"] in EMOTIONS_6

    def test_predict_too_few_channels_padded(self, model):
        """Input with 2 channels must not raise; missing channels are zero-padded."""
        rng = np.random.default_rng(55)
        eeg_2ch = rng.standard_normal((2, 256)) * 10.0
        result = model.predict(eeg_2ch)
        assert result["emotion"] in EMOTIONS_6

    def test_predict_short_signal_padded(self, model):
        """Signal shorter than T=256 samples must be zero-padded, not raise."""
        rng = np.random.default_rng(13)
        eeg_short = rng.standard_normal((4, 64)) * 10.0
        result = model.predict(eeg_short)
        assert result["emotion"] in EMOTIONS_6

    def test_predict_long_signal_trimmed(self, model):
        """Signal longer than T=256 samples must be trimmed, not raise."""
        rng = np.random.default_rng(17)
        eeg_long = rng.standard_normal((4, 1024)) * 10.0
        result = model.predict(eeg_long)
        assert result["emotion"] in EMOTIONS_6


# ---------------------------------------------------------------------------
# _count_params()
# ---------------------------------------------------------------------------


class TestCountParams:
    def test_returns_int(self, model):
        assert isinstance(model._count_params(), int)

    def test_positive_when_torch_available(self, model):
        if TORCH_AVAILABLE:
            assert model._count_params() > 0

    def test_zero_when_no_model(self):
        """If model attribute is None, _count_params must return 0."""
        m = EEGNetLite.__new__(EEGNetLite)
        m.model = None
        m.model_type = "feature-based"
        assert m._count_params() == 0

    def test_approximately_2600_params(self, model):
        """EEGNet-Lite for 4-channel Muse 2 should be ~2600 parameters."""
        if TORCH_AVAILABLE:
            n = model._count_params()
            assert 1000 < n < 10000, f"unexpected param count: {n}"


# ---------------------------------------------------------------------------
# fine_tune_last_layer()
# ---------------------------------------------------------------------------


class TestFineTuneLastLayer:
    def test_returns_dict(self, model, eeg_4ch):
        result = model.fine_tune_last_layer(eeg_4ch, label=0)
        assert isinstance(result, dict)

    def test_updated_key_present(self, model, eeg_4ch):
        result = model.fine_tune_last_layer(eeg_4ch, label=2)
        assert "updated" in result

    def test_updated_true_when_torch_available(self, model, eeg_4ch):
        result = model.fine_tune_last_layer(eeg_4ch, label=1)
        if TORCH_AVAILABLE:
            assert result["updated"] is True
        else:
            assert result["updated"] is False

    def test_loss_is_finite_when_torch_available(self, model, eeg_4ch):
        result = model.fine_tune_last_layer(eeg_4ch, label=3)
        if TORCH_AVAILABLE:
            assert result["loss"] is not None
            assert math.isfinite(result["loss"])

    def test_fine_tune_all_labels(self, model, eeg_4ch):
        """Fine-tuning with each of the 6 label indices must not raise."""
        for label in range(6):
            result = model.fine_tune_last_layer(eeg_4ch, label=label)
            assert "updated" in result

    def test_fine_tune_updates_weights(self, model, eeg_4ch):
        """After one SGD step, the fc layer weights must have changed."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        import torch
        before = model.model.fc.weight.detach().clone()
        model.fine_tune_last_layer(eeg_4ch, label=0)
        after = model.model.fc.weight.detach()
        assert not torch.allclose(before, after), "fc weights did not change after fine-tune"

    def test_fine_tune_no_torch_graceful(self):
        """Without torch, fine_tune returns updated=False with a reason."""
        m = EEGNetLite.__new__(EEGNetLite)
        m.model = None
        m.model_type = "feature-based"
        rng = np.random.default_rng(42)
        eeg = rng.standard_normal((4, 256)) * 10.0
        result = m.fine_tune_last_layer(eeg, label=0)
        assert result["updated"] is False
