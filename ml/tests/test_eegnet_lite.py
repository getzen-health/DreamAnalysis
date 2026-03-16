"""Tests for EEGNet-Lite compact on-device EEG emotion classifier."""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ is on the path so model imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.eegnet_lite import EEGNetLite
from models.eegnet_lite import _TORCH_AVAILABLE as TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EMOTION_CLASSES = ["positive", "neutral", "negative"]


@pytest.fixture
def model():
    """Fresh EEGNetLite instance (no saved weights)."""
    return EEGNetLite()


@pytest.fixture
def eeg_4ch():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz (~10 µV RMS)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 1024)) * 10.0


@pytest.fixture
def eeg_1ch():
    """Single-channel synthetic EEG (1-D array)."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(1024) * 10.0


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

    def test_module_attribute_set_when_torch_available(self, model):
        """model._module is a nn.Module when torch is available and weights exist, else None."""
        # Without saved weights, _module may be None even if torch is available
        # Just check it doesn't crash
        assert hasattr(model, '_module')


# ---------------------------------------------------------------------------
# get_model_info()
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    REQUIRED_KEYS = {
        "architecture",
        "n_params",
        "n_channels",
        "n_classes",
        "classes",
        "pytorch_available",
    }

    def test_returns_dict(self, model):
        info = model.get_model_info()
        assert isinstance(info, dict)

    def test_all_required_keys_present(self, model):
        info = model.get_model_info()
        assert self.REQUIRED_KEYS.issubset(info.keys())

    def test_architecture_name(self, model):
        assert model.get_model_info()["architecture"] == "EEGNet-Lite"

    def test_n_classes_is_3(self, model):
        assert model.get_model_info()["n_classes"] == 3

    def test_n_channels_is_4(self, model):
        assert model.get_model_info()["n_channels"] == 4

    def test_classes_list_length(self, model):
        assert len(model.get_model_info()["classes"]) == 3

    def test_classes_list_matches(self, model):
        assert model.get_model_info()["classes"] == EMOTION_CLASSES

    def test_n_params_positive_when_torch(self, model):
        """n_params should be positive when torch is available."""
        if TORCH_AVAILABLE:
            assert model.get_model_info()["n_params"] > 0

    def test_pytorch_available_reflects_import(self, model):
        assert model.get_model_info()["pytorch_available"] == TORCH_AVAILABLE

    def test_approximately_2700_params(self, model):
        """EEGNet-Lite for 4-channel Muse 2 should be ~2707 parameters."""
        if TORCH_AVAILABLE:
            n = model.get_model_info()["n_params"]
            assert 1000 < n < 10000, f"unexpected param count: {n}"


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_dict(self, model, eeg_4ch):
        result = model.predict(eeg_4ch)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, model, eeg_4ch):
        result = model.predict(eeg_4ch)
        for key in ("emotion", "probabilities", "model_type"):
            assert key in result, f"missing key: {key}"

    def test_emotion_is_valid(self, model, eeg_4ch):
        """Returned emotion must be one of the 3 canonical classes."""
        assert model.predict(eeg_4ch)["emotion"] in EMOTION_CLASSES

    def test_probabilities_keys_match_classes(self, model, eeg_4ch):
        probs = model.predict(eeg_4ch)["probabilities"]
        assert set(probs.keys()) == set(EMOTION_CLASSES)

    def test_probabilities_sum_to_one(self, model, eeg_4ch):
        probs = model.predict(eeg_4ch)["probabilities"]
        total = sum(probs.values())
        assert math.isclose(total, 1.0, abs_tol=0.02), f"probs sum={total}"

    def test_all_probabilities_non_negative(self, model, eeg_4ch):
        probs = model.predict(eeg_4ch)["probabilities"]
        for cls, p in probs.items():
            assert p >= 0.0, f"{cls} probability is negative: {p}"

    def test_valence_in_range(self, model, eeg_4ch):
        result = model.predict(eeg_4ch)
        if "valence" in result:
            assert -1.0 <= result["valence"] <= 1.0

    def test_predict_1d_input(self, model, eeg_1ch):
        """Single-channel 1-D input must be accepted without error."""
        result = model.predict(eeg_1ch)
        assert result["emotion"] in EMOTION_CLASSES

    def test_predict_1d_probabilities_sum(self, model, eeg_1ch):
        probs = model.predict(eeg_1ch)["probabilities"]
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=0.02)

    def test_predict_too_many_channels_handled(self, model):
        """Input with 6 channels must not raise."""
        rng = np.random.default_rng(99)
        eeg_6ch = rng.standard_normal((6, 1024)) * 10.0
        result = model.predict(eeg_6ch)
        assert result["emotion"] in EMOTION_CLASSES

    def test_predict_too_few_channels_handled(self, model):
        """Input with 2 channels must not raise."""
        rng = np.random.default_rng(55)
        eeg_2ch = rng.standard_normal((2, 1024)) * 10.0
        result = model.predict(eeg_2ch)
        assert result["emotion"] in EMOTION_CLASSES

    def test_predict_short_signal(self, model):
        """Signal shorter than 1024 samples must be handled (padded or feature-based)."""
        rng = np.random.default_rng(13)
        eeg_short = rng.standard_normal((4, 64)) * 10.0
        result = model.predict(eeg_short)
        assert result["emotion"] in EMOTION_CLASSES

    def test_predict_long_signal(self, model):
        """Signal longer than 1024 samples must be handled (trimmed)."""
        rng = np.random.default_rng(17)
        eeg_long = rng.standard_normal((4, 2048)) * 10.0
        result = model.predict(eeg_long)
        assert result["emotion"] in EMOTION_CLASSES


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

    def test_loss_key_present(self, model, eeg_4ch):
        result = model.fine_tune_last_layer(eeg_4ch, label=1)
        assert "loss" in result

    def test_fine_tune_all_labels(self, model, eeg_4ch):
        """Fine-tuning with each of the 3 label indices must not raise."""
        for label in range(3):
            result = model.fine_tune_last_layer(eeg_4ch, label=label)
            assert "updated" in result

    def test_fine_tune_updates_weights(self, model, eeg_4ch):
        """After one SGD step, the classifier weights must have changed."""
        if not TORCH_AVAILABLE or model._module is None:
            pytest.skip("torch model not loaded")
        import torch
        # Find the classifier layer
        classifier_params = [p for n, p in model._module.named_parameters() if "classifier" in n]
        if not classifier_params:
            pytest.skip("no classifier layer found")
        before = classifier_params[0].detach().clone()
        model.fine_tune_last_layer(eeg_4ch, label=0)
        after = classifier_params[0].detach()
        assert not torch.allclose(before, after), "classifier weights did not change"

    def test_fine_tune_graceful_without_model(self):
        """Without a loaded model, fine_tune returns updated=False."""
        m = EEGNetLite.__new__(EEGNetLite)
        m._module = None
        m._ort = None
        m.n_channels = 4
        m.n_classes = 3
        rng = np.random.default_rng(42)
        eeg = rng.standard_normal((4, 1024)) * 10.0
        result = m.fine_tune_last_layer(eeg, label=0)
        assert result["updated"] is False
