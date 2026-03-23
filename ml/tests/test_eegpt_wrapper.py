"""Tests for EEGPT foundation model wrapper."""
import numpy as np
import pytest

from models.eegpt_wrapper import (
    DEFAULT_FS,
    EEGPT_EXPECTED_CHANNELS,
    EMOTIONS_6,
    MUSE_CH_NAMES,
    MUSE_CHANNELS,
    EEGPTWrapper,
)


@pytest.fixture
def wrapper():
    return EEGPTWrapper()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_initializes_without_crash(self, wrapper):
        assert hasattr(wrapper, "available")
        assert hasattr(wrapper, "predict")

    def test_available_is_bool(self, wrapper):
        assert isinstance(wrapper.available, bool)

    def test_lazy_loading(self):
        """Model should not load until available/predict is called."""
        w = EEGPTWrapper()
        assert w._load_attempted is False
        _ = w.available  # triggers load
        assert w._load_attempted is True

    def test_custom_model_path(self, tmp_path):
        w = EEGPTWrapper(model_path=str(tmp_path / "custom.pt"))
        assert w._model_path == tmp_path / "custom.pt"


# ---------------------------------------------------------------------------
# predict — model not available (no fine-tuned weights)
# ---------------------------------------------------------------------------

class TestPredictUnavailable:
    def test_returns_none_when_unavailable(self, wrapper):
        if not wrapper.available:
            signals = np.random.randn(4, 1024)
            result = wrapper.predict(signals)
            assert result is None

    def test_none_input_returns_none(self, wrapper):
        result = wrapper.predict(None)
        assert result is None

    def test_1d_input_returns_none(self, wrapper):
        result = wrapper.predict(np.random.randn(256))
        assert result is None

    def test_too_few_channels_returns_none(self, wrapper):
        result = wrapper.predict(np.random.randn(2, 1024))
        assert result is None

    def test_too_short_signal_returns_none(self, wrapper):
        # Less than 2 seconds at 256 Hz
        result = wrapper.predict(np.random.randn(4, 100), fs=256)
        assert result is None


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_get_model_info(self):
        info = EEGPTWrapper.get_model_info()
        assert info["name"] == "EEGPT"
        assert info["input_channels"] == 4
        assert info["pretrained_channels"] == 64
        assert info["requires_finetuning"] is True
        assert "torch" in info["requires"]

    def test_get_finetuning_guide(self):
        guide = EEGPTWrapper.get_finetuning_guide()
        assert isinstance(guide, str)
        assert "Fine-tuning" in guide
        assert "Muse 2" in guide
        assert "eegpt_muse4ch.pt" in guide


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_fs(self):
        assert DEFAULT_FS == 256

    def test_muse_channels(self):
        assert MUSE_CHANNELS == 4

    def test_eegpt_expected_channels(self):
        assert EEGPT_EXPECTED_CHANNELS == 64

    def test_emotions_6(self):
        assert len(EMOTIONS_6) == 6
        expected = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
        assert set(EMOTIONS_6) == expected

    def test_muse_channel_names(self):
        assert MUSE_CH_NAMES == ["TP9", "AF7", "AF8", "TP10"]
