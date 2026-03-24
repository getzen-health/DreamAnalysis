"""Tests for FAA delta in temporal emotion features.

Verifies:
  1. With no history, delta_faa = 0 in temporal output
  2. With history, FAA delta reflects the change between epochs
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.emotion_features_enhanced import (
    ENHANCED_FEATURE_DIM,
    FEATURE_NAMES,
    extract_enhanced_emotion_features,
    extract_temporal_features,
    get_feature_names,
)


# FAA sits at index 35 in the 53-dim feature vector.
# In the 106-dim temporal output, delta_faa is at index 53 + 35 = 88.
FAA_IDX = FEATURE_NAMES.index("faa")
DELTA_FAA_IDX = ENHANCED_FEATURE_DIM + FAA_IDX


@pytest.fixture
def eeg_4ch():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 1024)) * 20.0


@pytest.fixture
def eeg_4ch_different():
    """A different 4-channel EEG sample (different seed)."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((4, 1024)) * 25.0


class TestFAADelta:
    """Tests for FAA delta in extract_temporal_features."""

    def test_feature_name_exists(self):
        """delta_faa should be a named feature in the temporal feature set."""
        names = get_feature_names(include_temporal=True)
        assert "delta_faa" in names
        assert names[DELTA_FAA_IDX] == "delta_faa"

    def test_no_history_faa_delta_is_zero(self, eeg_4ch):
        """Without history, FAA delta should be zero."""
        features = extract_enhanced_emotion_features(eeg_4ch, fs=256)
        temporal = extract_temporal_features(features, history=None)

        assert temporal[DELTA_FAA_IDX] == 0.0

    def test_empty_history_faa_delta_is_zero(self, eeg_4ch):
        """Empty history list should also produce zero FAA delta."""
        features = extract_enhanced_emotion_features(eeg_4ch, fs=256)
        temporal = extract_temporal_features(features, history=[])

        assert temporal[DELTA_FAA_IDX] == 0.0

    def test_with_history_faa_delta_reflects_change(self, eeg_4ch, eeg_4ch_different):
        """With a previous epoch, FAA delta should be non-zero (different inputs)."""
        f1 = extract_enhanced_emotion_features(eeg_4ch, fs=256)
        f2 = extract_enhanced_emotion_features(eeg_4ch_different, fs=256)

        # FAA values should differ between the two signals
        faa1 = f1[FAA_IDX]
        faa2 = f2[FAA_IDX]

        temporal = extract_temporal_features(f2, history=[f1], time_interval=2.0)

        # delta_faa = (faa2 - faa1) / time_interval
        expected_delta = (faa2 - faa1) / 2.0
        assert temporal[DELTA_FAA_IDX] == pytest.approx(expected_delta, abs=1e-10)

    def test_faa_delta_sign_matches_direction(self):
        """When right frontal alpha increases, FAA delta should be positive."""
        rng = np.random.default_rng(200)
        t = np.arange(1024) / 256.0

        # Epoch 1: symmetric alpha (equal AF7/AF8)
        signals1 = rng.standard_normal((4, 1024)) * 5.0
        signals1[1] += 15.0 * np.sin(2 * np.pi * 10 * t)  # AF7 alpha
        signals1[2] += 15.0 * np.sin(2 * np.pi * 10 * t)  # AF8 alpha (same)

        # Epoch 2: increased right alpha (AF8 > AF7)
        signals2 = rng.standard_normal((4, 1024)) * 5.0
        signals2[1] += 10.0 * np.sin(2 * np.pi * 10 * t)  # AF7 less alpha
        signals2[2] += 30.0 * np.sin(2 * np.pi * 10 * t)  # AF8 more alpha

        f1 = extract_enhanced_emotion_features(signals1, fs=256)
        f2 = extract_enhanced_emotion_features(signals2, fs=256)

        temporal = extract_temporal_features(f2, history=[f1])
        delta_faa = temporal[DELTA_FAA_IDX]

        # FAA = log(AF8_alpha) - log(AF7_alpha)
        # Epoch 2 has higher right alpha → higher FAA → positive delta
        assert delta_faa > 0, (
            f"Expected positive FAA delta when right alpha increases, got {delta_faa}"
        )

    def test_faa_delta_is_finite(self, eeg_4ch, eeg_4ch_different):
        """FAA delta must always be finite."""
        f1 = extract_enhanced_emotion_features(eeg_4ch, fs=256)
        f2 = extract_enhanced_emotion_features(eeg_4ch_different, fs=256)

        temporal = extract_temporal_features(f2, history=[f1])
        assert np.isfinite(temporal[DELTA_FAA_IDX])
