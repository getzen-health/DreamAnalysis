"""Tests for theta-beta phase-amplitude coupling (PAC) emotion features.

Verifies:
1. PAC features are present in the enhanced feature vector (84 dim, up from 80)
2. PAC feature names follow pac_tb_{channel} naming convention
3. PAC values are bounded [0, 1] (normalized Modulation Index)
4. PAC is higher for signals with genuine theta-beta coupling vs uncoupled noise
5. PAC is finite for edge cases (flat signal, very short signal, single channel)
6. PAC MI computation matches expected behavior from Tort et al. (2010)

References:
    - Wang (2021, Neuroscience Letters): CFC networks outperform synchronization for emotion
    - ACCNet (2025, Neural Networks): Adaptive CFC for emotion recognition
    - Tort et al. (2010): Modulation Index as standard PAC metric
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.emotion_features_enhanced import (
    ENHANCED_FEATURE_DIM,
    FEATURE_NAMES,
    _compute_theta_beta_pac,
    extract_enhanced_emotion_features,
    extract_temporal_features,
    get_feature_names,
)


@pytest.fixture
def four_channel_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 1024)) * 20.0


@pytest.fixture
def coupled_eeg():
    """4-channel EEG with genuine theta-beta phase-amplitude coupling.

    Beta amplitude is modulated by theta phase: when theta is at its peak,
    beta amplitude is maximal. This creates a strong PAC signal.
    """
    rng = np.random.default_rng(321)
    fs = 256
    t = np.arange(4 * fs) / fs  # 4 seconds

    signals = np.zeros((4, len(t)))
    for ch in range(4):
        # Theta carrier (6 Hz)
        theta = np.sin(2 * np.pi * 6.0 * t)
        theta_phase = np.angle(
            np.exp(1j * 2 * np.pi * 6.0 * t)
        )

        # Beta oscillation (20 Hz) with amplitude modulated by theta phase
        # Amplitude peaks when theta phase ~ 0 (peak of theta)
        beta_amp = 1.0 + 0.8 * np.cos(theta_phase)  # modulation depth 0.8
        beta = beta_amp * np.sin(2 * np.pi * 20.0 * t)

        signals[ch] = theta * 15.0 + beta * 10.0 + rng.standard_normal(len(t)) * 2.0

    return signals


@pytest.fixture
def uncoupled_eeg():
    """4-channel EEG with independent theta and beta (no coupling)."""
    rng = np.random.default_rng(654)
    fs = 256
    t = np.arange(4 * fs) / fs

    signals = np.zeros((4, len(t)))
    for ch in range(4):
        theta = np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
        beta = np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
        signals[ch] = theta * 15.0 + beta * 10.0 + rng.standard_normal(len(t)) * 5.0

    return signals


class TestPACFeatureDimension:
    """Tests for feature dimension after adding PAC features."""

    def test_enhanced_feature_dim_is_84(self):
        """Feature dimension should be 84 (80 original + 4 PAC)."""
        assert ENHANCED_FEATURE_DIM == 84

    def test_feature_names_count_matches_dim(self):
        """Feature name list must match the feature dimension."""
        assert len(FEATURE_NAMES) == ENHANCED_FEATURE_DIM

    def test_pac_feature_names_present(self):
        """PAC feature names should follow pac_tb_{channel} convention."""
        pac_names = [n for n in FEATURE_NAMES if n.startswith("pac_tb_")]
        assert len(pac_names) == 4
        assert "pac_tb_TP9" in FEATURE_NAMES
        assert "pac_tb_AF7" in FEATURE_NAMES
        assert "pac_tb_AF8" in FEATURE_NAMES
        assert "pac_tb_TP10" in FEATURE_NAMES

    def test_extract_returns_correct_shape(self, four_channel_eeg):
        """Feature vector must have exactly 84 dimensions."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert features.shape == (84,)

    def test_temporal_features_double_dim(self, four_channel_eeg):
        """Temporal features should be 168 dims (84 + 84 deltas)."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        temporal = extract_temporal_features(features, history=None)
        assert temporal.shape == (168,)

    def test_get_feature_names_base(self):
        """Base feature names should be 84."""
        names = get_feature_names(include_temporal=False)
        assert len(names) == 84

    def test_get_feature_names_temporal(self):
        """Temporal feature names should be 168."""
        names = get_feature_names(include_temporal=True)
        assert len(names) == 168


class TestPACValues:
    """Tests for PAC value correctness."""

    def test_pac_features_bounded_0_1(self, four_channel_eeg):
        """PAC features must be in [0, 1] (normalized MI)."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        pac_indices = [FEATURE_NAMES.index(f"pac_tb_{ch}") for ch in ["TP9", "AF7", "AF8", "TP10"]]
        for idx in pac_indices:
            assert 0.0 <= features[idx] <= 1.0, (
                f"PAC feature at index {idx} = {features[idx]}, expected [0, 1]"
            )

    def test_pac_features_finite(self, four_channel_eeg):
        """PAC features must be finite (no NaN/inf)."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        pac_indices = [FEATURE_NAMES.index(f"pac_tb_{ch}") for ch in ["TP9", "AF7", "AF8", "TP10"]]
        for idx in pac_indices:
            assert np.isfinite(features[idx])

    def test_coupled_signal_higher_pac_than_uncoupled(self, coupled_eeg, uncoupled_eeg):
        """Signals with theta-beta coupling should have higher PAC MI."""
        feat_coupled = extract_enhanced_emotion_features(coupled_eeg, fs=256)
        feat_uncoupled = extract_enhanced_emotion_features(uncoupled_eeg, fs=256)

        pac_indices = [FEATURE_NAMES.index(f"pac_tb_{ch}") for ch in ["TP9", "AF7", "AF8", "TP10"]]

        # At least one channel should show higher PAC for coupled signal
        coupled_pac = np.mean([feat_coupled[i] for i in pac_indices])
        uncoupled_pac = np.mean([feat_uncoupled[i] for i in pac_indices])

        assert coupled_pac > uncoupled_pac, (
            f"Coupled PAC mean={coupled_pac:.4f} should be > uncoupled={uncoupled_pac:.4f}"
        )


class TestPACEdgeCases:
    """Tests for PAC with edge-case inputs."""

    def test_flat_signal_pac_zero(self):
        """Flat (DC) signal should produce zero PAC."""
        flat = np.ones((4, 1024)) * 0.001
        features = extract_enhanced_emotion_features(flat, fs=256)
        pac_indices = [FEATURE_NAMES.index(f"pac_tb_{ch}") for ch in ["TP9", "AF7", "AF8", "TP10"]]
        for idx in pac_indices:
            assert features[idx] == 0.0, f"Flat signal PAC at {idx} should be 0"

    def test_short_signal_does_not_crash(self):
        """Very short signals should produce features without errors."""
        rng = np.random.default_rng(77)
        short = rng.standard_normal((4, 64)) * 20.0
        features = extract_enhanced_emotion_features(short, fs=256)
        assert features.shape == (84,)
        assert np.all(np.isfinite(features))

    def test_single_channel_input(self):
        """Single-channel input padded to 4 channels should work."""
        rng = np.random.default_rng(99)
        signal_1d = rng.standard_normal(1024) * 20.0
        features = extract_enhanced_emotion_features(signal_1d, fs=256)
        assert features.shape == (84,)
        assert np.all(np.isfinite(features))


class TestComputeThetaBetaPAC:
    """Tests for the _compute_theta_beta_pac helper function."""

    def test_returns_float_in_0_1(self):
        """PAC MI should be a float in [0, 1]."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1024) * 20.0
        mi = _compute_theta_beta_pac(signal, fs=256)
        assert isinstance(mi, float)
        assert 0.0 <= mi <= 1.0

    def test_flat_signal_returns_zero(self):
        """Flat signal should return 0 MI."""
        flat = np.ones(1024) * 0.001
        mi = _compute_theta_beta_pac(flat, fs=256)
        assert mi == 0.0

    def test_short_signal_returns_zero(self):
        """Signal shorter than 2 theta cycles should return 0."""
        short = np.random.default_rng(42).standard_normal(32)
        mi = _compute_theta_beta_pac(short, fs=256)
        assert mi == 0.0

    def test_coupled_higher_than_noise(self):
        """Coupled signal should have higher MI than pure noise."""
        fs = 256
        t = np.arange(4 * fs) / fs
        rng = np.random.default_rng(42)

        # Coupled: beta amplitude modulated by theta phase
        theta_phase = 2 * np.pi * 6.0 * t
        beta_amp = 1.0 + 0.8 * np.cos(theta_phase)
        coupled = 15.0 * np.sin(theta_phase) + 10.0 * beta_amp * np.sin(2 * np.pi * 20.0 * t)
        coupled += rng.standard_normal(len(t)) * 2.0

        # Noise
        noise = rng.standard_normal(len(t)) * 20.0

        mi_coupled = _compute_theta_beta_pac(coupled, fs=fs)
        mi_noise = _compute_theta_beta_pac(noise, fs=fs)

        assert mi_coupled > mi_noise
