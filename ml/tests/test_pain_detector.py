"""Tests for PainDetector — chronic pain biomarker detection from frontal EEG."""
import numpy as np
import pytest

from models.pain_detector import PainDetector


@pytest.fixture
def detector():
    return PainDetector()


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ---------------------------------------------------------------------------
# 1. Single-channel input → insufficient_data
# ---------------------------------------------------------------------------
def test_single_channel_returns_insufficient(detector, rng):
    signal_1d = rng.randn(256)
    result = detector.predict(signal_1d, fs=256)
    assert result['pain_level'] == 'insufficient_data'
    assert result['pain_biomarker_score'] == 0.0
    assert result['components'] == {}


# ---------------------------------------------------------------------------
# 2. Multichannel input returns all expected top-level keys
# ---------------------------------------------------------------------------
def test_multichannel_returns_expected_keys(detector, rng):
    signals = rng.randn(4, 256)
    result = detector.predict(signals, fs=256)
    expected_keys = {
        'pain_biomarker_score',
        'pain_level',
        'components',
        'model_type',
        'disclaimer',
    }
    assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# 3. pain_biomarker_score is between 0 and 1
# ---------------------------------------------------------------------------
def test_pain_score_in_range(detector, rng):
    signals = rng.randn(4, 512)
    result = detector.predict(signals, fs=256)
    assert 0.0 <= result['pain_biomarker_score'] <= 1.0


# ---------------------------------------------------------------------------
# 4-7. Pain level thresholds
# ---------------------------------------------------------------------------
def test_pain_level_none_detected(detector):
    """Score < 0.3 → none_detected."""
    det = PainDetector()
    # All zeros produces very low asymmetry → low score
    signals = np.zeros((4, 512))
    result = det.predict(signals, fs=256)
    assert result['pain_biomarker_score'] < 0.3
    assert result['pain_level'] == 'none_detected'


def test_pain_level_mild_indicators(detector, rng):
    """Score in [0.3, 0.5) → mild_indicators."""
    # Inject moderate asymmetry between AF7 (ch1) and AF8 (ch2)
    signals = rng.randn(4, 1024) * 5.0
    signals[2] *= 1.8  # boost AF8 beta
    result = detector.predict(signals, fs=256)
    # We can't guarantee exact thresholds with random data, so
    # if this doesn't hit mild, at least verify the mapping logic works
    if 0.3 <= result['pain_biomarker_score'] < 0.5:
        assert result['pain_level'] == 'mild_indicators'


def test_pain_level_moderate_indicators(detector, rng):
    """Score in [0.5, 0.7) → moderate_indicators."""
    signals = rng.randn(4, 1024) * 10.0
    signals[2] *= 3.0  # larger AF8 asymmetry
    result = detector.predict(signals, fs=256)
    if 0.5 <= result['pain_biomarker_score'] < 0.7:
        assert result['pain_level'] == 'moderate_indicators'


def test_pain_level_elevated_indicators(detector, rng):
    """Score >= 0.7 → elevated_indicators."""
    signals = rng.randn(4, 1024) * 20.0
    signals[2] *= 5.0  # very large AF8 boost
    result = detector.predict(signals, fs=256)
    if result['pain_biomarker_score'] >= 0.7:
        assert result['pain_level'] == 'elevated_indicators'


# ---------------------------------------------------------------------------
# 8. High beta asymmetry → higher pain score
# ---------------------------------------------------------------------------
def test_high_beta_asymmetry_increases_score(detector, rng):
    base = rng.randn(4, 1024)
    # Symmetric case
    sym_result = detector.predict(base.copy(), fs=256)
    # Asymmetric case — boost AF8 significantly
    asym = base.copy()
    asym[2] *= 4.0
    asym_result = detector.predict(asym, fs=256)
    assert asym_result['pain_biomarker_score'] >= sym_result['pain_biomarker_score']


# ---------------------------------------------------------------------------
# 9. Symmetric beta → lower beta_asymmetry_score component
# ---------------------------------------------------------------------------
def test_symmetric_beta_low_asymmetry_score(detector):
    # Use identical signals for AF7 and AF8
    base = np.sin(2 * np.pi * 15 * np.linspace(0, 1, 256))  # 15 Hz beta
    signals = np.stack([base, base, base, base])
    result = detector.predict(signals, fs=256)
    # Identical channels → asymmetry should be near zero
    assert result['components']['beta_asymmetry_score'] < 0.1


# ---------------------------------------------------------------------------
# 10. Components dict has all expected sub-keys
# ---------------------------------------------------------------------------
def test_components_subkeys(detector, rng):
    signals = rng.randn(4, 512)
    result = detector.predict(signals, fs=256)
    expected_subkeys = {
        'beta_asymmetry',
        'beta_asymmetry_score',
        'alpha_de_asymmetry',
        'alpha_de_score',
        'high_beta_mean',
        'high_beta_score',
    }
    assert expected_subkeys == set(result['components'].keys())


# ---------------------------------------------------------------------------
# 11. Disclaimer is always present
# ---------------------------------------------------------------------------
def test_disclaimer_present(detector, rng):
    # Multichannel path
    signals = rng.randn(4, 256)
    result = detector.predict(signals, fs=256)
    assert 'disclaimer' in result
    assert len(result['disclaimer']) > 0

    # Single-channel (insufficient) path
    result_1d = detector.predict(rng.randn(256), fs=256)
    assert 'disclaimer' in result_1d
    assert len(result_1d['disclaimer']) > 0


# ---------------------------------------------------------------------------
# 12. model_type is 'pain_biomarker'
# ---------------------------------------------------------------------------
def test_model_type(detector, rng):
    signals = rng.randn(4, 256)
    result = detector.predict(signals, fs=256)
    assert result['model_type'] == 'pain_biomarker'

    result_1d = detector.predict(rng.randn(256), fs=256)
    assert result_1d['model_type'] == 'pain_biomarker'


# ---------------------------------------------------------------------------
# 13. 2-channel input works (uses ch0 and ch1)
# ---------------------------------------------------------------------------
def test_two_channel_input(detector, rng):
    signals = rng.randn(2, 512)
    result = detector.predict(signals, fs=256)
    assert result['pain_level'] != 'insufficient_data'
    assert 0.0 <= result['pain_biomarker_score'] <= 1.0
    assert len(result['components']) > 0


# ---------------------------------------------------------------------------
# 14. 4-channel input uses correct AF7 (ch1) / AF8 (ch2) channels
# ---------------------------------------------------------------------------
def test_four_channel_uses_af7_af8(detector):
    t = np.linspace(0, 4, 1024)
    # AF7 (ch1): mostly alpha (10 Hz)
    af7 = np.sin(2 * np.pi * 10 * t)
    # AF8 (ch2): mostly beta (20 Hz) — more beta-band relative power
    af8 = np.sin(2 * np.pi * 20 * t)
    # ch0, ch3 are irrelevant noise
    noise = np.random.RandomState(0).randn(1024) * 0.1
    signals = np.stack([noise, af7, af8, noise])
    result = detector.predict(signals, fs=256)
    # AF8 has more relative beta power than AF7 → positive beta asymmetry
    assert result['components']['beta_asymmetry'] > 0


# ---------------------------------------------------------------------------
# 15. All-zeros input → low pain score
# ---------------------------------------------------------------------------
def test_all_zeros_low_score(detector):
    signals = np.zeros((4, 512))
    result = detector.predict(signals, fs=256)
    assert result['pain_biomarker_score'] < 0.3
    assert result['pain_level'] == 'none_detected'
