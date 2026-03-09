"""Tests for cross-modal EEG+Voice Optimal Transport fusion.

Covers:
- Instantiation
- Full bimodal prediction (EEG + audio)
- EEG-only fallback (audio=None)
- Voice-only fallback (eeg=None)
- Output structure and invariants
- Sinkhorn transport plan validity
- get_fusion_stats()
- Edge cases: empty audio, all-zeros EEG
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add ml/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.eeg_voice_fusion import (
    EEGVoiceFusionClassifier,
    EMOTIONS_6,
    _sinkhorn_transport,
    get_eeg_voice_fusion,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
EEG_FS = 256.0
VOICE_SR = 16000


@pytest.fixture
def eeg_4ch():
    """4 channels x 256 samples (1 second) of synthetic EEG at 256 Hz."""
    return RNG.standard_normal((4, 256)).astype(np.float32) * 20.0  # ~20 uV RMS


@pytest.fixture
def audio_1s():
    """1 second of synthetic audio at 16 kHz."""
    t = np.linspace(0, 1, VOICE_SR, endpoint=False)
    # Sinusoidal voice-like signal at 150 Hz + harmonics
    signal = (
        0.5 * np.sin(2 * np.pi * 150 * t)
        + 0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * RNG.standard_normal(VOICE_SR)
    ).astype(np.float32)
    return signal


@pytest.fixture
def fusion():
    """Fresh EEGVoiceFusionClassifier instance."""
    return EEGVoiceFusionClassifier()


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_instantiation_defaults():
    clf = EEGVoiceFusionClassifier()
    assert clf.n_classes == 6
    assert clf.eeg_fs == 256.0
    assert clf.voice_fs == 16000


def test_instantiation_custom():
    clf = EEGVoiceFusionClassifier(n_classes=4, eeg_fs=128.0, voice_fs=22050)
    assert clf.n_classes == 4
    assert clf.eeg_fs == 128.0
    assert clf.voice_fs == 22050


# ── Full bimodal prediction ───────────────────────────────────────────────────

def test_predict_bimodal_returns_dict(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    assert isinstance(result, dict)


def test_predict_bimodal_has_required_keys(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    required = {"emotion", "probabilities", "valence", "arousal", "fusion_weight", "model_type"}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


def test_predict_bimodal_emotion_valid(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    assert result["emotion"] in EMOTIONS_6


def test_predict_bimodal_probabilities_sum_to_one(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    probs = result["probabilities"]
    assert set(probs.keys()) == set(EMOTIONS_6)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected ~1.0"


def test_predict_bimodal_valence_range(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    assert -1.0 <= result["valence"] <= 1.0


def test_predict_bimodal_arousal_range(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    assert 0.0 <= result["arousal"] <= 1.0


def test_predict_bimodal_fusion_weight_range(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    assert 0.0 <= result["fusion_weight"] <= 1.0


def test_predict_bimodal_model_type(fusion, eeg_4ch, audio_1s):
    result = fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    assert "fusion" in result["model_type"] or "eeg" in result["model_type"]


# ── EEG-only fallback ─────────────────────────────────────────────────────────

def test_predict_eeg_only(fusion, eeg_4ch):
    result = fusion.predict(eeg_4ch, None, EEG_FS, VOICE_SR)
    assert result["emotion"] in EMOTIONS_6
    probs = result["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert "eeg" in result["model_type"]


def test_predict_eeg_only_fusion_weight_zero(fusion, eeg_4ch):
    result = fusion.predict(eeg_4ch, None, EEG_FS, VOICE_SR)
    # No audio -> OT not run -> fusion_weight is 0.0
    assert result["fusion_weight"] == 0.0


# ── Voice-only fallback ───────────────────────────────────────────────────────

def test_predict_voice_only(fusion, audio_1s):
    result = fusion.predict(None, audio_1s, EEG_FS, VOICE_SR)
    assert result["emotion"] in EMOTIONS_6
    probs = result["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert "voice" in result["model_type"]


def test_predict_voice_only_fusion_weight_zero(fusion, audio_1s):
    result = fusion.predict(None, audio_1s, EEG_FS, VOICE_SR)
    assert result["fusion_weight"] == 0.0


# ── Sinkhorn transport plan ────────────────────────────────────────────────────

def test_sinkhorn_transport_shape():
    n, m = 5, 4
    mu = np.ones(n) / n
    nu = np.ones(m) / m
    C = np.random.rand(n, m)
    T = _sinkhorn_transport(mu, nu, C, eps=0.1, n_iter=50)
    assert T.shape == (n, m)


def test_sinkhorn_transport_non_negative():
    n, m = 5, 4
    mu = np.ones(n) / n
    nu = np.ones(m) / m
    C = np.random.rand(n, m)
    T = _sinkhorn_transport(mu, nu, C, eps=0.1, n_iter=50)
    assert np.all(T >= -1e-10), "Transport plan must be non-negative"


def test_sinkhorn_transport_row_marginals():
    """Row sums of T should approximate source distribution mu."""
    n, m = 6, 4
    mu = np.array([0.3, 0.2, 0.1, 0.15, 0.15, 0.1])
    nu = np.ones(m) / m
    C = np.outer(np.arange(n, dtype=float) / n, np.ones(m))
    T = _sinkhorn_transport(mu, nu, C, eps=0.05, n_iter=100)
    row_sums = T.sum(axis=1)
    np.testing.assert_allclose(row_sums, mu, atol=0.05, err_msg="Row marginals should approximate mu")


def test_sinkhorn_transport_col_marginals():
    """Column sums of T should approximate target distribution nu."""
    n, m = 4, 5
    mu = np.ones(n) / n
    nu = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
    C = np.ones((n, m)) * 0.1
    T = _sinkhorn_transport(mu, nu, C, eps=0.05, n_iter=100)
    col_sums = T.sum(axis=0)
    np.testing.assert_allclose(col_sums, nu, atol=0.05, err_msg="Column marginals should approximate nu")


# ── get_fusion_stats ──────────────────────────────────────────────────────────

def test_get_fusion_stats_keys(fusion, eeg_4ch, audio_1s):
    fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    stats = fusion.get_fusion_stats()
    assert "transport_cost" in stats
    assert "alignment_quality" in stats


def test_get_fusion_stats_alignment_quality_range(fusion, eeg_4ch, audio_1s):
    fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    stats = fusion.get_fusion_stats()
    assert 0.0 <= stats["alignment_quality"] <= 1.0


def test_get_fusion_stats_transport_cost_non_negative(fusion, eeg_4ch, audio_1s):
    fusion.predict(eeg_4ch, audio_1s, EEG_FS, VOICE_SR)
    stats = fusion.get_fusion_stats()
    assert stats["transport_cost"] >= 0.0


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_predict_empty_audio_falls_back_to_eeg(fusion, eeg_4ch):
    empty_audio = np.array([], dtype=np.float32)
    result = fusion.predict(eeg_4ch, empty_audio, EEG_FS, VOICE_SR)
    # Empty audio treated as no audio -> EEG-only
    assert result["emotion"] in EMOTIONS_6
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_predict_all_zeros_eeg(fusion, audio_1s):
    zero_eeg = np.zeros((4, 256), dtype=np.float32)
    result = fusion.predict(zero_eeg, audio_1s, EEG_FS, VOICE_SR)
    assert result["emotion"] in EMOTIONS_6
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_predict_both_none_returns_neutral(fusion):
    result = fusion.predict(None, None, EEG_FS, VOICE_SR)
    assert result["emotion"] in EMOTIONS_6
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


# ── Singleton registry ────────────────────────────────────────────────────────

def test_get_eeg_voice_fusion_returns_instance():
    inst = get_eeg_voice_fusion("test_user")
    assert isinstance(inst, EEGVoiceFusionClassifier)


def test_get_eeg_voice_fusion_singleton():
    a = get_eeg_voice_fusion("singleton_test")
    b = get_eeg_voice_fusion("singleton_test")
    assert a is b, "Should return same instance for same user_id"


def test_get_eeg_voice_fusion_different_users():
    a = get_eeg_voice_fusion("user_alpha")
    b = get_eeg_voice_fusion("user_beta")
    assert a is not b, "Different user_ids should have separate instances"
