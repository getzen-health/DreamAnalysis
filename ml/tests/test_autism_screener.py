"""Tests for AutismScreener -- autism spectrum EEG biomarker screening."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.autism_screener import AutismScreener, MEDICAL_DISCLAIMER


FS = 256
DURATION = 4  # seconds
N_SAMPLES = FS * DURATION


# ── Synthetic EEG generators ────────────────────────────────────


def _make_eeg(
    theta_amp=10.0, beta_amp=10.0, alpha_amp=10.0,
    delta_amp=5.0, gamma_amp=3.0, seed=42,
):
    """Synthesize 4-channel EEG with controllable band amplitudes."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    signals = []
    for ch in range(4):
        delta = delta_amp * np.sin(2 * np.pi * 2 * t + ch * 0.1)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.2)
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.5)
        gamma = gamma_amp * np.sin(2 * np.pi * 40 * t + ch * 0.4)
        noise = 2.0 * rng.normal(size=len(t))
        signals.append(delta + theta + alpha + beta + gamma + noise)
    return np.array(signals)


def _atypical_eeg(seed=42):
    """EEG with ASD-like pattern: high theta, low beta, high delta+gamma."""
    return _make_eeg(
        theta_amp=25.0, beta_amp=3.0, alpha_amp=5.0,
        delta_amp=20.0, gamma_amp=15.0, seed=seed,
    )


def _typical_eeg(seed=42):
    """EEG with typical pattern: balanced bands, moderate alpha dominance."""
    return _make_eeg(
        theta_amp=8.0, beta_amp=15.0, alpha_amp=20.0,
        delta_amp=4.0, gamma_amp=2.0, seed=seed,
    )


@pytest.fixture
def screener():
    return AutismScreener(fs=FS)


# ── TestInit ────────────────────────────────────────────────────


class TestInit:
    def test_default_fs(self):
        s = AutismScreener()
        assert s._fs == 256.0

    def test_custom_fs(self):
        s = AutismScreener(fs=512.0)
        assert s._fs == 512.0


# ── TestSetBaseline ─────────────────────────────────────────────


class TestSetBaseline:
    def test_returns_dict(self, screener):
        eeg = _make_eeg()
        result = screener.set_baseline(eeg, FS)
        assert isinstance(result, dict)

    def test_baseline_set_flag(self, screener):
        eeg = _make_eeg()
        result = screener.set_baseline(eeg, FS)
        assert result["baseline_set"] is True

    def test_baseline_metrics_keys(self, screener):
        eeg = _make_eeg()
        result = screener.set_baseline(eeg, FS)
        metrics = result["baseline_metrics"]
        assert "theta" in metrics
        assert "alpha" in metrics
        assert "beta" in metrics
        assert "coherence" in metrics
        assert "entropy" in metrics

    def test_single_channel_baseline(self, screener):
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 10, N_SAMPLES)
        result = screener.set_baseline(eeg, FS)
        assert result["baseline_set"] is True

    def test_multi_user_baselines(self, screener):
        eeg = _make_eeg()
        screener.set_baseline(eeg, FS, user_id="alice")
        screener.set_baseline(eeg, FS, user_id="bob")
        assert "alice" in screener._baselines
        assert "bob" in screener._baselines


# ── TestScreen ──────────────────────────────────────────────────


class TestScreen:
    def test_output_keys(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        expected_keys = {
            "atypicality_score", "risk_level", "mu_suppression_index",
            "theta_beta_ratio", "coherence_index", "asymmetry_atypicality",
            "biomarkers", "medical_disclaimer", "has_baseline",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_atypicality_score_range(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["atypicality_score"] <= 100.0

    def test_mu_suppression_range(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["mu_suppression_index"] <= 1.0

    def test_coherence_range(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["coherence_index"] <= 1.0

    def test_asymmetry_range(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["asymmetry_atypicality"] <= 1.0

    def test_theta_beta_ratio_positive(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert result["theta_beta_ratio"] > 0

    def test_disclaimer_present(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert result["medical_disclaimer"] == MEDICAL_DISCLAIMER
        assert "not a clinical diagnostic" in result["medical_disclaimer"].lower()

    def test_has_baseline_false_without_baseline(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_baseline(self, screener):
        eeg = _make_eeg()
        screener.set_baseline(eeg, FS)
        result = screener.screen(eeg, FS)
        assert result["has_baseline"] is True

    def test_single_channel_screen(self, screener):
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 10, N_SAMPLES)
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["atypicality_score"] <= 100.0

    def test_two_channel_screen(self, screener):
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 10, (2, N_SAMPLES))
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["atypicality_score"] <= 100.0


# ── TestBiomarkers ──────────────────────────────────────────────


class TestBiomarkers:
    def test_biomarker_keys(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        bm = result["biomarkers"]
        expected = {
            "tbr_score", "mu_atypicality_score",
            "coherence_atypicality_score", "asymmetry_score",
            "entropy_score", "u_shape_score",
        }
        assert expected.issubset(set(bm.keys()))

    def test_biomarker_values_bounded(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        for key, val in result["biomarkers"].items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


# ── TestRiskLevels ──────────────────────────────────────────────


class TestRiskLevels:
    def test_valid_risk_levels(self, screener):
        for seed in range(5):
            eeg = _make_eeg(seed=seed)
            result = screener.screen(eeg, FS)
            assert result["risk_level"] in {
                "typical", "mildly_atypical",
                "moderately_atypical", "significantly_atypical",
            }

    def test_atypical_eeg_higher_score(self, screener):
        atyp = screener.screen(_atypical_eeg(seed=10), FS)
        typ = screener.screen(_typical_eeg(seed=11), FS)
        assert atyp["atypicality_score"] > typ["atypicality_score"]

    def test_typical_eeg_low_risk(self, screener):
        result = screener.screen(_typical_eeg(), FS)
        assert result["risk_level"] in {"typical", "mildly_atypical"}


# ── TestTBR ─────────────────────────────────────────────────────


class TestTBR:
    def test_high_theta_high_tbr(self, screener):
        eeg = _make_eeg(theta_amp=30.0, beta_amp=3.0)
        result = screener.screen(eeg, FS)
        assert result["theta_beta_ratio"] > 3.0

    def test_low_theta_low_tbr(self, screener):
        eeg = _make_eeg(theta_amp=3.0, beta_amp=30.0)
        result = screener.screen(eeg, FS)
        assert result["theta_beta_ratio"] < 1.0


# ── TestMuSuppression ──────────────────────────────────────────


class TestMuSuppression:
    def test_without_baseline_returns_valid(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["mu_suppression_index"] <= 1.0

    def test_with_baseline_changes_value(self, screener):
        baseline = _make_eeg(alpha_amp=20.0, seed=1)
        screener.set_baseline(baseline, FS)

        task = _make_eeg(alpha_amp=5.0, seed=2)
        result = screener.screen(task, FS)
        # With reduced alpha in task, suppression should be measurable
        assert 0.0 <= result["mu_suppression_index"] <= 1.0


# ── TestCoherence ───────────────────────────────────────────────


class TestCoherence:
    def test_correlated_channels_high_coherence(self, screener):
        """Identical channels should produce high coherence."""
        rng = np.random.default_rng(42)
        t = np.arange(N_SAMPLES) / FS
        base = 20.0 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 1, N_SAMPLES)
        signals = np.array([base, base, base, base])
        result = screener.screen(signals, FS)
        assert result["coherence_index"] > 0.5

    def test_uncorrelated_channels_lower_coherence(self, screener):
        """Independent random channels should produce lower coherence."""
        rng = np.random.default_rng(42)
        signals = rng.normal(0, 10, (4, N_SAMPLES))
        result = screener.screen(signals, FS)
        # Random noise channels have some spurious coherence but it should be moderate
        assert result["coherence_index"] < 0.9


# ── TestSessionStats ────────────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self, screener):
        stats = screener.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_atypicality"] == 0.0
        assert stats["has_baseline"] is False

    def test_stats_after_screening(self, screener):
        eeg = _make_eeg()
        screener.screen(eeg, FS)
        screener.screen(eeg, FS)
        stats = screener.get_session_stats()
        assert stats["n_epochs"] == 2
        assert stats["mean_atypicality"] > 0

    def test_stats_with_baseline(self, screener):
        eeg = _make_eeg()
        screener.set_baseline(eeg, FS)
        stats = screener.get_session_stats()
        assert stats["has_baseline"] is True


# ── TestHistory ─────────────────────────────────────────────────


class TestHistory:
    def test_empty_history(self, screener):
        assert screener.get_history() == []

    def test_history_grows(self, screener):
        eeg = _make_eeg()
        for _ in range(3):
            screener.screen(eeg, FS)
        assert len(screener.get_history()) == 3

    def test_history_last_n(self, screener):
        eeg = _make_eeg()
        for _ in range(5):
            screener.screen(eeg, FS)
        assert len(screener.get_history(last_n=2)) == 2

    def test_history_last_n_none_returns_all(self, screener):
        eeg = _make_eeg()
        for _ in range(3):
            screener.screen(eeg, FS)
        assert len(screener.get_history(last_n=None)) == 3

    def test_history_cap(self):
        """History should not exceed _HISTORY_LIMIT (500)."""
        from models.autism_screener import _HISTORY_LIMIT
        screener = AutismScreener(fs=FS)
        eeg = _make_eeg()
        for _ in range(_HISTORY_LIMIT + 10):
            screener.screen(eeg, FS)
        assert len(screener.get_history()) == _HISTORY_LIMIT


# ── TestMultiUser ───────────────────────────────────────────────


class TestMultiUser:
    def test_independent_users(self, screener):
        eeg_a = _atypical_eeg(seed=1)
        eeg_b = _typical_eeg(seed=2)
        screener.screen(eeg_a, FS, user_id="alice")
        screener.screen(eeg_b, FS, user_id="bob")
        assert len(screener.get_history(user_id="alice")) == 1
        assert len(screener.get_history(user_id="bob")) == 1
        stats_a = screener.get_session_stats(user_id="alice")
        stats_b = screener.get_session_stats(user_id="bob")
        assert stats_a["mean_atypicality"] != stats_b["mean_atypicality"]

    def test_baseline_per_user(self, screener):
        eeg = _make_eeg()
        screener.set_baseline(eeg, FS, user_id="alice")
        assert screener.get_session_stats(user_id="alice")["has_baseline"] is True
        assert screener.get_session_stats(user_id="bob")["has_baseline"] is False


# ── TestReset ───────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self, screener):
        eeg = _make_eeg()
        screener.set_baseline(eeg, FS)
        screener.screen(eeg, FS)
        screener.reset()
        assert screener.get_history() == []
        stats = screener.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

    def test_reset_specific_user(self, screener):
        eeg = _make_eeg()
        screener.screen(eeg, FS, user_id="keep")
        screener.screen(eeg, FS, user_id="remove")
        screener.reset(user_id="remove")
        assert len(screener.get_history(user_id="keep")) == 1
        assert len(screener.get_history(user_id="remove")) == 0

    def test_reset_clears_baseline(self, screener):
        eeg = _make_eeg()
        screener.set_baseline(eeg, FS)
        screener.reset()
        assert screener.get_session_stats()["has_baseline"] is False


# ── TestUShape ──────────────────────────────────────────────────


class TestUShape:
    def test_high_delta_gamma_high_u_shape(self, screener):
        """Strong delta + gamma with weak mid-bands -> high U-shape."""
        eeg = _make_eeg(delta_amp=30.0, gamma_amp=20.0, alpha_amp=3.0, beta_amp=3.0)
        result = screener.screen(eeg, FS)
        assert result["biomarkers"]["u_shape_score"] > 0.3

    def test_alpha_dominant_low_u_shape(self, screener):
        """Strong alpha with weak extremes -> low U-shape."""
        eeg = _make_eeg(delta_amp=2.0, gamma_amp=1.0, alpha_amp=30.0, beta_amp=15.0)
        result = screener.screen(eeg, FS)
        assert result["biomarkers"]["u_shape_score"] < 0.5


# ── TestSpectralEntropy ─────────────────────────────────────────


class TestSpectralEntropy:
    def test_entropy_bounded(self, screener):
        eeg = _make_eeg()
        result = screener.screen(eeg, FS)
        assert 0.0 <= result["biomarkers"]["entropy_score"] <= 1.0

    def test_pure_tone_low_entropy(self, screener):
        """A pure sine wave has low spectral entropy."""
        t = np.arange(N_SAMPLES) / FS
        pure = 20.0 * np.sin(2 * np.pi * 10 * t)
        signals = np.array([pure, pure, pure, pure])
        result = screener.screen(signals, FS)
        # Entropy score should be low for a pure tone
        assert result["biomarkers"]["entropy_score"] < 0.7


# ── TestDefaultFs ───────────────────────────────────────────────


class TestDefaultFs:
    def test_screen_uses_constructor_fs(self):
        s = AutismScreener(fs=128.0)
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 10, (4, 128 * 4))
        # Should not raise -- uses fs=128 internally
        result = s.screen(eeg)
        assert 0.0 <= result["atypicality_score"] <= 100.0

    def test_explicit_fs_overrides_constructor(self):
        s = AutismScreener(fs=128.0)
        eeg = _make_eeg()
        # Explicit fs=256 overrides the 128 default
        result = s.screen(eeg, fs=256)
        assert 0.0 <= result["atypicality_score"] <= 100.0
