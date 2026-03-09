"""Tests for GRUSleepStager — GRU-based sleep staging with permutation channel selection."""

import numpy as np
import pytest

from models.gru_sleep_stager import (
    GRUSleepStager,
    SLEEP_STAGES,
    get_gru_sleep_stager,
    _extract_bands,
    _bandpower,
    _rank_channels_by_delta_variance,
    _classify_stage,
    _scores_to_probs,
)

FS = 256.0
# 2-second window at 256 Hz
N_SAMPLES = 512


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_sine(freq_hz: float, duration_s: float = 2.0, amplitude: float = 10.0) -> np.ndarray:
    """Generate a pure sine wave at the given frequency."""
    t = np.linspace(0, duration_s, int(duration_s * FS), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _make_multichannel(n_channels: int = 4, freq_hz: float = 2.0) -> np.ndarray:
    """Generate multi-channel EEG (n_channels, N_SAMPLES) with the given freq."""
    rng = np.random.default_rng(42)
    signals = np.stack([
        _make_sine(freq_hz) + rng.normal(0, 0.5, N_SAMPLES)
        for _ in range(n_channels)
    ])
    return signals


# ── Init / status ─────────────────────────────────────────────────────────────

def test_init_status_has_required_keys():
    stager = GRUSleepStager()
    status = stager.get_status()
    for key in ("model", "buffer_size", "buffer_capacity", "prediction_count",
                "ema_initialised", "stages"):
        assert key in status, f"Missing key in status: {key}"


def test_init_buffer_empty():
    stager = GRUSleepStager()
    assert stager.get_status()["buffer_size"] == 0


def test_init_prediction_count_zero():
    stager = GRUSleepStager()
    assert stager.get_status()["prediction_count"] == 0


def test_init_ema_not_initialised():
    stager = GRUSleepStager()
    assert stager.get_status()["ema_initialised"] is False


# ── predict: required output keys ─────────────────────────────────────────────

def test_predict_returns_required_keys():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    for key in ("stage", "probabilities", "confidence", "channel_ranking"):
        assert key in result, f"Missing key: {key}"


def test_predict_stage_is_valid_label():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] in SLEEP_STAGES


def test_predict_probabilities_sum_to_one():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 2e-4, f"Probabilities sum to {total}, expected ~1.0"


def test_predict_probabilities_contain_all_stages():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    for stage in SLEEP_STAGES:
        assert stage in result["probabilities"]


def test_predict_confidence_in_zero_one():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_channel_ranking_is_list():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    assert isinstance(result["channel_ranking"], list)


def test_predict_channel_ranking_indices_valid():
    """Channel ranking indices must be valid channel indices."""
    stager = GRUSleepStager()
    n_ch = 4
    eeg = _make_multichannel(n_ch, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    for idx in result["channel_ranking"]:
        assert 0 <= idx < n_ch


# ── predict: input shape variants ────────────────────────────────────────────

def test_predict_1d_input_works():
    stager = GRUSleepStager()
    eeg_1d = _make_sine(2.0)
    result = stager.predict(eeg_1d, fs=FS)
    assert result["stage"] in SLEEP_STAGES
    assert result["channel_ranking"] == [0]


def test_predict_multichannel_4ch_works():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4, freq_hz=2.0)
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] in SLEEP_STAGES
    assert len(result["channel_ranking"]) <= 2


def test_predict_single_channel_2d_works():
    stager = GRUSleepStager()
    eeg = _make_multichannel(1, freq_hz=5.0)  # shape (1, N)
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] in SLEEP_STAGES


def test_predict_short_signal_handled_gracefully():
    """Signal shorter than 4 samples should not raise."""
    stager = GRUSleepStager()
    eeg = np.array([[1.0, 2.0, 1.5]])  # 3 samples — below threshold
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] in SLEEP_STAGES
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-6


# ── Buffer behaviour ──────────────────────────────────────────────────────────

def test_buffer_accumulates_after_predictions():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4)
    for i in range(5):
        stager.predict(eeg, fs=FS)
    assert stager.get_status()["buffer_size"] == 5


def test_buffer_capped_at_capacity():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4)
    capacity = stager.get_status()["buffer_capacity"]
    for _ in range(capacity + 5):
        stager.predict(eeg, fs=FS)
    assert stager.get_status()["buffer_size"] == capacity


def test_prediction_count_increments():
    stager = GRUSleepStager()
    eeg = _make_multichannel(2)
    for i in range(3):
        stager.predict(eeg, fs=FS)
    assert stager.get_status()["prediction_count"] == 3


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_clears_buffer():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4)
    stager.predict(eeg, fs=FS)
    stager.reset()
    assert stager.get_status()["buffer_size"] == 0


def test_reset_clears_prediction_count():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4)
    stager.predict(eeg, fs=FS)
    stager.reset()
    assert stager.get_status()["prediction_count"] == 0


def test_reset_clears_ema_state():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4)
    stager.predict(eeg, fs=FS)
    assert stager.get_status()["ema_initialised"] is True
    stager.reset()
    assert stager.get_status()["ema_initialised"] is False


def test_predict_works_after_reset():
    stager = GRUSleepStager()
    eeg = _make_multichannel(4)
    stager.predict(eeg, fs=FS)
    stager.reset()
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] in SLEEP_STAGES


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance():
    a = get_gru_sleep_stager()
    b = get_gru_sleep_stager()
    assert a is b


# ── Stage-logic / band-power helpers ─────────────────────────────────────────

def test_delta_dominated_signal_classifies_as_n3():
    """High-amplitude 2 Hz signal → delta dominates → N3."""
    stager = GRUSleepStager()
    # Strong delta (2 Hz), weak everything else
    delta_sig = 50.0 * np.sin(2 * np.pi * 2.0 * np.linspace(0, 4, 4 * int(FS)))
    eeg = np.stack([delta_sig] * 2)
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] == "N3", (
        f"Expected N3 for delta-dominant signal, got {result['stage']}"
    )


def test_high_beta_signal_classifies_as_wake():
    """High-amplitude 20 Hz signal → beta dominates → Wake."""
    stager = GRUSleepStager()
    beta_sig = 50.0 * np.sin(2 * np.pi * 20.0 * np.linspace(0, 4, 4 * int(FS)))
    eeg = np.stack([beta_sig] * 2)
    result = stager.predict(eeg, fs=FS)
    assert result["stage"] == "Wake", (
        f"Expected Wake for beta-dominant signal, got {result['stage']}"
    )


def test_bandpower_returns_positive():
    sig = _make_sine(2.0)
    bp = _bandpower(sig, FS, 0.5, 4.0)
    assert bp > 0.0


def test_extract_bands_returns_all_keys():
    sig = _make_sine(2.0)
    bands = _extract_bands(sig, FS)
    for k in ("delta", "theta", "alpha", "beta"):
        assert k in bands
        assert bands[k] > 0


def test_scores_to_probs_sum_to_one():
    scores = {"Wake": 1.2, "N1": 0.5, "N2": 0.3, "N3": 2.1, "REM": 0.8}
    probs = _scores_to_probs(scores)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_channel_ranking_returns_top_two():
    eeg = _make_multichannel(4, freq_hz=2.0)
    ranking = _rank_channels_by_delta_variance(eeg, FS, n_top=2)
    assert len(ranking) == 2
    assert len(set(ranking)) == 2  # no duplicates
