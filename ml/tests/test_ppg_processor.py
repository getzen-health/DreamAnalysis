"""Tests for PPG/HRV feature extraction."""
import numpy as np
import pytest
from processing.ppg_processor import extract_hrv_features, _empty_hrv

FS = 64


def _synthetic_ppg(n_seconds: int = 30, hr_bpm: float = 70.0) -> np.ndarray:
    """Generate synthetic PPG with periodic peaks at given HR."""
    n = int(n_seconds * FS)
    t = np.linspace(0, n_seconds, n)
    period = 60.0 / hr_bpm
    # Sawtooth-ish PPG shape
    ppg = np.sin(2 * np.pi * t / period) + 0.3 * np.sin(4 * np.pi * t / period)
    noise = np.random.default_rng(42).normal(0, 0.05, n)
    return (ppg + noise).astype(np.float32)


def test_basic_hrv():
    ppg = _synthetic_ppg(30, hr_bpm=70)
    hrv = extract_hrv_features(ppg, fs=FS)
    assert "mean_hr" in hrv
    assert "rmssd" in hrv
    assert "stress_index" in hrv
    assert 0.0 <= hrv["stress_index"] <= 1.0
    assert hrv["n_rr_intervals"] >= 2


def test_too_short_returns_empty():
    short = np.random.randn(50).astype(np.float32)
    result = extract_hrv_features(short, fs=FS)
    assert result["n_rr_intervals"] == 0
    assert result["stress_index"] == 0.0


def test_stress_index_range():
    for hr in [60, 80, 100]:
        ppg = _synthetic_ppg(30, hr_bpm=float(hr))
        hrv = extract_hrv_features(ppg, fs=FS)
        assert 0.0 <= hrv["stress_index"] <= 1.0


def test_empty_hrv_structure():
    empty = _empty_hrv()
    required_keys = ["mean_hr", "sdnn", "rmssd", "pnn50", "lf_power", "hf_power", "lf_hf_ratio", "stress_index"]
    for k in required_keys:
        assert k in empty
