"""Tests for AttentionScreener — aperiodic + TBR attention screening."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.attention_screener import AttentionScreener, DISCLAIMER


def test_output_keys():
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 10, (4, 1024)).astype(np.float32)
    screener = AttentionScreener()
    result = screener.predict(eeg, fs=256)
    assert "attention_risk_index" in result
    assert "risk_level" in result
    assert "tbr" in result
    assert "aperiodic_exponent" in result
    assert "disclaimer" in result
    assert result["disclaimer"] == DISCLAIMER


def test_risk_range():
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    screener = AttentionScreener()
    result = screener.predict(eeg, fs=256)
    assert 0.0 <= result["attention_risk_index"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "elevated"}


def test_single_channel():
    rng = np.random.default_rng(2)
    eeg = rng.normal(0, 10, 1024).astype(np.float32)
    screener = AttentionScreener()
    result = screener.predict(eeg, fs=256)
    assert 0.0 <= result["attention_risk_index"] <= 1.0


def test_rest_baseline_enables_dynamic():
    rng = np.random.default_rng(3)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    screener = AttentionScreener()
    screener.record_rest_baseline(eeg, fs=256)
    result = screener.predict(eeg, fs=256)
    assert result["dynamic_pattern"] in {"typical", "borderline", "atypical"}


def test_tbr_positive():
    rng = np.random.default_rng(4)
    eeg = rng.normal(0, 10, (4, 1024)).astype(np.float32)
    screener = AttentionScreener()
    result = screener.predict(eeg, fs=256)
    assert result["tbr"] > 0
