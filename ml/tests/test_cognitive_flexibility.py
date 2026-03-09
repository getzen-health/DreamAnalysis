"""Tests for CognitiveFlexibilityDetector."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.cognitive_flexibility_detector import CognitiveFlexibilityDetector


def test_output_keys():
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 10, (4, 1024)).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    result = det.predict(eeg, fs=256)
    assert "flexibility_index" in result
    assert "level" in result
    assert "metacontrol_bias" in result
    assert "aperiodic_exponent" in result
    assert "fmt_power" in result


def test_flexibility_range():
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    result = det.predict(eeg, fs=256)
    assert 0.0 <= result["flexibility_index"] <= 1.0
    assert result["level"] in {"rigid", "moderate", "flexible"}


def test_single_channel():
    rng = np.random.default_rng(2)
    eeg = rng.normal(0, 10, 1024).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    result = det.predict(eeg, fs=256)
    assert 0.0 <= result["flexibility_index"] <= 1.0


def test_baseline_recording():
    rng = np.random.default_rng(3)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    baseline = det.record_baseline(eeg, fs=256)
    assert "rest_exponent" in baseline
    assert baseline["rest_exponent"] > 0


def test_dynamic_measurement():
    rng = np.random.default_rng(4)
    rest_eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    task_eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    result = det.measure_dynamic_flexibility(rest_eeg, task_eeg, fs=256)
    assert "exponent_change" in result
    assert "flexibility_response" in result
    assert result["flexibility_response"] in {
        "high_flexibility",
        "moderate_flexibility",
        "balanced",
        "low_flexibility",
    }


def test_dynamic_updates_baseline():
    rng = np.random.default_rng(5)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    det.record_baseline(eeg, fs=256)
    result = det.predict(eeg, fs=256)
    # After baseline, dynamic_score should be set
    assert result["dynamic_score"] is not None


def test_metacontrol_bias_values():
    rng = np.random.default_rng(6)
    eeg = rng.normal(0, 10, (4, 1024)).astype(np.float32)
    det = CognitiveFlexibilityDetector()
    result = det.predict(eeg, fs=256)
    assert result["metacontrol_bias"] in {
        "flexible",
        "moderately_flexible",
        "balanced",
        "persistent",
        "unknown",
    }
