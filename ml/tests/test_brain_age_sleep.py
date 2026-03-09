"""Tests for BrainAgeEstimator and MemoryConsolidationTracker."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.brain_age_estimator import BrainAgeEstimator, DISCLAIMER as BRAIN_AGE_DISCLAIMER
from models.memory_consolidation_tracker import MemoryConsolidationTracker


# ── Brain Age ────────────────────────────────────────────────────────────────

def test_brain_age_output_keys():
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 10, (4, 1024)).astype(np.float32)
    estimator = BrainAgeEstimator()
    result = estimator.predict(eeg, fs=256)
    assert "predicted_age" in result
    assert "brain_age_gap" in result  # None when no chronological_age
    assert "aperiodic_exponent" in result
    assert "disclaimer" in result
    assert result["disclaimer"] == BRAIN_AGE_DISCLAIMER


def test_brain_age_range():
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    estimator = BrainAgeEstimator()
    result = estimator.predict(eeg, fs=256)
    assert 15 <= result["predicted_age"] <= 90


def test_brain_age_gap_computed():
    rng = np.random.default_rng(2)
    eeg = rng.normal(0, 10, (4, 2048)).astype(np.float32)
    estimator = BrainAgeEstimator()
    result = estimator.predict(eeg, fs=256, chronological_age=30.0)
    assert result["brain_age_gap"] is not None
    assert result["gap_interpretation"] is not None
    assert result["percentile"] is not None


def test_brain_age_single_channel():
    rng = np.random.default_rng(3)
    eeg = rng.normal(0, 10, 1024).astype(np.float32)
    estimator = BrainAgeEstimator()
    result = estimator.predict(eeg, fs=256)
    assert 15 <= result["predicted_age"] <= 90


# ── Memory Consolidation ─────────────────────────────────────────────────────

def test_epoch_score_keys():
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 20, (4, 7680)).astype(np.float32)  # 30 sec at 256 Hz
    tracker = MemoryConsolidationTracker()
    result = tracker.score_epoch(eeg, fs=256, sleep_stage="N2")
    assert "consolidation_quality" in result
    assert "spindle_density_per_min" in result
    assert "coupling_strength" in result
    assert "so_rate_per_min" in result
    assert 0.0 <= result["consolidation_quality"] <= 1.0


def test_epoch_score_range():
    rng = np.random.default_rng(2)
    eeg = rng.normal(0, 15, (4, 7680)).astype(np.float32)
    tracker = MemoryConsolidationTracker()
    result = tracker.score_epoch(eeg, fs=256, sleep_stage="N3")
    assert 0.0 <= result["consolidation_quality"] <= 1.0
    assert result["spindle_density_per_min"] >= 0
    assert 0.0 <= result["coupling_strength"] <= 1.0


def test_session_summary():
    rng = np.random.default_rng(5)
    tracker = MemoryConsolidationTracker()
    for _ in range(3):
        eeg = rng.normal(0, 15, (4, 7680)).astype(np.float32)
        tracker.score_epoch(eeg, fs=256, sleep_stage="N2")
    summary = tracker.score_session()
    assert "session_quality" in summary
    assert "quality_label" in summary
    assert summary["n_epochs"] == 3
    assert 0.0 <= summary["session_quality"] <= 1.0


def test_session_empty():
    tracker = MemoryConsolidationTracker()
    summary = tracker.score_session()
    assert summary["n_epochs"] == 0
    assert summary["session_quality"] == 0.0


def test_tmr_trigger():
    rng = np.random.default_rng(6)
    eeg = rng.normal(0, 20, (4, 512)).astype(np.float32)
    tracker = MemoryConsolidationTracker()
    result = tracker.get_tmr_trigger(eeg, fs=256, sleep_stage="N2")
    assert "trigger_cue" in result
    assert isinstance(result["trigger_cue"], bool)
