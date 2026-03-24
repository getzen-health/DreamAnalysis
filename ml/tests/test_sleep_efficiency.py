"""Tests for sleep_efficiency in compute_sleep_stats().

Verifies:
  1. sleep_efficiency is present in compute_sleep_stats output
  2. All-sleep epochs yield 100% efficiency (1.0)
  3. Half-wake epochs yield ~50% efficiency
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sleep_staging import compute_sleep_stats


def _make_epochs(stages: list[str]) -> list[dict]:
    """Build epoch dicts from a list of stage names."""
    stage_to_idx = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    return [
        {"stage": s, "stage_index": stage_to_idx[s], "confidence": 0.8}
        for s in stages
    ]


class TestSleepEfficiency:
    """Tests for sleep_efficiency in compute_sleep_stats."""

    def test_sleep_efficiency_present_in_output(self):
        """compute_sleep_stats output must contain 'sleep_efficiency' key."""
        epochs = _make_epochs(["Wake", "N1", "N2", "N3", "REM"] * 4)
        stats = compute_sleep_stats(epochs)
        assert stats is not None
        assert "sleep_efficiency" in stats

    def test_all_sleep_is_100_percent(self):
        """If every epoch is a sleep stage (no Wake), efficiency should be 1.0."""
        # 20 epochs all N2 — no Wake at all
        epochs = _make_epochs(["N2"] * 20)
        stats = compute_sleep_stats(epochs)
        assert stats is not None
        assert stats["sleep_efficiency"] == pytest.approx(1.0, abs=1e-4)

    def test_half_wake_is_50_percent(self):
        """If half the epochs are Wake and half are sleep, efficiency ~0.5."""
        # 10 Wake + 10 N2 (alternating would complicate WASO; contiguous is fine)
        stages = ["Wake"] * 10 + ["N2"] * 10
        epochs = _make_epochs(stages)
        stats = compute_sleep_stats(epochs)
        assert stats is not None
        assert stats["sleep_efficiency"] == pytest.approx(0.5, abs=1e-4)

    def test_all_wake_is_zero(self):
        """All-Wake epochs should return sleep_efficiency = 0."""
        epochs = _make_epochs(["Wake"] * 20)
        stats = compute_sleep_stats(epochs)
        assert stats is not None
        assert stats["sleep_efficiency"] == pytest.approx(0.0, abs=1e-4)

    def test_empty_epochs_returns_none(self):
        """Empty epoch list should return None (no data)."""
        stats = compute_sleep_stats([])
        assert stats is None

    def test_sleep_efficiency_is_ratio_not_percentage(self):
        """sleep_efficiency should be 0-1 ratio, not 0-100 percentage."""
        epochs = _make_epochs(["Wake"] * 5 + ["N3"] * 15)
        stats = compute_sleep_stats(epochs)
        assert stats is not None
        assert 0.0 <= stats["sleep_efficiency"] <= 1.0
        assert stats["sleep_efficiency"] == pytest.approx(0.75, abs=1e-4)
