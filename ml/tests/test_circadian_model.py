"""Unit tests for the circadian neural signature model (issue #410)."""
from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

# Add ml/ to path so models can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.circadian_model import (
    CosinorFit,
    CircadianProfile,
    _fit_cosinor,
    _classify_chronotype,
    _format_time,
    _predict_windows,
    compute_circadian_profile,
    profile_to_dict,
)


# ─── _fit_cosinor ─────────────────────────────────────────────────────────────

class TestFitCosinor:
    """Test the core cosinor fitting function."""

    def test_perfect_cosine_signal(self):
        """A perfect cosine should yield amplitude ~1, R² ~1, low p-value."""
        np.random.seed(42)
        n = 200
        times = np.linspace(0, 24, n, endpoint=False)
        # y = 0.5 + 1.0 * cos(2*pi*(t - 10)/24)  → peak at t=10h
        values = 0.5 + 1.0 * np.cos(2.0 * math.pi * (times - 10.0) / 24.0)

        fit = _fit_cosinor(times, values, period_h=24.0)

        assert fit.n_samples == n
        assert abs(fit.mesor - 0.5) < 0.01
        assert abs(fit.amplitude - 1.0) < 0.01
        assert abs(fit.acrophase_h - 10.0) < 0.1
        assert fit.r_squared > 0.99
        assert fit.p_value < 0.001

    def test_noisy_signal(self):
        """A noisy cosine should still find approximate parameters."""
        np.random.seed(123)
        n = 100
        times = np.random.uniform(0, 24, n)
        # Signal + noise
        values = 2.0 + 0.8 * np.cos(2.0 * math.pi * (times - 14.0) / 24.0)
        values += np.random.normal(0, 0.3, n)

        fit = _fit_cosinor(times, values, period_h=24.0)

        assert abs(fit.mesor - 2.0) < 0.2
        assert abs(fit.amplitude - 0.8) < 0.3
        assert abs(fit.acrophase_h - 14.0) < 2.0  # within 2 hours
        assert fit.r_squared > 0.3  # noisy but detectable

    def test_flat_signal_no_rhythm(self):
        """A flat signal should yield near-zero amplitude and high p-value."""
        np.random.seed(99)
        n = 50
        times = np.random.uniform(0, 24, n)
        values = np.random.normal(5.0, 0.1, n)  # constant + tiny noise

        fit = _fit_cosinor(times, values, period_h=24.0)

        assert fit.amplitude < 0.3
        assert fit.p_value > 0.05  # not significant

    def test_too_few_samples_returns_defaults(self):
        """With < 6 samples, should return default values without crashing."""
        times = np.array([1.0, 5.0, 10.0])
        values = np.array([1.0, 2.0, 3.0])

        fit = _fit_cosinor(times, values)

        assert fit.n_samples == 3
        assert fit.amplitude == 0.0
        assert fit.p_value == 1.0
        assert fit.r_squared == 0.0

    def test_empty_input(self):
        """Empty arrays should not crash."""
        fit = _fit_cosinor(np.array([]), np.array([]))
        assert fit.n_samples == 0
        assert fit.amplitude == 0.0


# ─── _classify_chronotype ─────────────────────────────────────────────────────

class TestClassifyChronotype:

    def test_early_bird(self):
        label, conf = _classify_chronotype(8.0)
        assert label == "early_bird"
        assert conf > 0.5

    def test_night_owl(self):
        label, conf = _classify_chronotype(15.0)
        assert label == "night_owl"
        assert conf > 0.5

    def test_intermediate(self):
        label, conf = _classify_chronotype(11.5)
        assert label == "intermediate"
        assert conf > 0.5

    def test_boundary_early(self):
        label, _ = _classify_chronotype(10.0)
        # Exactly at boundary — could be either, but should not crash
        assert label in ("early_bird", "intermediate")

    def test_boundary_late(self):
        label, _ = _classify_chronotype(13.0)
        assert label in ("intermediate", "night_owl")


# ─── _format_time ──────────────────────────────────────────────────────────────

class TestFormatTime:

    def test_morning(self):
        assert _format_time(9.5) == "9:30am"

    def test_afternoon(self):
        assert _format_time(14.0) == "2:00pm"

    def test_midnight(self):
        assert _format_time(0.0) == "12:00am"

    def test_noon(self):
        assert _format_time(12.0) == "12:00pm"

    def test_late_night(self):
        assert _format_time(23.75) == "11:45pm"


# ─── _predict_windows ─────────────────────────────────────────────────────────

class TestPredictWindows:

    def test_typical_morning_peak(self):
        focus, slump = _predict_windows(10.0, 0.5)
        assert "8:30am" in focus
        assert "11:30am" in focus
        # Slump ~6h later
        assert "pm" in slump

    def test_afternoon_peak(self):
        focus, slump = _predict_windows(14.0, 0.5)
        assert "12:30pm" in focus
        assert "3:30pm" in focus


# ─── compute_circadian_profile ─────────────────────────────────────────────────

class TestComputeCircadianProfile:

    def _make_cosine_stream(self, acrophase: float, n: int = 100, noise: float = 0.1):
        """Generate a cosine feature stream with given acrophase."""
        np.random.seed(42)
        times = np.random.uniform(0, 24, n)
        values = np.cos(2.0 * math.pi * (times - acrophase) / 24.0)
        values += np.random.normal(0, noise, n)
        return [{"time_h": float(t), "value": float(v)} for t, v in zip(times, values)]

    def test_consistent_streams_produce_clear_profile(self):
        """When all streams agree on acrophase, profile should be clear."""
        acrophase = 10.0
        streams = {
            "alpha_beta_ratio": self._make_cosine_stream(acrophase, n=150),
            "valence": self._make_cosine_stream(acrophase, n=120),
            "arousal": self._make_cosine_stream(acrophase, n=100),
        }

        profile = compute_circadian_profile(streams, current_hour=10.0)

        assert abs(profile.acrophase_h - acrophase) < 2.0
        assert profile.chronotype == "early_bird"
        assert profile.phase_stability > 0.5
        assert profile.amplitude > 0.0
        assert len(profile.fits) >= 2

    def test_night_owl_profile(self):
        """Late acrophase should classify as night owl."""
        streams = {
            "alpha_beta_ratio": self._make_cosine_stream(15.0, n=150),
            "valence": self._make_cosine_stream(15.0, n=120),
        }

        profile = compute_circadian_profile(streams, current_hour=12.0)

        assert profile.chronotype == "night_owl"

    def test_empty_streams_fallback(self):
        """Empty input should produce a fallback profile."""
        profile = compute_circadian_profile({}, current_hour=12.0)

        assert profile.chronotype == "intermediate"
        assert profile.acrophase_h == 10.5  # population average
        assert profile.phase_stability == 0.0
        assert profile.amplitude == 0.0

    def test_phase_shift_detection(self):
        """When baseline_acrophase differs from current, phase_shift should be non-zero."""
        streams = {
            "valence": self._make_cosine_stream(12.0, n=150),
        }

        profile = compute_circadian_profile(
            streams,
            current_hour=12.0,
            baseline_acrophase=10.0,
        )

        # Phase shift should be approximately +2 hours
        assert abs(profile.phase_shift_hours - 2.0) < 1.5

    def test_current_phase_at_peak(self):
        """At acrophase, current_phase should be near 1.0 (peak)."""
        streams = {
            "alpha_beta_ratio": self._make_cosine_stream(10.0, n=200, noise=0.05),
        }

        profile = compute_circadian_profile(streams, current_hour=10.0)

        # current_phase near 1.0 when at peak
        assert profile.current_phase > 0.7

    def test_current_phase_at_trough(self):
        """At trough (acrophase + 12h), current_phase should be near 0."""
        streams = {
            "alpha_beta_ratio": self._make_cosine_stream(10.0, n=200, noise=0.05),
        }

        profile = compute_circadian_profile(streams, current_hour=22.0)

        assert profile.current_phase < 0.3

    def test_predicted_windows_present(self):
        """Profile should have non-empty focus and slump windows."""
        streams = {
            "valence": self._make_cosine_stream(11.0, n=100),
        }

        profile = compute_circadian_profile(streams, current_hour=9.0)

        assert len(profile.predicted_focus_window) > 0
        assert len(profile.predicted_slump_window) > 0
        assert "–" in profile.predicted_focus_window


# ─── profile_to_dict ───────────────────────────────────────────────────────────

class TestProfileToDict:

    def test_serialization_roundtrip(self):
        """profile_to_dict should produce a JSON-safe dict."""
        fit = CosinorFit(
            mesor=0.5, amplitude=0.8, acrophase_h=10.0,
            period_h=24.0, r_squared=0.95, p_value=0.001, n_samples=100,
        )
        profile = CircadianProfile(
            chronotype="early_bird",
            chronotype_confidence=0.8,
            acrophase_h=10.0,
            amplitude=0.8,
            period_h=24.0,
            phase_stability=0.9,
            current_phase=0.75,
            predicted_focus_window="8:30am – 11:30am",
            predicted_slump_window="4:00pm – 5:30pm",
            phase_shift_hours=0.5,
            fits={"alpha_beta_ratio": fit},
            data_days=14,
        )

        d = profile_to_dict(profile)

        assert d["chronotype"] == "early_bird"
        assert d["acrophase_h"] == 10.0
        assert "alpha_beta_ratio" in d["fits"]
        assert d["fits"]["alpha_beta_ratio"]["n_samples"] == 100
        assert d["data_days"] == 14

    def test_empty_fits(self):
        """Should handle empty fits dict."""
        profile = CircadianProfile(
            chronotype="intermediate",
            chronotype_confidence=0.0,
            acrophase_h=10.5,
            amplitude=0.0,
            period_h=24.0,
            phase_stability=0.0,
            current_phase=0.5,
            predicted_focus_window="9:00am – 12:00pm",
            predicted_slump_window="3:00pm – 4:30pm",
            phase_shift_hours=0.0,
            fits={},
            data_days=0,
        )

        d = profile_to_dict(profile)
        assert d["fits"] == {}
        assert d["amplitude"] == 0.0
