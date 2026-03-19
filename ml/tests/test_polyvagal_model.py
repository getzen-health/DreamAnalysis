"""Tests for polyvagal_model -- polyvagal state tracking (#439).

Covers:
  - State classification for each polyvagal state
  - Probability distribution properties
  - Trajectory computation from sample sequences
  - Transition matrix computation and normalization
  - Dwell time calculation
  - Autonomic flexibility scoring
  - Full profile computation
  - Profile serialization (profile_to_dict)
  - Edge cases (identical samples, minimal input, boundary values)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from models.polyvagal_model import (
    AutonomicSample,
    PolyvagalProfile,
    PolyvagalState,
    STATES,
    classify_polyvagal_state,
    compute_autonomic_flexibility,
    compute_polyvagal_profile,
    compute_state_trajectory,
    profile_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ventral_sample(ts: float = 0.0) -> AutonomicSample:
    """Prototypical ventral vagal sample: high HRV, moderate HR, high alpha."""
    return AutonomicSample(
        hrv_rmssd=55.0,
        heart_rate=68.0,
        alpha_power=0.40,
        beta_alpha_ratio=0.7,
        resp_rate=14.0,
        timestamp=ts or time.time(),
    )


def _sympathetic_sample(ts: float = 0.0) -> AutonomicSample:
    """Prototypical sympathetic sample: low HRV, high HR, low alpha, high beta."""
    return AutonomicSample(
        hrv_rmssd=18.0,
        heart_rate=105.0,
        alpha_power=0.10,
        beta_alpha_ratio=2.5,
        resp_rate=24.0,
        timestamp=ts or time.time(),
    )


def _dorsal_sample(ts: float = 0.0) -> AutonomicSample:
    """Prototypical dorsal vagal sample: very low HRV, low HR, low alpha, low arousal."""
    return AutonomicSample(
        hrv_rmssd=10.0,
        heart_rate=52.0,
        alpha_power=0.10,
        beta_alpha_ratio=0.6,
        resp_rate=8.0,
        timestamp=ts or time.time(),
    )


def _mixed_trajectory(n: int = 20) -> list:
    """Create a realistic mixed trajectory cycling through states."""
    samples = []
    base_ts = 1700000000.0
    # ventral -> sympathetic -> ventral -> dorsal -> ventral pattern
    pattern = [
        _ventral_sample,
        _ventral_sample,
        _ventral_sample,
        _sympathetic_sample,
        _sympathetic_sample,
        _ventral_sample,
        _ventral_sample,
        _dorsal_sample,
        _ventral_sample,
        _ventral_sample,
    ]
    for i in range(n):
        fn = pattern[i % len(pattern)]
        samples.append(fn(ts=base_ts + i * 30.0))
    return samples


# ---------------------------------------------------------------------------
# Tests: classify_polyvagal_state
# ---------------------------------------------------------------------------

class TestClassifyPolyvagalState:
    """Tests for single-sample state classification."""

    def test_ventral_vagal_classification(self):
        """Prototypical ventral vagal sample should classify as ventral_vagal."""
        result = classify_polyvagal_state(_ventral_sample())
        assert result.state == "ventral_vagal"
        assert result.confidence > 0.4

    def test_sympathetic_classification(self):
        """Prototypical sympathetic sample should classify as sympathetic."""
        result = classify_polyvagal_state(_sympathetic_sample())
        assert result.state == "sympathetic"
        assert result.confidence > 0.4

    def test_dorsal_vagal_classification(self):
        """Prototypical dorsal vagal sample should classify as dorsal_vagal."""
        result = classify_polyvagal_state(_dorsal_sample())
        assert result.state == "dorsal_vagal"
        assert result.confidence > 0.3

    def test_probabilities_sum_to_one(self):
        """Probabilities across all states should sum to 1.0."""
        for fn in (_ventral_sample, _sympathetic_sample, _dorsal_sample):
            result = classify_polyvagal_state(fn())
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 0.01, f"Sum={total}, state={result.state}"

    def test_all_states_in_probabilities(self):
        """All three states should appear in the probabilities dict."""
        result = classify_polyvagal_state(_ventral_sample())
        for s in STATES:
            assert s in result.probabilities

    def test_probabilities_non_negative(self):
        """No probability should be negative."""
        for fn in (_ventral_sample, _sympathetic_sample, _dorsal_sample):
            result = classify_polyvagal_state(fn())
            for s, p in result.probabilities.items():
                assert p >= 0.0, f"{s} has negative prob {p}"

    def test_timestamp_propagated(self):
        """Timestamp from the sample should be reflected in the result."""
        ts = 1700000000.0
        sample = _ventral_sample(ts=ts)
        result = classify_polyvagal_state(sample)
        assert result.timestamp == ts

    def test_returns_polyvagal_state_instance(self):
        """Return type should be PolyvagalState dataclass."""
        result = classify_polyvagal_state(_ventral_sample())
        assert isinstance(result, PolyvagalState)


# ---------------------------------------------------------------------------
# Tests: compute_state_trajectory
# ---------------------------------------------------------------------------

class TestComputeStateTrajectory:
    """Tests for trajectory computation from sample sequences."""

    def test_trajectory_length_matches_samples(self):
        """Trajectory should have one state per input sample."""
        samples = [_ventral_sample(ts=i) for i in range(5)]
        traj = compute_state_trajectory(samples)
        assert len(traj) == 5

    def test_homogeneous_trajectory(self):
        """All-ventral samples should produce all-ventral trajectory."""
        samples = [_ventral_sample(ts=i) for i in range(10)]
        traj = compute_state_trajectory(samples)
        assert all(ps.state == "ventral_vagal" for ps in traj)

    def test_mixed_trajectory_has_multiple_states(self):
        """Mixed input should produce multiple distinct states."""
        samples = _mixed_trajectory(10)
        traj = compute_state_trajectory(samples)
        states_seen = {ps.state for ps in traj}
        assert len(states_seen) >= 2

    def test_empty_input_returns_empty(self):
        """Empty sample list should return empty trajectory."""
        traj = compute_state_trajectory([])
        assert traj == []


# ---------------------------------------------------------------------------
# Tests: compute_autonomic_flexibility
# ---------------------------------------------------------------------------

class TestComputeAutonomicFlexibility:
    """Tests for autonomic flexibility scoring."""

    def test_mixed_trajectory_moderate_flexibility(self):
        """A trajectory with state changes and recovery should have moderate-high flexibility."""
        samples = _mixed_trajectory(20)
        traj = compute_state_trajectory(samples)
        flex = compute_autonomic_flexibility(traj)
        assert 0.2 <= flex <= 1.0

    def test_stuck_in_one_state_low_flexibility(self):
        """All-same-state trajectory should have low flexibility."""
        traj = [
            PolyvagalState(state="sympathetic", confidence=0.8, probabilities={}, timestamp=i)
            for i in range(20)
        ]
        flex = compute_autonomic_flexibility(traj)
        assert flex < 0.15

    def test_too_few_samples_returns_zero(self):
        """Fewer than 3 samples should return 0.0 flexibility."""
        traj = [
            PolyvagalState(state="ventral_vagal", confidence=0.8, probabilities={}, timestamp=0)
        ]
        assert compute_autonomic_flexibility(traj) == 0.0
        assert compute_autonomic_flexibility([]) == 0.0

    def test_flexibility_in_valid_range(self):
        """Flexibility should always be between 0 and 1."""
        for _ in range(5):
            samples = _mixed_trajectory(30)
            traj = compute_state_trajectory(samples)
            flex = compute_autonomic_flexibility(traj)
            assert 0.0 <= flex <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_polyvagal_profile
# ---------------------------------------------------------------------------

class TestComputePolyvagalProfile:
    """Tests for full profile computation."""

    def test_profile_returns_dataclass(self):
        """Profile should return a PolyvagalProfile instance."""
        samples = _mixed_trajectory(10)
        profile = compute_polyvagal_profile(samples)
        assert isinstance(profile, PolyvagalProfile)

    def test_profile_state_distribution_sums_to_one(self):
        """State distribution should sum to 1.0."""
        samples = _mixed_trajectory(20)
        profile = compute_polyvagal_profile(samples)
        total = sum(profile.state_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_profile_dominant_state_in_distribution(self):
        """Dominant state should be the state with highest distribution."""
        samples = _mixed_trajectory(20)
        profile = compute_polyvagal_profile(samples)
        assert profile.dominant_state in STATES
        # Dominant should have the highest distribution value
        max_state = max(
            profile.state_distribution,
            key=profile.state_distribution.get,  # type: ignore
        )
        assert profile.dominant_state == max_state

    def test_profile_transition_matrix_rows_sum_to_one(self):
        """Each row in the transition matrix should sum to 1.0 (if state appeared)."""
        samples = _mixed_trajectory(20)
        profile = compute_polyvagal_profile(samples)
        for from_state, row in profile.transition_matrix.items():
            row_sum = sum(row.values())
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.02, (
                    f"Row for {from_state} sums to {row_sum}"
                )

    def test_profile_n_samples_matches(self):
        """n_samples should match input length."""
        samples = _mixed_trajectory(15)
        profile = compute_polyvagal_profile(samples)
        assert profile.n_samples == 15

    def test_profile_trajectory_length(self):
        """Trajectory in profile should match n_samples."""
        samples = _mixed_trajectory(10)
        profile = compute_polyvagal_profile(samples)
        assert len(profile.trajectory) == profile.n_samples

    def test_profile_dwell_times_non_negative(self):
        """Mean dwell times should be non-negative."""
        samples = _mixed_trajectory(20)
        profile = compute_polyvagal_profile(samples)
        for s, dwell in profile.mean_dwell_times.items():
            assert dwell >= 0.0, f"Dwell time for {s} is {dwell}"

    def test_profile_raises_on_too_few_samples(self):
        """Profile should raise ValueError with fewer than 3 samples."""
        with pytest.raises(ValueError, match="at least 3"):
            compute_polyvagal_profile([_ventral_sample()])

        with pytest.raises(ValueError, match="at least 3"):
            compute_polyvagal_profile([])


# ---------------------------------------------------------------------------
# Tests: profile_to_dict
# ---------------------------------------------------------------------------

class TestProfileToDict:
    """Tests for profile serialization."""

    def test_dict_has_expected_keys(self):
        """Serialized profile should contain all expected top-level keys."""
        samples = _mixed_trajectory(10)
        profile = compute_polyvagal_profile(samples)
        d = profile_to_dict(profile)
        expected_keys = {
            "dominant_state",
            "state_distribution",
            "transition_matrix",
            "mean_dwell_times",
            "autonomic_flexibility",
            "n_samples",
            "n_transitions",
            "trajectory",
        }
        assert expected_keys.issubset(d.keys())

    def test_trajectory_items_are_dicts(self):
        """Each trajectory entry should be a plain dict with state/confidence/probabilities."""
        samples = _mixed_trajectory(5)
        profile = compute_polyvagal_profile(samples)
        d = profile_to_dict(profile)
        for entry in d["trajectory"]:
            assert isinstance(entry, dict)
            assert "state" in entry
            assert "confidence" in entry
            assert "probabilities" in entry
            assert "timestamp" in entry

    def test_dict_is_json_serializable(self):
        """The dict should be JSON-serializable (no numpy types, no dataclasses)."""
        import json

        samples = _mixed_trajectory(10)
        profile = compute_polyvagal_profile(samples)
        d = profile_to_dict(profile)
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_all_identical_samples(self):
        """All-identical samples should produce valid profile without error."""
        samples = [_ventral_sample(ts=1700000000.0 + i) for i in range(5)]
        profile = compute_polyvagal_profile(samples)
        assert profile.n_transitions == 0
        assert profile.dominant_state == "ventral_vagal"

    def test_boundary_hrv_values(self):
        """Samples at HRV thresholds should not crash."""
        for hrv in (0.0, 20.0, 40.0, 100.0):
            sample = AutonomicSample(
                hrv_rmssd=hrv,
                heart_rate=70.0,
                alpha_power=0.25,
                beta_alpha_ratio=1.0,
                resp_rate=15.0,
            )
            result = classify_polyvagal_state(sample)
            assert result.state in STATES

    def test_extreme_values(self):
        """Extreme physiological values should not crash."""
        sample = AutonomicSample(
            hrv_rmssd=200.0,
            heart_rate=200.0,
            alpha_power=1.0,
            beta_alpha_ratio=10.0,
            resp_rate=40.0,
        )
        result = classify_polyvagal_state(sample)
        assert result.state in STATES
        assert 0.0 <= result.confidence <= 1.0

    def test_exactly_three_samples(self):
        """Minimum 3 samples should produce a valid profile."""
        samples = [
            _ventral_sample(ts=1.0),
            _sympathetic_sample(ts=2.0),
            _ventral_sample(ts=3.0),
        ]
        profile = compute_polyvagal_profile(samples)
        assert profile.n_samples == 3
        assert profile.n_transitions >= 1
        assert profile.autonomic_flexibility > 0.0
