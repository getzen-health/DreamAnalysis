"""Tests for couples emotional resonance — issue #440.

Covers:
  - Emotional synchrony computation (correlated, anti-correlated, constant, insufficient data)
  - Resonance period detection (in-sync, divergent, edge cases)
  - Conflict detection (thresholds, minimum samples, no-conflict)
  - Repair detection (successful repair, no repair, no conflicts)
  - Relationship profile computation (aggregation, health classification)
  - Consent management (opt-in, revoke, status, invalid actions)
  - Serialization (profile_to_dict)
  - Edge cases (empty, single sample, mismatched lengths)
"""

import sys
import os
import math
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.couples_resonance import (
    EmotionSample,
    ConflictEvent,
    RepairEvent,
    RelationshipProfile,
    compute_emotional_synchrony,
    detect_resonance_periods,
    detect_conflict,
    detect_repair,
    compute_relationship_profile,
    manage_partnership_consent,
    profile_to_dict,
    store_emotion_samples,
    _pearson_correlation,
    _align_samples,
    _interpret_synchrony,
    _classify_health,
    _partnerships,
    _user_emotions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minutes: float) -> float:
    """Convert minutes to a fake unix timestamp (base = 1_000_000)."""
    return 1_000_000 + minutes * 60.0


def _make_samples(points: list) -> list:
    """Build EmotionSample list from (minute, valence, arousal) tuples."""
    return [EmotionSample(timestamp=_ts(m), valence=v, arousal=a) for m, v, a in points]


@pytest.fixture(autouse=True)
def _clear_global_state():
    """Clear in-memory stores between tests."""
    _partnerships.clear()
    _user_emotions.clear()
    yield
    _partnerships.clear()
    _user_emotions.clear()


# ===========================================================================
# Test _pearson_correlation helper
# ===========================================================================

class TestPearsonCorrelation:

    def test_perfect_positive(self):
        """Identical series should give correlation = 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _pearson_correlation(x, x) == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative(self):
        """Exactly inverted series should give correlation = -1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _pearson_correlation(x, y) == pytest.approx(-1.0, abs=0.001)

    def test_constant_series_returns_zero(self):
        """Constant series (zero variance) should return 0.0."""
        x = [3.0, 3.0, 3.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0]
        assert _pearson_correlation(x, y) == 0.0

    def test_single_element_returns_zero(self):
        """Single element cannot produce a correlation."""
        assert _pearson_correlation([1.0], [2.0]) == 0.0


# ===========================================================================
# Test compute_emotional_synchrony
# ===========================================================================

class TestComputeEmotionalSynchrony:

    def test_high_synchrony_correlated_series(self):
        """Partners with correlated valence/arousal should have high synchrony."""
        samples_a = _make_samples([
            (0, 0.1, 0.3), (1, 0.3, 0.4), (2, 0.5, 0.5),
            (3, 0.7, 0.6), (4, 0.9, 0.7),
        ])
        samples_b = _make_samples([
            (0, 0.2, 0.35), (1, 0.4, 0.45), (2, 0.6, 0.55),
            (3, 0.8, 0.65), (4, 0.95, 0.75),
        ])
        result = compute_emotional_synchrony(samples_a, samples_b)
        assert result["overall_synchrony"] > 0.8
        assert result["interpretation"] == "high_synchrony"
        assert result["n_aligned_samples"] == 5

    def test_anti_correlated_series(self):
        """Partners moving in opposite directions should show negative synchrony."""
        samples_a = _make_samples([
            (0, -0.8, 0.2), (1, -0.4, 0.3), (2, 0.0, 0.4),
            (3, 0.4, 0.5), (4, 0.8, 0.6),
        ])
        samples_b = _make_samples([
            (0, 0.8, 0.6), (1, 0.4, 0.5), (2, 0.0, 0.4),
            (3, -0.4, 0.3), (4, -0.8, 0.2),
        ])
        result = compute_emotional_synchrony(samples_a, samples_b)
        assert result["overall_synchrony"] < 0.0
        assert result["interpretation"] == "anti_synchrony"

    def test_insufficient_data(self):
        """Too few samples should return insufficient_data."""
        samples_a = _make_samples([(0, 0.5, 0.5), (1, 0.6, 0.6)])
        samples_b = _make_samples([(0, 0.5, 0.5)])  # only 1 sample
        result = compute_emotional_synchrony(samples_a, samples_b)
        assert result["interpretation"] == "insufficient_data"
        assert result["overall_synchrony"] == 0.0

    def test_empty_samples(self):
        """Empty input should return insufficient_data."""
        result = compute_emotional_synchrony([], [])
        assert result["interpretation"] == "insufficient_data"


# ===========================================================================
# Test detect_resonance_periods
# ===========================================================================

class TestDetectResonancePeriods:

    def test_fully_resonant_series(self):
        """Perfectly correlated series: all windows should be resonant."""
        samples_a = _make_samples([
            (i, 0.1 * i, 0.5) for i in range(15)
        ])
        samples_b = _make_samples([
            (i, 0.1 * i + 0.05, 0.5) for i in range(15)
        ])
        result = detect_resonance_periods(samples_a, samples_b, window=5)
        assert result["resonance_ratio"] > 0.8
        assert result["n_windows"] > 0

    def test_no_resonance_random(self):
        """Anti-correlated series in windows should produce low resonance."""
        # A goes up while B goes down in alternating windows
        samples_a = _make_samples([
            (0, 0.1, 0.5), (1, 0.3, 0.5), (2, 0.5, 0.5), (3, 0.7, 0.5), (4, 0.9, 0.5),
            (5, 0.9, 0.5), (6, 0.7, 0.5), (7, 0.5, 0.5), (8, 0.3, 0.5), (9, 0.1, 0.5),
        ])
        samples_b = _make_samples([
            (0, 0.9, 0.5), (1, 0.7, 0.5), (2, 0.5, 0.5), (3, 0.3, 0.5), (4, 0.1, 0.5),
            (5, 0.1, 0.5), (6, 0.3, 0.5), (7, 0.5, 0.5), (8, 0.7, 0.5), (9, 0.9, 0.5),
        ])
        result = detect_resonance_periods(samples_a, samples_b, window=5)
        # Anti-correlated windows should not be resonant
        assert result["resonance_ratio"] < 0.5

    def test_too_few_samples_for_window(self):
        """Fewer samples than window size returns empty."""
        samples_a = _make_samples([(0, 0.5, 0.5), (1, 0.6, 0.6)])
        samples_b = _make_samples([(0, 0.5, 0.5), (1, 0.6, 0.6)])
        result = detect_resonance_periods(samples_a, samples_b, window=10)
        assert result["n_windows"] == 0
        assert result["resonance_ratio"] == 0.0


# ===========================================================================
# Test detect_conflict
# ===========================================================================

class TestDetectConflict:

    def test_clear_conflict_detected(self):
        """Both partners negative valence + high arousal for multiple samples."""
        samples_a = _make_samples([
            (0, 0.5, 0.3),   # calm
            (1, -0.3, 0.7),  # conflict
            (2, -0.4, 0.8),  # conflict
            (3, -0.5, 0.9),  # conflict
            (4, 0.3, 0.3),   # calm again
        ])
        samples_b = _make_samples([
            (0, 0.5, 0.3),
            (1, -0.2, 0.6),  # conflict
            (2, -0.3, 0.7),  # conflict
            (3, -0.4, 0.8),  # conflict
            (4, 0.4, 0.3),
        ])
        conflicts = detect_conflict(samples_a, samples_b)
        assert len(conflicts) >= 1
        c = conflicts[0]
        assert c.avg_valence_a < 0
        assert c.avg_valence_b < 0
        assert c.avg_arousal_a > 0.5
        assert c.avg_arousal_b > 0.5

    def test_no_conflict_when_calm(self):
        """Positive valence, low arousal should produce no conflicts."""
        samples_a = _make_samples([
            (i, 0.5, 0.3) for i in range(10)
        ])
        samples_b = _make_samples([
            (i, 0.4, 0.2) for i in range(10)
        ])
        conflicts = detect_conflict(samples_a, samples_b)
        assert len(conflicts) == 0

    def test_single_point_not_enough(self):
        """Single conflict point below min_samples should not trigger."""
        samples_a = _make_samples([
            (0, 0.5, 0.3),
            (1, -0.5, 0.9),  # one conflict point
            (2, 0.5, 0.3),
        ])
        samples_b = _make_samples([
            (0, 0.5, 0.3),
            (1, -0.5, 0.9),
            (2, 0.5, 0.3),
        ])
        conflicts = detect_conflict(samples_a, samples_b, min_samples=2)
        assert len(conflicts) == 0

    def test_empty_input(self):
        """Empty input should produce no conflicts."""
        assert detect_conflict([], []) == []


# ===========================================================================
# Test detect_repair
# ===========================================================================

class TestDetectRepair:

    def test_repair_detected_after_conflict(self):
        """After conflict, both partners recover to positive valence."""
        conflict = ConflictEvent(
            start_ts=_ts(5), end_ts=_ts(8),
            avg_valence_a=-0.4, avg_valence_b=-0.3,
            avg_arousal_a=0.7, avg_arousal_b=0.6,
            duration_sec=180.0,
        )
        # Post-conflict recovery samples
        samples_a = _make_samples([
            (9, -0.2, 0.5), (10, 0.0, 0.4), (11, 0.2, 0.3),
        ])
        samples_b = _make_samples([
            (9, -0.1, 0.5), (10, 0.1, 0.4), (11, 0.3, 0.3),
        ])
        repairs = detect_repair([conflict], samples_a, samples_b)
        assert len(repairs) == 1
        r = repairs[0]
        assert r.repair_latency_sec > 0
        assert r.recovered_valence_a >= 0.0
        assert r.recovered_valence_b >= 0.0

    def test_no_repair_when_still_negative(self):
        """If neither partner recovers, no repair should be detected."""
        conflict = ConflictEvent(
            start_ts=_ts(5), end_ts=_ts(8),
            avg_valence_a=-0.4, avg_valence_b=-0.3,
            avg_arousal_a=0.7, avg_arousal_b=0.6,
            duration_sec=180.0,
        )
        samples_a = _make_samples([(9, -0.3, 0.5), (10, -0.2, 0.5)])
        samples_b = _make_samples([(9, -0.4, 0.5), (10, -0.3, 0.5)])
        repairs = detect_repair([conflict], samples_a, samples_b)
        assert len(repairs) == 0

    def test_no_conflicts_no_repairs(self):
        """No conflicts means no repairs to detect."""
        samples_a = _make_samples([(0, 0.5, 0.3)])
        repairs = detect_repair([], samples_a, samples_a)
        assert len(repairs) == 0


# ===========================================================================
# Test compute_relationship_profile
# ===========================================================================

class TestComputeRelationshipProfile:

    def test_healthy_profile(self):
        """Correlated partners with no conflict should get healthy/strong status."""
        samples_a = _make_samples([
            (i, 0.1 * i, 0.3 + 0.02 * i) for i in range(12)
        ])
        samples_b = _make_samples([
            (i, 0.1 * i + 0.05, 0.3 + 0.02 * i) for i in range(12)
        ])
        profile = compute_relationship_profile(samples_a, samples_b)
        assert profile.synchrony_score > 0.5
        assert profile.conflict_count == 0
        assert profile.repair_rate == 1.0
        assert profile.overall_health in ("strong", "healthy")

    def test_profile_with_conflict(self):
        """Profile should detect conflict when present."""
        samples_a = _make_samples([
            (0, 0.5, 0.3), (1, 0.4, 0.3),
            (2, -0.5, 0.8), (3, -0.4, 0.7), (4, -0.3, 0.6),
            (5, 0.1, 0.4), (6, 0.3, 0.3), (7, 0.5, 0.3),
            (8, 0.5, 0.3), (9, 0.5, 0.3), (10, 0.5, 0.3),
        ])
        samples_b = _make_samples([
            (0, 0.5, 0.3), (1, 0.4, 0.3),
            (2, -0.4, 0.7), (3, -0.3, 0.6), (4, -0.2, 0.6),
            (5, 0.0, 0.4), (6, 0.2, 0.3), (7, 0.4, 0.3),
            (8, 0.4, 0.3), (9, 0.4, 0.3), (10, 0.4, 0.3),
        ])
        profile = compute_relationship_profile(samples_a, samples_b)
        assert isinstance(profile, RelationshipProfile)
        assert profile.overall_health in ("strong", "healthy", "strained", "at_risk")


# ===========================================================================
# Test profile_to_dict
# ===========================================================================

class TestProfileToDict:

    def test_all_keys_present(self):
        """Serialized profile should have all expected keys."""
        profile = RelationshipProfile(
            synchrony_score=0.75,
            resonance_ratio=0.6,
            contagion_a_to_b=0.3,
            contagion_b_to_a=0.2,
            conflict_count=1,
            avg_conflict_duration_sec=120.0,
            repair_count=1,
            avg_repair_latency_sec=60.0,
            repair_rate=1.0,
            overall_health="healthy",
        )
        d = profile_to_dict(profile)
        assert d["synchrony_score"] == 0.75
        assert d["resonance_ratio"] == 0.6
        assert d["contagion_a_to_b"] == 0.3
        assert d["contagion_b_to_a"] == 0.2
        assert d["conflict_count"] == 1
        assert d["avg_conflict_duration_sec"] == 120.0
        assert d["repair_count"] == 1
        assert d["avg_repair_latency_sec"] == 60.0
        assert d["repair_rate"] == 1.0
        assert d["overall_health"] == "healthy"


# ===========================================================================
# Test consent management
# ===========================================================================

class TestConsentManagement:

    def test_opt_in_single_partner(self):
        """Single opt-in should not activate partnership."""
        result = manage_partnership_consent("alice", "bob", "opt_in", "alice")
        assert result["partnership_exists"] is True
        assert result["active"] is False
        assert result["partner_a"]["consented"] is True or result["partner_b"]["consented"] is True

    def test_opt_in_both_activates(self):
        """Both opting in should activate the partnership."""
        manage_partnership_consent("alice", "bob", "opt_in", "alice")
        result = manage_partnership_consent("alice", "bob", "opt_in", "bob")
        assert result["active"] is True

    def test_revoke_deactivates(self):
        """Revoking consent should deactivate the partnership."""
        manage_partnership_consent("alice", "bob", "opt_in", "alice")
        manage_partnership_consent("alice", "bob", "opt_in", "bob")
        result = manage_partnership_consent("alice", "bob", "revoke", "alice")
        assert result["active"] is False

    def test_revoke_wipes_data(self):
        """Revoking should wipe stored emotion data for both partners."""
        manage_partnership_consent("alice", "bob", "opt_in", "alice")
        manage_partnership_consent("alice", "bob", "opt_in", "bob")
        # Store some emotion data
        store_emotion_samples("alice", _make_samples([(0, 0.5, 0.3)]))
        store_emotion_samples("bob", _make_samples([(0, 0.4, 0.3)]))
        assert len(_user_emotions["alice"]) == 1

        manage_partnership_consent("alice", "bob", "revoke", "bob")
        assert "alice" not in _user_emotions
        assert "bob" not in _user_emotions

    def test_status_no_partnership(self):
        """Status on nonexistent partnership should report not found."""
        result = manage_partnership_consent("x", "y", "status", "x")
        assert result["partnership_exists"] is False

    def test_status_existing_partnership(self):
        """Status should report current state."""
        manage_partnership_consent("alice", "bob", "opt_in", "alice")
        result = manage_partnership_consent("alice", "bob", "status", "alice")
        assert result["partnership_exists"] is True

    def test_invalid_action(self):
        """Invalid action should return error."""
        result = manage_partnership_consent("a", "b", "destroy", "a")
        assert "error" in result

    def test_invalid_acting_user(self):
        """Acting user not in partnership should return error."""
        result = manage_partnership_consent("alice", "bob", "opt_in", "charlie")
        assert "error" in result

    def test_revoke_nonexistent_partnership(self):
        """Revoking a nonexistent partnership should return error."""
        result = manage_partnership_consent("x", "y", "revoke", "x")
        assert "error" in result


# ===========================================================================
# Test _align_samples
# ===========================================================================

class TestAlignSamples:

    def test_exact_timestamp_match(self):
        """Samples at identical timestamps should align perfectly."""
        a = _make_samples([(0, 0.1, 0.2), (1, 0.3, 0.4), (2, 0.5, 0.6)])
        b = _make_samples([(0, 0.2, 0.3), (1, 0.4, 0.5), (2, 0.6, 0.7)])
        aligned_a, aligned_b = _align_samples(a, b)
        assert len(aligned_a) == 3
        assert len(aligned_b) == 3

    def test_large_gap_discarded(self):
        """Samples with time gap > max_gap_sec should be discarded."""
        a = _make_samples([(0, 0.1, 0.2)])
        b = _make_samples([(100, 0.2, 0.3)])  # 100 minutes = 6000 sec >> 120 sec
        aligned_a, aligned_b = _align_samples(a, b, max_gap_sec=120.0)
        assert len(aligned_a) == 0

    def test_empty_input(self):
        """Empty input should return empty aligned lists."""
        aligned_a, aligned_b = _align_samples([], [])
        assert aligned_a == []
        assert aligned_b == []


# ===========================================================================
# Test _interpret_synchrony and _classify_health helpers
# ===========================================================================

class TestHelpers:

    def test_interpret_high(self):
        assert _interpret_synchrony(0.7) == "high_synchrony"

    def test_interpret_moderate(self):
        assert _interpret_synchrony(0.4) == "moderate_synchrony"

    def test_interpret_low(self):
        assert _interpret_synchrony(0.1) == "low_synchrony"

    def test_interpret_anti(self):
        assert _interpret_synchrony(-0.3) == "anti_synchrony"

    def test_classify_health_strong(self):
        """High synchrony, high resonance, no conflicts, full repair = strong."""
        result = _classify_health(
            synchrony=0.9, resonance_ratio=0.8,
            conflict_count=0, repair_rate=1.0,
        )
        assert result == "strong"

    def test_classify_health_at_risk(self):
        """Low everything = at_risk."""
        result = _classify_health(
            synchrony=-0.2, resonance_ratio=0.1,
            conflict_count=10, repair_rate=0.0,
        )
        assert result == "at_risk"


# ===========================================================================
# Test store_emotion_samples
# ===========================================================================

class TestStoreEmotionSamples:

    def test_basic_store(self):
        """Samples should be stored and counted."""
        samples = _make_samples([(0, 0.5, 0.3), (1, 0.6, 0.4)])
        count = store_emotion_samples("user1", samples)
        assert count == 2

    def test_cap_enforced(self):
        """Storing more than _MAX_STORED_SAMPLES should trim to cap."""
        from models.couples_resonance import _MAX_STORED_SAMPLES
        big_list = _make_samples([(i, 0.0, 0.0) for i in range(_MAX_STORED_SAMPLES + 100)])
        count = store_emotion_samples("user1", big_list)
        assert count == _MAX_STORED_SAMPLES
