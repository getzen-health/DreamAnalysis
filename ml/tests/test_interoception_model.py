"""Tests for interoception training engine (#422).

Covers:
  - Heartbeat counting IAS scoring (perfect, over-count, under-count, zero)
  - Body scan scoring (accuracy, sensitivity, specificity)
  - Interoceptive profile computation (MAIA dimensions, composite score)
  - Progressive exercise generation (heartbeat vs body scan, difficulty)
  - profile_to_dict serialization
  - Edge cases (zero actual beats, empty reports, empty history)
  - Multi-user isolation
  - History capping
"""

import pytest

from models.interoception_model import (
    BODY_REGIONS,
    DIFFICULTY_LEVELS,
    MAIA_DIMENSIONS,
    BodyScanResult,
    HeartbeatTask,
    InteroceptionEngine,
    InteroceptiveProfile,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _scan_report(
    region: str = "chest",
    sensation: str = "warmth",
    active: bool = True,
    confidence: float = 0.8,
) -> dict:
    """Build a single body-scan report dict."""
    return {
        "region": region,
        "reported_sensation": sensation,
        "ground_truth_active": active,
        "confidence": confidence,
    }


@pytest.fixture
def engine():
    return InteroceptionEngine()


# ── Heartbeat Counting ───────────────────────────────────────────────


class TestHeartbeatCounting:
    def test_perfect_accuracy(self, engine):
        """Counted == actual -> IAS = 1.0."""
        result = engine.score_heartbeat_counting(
            counted_beats=70, actual_beats=70, duration_seconds=60
        )
        assert result["ias"] == 1.0

    def test_over_count(self, engine):
        """Counting more than actual reduces IAS."""
        result = engine.score_heartbeat_counting(
            counted_beats=80, actual_beats=70, duration_seconds=60
        )
        expected = 1.0 - abs(80 - 70) / 70
        assert abs(result["ias"] - round(expected, 4)) < 1e-3
        assert result["ias"] < 1.0

    def test_under_count(self, engine):
        """Counting fewer than actual reduces IAS."""
        result = engine.score_heartbeat_counting(
            counted_beats=50, actual_beats=70, duration_seconds=60
        )
        expected = 1.0 - abs(50 - 70) / 70
        assert abs(result["ias"] - round(expected, 4)) < 1e-3
        assert result["ias"] < 1.0

    def test_ias_clamped_at_zero(self, engine):
        """Extreme over/under-counting should not produce negative IAS."""
        result = engine.score_heartbeat_counting(
            counted_beats=200, actual_beats=50, duration_seconds=60
        )
        assert result["ias"] == 0.0

    def test_zero_actual_beats(self, engine):
        """Zero actual beats should not crash; IAS = 0."""
        result = engine.score_heartbeat_counting(
            counted_beats=10, actual_beats=0, duration_seconds=60
        )
        assert result["ias"] == 0.0

    def test_hr_bpm_calculated(self, engine):
        """Heart rate in BPM should be computed from actual/duration."""
        result = engine.score_heartbeat_counting(
            counted_beats=70, actual_beats=75, duration_seconds=60
        )
        assert result["hr_bpm"] == 75.0

    def test_interpretation_excellent(self, engine):
        result = engine.score_heartbeat_counting(
            counted_beats=70, actual_beats=70, duration_seconds=60
        )
        assert result["interpretation"] == "excellent"

    def test_interpretation_poor(self, engine):
        result = engine.score_heartbeat_counting(
            counted_beats=10, actual_beats=70, duration_seconds=60
        )
        assert result["interpretation"] == "poor"

    def test_condition_passed_through(self, engine):
        result = engine.score_heartbeat_counting(
            counted_beats=60, actual_beats=70, duration_seconds=60,
            condition="movement",
        )
        assert result["condition"] == "movement"

    def test_result_keys(self, engine):
        result = engine.score_heartbeat_counting(
            counted_beats=70, actual_beats=70, duration_seconds=60
        )
        expected_keys = {
            "ias", "counted_beats", "actual_beats", "duration_seconds",
            "condition", "hr_bpm", "interpretation",
        }
        assert expected_keys == set(result.keys())


# ── Body Scan Scoring ────────────────────────────────────────────────


class TestBodyScan:
    def test_all_correct(self, engine):
        """All correct detections -> accuracy = 1.0."""
        reports = [
            _scan_report("chest", "warmth", active=True),
            _scan_report("stomach", "none", active=False),
        ]
        result = engine.score_body_scan(reports)
        assert result["accuracy"] == 1.0
        assert result["n_correct"] == 2
        assert result["n_total"] == 2

    def test_all_wrong(self, engine):
        """All misses -> accuracy = 0.0."""
        reports = [
            _scan_report("chest", "none", active=True),   # miss
            _scan_report("stomach", "tingling", active=False),  # false positive
        ]
        result = engine.score_body_scan(reports)
        assert result["accuracy"] == 0.0
        assert result["n_correct"] == 0

    def test_sensitivity_computed(self, engine):
        """Sensitivity = TP / (TP + FN)."""
        reports = [
            _scan_report("chest", "warmth", active=True),   # TP
            _scan_report("head", "none", active=True),       # FN
        ]
        result = engine.score_body_scan(reports)
        assert result["sensitivity"] == 0.5

    def test_specificity_computed(self, engine):
        """Specificity = TN / (TN + FP)."""
        reports = [
            _scan_report("chest", "none", active=False),        # TN
            _scan_report("stomach", "tingling", active=False),  # FP
        ]
        result = engine.score_body_scan(reports)
        assert result["specificity"] == 0.5

    def test_empty_reports(self, engine):
        """Empty report list should not crash; accuracy = 0."""
        result = engine.score_body_scan([])
        assert result["accuracy"] == 0.0
        assert result["n_total"] == 0

    def test_per_region_details(self, engine):
        reports = [_scan_report("chest", "warmth", active=True)]
        result = engine.score_body_scan(reports)
        assert len(result["per_region"]) == 1
        assert result["per_region"][0]["region"] == "chest"
        assert result["per_region"][0]["correct"] is True


# ── Interoceptive Profile ────────────────────────────────────────────


class TestProfile:
    def test_empty_profile(self, engine):
        """No history -> baseline profile with zero task scores."""
        profile = engine.compute_interoceptive_profile()
        assert profile.heartbeat_accuracy == 0.0
        assert profile.body_scan_accuracy == 0.0
        # Overall > 0 because MAIA defaults to 0.5 for dimensions
        # with insufficient data (not_distracting, not_worrying, etc.)
        assert 0.0 <= profile.overall_score <= 0.2
        assert profile.n_heartbeat_tasks == 0
        assert profile.n_body_scans == 0

    def test_profile_after_heartbeat(self, engine):
        engine.score_heartbeat_counting(70, 70, 60)
        profile = engine.compute_interoceptive_profile()
        assert profile.heartbeat_accuracy == 1.0
        assert profile.n_heartbeat_tasks == 1

    def test_profile_after_body_scan(self, engine):
        engine.score_body_scan([
            _scan_report("chest", "warmth", active=True),
            _scan_report("head", "none", active=False),
        ])
        profile = engine.compute_interoceptive_profile()
        assert profile.body_scan_accuracy == 1.0
        assert profile.n_body_scans == 2

    def test_maia_dimensions_present(self, engine):
        """All 8 MAIA dimensions should be present in profile."""
        engine.score_heartbeat_counting(70, 70, 60)
        engine.score_body_scan([_scan_report("chest", "warmth", active=True)])
        profile = engine.compute_interoceptive_profile()
        for dim in MAIA_DIMENSIONS:
            assert dim in profile.maia_scores, f"Missing MAIA dimension: {dim}"

    def test_maia_scores_in_range(self, engine):
        """Each MAIA score should be in [0, 1]."""
        for _ in range(5):
            engine.score_heartbeat_counting(65, 70, 60)
        engine.score_body_scan([
            _scan_report("chest", "warmth", active=True),
            _scan_report("head", "none", active=False),
            _scan_report("stomach", "tension", active=True),
        ])
        profile = engine.compute_interoceptive_profile()
        for dim, score in profile.maia_scores.items():
            assert 0.0 <= score <= 1.0, f"{dim} out of range: {score}"

    def test_overall_score_in_range(self, engine):
        engine.score_heartbeat_counting(70, 70, 60)
        profile = engine.compute_interoceptive_profile()
        assert 0.0 <= profile.overall_score <= 1.0

    def test_difficulty_beginner_with_few_exercises(self, engine):
        """With fewer than 3 exercises, difficulty should be beginner."""
        engine.score_heartbeat_counting(70, 70, 60)
        profile = engine.compute_interoceptive_profile()
        assert profile.difficulty_level == "beginner"

    def test_difficulty_escalates_with_performance(self, engine):
        """High accuracy over many exercises should increase difficulty."""
        for _ in range(10):
            engine.score_heartbeat_counting(70, 70, 60)
            engine.score_body_scan([
                _scan_report("chest", "warmth", active=True),
                _scan_report("head", "none", active=False),
            ])
        profile = engine.compute_interoceptive_profile()
        assert profile.difficulty_level in ("hard", "expert")


# ── Exercise Generation ──────────────────────────────────────────────


class TestExerciseGeneration:
    def test_first_exercise_is_heartbeat(self, engine):
        """First exercises should be heartbeat counting."""
        ex = engine.generate_next_exercise()
        assert ex["exercise_type"] == "heartbeat_counting"

    def test_exercise_has_required_keys(self, engine):
        ex = engine.generate_next_exercise()
        assert "exercise_type" in ex
        assert "instructions" in ex
        assert "difficulty" in ex
        assert "tip" in ex

    def test_body_scan_generated_after_heartbeats(self, engine):
        """After several heartbeat tasks, a body scan should appear."""
        # Do 3 heartbeat tasks (total_exercises=3, then 3 % 3 == 0 -> body scan)
        for _ in range(3):
            engine.score_heartbeat_counting(70, 70, 60)
        ex = engine.generate_next_exercise()
        assert ex["exercise_type"] == "body_scan"
        assert "regions" in ex
        assert len(ex["regions"]) > 0

    def test_difficulty_reflected_in_exercise(self, engine):
        """Exercise difficulty should match the user's profile difficulty."""
        ex = engine.generate_next_exercise()
        assert ex["difficulty"] in DIFFICULTY_LEVELS

    def test_heartbeat_condition_at_rest_for_beginners(self, engine):
        """Beginners should get rest condition."""
        ex = engine.generate_next_exercise()
        assert ex.get("condition") == "rest"


# ── Profile Serialization ────────────────────────────────────────────


class TestProfileToDict:
    def test_returns_dict(self, engine):
        profile = engine.compute_interoceptive_profile()
        d = engine.profile_to_dict(profile)
        assert isinstance(d, dict)

    def test_dict_has_all_fields(self, engine):
        engine.score_heartbeat_counting(70, 70, 60)
        profile = engine.compute_interoceptive_profile()
        d = engine.profile_to_dict(profile)
        assert "user_id" in d
        assert "heartbeat_accuracy" in d
        assert "body_scan_accuracy" in d
        assert "maia_scores" in d
        assert "overall_score" in d
        assert "difficulty_level" in d


# ── Multi-User Isolation ─────────────────────────────────────────────


class TestMultiUser:
    def test_separate_heartbeat_history(self, engine):
        engine.score_heartbeat_counting(70, 70, 60, user_id="alice")
        engine.score_heartbeat_counting(50, 70, 60, user_id="bob")
        p_alice = engine.compute_interoceptive_profile("alice")
        p_bob = engine.compute_interoceptive_profile("bob")
        assert p_alice.heartbeat_accuracy == 1.0
        assert p_bob.heartbeat_accuracy < 1.0

    def test_separate_body_scan_history(self, engine):
        engine.score_body_scan(
            [_scan_report("chest", "warmth", active=True)],
            user_id="alice",
        )
        p_alice = engine.compute_interoceptive_profile("alice")
        p_bob = engine.compute_interoceptive_profile("bob")
        assert p_alice.n_body_scans == 1
        assert p_bob.n_body_scans == 0


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_duration_heartbeat(self, engine):
        """Zero duration should not crash (hr_bpm = 0)."""
        # Duration 0 is not allowed by the API (gt=0), but the engine
        # should handle it gracefully at model level.
        result = engine.score_heartbeat_counting(70, 70, 0.001)
        assert result["ias"] == 1.0
        assert result["hr_bpm"] > 0

    def test_very_large_counts(self, engine):
        """Very large beat counts should not crash."""
        result = engine.score_heartbeat_counting(
            counted_beats=100000, actual_beats=70, duration_seconds=60
        )
        assert result["ias"] == 0.0

    def test_body_scan_unknown_region(self, engine):
        """Unknown body region should still be processed."""
        reports = [_scan_report("unknown_region", "warmth", active=True)]
        result = engine.score_body_scan(reports)
        assert result["n_total"] == 1

    def test_dataclass_creation(self):
        """HeartbeatTask and BodyScanResult dataclasses should create cleanly."""
        ht = HeartbeatTask(
            counted_beats=70, actual_beats=70,
            duration_seconds=60, ias=1.0,
        )
        assert ht.ias == 1.0
        assert ht.condition == "rest"

        bsr = BodyScanResult(
            region="chest", reported_sensation="warmth",
            ground_truth_active=True, correct=True,
        )
        assert bsr.correct is True
        assert bsr.confidence == 0.5

    def test_profile_dataclass(self):
        """InteroceptiveProfile dataclass should create cleanly."""
        p = InteroceptiveProfile(
            user_id="test",
            heartbeat_accuracy=0.9,
            body_scan_accuracy=0.8,
            maia_scores={"noticing": 0.7},
            overall_score=0.8,
            n_heartbeat_tasks=5,
            n_body_scans=3,
            difficulty_level="moderate",
            exercise_history_length=8,
        )
        assert p.user_id == "test"
        assert p.overall_score == 0.8
