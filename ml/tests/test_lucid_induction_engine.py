"""Tests for the lucid dream induction engine — issue #452.

Coverage:
  - SleepEEGFeatures extraction from EEG data
  - REM state detection and scoring
  - Cue timing computation
  - Cue type selection (history-based)
  - Induction attempt tracking and profile building
  - Reality test scheduling
  - Profile serialization
  - FastAPI endpoint smoke tests
"""

import sys
import os
import time
import unittest
from unittest.mock import patch

import numpy as np

# ---- Path setup ----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- Import module under test --------------------------------------------
from models.lucid_induction_engine import (
    LucidInductionEngine,
    SleepEEGFeatures,
    CueConfig,
    CueType,
    InductionAttempt,
    InductionTechnique,
    LucidProfile,
    REMState,
    STABLE_REM_MIN_DURATION_S,
    K_COMPLEX_AMPLITUDE_UV,
    MIN_THETA_PEAK_RATIO,
    get_lucid_induction_engine,
)


# ---- Helpers -------------------------------------------------------------

def make_eeg(n_ch=4, n_samples=256, seed=42):
    """Generate random EEG-like data."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 10, (n_ch, n_samples)).astype(np.float32)


def make_rem_like_eeg(n_ch=4, n_samples=512, fs=256.0, seed=99):
    """Generate EEG that looks REM-like: theta-dominant, low amplitude, eye movements."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    # Strong theta (6 Hz) + weak alpha (10 Hz) + noise
    theta = 15.0 * np.sin(2 * np.pi * 6 * t)
    alpha = 3.0 * np.sin(2 * np.pi * 10 * t)
    noise = rng.normal(0, 2, n_samples)

    eeg = np.zeros((n_ch, n_samples), dtype=np.float32)
    for ch in range(n_ch):
        eeg[ch] = theta + alpha + noise + rng.normal(0, 1, n_samples)

    # Add eye movement artifacts to TP9/TP10 (ch0, ch3)
    saccade_times = np.array([0.5, 1.0, 1.5])
    for st in saccade_times:
        idx = int(st * fs)
        if idx + 10 < n_samples:
            eeg[0, idx:idx+10] += 30
            eeg[3, idx:idx+10] -= 30

    return eeg


# ---- Test: Feature Extraction --------------------------------------------

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_extract_features_returns_correct_type(self):
        eeg = make_eeg()
        features = self.engine._extract_sleep_features(eeg, 256.0)
        self.assertIsInstance(features, SleepEEGFeatures)

    def test_features_have_valid_ranges(self):
        eeg = make_eeg()
        f = self.engine._extract_sleep_features(eeg, 256.0)
        self.assertGreaterEqual(f.theta_power, 0.0)
        self.assertLessEqual(f.theta_power, 1.0)
        self.assertGreaterEqual(f.alpha_power, 0.0)
        self.assertLessEqual(f.alpha_power, 1.0)
        self.assertGreaterEqual(f.spectral_entropy, 0.0)
        self.assertLessEqual(f.spectral_entropy, 1.0)
        self.assertGreaterEqual(f.emg_amplitude, 0.0)

    def test_single_channel_input(self):
        """Should work with 1D (single-channel) input."""
        signal = np.random.default_rng(42).normal(0, 10, 256).astype(np.float32)
        f = self.engine._extract_sleep_features(signal, 256.0)
        self.assertIsInstance(f, SleepEEGFeatures)
        # Eye movement score should be 0 without multichannel
        self.assertEqual(f.eye_movement_score, 0.0)

    def test_epoch_duration_computed(self):
        eeg = make_eeg(n_samples=512)
        f = self.engine._extract_sleep_features(eeg, 256.0)
        self.assertAlmostEqual(f.epoch_duration_s, 2.0, places=1)

    def test_k_complex_detection(self):
        """Signal with a large spike should flag K-complex."""
        eeg = make_eeg(n_samples=256)
        # Inject a large spike above threshold
        eeg[0, 100] = K_COMPLEX_AMPLITUDE_UV + 20
        f = self.engine._extract_sleep_features(eeg, 256.0)
        self.assertTrue(f.has_k_complex)


# ---- Test: REM Detection -------------------------------------------------

class TestREMDetection(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_rem_score_range(self):
        eeg = make_eeg()
        result = self.engine.detect_rem_state(eeg, 256.0, user_id="u1")
        self.assertGreaterEqual(result["rem_score"], 0.0)
        self.assertLessEqual(result["rem_score"], 1.0)

    def test_rem_state_in_valid_enum(self):
        eeg = make_eeg()
        result = self.engine.detect_rem_state(eeg, 256.0, user_id="u2")
        valid_states = {s.value for s in REMState}
        self.assertIn(result["rem_state"], valid_states)

    def test_rem_like_eeg_scores_higher(self):
        """REM-like EEG should score higher than random noise."""
        random_eeg = make_eeg(seed=10)
        rem_eeg = make_rem_like_eeg()

        r_random = self.engine.detect_rem_state(random_eeg, 256.0, user_id="cmp_rand")
        r_rem = self.engine.detect_rem_state(rem_eeg, 256.0, user_id="cmp_rem")
        self.assertGreater(r_rem["rem_score"], r_random["rem_score"])

    def test_rem_duration_tracking(self):
        """Consecutive REM detections should accumulate duration."""
        eeg = make_rem_like_eeg()
        # Process multiple epochs quickly — duration tracked via time.time()
        r1 = self.engine.detect_rem_state(eeg, 256.0, user_id="dur_test")
        # Duration should be >= 0
        self.assertGreaterEqual(r1["rem_duration_s"], 0.0)

    def test_non_rem_resets_duration(self):
        """A non-REM epoch should reset cumulative duration to 0."""
        # First, get a REM detection going
        rem_eeg = make_rem_like_eeg()
        self.engine.detect_rem_state(rem_eeg, 256.0, user_id="reset_test")

        # Now send flat signal (all zeros -> very low scores)
        flat_eeg = np.zeros((4, 256), dtype=np.float32)
        result = self.engine.detect_rem_state(flat_eeg, 256.0, user_id="reset_test")
        self.assertEqual(result["rem_duration_s"], 0.0)

    def test_features_dict_in_result(self):
        eeg = make_eeg()
        result = self.engine.detect_rem_state(eeg, 256.0, user_id="feat_check")
        self.assertIn("features", result)
        self.assertIn("theta_power", result["features"])
        self.assertIn("alpha_power", result["features"])
        self.assertIn("emg_amplitude", result["features"])


# ---- Test: Cue Timing ---------------------------------------------------

class TestCueTiming(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_not_ready_insufficient_rem(self):
        """Cue should not be ready when REM duration is too short."""
        features = SleepEEGFeatures(
            theta_power=0.35, alpha_power=0.10,
        )
        result = self.engine.compute_cue_timing(features, rem_duration_s=60.0)
        self.assertFalse(result["ready"])
        self.assertFalse(result["rem_stable"])

    def test_not_ready_low_theta(self):
        """Cue should not be ready when theta is not peaking."""
        features = SleepEEGFeatures(
            theta_power=0.10, alpha_power=0.30,
        )
        result = self.engine.compute_cue_timing(
            features, rem_duration_s=STABLE_REM_MIN_DURATION_S + 10
        )
        self.assertFalse(result["ready"])

    def test_not_ready_k_complex(self):
        """K-complex present should block cue delivery."""
        features = SleepEEGFeatures(
            theta_power=0.40, alpha_power=0.05,
            has_k_complex=True,
        )
        result = self.engine.compute_cue_timing(
            features, rem_duration_s=STABLE_REM_MIN_DURATION_S + 10
        )
        self.assertFalse(result["ready"])
        self.assertIn("K-complex", result["reason"])

    def test_ready_when_all_conditions_met(self):
        """All conditions met should produce ready=True."""
        features = SleepEEGFeatures(
            theta_power=0.40, alpha_power=0.05,
            has_k_complex=False,
        )
        result = self.engine.compute_cue_timing(
            features, rem_duration_s=STABLE_REM_MIN_DURATION_S + 10
        )
        self.assertTrue(result["ready"])
        self.assertTrue(result["rem_stable"])

    def test_optimal_intensity_in_range(self):
        features = SleepEEGFeatures(theta_power=0.35, alpha_power=0.10)
        result = self.engine.compute_cue_timing(features, rem_duration_s=400)
        self.assertGreaterEqual(result["optimal_intensity"], 0.1)
        self.assertLessEqual(result["optimal_intensity"], 0.7)


# ---- Test: Cue Selection -------------------------------------------------

class TestCueSelection(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_default_cue_is_audio(self):
        cue = self.engine.select_cue_type(user_id="new_user")
        self.assertEqual(cue.cue_type, CueType.AUDIO)
        self.assertGreater(cue.intensity, 0.0)

    def test_preferred_cue_respected(self):
        cue = self.engine.select_cue_type(
            user_id="pref_user", preferred=CueType.HAPTIC
        )
        self.assertEqual(cue.cue_type, CueType.HAPTIC)

    def test_history_based_selection(self):
        """After recording successes with LED, engine should prefer LED."""
        for _ in range(5):
            self.engine.track_induction_success(
                user_id="hist_user",
                technique=InductionTechnique.EXTERNAL_CUE,
                cue_config=CueConfig(cue_type=CueType.LED, intensity=0.3),
                lucid_reported=True,
            )
        for _ in range(5):
            self.engine.track_induction_success(
                user_id="hist_user",
                technique=InductionTechnique.EXTERNAL_CUE,
                cue_config=CueConfig(cue_type=CueType.AUDIO, intensity=0.3),
                lucid_reported=False,
            )
        cue = self.engine.select_cue_type(user_id="hist_user")
        self.assertEqual(cue.cue_type, CueType.LED)


# ---- Test: Success Tracking ----------------------------------------------

class TestSuccessTracking(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_record_single_attempt(self):
        result = self.engine.track_induction_success(
            user_id="track_u1",
            technique=InductionTechnique.MILD,
            cue_config=None,
            lucid_reported=True,
        )
        self.assertTrue(result["recorded"])
        self.assertEqual(result["profile_summary"]["total_attempts"], 1)
        self.assertEqual(result["profile_summary"]["total_lucid"], 1)
        self.assertAlmostEqual(result["profile_summary"]["success_rate"], 1.0)

    def test_success_rate_calculation(self):
        """Record 2 successes and 3 failures -> 0.4 rate."""
        for lucid in [True, True, False, False, False]:
            self.engine.track_induction_success(
                user_id="rate_user",
                technique=InductionTechnique.WBTB,
                cue_config=None,
                lucid_reported=lucid,
            )
        profile = self.engine.compute_lucid_profile("rate_user")
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.success_rate, 0.4)
        self.assertEqual(profile.total_attempts, 5)
        self.assertEqual(profile.total_lucid, 2)

    def test_best_technique_tracked(self):
        """MILD with higher success should be selected as best."""
        for _ in range(3):
            self.engine.track_induction_success(
                "best_tech", InductionTechnique.MILD, None, lucid_reported=True
            )
        for _ in range(3):
            self.engine.track_induction_success(
                "best_tech", InductionTechnique.WBTB, None, lucid_reported=False
            )
        profile = self.engine.compute_lucid_profile("best_tech")
        self.assertEqual(profile.best_technique, "mild")


# ---- Test: Reality Test Scheduling ----------------------------------------

class TestRealityTestScheduling(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_schedule_returns_correct_count(self):
        result = self.engine.schedule_reality_tests(
            user_id="sched_u1", tests_per_day=8
        )
        self.assertEqual(len(result["times_h"]), 8)
        self.assertEqual(len(result["times_formatted"]), 8)
        self.assertEqual(result["tests_per_day"], 8)

    def test_schedule_times_in_window(self):
        result = self.engine.schedule_reality_tests(
            user_id="sched_u2", tests_per_day=5,
            window_start_h=9, window_end_h=21,
        )
        for t in result["times_h"]:
            self.assertGreaterEqual(t, 9.0)
            self.assertLessEqual(t, 21.0)

    def test_schedule_times_sorted(self):
        result = self.engine.schedule_reality_tests(
            user_id="sched_u3", tests_per_day=10,
        )
        times = result["times_h"]
        self.assertEqual(times, sorted(times))

    def test_record_reality_test(self):
        self.engine.record_reality_test("rt_user")
        self.engine.record_reality_test("rt_user")
        result = self.engine.record_reality_test("rt_user")
        self.assertEqual(result["reality_tests_completed"], 3)


# ---- Test: Profile Serialization -----------------------------------------

class TestProfileSerialization(unittest.TestCase):

    def setUp(self):
        self.engine = LucidInductionEngine()

    def test_profile_to_dict(self):
        self.engine.track_induction_success(
            "serial_u", InductionTechnique.MILD, None, lucid_reported=True
        )
        profile = self.engine.compute_lucid_profile("serial_u")
        d = self.engine.profile_to_dict(profile)
        self.assertEqual(d["user_id"], "serial_u")
        self.assertEqual(d["total_attempts"], 1)
        self.assertEqual(d["total_lucid"], 1)
        self.assertIn("technique_rates", d)
        self.assertIn("cue_type_rates", d)
        self.assertIn("last_updated", d)

    def test_profile_not_found_returns_none(self):
        profile = self.engine.compute_lucid_profile("nonexistent_user")
        self.assertIsNone(profile)


# ---- Test: Singleton Accessor --------------------------------------------

class TestSingleton(unittest.TestCase):

    def test_get_engine_returns_instance(self):
        engine = get_lucid_induction_engine()
        self.assertIsInstance(engine, LucidInductionEngine)

    def test_singleton_is_same_instance(self):
        e1 = get_lucid_induction_engine()
        e2 = get_lucid_induction_engine()
        self.assertIs(e1, e2)


# ---- Test: Endpoints (smoke tests) --------------------------------------

class TestEndpoints(unittest.TestCase):
    """Smoke tests for FastAPI route handlers."""

    def setUp(self):
        from api.routes.lucid_induction_engine import (
            detect_rem,
            compute_cue,
            record_attempt,
            get_profile,
            engine_status,
        )
        self._detect_rem = detect_rem
        self._compute_cue = compute_cue
        self._record_attempt = record_attempt
        self._get_profile = get_profile
        self._engine_status = engine_status

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_status_endpoint(self):
        result = self._run(self._engine_status())
        self.assertTrue(result["available"])
        self.assertEqual(result["model"], "LucidInductionEngine")

    def test_detect_rem_endpoint(self):
        from api.routes.lucid_induction_engine import DetectREMRequest
        eeg = make_eeg().tolist()
        req = DetectREMRequest(user_id="ep_rem", eeg_data=eeg, fs=256.0)
        result = self._run(self._detect_rem(req))
        self.assertIn("rem_state", result)
        self.assertIn("rem_score", result)

    def test_compute_cue_endpoint(self):
        from api.routes.lucid_induction_engine import ComputeCueRequest
        req = ComputeCueRequest(
            user_id="ep_cue",
            theta_power=0.35,
            alpha_power=0.05,
            rem_duration_s=400.0,
        )
        result = self._run(self._compute_cue(req))
        self.assertIn("ready", result)
        self.assertIn("recommended_cue", result)

    def test_record_attempt_endpoint(self):
        from api.routes.lucid_induction_engine import RecordAttemptRequest
        req = RecordAttemptRequest(
            user_id="ep_record",
            technique="mild",
            lucid_reported=True,
        )
        result = self._run(self._record_attempt(req))
        self.assertTrue(result["recorded"])

    def test_profile_404_on_missing(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self._run(self._get_profile("no_such_user"))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
