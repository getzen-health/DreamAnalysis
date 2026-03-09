"""Tests for LuciEntry-style lucid dream induction.

Coverage:
- LucidDreamInducer session lifecycle
- REM detection (feature-based, sleep-stage label shortcut)
- LR eye signal detection
- State machine transitions
- FastAPI endpoint smoke tests
"""

import sys
import os
import time
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Stub heavy dependencies before importing project code ────────────────────

def _make_stub_module(*names):
    mod = types.ModuleType(names[-1])
    for attr in ("preprocess", "extract_band_powers", "extract_features",
                 "butter", "filtfilt"):
        setattr(mod, attr, MagicMock(return_value=np.zeros(256)))
    return mod


for mod_name in ("brainflow", "lightgbm", "onnxruntime", "mne", "torch",
                 "sklearn", "sklearn.preprocessing", "sklearn.pipeline"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _make_stub_module(mod_name)

# Stub eeg_processor so imports don't fail if scipy absent
import types as _types
_proc = _types.ModuleType("processing.eeg_processor")

def _fake_preprocess(sig, fs):
    return np.asarray(sig, dtype=float)

def _fake_band_powers(sig, fs):
    # Return theta-dominant pattern (REM-like)
    return {"delta": 0.05, "theta": 0.35, "alpha": 0.15,
            "beta": 0.10, "high_beta": 0.05, "gamma": 0.03}

_proc.preprocess = _fake_preprocess
_proc.extract_band_powers = _fake_band_powers
_proc.extract_features = MagicMock(return_value={})
sys.modules["processing.eeg_processor"] = _proc

# Stub scipy.signal for LR detection path
import scipy.signal as _ss_real
_ss_stub = _types.ModuleType("scipy.signal")
def _fake_butter(order, freq, btype="low"):
    return (np.array([1.0]), np.array([1.0]))
def _fake_filtfilt(b, a, x):
    return np.asarray(x, dtype=float)
_ss_stub.butter = _fake_butter
_ss_stub.filtfilt = _fake_filtfilt
sys.modules["scipy.signal"] = _ss_stub

# Now import the module under test
from models.lucid_dream_inducer import (
    LucidDreamInducer,
    InductionState,
    REM_CONSECUTIVE_EPOCHS,
    CUE_DURATION_S,
    LR_WINDOW_S,
    RETRY_WAIT_S,
    REM_STABLE_DURATION_S,
)

# Restore real scipy.signal so other test files in the same pytest run
# are not broken by the stub (which lacks welch, hilbert, etc.)
sys.modules["scipy.signal"] = _ss_real
# Remove the eeg_processor stub so the real module can be imported later
sys.modules.pop("processing.eeg_processor", None)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_eeg(n_ch=4, n_samples=256, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 10, (n_ch, n_samples)).astype(np.float32)


def make_lr_eog(n_samples=512, fs=256.0, n_cycles=2, amplitude=120.0):
    """Synthesise a deliberate LRLR signal in the TP9/TP10 differential."""
    eeg = np.zeros((4, n_samples), dtype=np.float32)
    gap = int(0.6 * fs)   # 600 ms between saccades
    directions = [1, -1, 1, -1][:n_cycles * 2]
    pulse_len = int(0.15 * fs)
    start = int(0.2 * fs)
    for i, d in enumerate(directions):
        s = start + i * gap
        e = min(s + pulse_len, n_samples)
        eeg[0, s:e] += d * amplitude   # TP9
        eeg[3, s:e] -= d * amplitude   # TP10 (opposite polarity)
    return eeg


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestSessionLifecycle(unittest.TestCase):

    def setUp(self):
        self.inducer = LucidDreamInducer()

    def test_start_session(self):
        result = self.inducer.start_session("u1", fs=256.0)
        self.assertEqual(result["user_id"], "u1")
        self.assertEqual(result["state"], InductionState.REM_MONITORING.value)

    def test_stop_nonexistent_session(self):
        result = self.inducer.stop_session("ghost")
        self.assertEqual(result["status"], "not_found")

    def test_stop_session_returns_summary(self):
        self.inducer.start_session("u2")
        result = self.inducer.stop_session("u2")
        self.assertEqual(result["status"], "stopped")
        self.assertIn("lucid_episodes", result)
        self.assertNotIn("u2", self.inducer._sessions)

    def test_get_status_no_session(self):
        result = self.inducer.get_status("nobody")
        self.assertEqual(result["status"], "no_session")

    def test_get_status_active(self):
        self.inducer.start_session("u3")
        result = self.inducer.get_status("u3")
        self.assertEqual(result["state"], InductionState.REM_MONITORING.value)

    def test_list_sessions(self):
        self.inducer.start_session("a")
        self.inducer.start_session("b")
        sessions = self.inducer.list_sessions()
        user_ids = {s["user_id"] for s in sessions}
        self.assertIn("a", user_ids)
        self.assertIn("b", user_ids)

    def test_confirm_lucidity_no_session(self):
        result = self.inducer.confirm_lucidity("nobody")
        self.assertEqual(result["status"], "no_session")

    def test_confirm_lucidity(self):
        self.inducer.start_session("u4")
        result = self.inducer.confirm_lucidity("u4")
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["lucid_episodes"], 1)


class TestREMDetection(unittest.TestCase):

    def setUp(self):
        self.inducer = LucidDreamInducer()
        self.eeg = make_eeg()

    def test_rem_label_shortcut(self):
        self.inducer.start_session("rem_user")
        result = self.inducer.process_epoch("rem_user", self.eeg, sleep_stage="REM")
        self.assertGreater(result["rem_score"], 0.8)

    def test_wake_label_suppresses_rem(self):
        self.inducer.start_session("wake_user")
        result = self.inducer.process_epoch("wake_user", self.eeg, sleep_stage="Wake")
        self.assertLess(result["rem_score"], 0.1)

    def test_n3_label_suppresses_rem(self):
        self.inducer.start_session("n3_user")
        result = self.inducer.process_epoch("n3_user", self.eeg, sleep_stage="N3")
        self.assertLess(result["rem_score"], 0.1)

    def test_feature_based_rem_detection(self):
        # band powers stub returns theta=0.35 which should yield moderate REM score
        self.inducer.start_session("feat_user")
        result = self.inducer.process_epoch("feat_user", self.eeg, sleep_stage=None)
        self.assertGreater(result["rem_score"], 0.0)
        self.assertLessEqual(result["rem_score"], 1.0)

    def test_epoch_count_increments(self):
        self.inducer.start_session("cnt")
        for _ in range(5):
            result = self.inducer.process_epoch("cnt", self.eeg)
        self.assertEqual(result["epoch"], 5)


class TestStateMachine(unittest.TestCase):

    def setUp(self):
        self.inducer = LucidDreamInducer()

    def test_rem_consecutive_accumulates_on_high_score(self):
        self.inducer.start_session("sm1")
        for _ in range(10):
            r = self.inducer.process_epoch("sm1", make_eeg(), sleep_stage="REM")
        self.assertGreater(r["rem_consecutive_epochs"], 0)
        self.assertEqual(r["state"], InductionState.REM_MONITORING.value)

    def test_rem_stable_transition(self):
        """Force transition to REM_STABLE by exceeding REM_CONSECUTIVE_EPOCHS."""
        self.inducer.start_session("sm2")
        session = self.inducer._sessions["sm2"]
        session.rem_consecutive = REM_CONSECUTIVE_EPOCHS
        # Single REM epoch triggers state advancement
        r = self.inducer.process_epoch("sm2", make_eeg(), sleep_stage="REM")
        self.assertEqual(r["state"], InductionState.REM_STABLE.value)

    def test_cues_trigger_from_cues_scheduled(self):
        self.inducer.start_session("sm3")
        session = self.inducer._sessions["sm3"]
        session.state = InductionState.CUES_SCHEDULED
        r = self.inducer.process_epoch("sm3", make_eeg())
        self.assertTrue(r["trigger_cue"])
        self.assertEqual(r["state"], InductionState.CUES_DELIVERED.value)

    def test_no_cue_trigger_in_monitoring(self):
        self.inducer.start_session("sm4")
        r = self.inducer.process_epoch("sm4", make_eeg(), sleep_stage="N1")
        self.assertFalse(r["trigger_cue"])

    def test_lr_detection_moves_to_lucid_confirmed(self):
        self.inducer.start_session("sm5")
        session = self.inducer._sessions["sm5"]
        session.state = InductionState.LR_DETECTION
        session.lr_window_started_at = time.time()
        eeg = make_lr_eog()
        r = self.inducer.process_epoch("sm5", eeg)
        self.assertEqual(r["state"], InductionState.LUCID_CONFIRMED.value)
        self.assertEqual(r["lucid_episodes"], 1)

    def test_retry_after_lr_timeout(self):
        self.inducer.start_session("sm6")
        session = self.inducer._sessions["sm6"]
        session.state = InductionState.LR_DETECTION
        # Simulate timeout
        session.lr_window_started_at = time.time() - LR_WINDOW_S - 1
        r = self.inducer.process_epoch("sm6", make_eeg())
        self.assertEqual(r["state"], InductionState.RETRY.value)
        self.assertEqual(r["retry_count"], 1)

    def test_retry_returns_to_rem_monitoring_after_wait(self):
        self.inducer.start_session("sm7")
        session = self.inducer._sessions["sm7"]
        session.state = InductionState.RETRY
        session.last_retry_at = time.time() - RETRY_WAIT_S - 1
        r = self.inducer.process_epoch("sm7", make_eeg())
        self.assertEqual(r["state"], InductionState.REM_MONITORING.value)

    def test_rem_stable_degrades_on_low_score(self):
        """Low REM score in REM_STABLE state should eventually revert to monitoring."""
        self.inducer.start_session("sm8")
        session = self.inducer._sessions["sm8"]
        session.state = InductionState.REM_STABLE
        session.rem_consecutive = REM_CONSECUTIVE_EPOCHS
        session.rem_stable_since = time.time()
        # Low-theta EEG → low rem_score from feature-based detector
        # Patch band powers to return low theta
        with patch(
            "models.lucid_dream_inducer.extract_band_powers",
            return_value={"delta": 0.5, "theta": 0.02, "alpha": 0.1,
                          "beta": 0.3, "high_beta": 0.05, "gamma": 0.02},
        ):
            for _ in range(50):
                r = self.inducer.process_epoch("sm8", make_eeg(), sleep_stage=None)
                if r["state"] == InductionState.REM_MONITORING.value:
                    break
        self.assertEqual(r["state"], InductionState.REM_MONITORING.value)


class TestLRDetection(unittest.TestCase):

    def setUp(self):
        self.inducer = LucidDreamInducer()

    def test_no_detection_on_flat_eeg(self):
        eeg = np.zeros((4, 512), dtype=np.float32)
        score, detected = self.inducer._detect_lr_signal(eeg, 256.0)
        self.assertFalse(detected)
        self.assertEqual(score, 0.0)

    def test_detection_on_lrlr_signal(self):
        eeg = make_lr_eog(n_cycles=2)
        score, detected = self.inducer._detect_lr_signal(eeg, 256.0)
        self.assertTrue(detected)
        self.assertGreater(score, 0.9)

    def test_partial_lr_not_detected(self):
        """Only 1 alternation (2 crossings) should NOT trigger detection."""
        eeg = make_lr_eog(n_cycles=1)  # 2 crossings
        score, detected = self.inducer._detect_lr_signal(eeg, 256.0)
        self.assertFalse(detected)

    def test_insufficient_channels_returns_zero(self):
        eeg = np.zeros((2, 256), dtype=np.float32)
        score, detected = self.inducer._detect_lr_signal(eeg, 256.0)
        self.assertFalse(detected)
        self.assertEqual(score, 0.0)

    def test_score_between_zero_and_one(self):
        for seed in range(5):
            eeg = make_eeg(seed=seed)
            score, _ = self.inducer._detect_lr_signal(eeg, 256.0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestEndpoints(unittest.TestCase):
    """Smoke tests for FastAPI route handlers (no HTTP server needed)."""

    def setUp(self):
        # Import here so that the stubs above are already in place
        from api.routes.lucid_induction import (
            start_induction, process_epoch, get_induction_status,
            confirm_lucidity, stop_induction, list_induction_sessions,
            _inducer as _ep_inducer,
        )
        self._start = start_induction
        self._process = process_epoch
        self._status = get_induction_status
        self._confirm = confirm_lucidity
        self._stop = stop_induction
        self._list = list_induction_sessions
        self._inducer = _ep_inducer
        # Clean slate
        self._inducer._sessions.clear()

    def _run(self, coro):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_start_and_status(self):
        from api.routes.lucid_induction import StartInductionRequest
        req = StartInductionRequest(user_id="ep1", fs=256.0)
        result = self._run(self._start(req))
        self.assertEqual(result["state"], InductionState.REM_MONITORING.value)
        status = self._run(self._status("ep1"))
        self.assertEqual(status["state"], InductionState.REM_MONITORING.value)

    def test_process_epoch_endpoint(self):
        from api.routes.lucid_induction import StartInductionRequest, ProcessEpochRequest
        self._run(self._start(StartInductionRequest(user_id="ep2")))
        signals = make_eeg().tolist()
        req = ProcessEpochRequest(user_id="ep2", signals=signals, fs=256.0)
        result = self._run(self._process(req))
        self.assertIn("state", result)
        self.assertIn("trigger_cue", result)

    def test_stop_endpoint(self):
        from api.routes.lucid_induction import StartInductionRequest
        self._run(self._start(StartInductionRequest(user_id="ep3")))
        result = self._run(self._stop("ep3"))
        self.assertEqual(result["status"], "stopped")

    def test_status_404_on_missing(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self._run(self._status("missing_user"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_confirm_404_on_missing(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self._run(self._confirm("nobody"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_list_sessions(self):
        from api.routes.lucid_induction import StartInductionRequest
        self._run(self._start(StartInductionRequest(user_id="la")))
        self._run(self._start(StartInductionRequest(user_id="lb")))
        result = self._run(self._list())
        ids = {s["user_id"] for s in result["sessions"]}
        self.assertIn("la", ids)
        self.assertIn("lb", ids)


if __name__ == "__main__":
    unittest.main()
