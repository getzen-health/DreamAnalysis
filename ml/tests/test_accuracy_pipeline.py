"""Tests for the 5 accuracy improvement modules."""

import os
import numpy as np


# ─── Signal Quality ───

class TestSignalQuality:
    def setup_method(self):
        from processing.signal_quality import SignalQualityChecker
        self.checker = SignalQualityChecker(fs=256)

    def test_clean_signal_usable(self, sample_eeg):
        result = self.checker.check_quality(sample_eeg)
        assert result["is_usable"] is True
        assert result["quality_score"] > 0.4

    def test_flat_signal_detected(self, flat_signal):
        result = self.checker.check_quality(flat_signal)
        assert result["details"]["amplitude_score"] == 0.0

    def test_railed_signal_rejected(self, railed_signal):
        result = self.checker.check_quality(railed_signal)
        assert len(result["rejection_reasons"]) > 0

    def test_multichannel(self, multichannel_eeg):
        result = self.checker.check_multichannel(multichannel_eeg)
        assert result["total_channels"] == 4
        assert "usable_channels" in result
        assert len(result["channel_scores"]) == 4

    def test_output_keys(self, sample_eeg):
        result = self.checker.check_quality(sample_eeg)
        assert "quality_score" in result
        assert "is_usable" in result
        assert "rejection_reasons" in result
        assert "details" in result
        assert "metrics" in result

    def test_score_range(self, sample_eeg):
        result = self.checker.check_quality(sample_eeg)
        assert 0 <= result["quality_score"] <= 1


# ─── Calibration ───

class TestCalibration:
    def test_calibration_roundtrip(self):
        from processing.calibration import (
            UserCalibration, CalibrationRunner, CALIBRATION_DIR,
        )

        runner = CalibrationRunner(fs=256)
        for cond in ["eyes_open", "eyes_closed", "focused_task", "relaxed_breathing"]:
            for _ in range(5):
                runner.add_epoch(cond, np.random.randn(1024) * 20)

        cal = runner.compute_calibration("_pytest_user")
        assert cal.is_calibrated is True
        assert cal.alpha_reactivity is not None
        assert len(cal.global_band_means) == 5

        # Save and reload
        cal.save()
        cal2 = UserCalibration.load("_pytest_user")
        assert cal2.is_calibrated is True
        assert cal2.alpha_reactivity == cal.alpha_reactivity

        # Cleanup
        os.unlink(CALIBRATION_DIR / "_pytest_user.json")

    def test_normalize_band_powers(self):
        from processing.calibration import CalibrationRunner, CALIBRATION_DIR

        runner = CalibrationRunner(fs=256)
        for cond in ["eyes_open", "eyes_closed", "focused_task", "relaxed_breathing"]:
            for _ in range(5):
                runner.add_epoch(cond, np.random.randn(1024) * 20)
        cal = runner.compute_calibration("_pytest_norm")

        raw = {"delta": 15, "theta": 8, "alpha": 12, "beta": 5, "gamma": 2}
        normed = cal.normalize_band_powers(raw)
        assert set(normed.keys()) == set(raw.keys())
        # Normalized values should be z-scores (not same as raw)
        assert normed != raw

        os.unlink(CALIBRATION_DIR / "_pytest_norm.json")

    def test_uncalibrated_passthrough(self):
        from processing.calibration import UserCalibration

        cal = UserCalibration("nobody")
        raw = {"delta": 15, "theta": 8}
        assert cal.normalize_band_powers(raw) == raw

    def test_progress_tracking(self):
        from processing.calibration import CalibrationRunner

        runner = CalibrationRunner(fs=256)
        progress = runner.get_progress()
        assert progress["completed_steps"] == 0
        assert progress["is_complete"] is False

        for cond in ["eyes_open", "eyes_closed", "focused_task", "relaxed_breathing"]:
            for _ in range(5):
                runner.add_epoch(cond, np.random.randn(1024) * 20)

        progress = runner.get_progress()
        assert progress["completed_steps"] == 4
        assert progress["is_complete"] is True


# ─── State Transitions ───

class TestStateTransitions:
    def setup_method(self):
        from processing.state_transitions import BrainStateEngine
        self.engine = BrainStateEngine()

    def _baseline(self):
        return self.engine.update({
            "sleep": {"stage": "Wake", "confidence": 0.8},
            "flow": {"state": "no_flow", "flow_score": 0.2},
            "emotion": {"emotion": "relaxed", "confidence": 0.6},
            "creativity": {"state": "analytical", "creativity_score": 0.3},
            "memory": {"state": "weak_encoding", "encoding_score": 0.4},
            "dream": {"is_dreaming": False, "probability": 0.1},
        })

    def test_blocks_impossible_flow_jump(self):
        self._baseline()
        r = self.engine.update({
            "sleep": {"stage": "Wake", "confidence": 0.7},
            "flow": {"state": "deep_flow", "flow_score": 0.95},
            "emotion": {"emotion": "focused", "confidence": 0.7},
            "creativity": {"state": "analytical", "creativity_score": 0.5},
            "memory": {"state": "weak_encoding", "encoding_score": 0.5},
            "dream": {"is_dreaming": False, "probability": 0.05},
        })
        assert r["flow"]["was_overridden"] is True
        assert r["flow"]["smoothed_state"] != "deep_flow"

    def test_blocks_impossible_creativity_jump(self):
        self._baseline()
        r = self.engine.update({
            "sleep": {"stage": "Wake", "confidence": 0.7},
            "flow": {"state": "no_flow", "flow_score": 0.2},
            "emotion": {"emotion": "relaxed", "confidence": 0.6},
            "creativity": {"state": "insight", "creativity_score": 0.9},
            "memory": {"state": "weak_encoding", "encoding_score": 0.4},
            "dream": {"is_dreaming": False, "probability": 0.1},
        })
        assert r["creativity"]["was_overridden"] is True

    def test_coherence_check(self):
        self._baseline()
        coh = self.engine.get_cross_state_coherence()
        assert "is_coherent" in coh
        assert "warnings" in coh

    def test_summary_has_all_trackers(self):
        self._baseline()
        summary = self.engine.get_summary()
        assert len(summary) == 5  # sleep, flow, emotion, creativity, memory

    def test_dream_ema_smoothing(self):
        self._baseline()
        r = self.engine.update({
            "sleep": {"stage": "Wake", "confidence": 0.8},
            "flow": {"state": "no_flow", "flow_score": 0.2},
            "emotion": {"emotion": "relaxed", "confidence": 0.6},
            "creativity": {"state": "analytical", "creativity_score": 0.3},
            "memory": {"state": "weak_encoding", "encoding_score": 0.4},
            "dream": {"is_dreaming": True, "probability": 0.9},
        })
        # EMA should smooth — won't jump to 0.9 immediately
        assert r["dream"]["smoothed_probability"] < 0.9

    def test_handles_dict_probabilities(self):
        """Models return probabilities as dicts, not arrays."""
        self._baseline()
        r = self.engine.update({
            "sleep": {"stage": "N2", "confidence": 0.7,
                      "probabilities": {"Wake": 0.1, "N1": 0.1, "N2": 0.6, "N3": 0.1, "REM": 0.1}},
            "flow": {"state": "flow", "flow_score": 0.7},
            "emotion": {"emotion": "focused", "confidence": 0.6},
            "creativity": {"state": "transitional", "creativity_score": 0.5},
            "memory": {"state": "active_encoding", "encoding_score": 0.6},
            "dream": {"is_dreaming": False, "probability": 0.1},
        })
        assert "smoothed_state" in r["sleep"]


# ─── Confidence Calibration ───

class TestConfidenceCalibration:
    def setup_method(self):
        from processing.confidence_calibration import ConfidenceCalibrator
        self.cal = ConfidenceCalibrator()

    def test_compresses_overconfidence(self):
        r = self.cal.calibrate("sleep_staging", 0.95)
        assert r["calibrated_confidence"] < 0.95

    def test_low_confidence_uncertain(self):
        r = self.cal.calibrate("emotion", 0.1)
        assert r["is_uncertain"]  # np.True_ == True but is not True

    def test_dict_probabilities(self):
        r = self.cal.calibrate("sleep_staging", 0.7,
                               {"Wake": 0.7, "N1": 0.1, "N2": 0.1, "N3": 0.05, "REM": 0.05})
        assert r["calibrated_probs"] is not None
        assert len(r["calibrated_probs"]) == 5

    def test_list_probabilities(self):
        r = self.cal.calibrate("sleep_staging", 0.7, [0.7, 0.1, 0.1, 0.05, 0.05])
        assert r["calibrated_probs"] is not None

    def test_reliability_tiers(self):
        rel = self.cal.get_all_reliability()
        assert rel["sleep_staging"]["reliability_tier"] == "high"
        assert rel["creativity"]["reliability_tier"] == "low"

    def test_calibrate_prediction_inplace(self):
        pred = {"stage": "Wake", "confidence": 0.8}
        self.cal.calibrate_prediction("sleep_staging", pred)
        assert "calibrated_confidence" in pred
        assert "is_uncertain" in pred

    def test_add_uncertainty_labels(self):
        from processing.confidence_calibration import add_uncertainty_labels

        analysis = {
            "sleep_staging": {"stage": "Wake", "confidence": 0.7},
            "emotions": {"emotion": "happy", "confidence": 0.5},
            "flow_state": {"state": "flow", "flow_score": 0.6},
            "creativity": {"state": "creative", "creativity_score": 0.4},
            "memory_encoding": {"state": "active_encoding", "encoding_score": 0.5},
            "dream_detection": {"is_dreaming": False, "probability": 0.1},
        }
        labeled = add_uncertainty_labels(analysis, self.cal)
        assert "_confidence_summary" in labeled
        assert "mean_confidence" in labeled["_confidence_summary"]


# ─── User Feedback ───

class TestUserFeedback:
    def test_feedback_collection(self, tmp_path):
        from processing.user_feedback import FeedbackCollector, FEEDBACK_DIR

        fc = FeedbackCollector("_pytest_fb")
        for i in range(20):
            fc.record_state_correction("flow_state", "no_flow",
                                       "flow" if i % 3 == 0 else "no_flow",
                                       np.random.randn(17))
        stats = fc.get_feedback_stats()
        assert stats["total_entries"] == 20
        assert stats["can_personalize"]["flow_state"] is True

        # Cleanup
        os.unlink(FEEDBACK_DIR / "_pytest_fb_feedback.jsonl")

    def test_personal_model_fit(self):
        from processing.user_feedback import PersonalizedModel

        X = np.random.randn(20, 17)
        y = np.array(["flow"] * 7 + ["no_flow"] * 13)
        pm = PersonalizedModel("test", "flow_state")
        pm.fit(X, y)
        assert pm.is_fitted is True
        assert 0 < pm.get_blend_weight() < 1

    def test_personal_model_predict(self):
        from processing.user_feedback import PersonalizedModel

        X = np.random.randn(20, 17)
        y = np.array(["flow"] * 7 + ["no_flow"] * 13)
        pm = PersonalizedModel("test", "flow_state")
        pm.fit(X, y)
        pred = pm.predict(np.random.randn(17))
        assert pred is not None
        assert pred["state"] in ["flow", "no_flow"]

    def test_not_enough_samples(self):
        from processing.user_feedback import PersonalizedModel

        X = np.random.randn(5, 17)
        y = np.array(["flow"] * 5)
        pm = PersonalizedModel("test", "flow_state")
        pm.fit(X, y)
        assert pm.is_fitted is False

    def test_accuracy_no_double_count(self):
        """user_perceived_accuracy must not double-count state corrections."""
        from processing.user_feedback import FeedbackCollector, FEEDBACK_DIR

        fc = FeedbackCollector("_pytest_acc")
        # 8 corrections: predicted == corrected (was_correct=True) for 6,
        # predicted != corrected (was_correct=False) for 2 → accuracy = 6/8 = 0.75
        for i in range(8):
            correct = "happy" if i < 6 else "sad"
            fc.record_state_correction(
                "emotion", "happy", correct, np.random.randn(17)
            )
        # 2 self-reports (should NOT affect accuracy denominator)
        for _ in range(2):
            fc.record_self_report("neutral", "emotion", np.random.randn(17))

        stats = fc.get_feedback_stats()
        model = stats["models"]["emotion"]
        # Total = 10 (8 corrections + 2 reports), rated = 8 (total - reports)
        assert model["total"] == 10
        assert model["reports"] == 2
        assert model["correct"] == 6
        assert model["user_perceived_accuracy"] == 0.75

        # Cleanup
        os.unlink(FEEDBACK_DIR / "_pytest_acc_feedback.jsonl")

    def test_binary_feedback_accuracy(self):
        """Binary feedback (thumbs up/down) should be counted in accuracy."""
        from processing.user_feedback import FeedbackCollector, FEEDBACK_DIR

        fc = FeedbackCollector("_pytest_bin")
        # 4 binary correct, 1 binary wrong → accuracy = 4/5 = 0.8
        for i in range(5):
            fc.record_binary_feedback("stress", "high", was_correct=(i < 4))

        stats = fc.get_feedback_stats()
        model = stats["models"]["stress"]
        assert model["correct"] == 4
        assert model["user_perceived_accuracy"] == 0.8

        os.unlink(FEEDBACK_DIR / "_pytest_bin_feedback.jsonl")
