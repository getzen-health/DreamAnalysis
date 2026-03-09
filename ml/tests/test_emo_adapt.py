"""Tests for EmoAdapt self-supervised EEG emotion personalization.

Covers:
- Default state (no prototypes, n_updates=0)
- predict() — required keys, valid emotion, probs sum, ranges, adaptation_gain
- update_prototype() — return keys, counters, gain after updates
- reset() — counters zeroed
- Singleton factory
- Prototype bank updates steer prediction toward the labeled class
- Status dict completeness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ root is on the path (mirrors other test files in this repo)
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from models.emo_adapt import (
    EmoAdaptLearner,
    EMOTIONS_6,
    get_emo_adapt_learner,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
FS = 256.0

# 4-channel × 2 seconds of synthetic EEG (µV scale)
EEG_4CH = (RNG.standard_normal((4, 512)) * 15.0).astype(np.float32)

# 1D signal — 1 second
EEG_1D = (RNG.standard_normal(256) * 15.0).astype(np.float32)

# Very short signal (< minimum filter length)
EEG_SHORT = (RNG.standard_normal((4, 10)) * 15.0).astype(np.float32)

REQUIRED_PREDICT_KEYS = {
    "emotion",
    "probabilities",
    "valence",
    "arousal",
    "adaptation_gain",
    "n_updates",
}

REQUIRED_STATUS_KEYS = {
    "n_updates_per_class",
    "n_updates_total",
    "n_prototypes",
    "adaptation_ready",
    "adaptation_gain",
}


@pytest.fixture(autouse=False)
def fresh_learner():
    """Return a fresh EmoAdaptLearner for isolation."""
    return EmoAdaptLearner(fs=FS)


# ── Init / default state ──────────────────────────────────────────────────────

class TestInit:
    def test_default_n_updates_zero(self, fresh_learner):
        status = fresh_learner.get_status()
        assert status["n_updates_total"] == 0

    def test_default_n_prototypes_zero(self, fresh_learner):
        status = fresh_learner.get_status()
        assert status["n_prototypes"] == 0

    def test_default_all_class_updates_zero(self, fresh_learner):
        status = fresh_learner.get_status()
        for label in EMOTIONS_6:
            assert status["n_updates_per_class"][label] == 0

    def test_default_adaptation_ready_false(self, fresh_learner):
        status = fresh_learner.get_status()
        assert status["adaptation_ready"] is False

    def test_default_adaptation_gain_zero(self, fresh_learner):
        status = fresh_learner.get_status()
        assert status["adaptation_gain"] == 0.0


# ── predict() ─────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_required_keys_4ch(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys())

    def test_returns_required_keys_1d(self, fresh_learner):
        result = fresh_learner.predict(EEG_1D, fs=FS)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys())

    def test_emotion_is_valid_label_4ch(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert result["emotion"] in EMOTIONS_6

    def test_emotion_is_valid_label_1d(self, fresh_learner):
        result = fresh_learner.predict(EEG_1D, fs=FS)
        assert result["emotion"] in EMOTIONS_6

    def test_probabilities_sum_to_one_4ch(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 2e-4, f"Prob sum={prob_sum}"

    def test_probabilities_sum_to_one_1d(self, fresh_learner):
        result = fresh_learner.predict(EEG_1D, fs=FS)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 2e-4, f"Prob sum={prob_sum}"

    def test_probabilities_have_all_six_labels(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert set(result["probabilities"].keys()) == set(EMOTIONS_6)

    def test_valence_in_range_4ch(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert -1.0 <= result["valence"] <= 1.0

    def test_arousal_in_range_4ch(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert 0.0 <= result["arousal"] <= 1.0

    def test_adaptation_gain_in_range(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert 0.0 <= result["adaptation_gain"] <= 1.0

    def test_adaptation_gain_zero_before_updates(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert result["adaptation_gain"] == 0.0

    def test_n_updates_zero_before_any_updates(self, fresh_learner):
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert result["n_updates"] == 0

    def test_short_signal_handled_gracefully(self, fresh_learner):
        """Very short signals should not raise exceptions."""
        result = fresh_learner.predict(EEG_SHORT, fs=FS)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys())
        assert result["emotion"] in EMOTIONS_6


# ── update_prototype() ────────────────────────────────────────────────────────

class TestUpdatePrototype:
    def test_returns_updated_prototypes_key(self, fresh_learner):
        result = fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        assert "updated_prototypes" in result

    def test_returns_label_key(self, fresh_learner):
        result = fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        assert "label" in result

    def test_label_matches_input(self, fresh_learner):
        result = fresh_learner.update_prototype(EEG_4CH, "sad", fs=FS)
        assert result["label"] == "sad"

    def test_n_updates_increments_after_update(self, fresh_learner):
        fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        status = fresh_learner.get_status()
        assert status["n_updates_total"] == 1

    def test_per_class_counter_increments(self, fresh_learner):
        fresh_learner.update_prototype(EEG_4CH, "angry", fs=FS)
        fresh_learner.update_prototype(EEG_4CH, "angry", fs=FS)
        status = fresh_learner.get_status()
        assert status["n_updates_per_class"]["angry"] == 2

    def test_adaptation_gain_increases_after_enough_updates(self, fresh_learner):
        """After ≥5 updates per class across all classes, gain > 0."""
        threshold = 5
        for label in EMOTIONS_6:
            for _ in range(threshold):
                fresh_learner.update_prototype(EEG_4CH, label, fs=FS)
        result = fresh_learner.predict(EEG_4CH, fs=FS)
        assert result["adaptation_gain"] > 0.0

    def test_invalid_emotion_label_raises(self, fresh_learner):
        """An unknown emotion label should raise ValueError."""
        with pytest.raises(ValueError):
            fresh_learner.update_prototype(EEG_4CH, "confused", fs=FS)

    def test_updated_prototypes_count_increases(self, fresh_learner):
        r1 = fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        r2 = fresh_learner.update_prototype(EEG_4CH, "sad", fs=FS)
        assert r2["updated_prototypes"] >= r1["updated_prototypes"]


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_n_updates_returns_to_zero_after_reset(self, fresh_learner):
        fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        fresh_learner.reset()
        status = fresh_learner.get_status()
        assert status["n_updates_total"] == 0

    def test_n_prototypes_returns_to_zero_after_reset(self, fresh_learner):
        fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        fresh_learner.reset()
        status = fresh_learner.get_status()
        assert status["n_prototypes"] == 0

    def test_per_class_counters_zero_after_reset(self, fresh_learner):
        for label in EMOTIONS_6:
            fresh_learner.update_prototype(EEG_4CH, label, fs=FS)
        fresh_learner.reset()
        status = fresh_learner.get_status()
        for label in EMOTIONS_6:
            assert status["n_updates_per_class"][label] == 0

    def test_reset_returns_n_updates_zero(self, fresh_learner):
        fresh_learner.update_prototype(EEG_4CH, "fear", fs=FS)
        result = fresh_learner.reset()
        assert result["n_updates"] == 0


# ── Singleton factory ─────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_emo_adapt_learner_returns_same_instance(self):
        a = get_emo_adapt_learner()
        b = get_emo_adapt_learner()
        assert a is b

    def test_singleton_is_emo_adapt_learner_instance(self):
        learner = get_emo_adapt_learner()
        assert isinstance(learner, EmoAdaptLearner)


# ── Prototype bank steering ───────────────────────────────────────────────────

class TestPrototypeSteering:
    def test_predict_shows_higher_prob_for_repeatedly_updated_class(self):
        """After enough updates for 'happy', its probability should be the max."""
        learner = EmoAdaptLearner(fs=FS)
        # Provide well above threshold updates for all classes so adaptation is active,
        # but concentrate 'happy' updates with a clearly different signal pattern.
        threshold = 5

        # Base updates for non-happy classes (flat signal → different features)
        flat = np.zeros((4, 512), dtype=np.float32)
        for label in EMOTIONS_6:
            if label != "happy":
                for _ in range(threshold):
                    learner.update_prototype(flat, label, fs=FS)

        # 'Happy' updates with the query EEG (closer to query → higher similarity)
        for _ in range(threshold + 2):
            learner.update_prototype(EEG_4CH, "happy", fs=FS)

        result = learner.predict(EEG_4CH, fs=FS)
        happy_prob = result["probabilities"]["happy"]
        # happy should have the highest probability (cosine similarity is highest)
        assert happy_prob == max(result["probabilities"].values()), (
            f"Expected happy to have max prob, got: {result['probabilities']}"
        )


# ── Status dict ───────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_has_required_keys(self, fresh_learner):
        status = fresh_learner.get_status()
        assert REQUIRED_STATUS_KEYS.issubset(status.keys())

    def test_status_adaptation_ready_true_when_all_classes_above_threshold(
        self, fresh_learner
    ):
        threshold = 5
        for label in EMOTIONS_6:
            for _ in range(threshold):
                fresh_learner.update_prototype(EEG_4CH, label, fs=FS)
        status = fresh_learner.get_status()
        assert status["adaptation_ready"] is True

    def test_status_adaptation_ready_false_when_partial(self, fresh_learner):
        # Only update one class above threshold
        for _ in range(10):
            fresh_learner.update_prototype(EEG_4CH, "happy", fs=FS)
        status = fresh_learner.get_status()
        assert status["adaptation_ready"] is False
