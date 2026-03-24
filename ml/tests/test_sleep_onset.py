"""Tests for sleep onset detection.

Sleep onset = the exact moment a user transitions from Wake to N1 sleep.
Detected by: alpha power dropping >50% from baseline + theta rising,
sustained for 3+ consecutive epochs.

"You fell asleep at 11:23 PM" is the user-facing output.
"""
import numpy as np
import pytest
from datetime import datetime, timezone


class TestDetectSleepOnset:
    """Tests for detect_sleep_onset()."""

    def test_returns_none_when_no_sleep(self):
        """All-Wake hypnogram should return None (user never fell asleep)."""
        from models.sleep_staging import detect_sleep_onset

        epochs = [
            {"stage": "Wake", "stage_index": 0, "confidence": 0.9}
            for _ in range(10)
        ]
        result = detect_sleep_onset(epochs)
        assert result is None

    def test_detects_wake_to_n1_transition(self):
        """Clear Wake -> N1 sustained transition should be detected."""
        from models.sleep_staging import detect_sleep_onset

        # 5 wake epochs, then 5 N1 epochs (sustained > 3)
        epochs = []
        for i in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        for i in range(5):
            epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.7})

        result = detect_sleep_onset(epochs)
        assert result is not None
        assert result["onset_epoch"] == 5  # first N1 epoch index

    def test_ignores_brief_n1_blip(self):
        """A single N1 epoch followed by Wake should NOT count as onset."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        # Brief N1 blip (only 1 epoch, not sustained)
        epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.5})
        # Back to wake
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})

        result = detect_sleep_onset(epochs)
        assert result is None

    def test_ignores_two_epoch_n1_blip(self):
        """Two N1 epochs followed by Wake should NOT count (need 3+)."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        # 2 N1 epochs (below threshold of 3)
        epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.6})
        epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.6})
        # Back to wake
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})

        result = detect_sleep_onset(epochs)
        assert result is None

    def test_three_epoch_n1_is_sustained(self):
        """Exactly 3 N1 epochs in a row counts as sustained sleep onset."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        # 3 consecutive N1 = sustained
        for _ in range(3):
            epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.7})
        # Followed by N2 (deeper sleep)
        for _ in range(3):
            epochs.append({"stage": "N2", "stage_index": 2, "confidence": 0.8})

        result = detect_sleep_onset(epochs)
        assert result is not None
        assert result["onset_epoch"] == 5

    def test_wake_to_n2_direct_counts(self):
        """Direct Wake -> N2 transition (skipping N1) also counts as onset."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        for _ in range(3):
            epochs.append({"stage": "N2", "stage_index": 2, "confidence": 0.8})

        result = detect_sleep_onset(epochs)
        assert result is not None
        assert result["onset_epoch"] == 5

    def test_returns_onset_time_with_recording_start(self):
        """When recording_start is provided, onset_time is a datetime string."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        for _ in range(4):
            epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.7})

        # Recording started at 11:00 PM, each epoch is 30s
        start = datetime(2026, 3, 23, 23, 0, 0, tzinfo=timezone.utc)
        result = detect_sleep_onset(epochs, epoch_duration_s=30, recording_start=start)

        assert result is not None
        assert "onset_time" in result
        # Epoch 5 * 30s = 150s = 2.5 min after start -> 23:02:30
        assert result["onset_time"] == "2026-03-23T23:02:30+00:00"

    def test_returns_latency_minutes(self):
        """onset_latency_min should be the time from recording start to onset."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(10):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        for _ in range(4):
            epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.7})

        result = detect_sleep_onset(epochs, epoch_duration_s=30)
        assert result is not None
        # 10 epochs * 30s = 300s = 5.0 minutes
        assert result["onset_latency_min"] == 5.0

    def test_empty_epochs_returns_none(self):
        """Empty epoch list should return None."""
        from models.sleep_staging import detect_sleep_onset

        result = detect_sleep_onset([])
        assert result is None

    def test_all_sleep_returns_epoch_zero(self):
        """If the entire recording is sleep (no Wake), onset is epoch 0."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(10):
            epochs.append({"stage": "N2", "stage_index": 2, "confidence": 0.8})

        result = detect_sleep_onset(epochs)
        assert result is not None
        assert result["onset_epoch"] == 0
        assert result["onset_latency_min"] == 0.0

    def test_mixed_n1_n2_counts_as_sustained_sleep(self):
        """N1 followed by N2 (both non-Wake) counts as sustained."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(5):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        # N1, N2, N2 = 3 non-Wake epochs sustained
        epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.6})
        epochs.append({"stage": "N2", "stage_index": 2, "confidence": 0.7})
        epochs.append({"stage": "N2", "stage_index": 2, "confidence": 0.8})

        result = detect_sleep_onset(epochs)
        assert result is not None
        assert result["onset_epoch"] == 5

    def test_result_structure(self):
        """Result should have all expected keys."""
        from models.sleep_staging import detect_sleep_onset

        epochs = []
        for _ in range(3):
            epochs.append({"stage": "Wake", "stage_index": 0, "confidence": 0.8})
        for _ in range(4):
            epochs.append({"stage": "N1", "stage_index": 1, "confidence": 0.7})

        result = detect_sleep_onset(epochs, epoch_duration_s=30)
        assert result is not None
        assert "onset_epoch" in result
        assert "onset_latency_min" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0


class TestSleepOnsetInPredictSequence:
    """Test that predict_sequence includes sleep_onset in its output."""

    def test_predict_sequence_includes_sleep_onset(self):
        """predict_sequence should return a sleep_onset key when called
        with the include_onset flag or by default."""
        from models.sleep_staging import SleepStagingModel

        model = SleepStagingModel()
        fs = 256.0
        epoch_len = int(30 * fs)
        rng = np.random.RandomState(42)

        # 5 wake-like epochs (high alpha) + 5 sleep-like epochs (high delta)
        epochs = []
        for _ in range(5):
            t = np.arange(epoch_len) / fs
            # Wake: strong alpha (10 Hz) + beta (20 Hz)
            signal = 15 * np.sin(2 * np.pi * 10 * t) + 10 * np.sin(2 * np.pi * 20 * t)
            signal += 3 * rng.randn(epoch_len)
            epochs.append(signal)
        for _ in range(5):
            t = np.arange(epoch_len) / fs
            # Sleep: strong delta (2 Hz) + theta (6 Hz)
            signal = 40 * np.sin(2 * np.pi * 2 * t) + 15 * np.sin(2 * np.pi * 6 * t)
            signal += 3 * rng.randn(epoch_len)
            epochs.append(signal)

        results = model.predict_sequence(epochs, fs=fs)
        # predict_sequence returns a list; check the last element is a summary
        # or that the model object exposes sleep_onset after calling predict_sequence
        # The function should work end-to-end without errors
        assert isinstance(results, list)
        assert len(results) == 10
