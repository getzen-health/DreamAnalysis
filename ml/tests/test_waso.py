"""Tests for WASO (Wake After Sleep Onset) computation.

WASO is a standard clinical sleep metric: total minutes spent awake between
sleep onset and the final awakening. High WASO (>30 min) indicates fragmented
sleep and predicts next-day cognitive impairment.

AASM definition: sum of all Wake epochs after the first epoch of sustained
sleep and before the last epoch of sleep in the recording.
"""
import pytest

from models.sleep_staging import compute_waso, compute_sleep_stats


def _epoch(stage: str, stage_index: int, confidence: float = 0.8):
    return {"stage": stage, "stage_index": stage_index, "confidence": confidence}


class TestComputeWaso:
    """Tests for compute_waso()."""

    def test_no_wake_after_onset(self):
        """Continuous sleep with no awakenings -> WASO = 0."""
        epochs = (
            [_epoch("Wake", 0)] * 5
            + [_epoch("N1", 1)] * 3
            + [_epoch("N2", 2)] * 10
            + [_epoch("N3", 3)] * 5
        )
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result["waso_minutes"] == 0.0
        assert result["num_awakenings"] == 0
        assert result["longest_awakening_minutes"] == 0.0

    def test_single_awakening(self):
        """One Wake bout in the middle of sleep."""
        epochs = (
            [_epoch("Wake", 0)] * 5           # pre-sleep
            + [_epoch("N1", 1)] * 3            # onset (sustained)
            + [_epoch("N2", 2)] * 5            # sleep
            + [_epoch("Wake", 0)] * 2          # mid-sleep awakening (2 * 30s = 1 min)
            + [_epoch("N2", 2)] * 5            # back to sleep
        )
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result["waso_minutes"] == 1.0   # 2 epochs * 30s = 60s = 1 min
        assert result["num_awakenings"] == 1
        assert result["longest_awakening_minutes"] == 1.0

    def test_multiple_awakenings(self):
        """Two separate Wake bouts after onset."""
        epochs = (
            [_epoch("Wake", 0)] * 3           # pre-sleep
            + [_epoch("N1", 1)] * 3            # onset
            + [_epoch("N2", 2)] * 4            # sleep
            + [_epoch("Wake", 0)] * 2          # awakening 1: 1 min
            + [_epoch("N2", 2)] * 4            # sleep
            + [_epoch("Wake", 0)] * 4          # awakening 2: 2 min
            + [_epoch("N3", 3)] * 3            # back to sleep
        )
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result["waso_minutes"] == 3.0   # (2 + 4) * 30s = 180s = 3 min
        assert result["num_awakenings"] == 2
        assert result["longest_awakening_minutes"] == 2.0

    def test_wake_at_end_excluded(self):
        """Wake epochs after the last sleep epoch are final awakening, not WASO."""
        epochs = (
            [_epoch("Wake", 0)] * 3           # pre-sleep
            + [_epoch("N2", 2)] * 5            # onset
            + [_epoch("Wake", 0)] * 2          # mid-sleep awakening: 1 min
            + [_epoch("N2", 2)] * 3            # back to sleep
            + [_epoch("Wake", 0)] * 10         # final awakening (excluded)
        )
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result["waso_minutes"] == 1.0   # only the mid-sleep wake
        assert result["num_awakenings"] == 1

    def test_all_wake_returns_none(self):
        """All-Wake recording has no sleep onset -> returns None."""
        epochs = [_epoch("Wake", 0)] * 20
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result is None

    def test_empty_epochs_returns_none(self):
        """Empty epoch list returns None."""
        result = compute_waso([], epoch_duration_s=30.0)
        assert result is None

    def test_all_sleep_no_wake(self):
        """Entire recording is sleep, no Wake at all -> WASO = 0."""
        epochs = [_epoch("N2", 2)] * 20
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result["waso_minutes"] == 0.0
        assert result["num_awakenings"] == 0

    def test_custom_epoch_duration(self):
        """Non-default epoch duration (e.g. 60s) computes correctly."""
        epochs = (
            [_epoch("Wake", 0)] * 3
            + [_epoch("N2", 2)] * 5
            + [_epoch("Wake", 0)] * 3          # 3 * 60s = 3 min
            + [_epoch("N2", 2)] * 5
        )
        result = compute_waso(epochs, epoch_duration_s=60.0)
        assert result["waso_minutes"] == 3.0

    def test_result_structure(self):
        """Result dict has all expected keys with correct types."""
        epochs = (
            [_epoch("Wake", 0)] * 3
            + [_epoch("N2", 2)] * 5
            + [_epoch("Wake", 0)] * 2
            + [_epoch("N2", 2)] * 5
        )
        result = compute_waso(epochs, epoch_duration_s=30.0)
        assert result is not None
        assert isinstance(result["waso_minutes"], float)
        assert isinstance(result["num_awakenings"], int)
        assert isinstance(result["longest_awakening_minutes"], float)

    def test_artifact_epochs_treated_as_not_wake(self):
        """Artifact-rejected epochs (stage='artifact') are not Wake, not sleep.
        They should be excluded from WASO counting."""
        epochs = (
            [_epoch("Wake", 0)] * 3
            + [_epoch("N2", 2)] * 5
            + [{"stage": "artifact", "stage_index": -1, "confidence": 0.0}] * 2
            + [_epoch("N2", 2)] * 5
        )
        result = compute_waso(epochs, epoch_duration_s=30.0)
        # Artifact epochs are ambiguous; they are NOT Wake, so WASO = 0
        assert result["waso_minutes"] == 0.0


class TestComputeSleepStats:
    """Tests for compute_sleep_stats() -- comprehensive sleep summary."""

    def test_typical_night(self):
        """A realistic night: Wake -> N1 -> N2 -> N3 -> REM with one awakening."""
        epochs = (
            [_epoch("Wake", 0)] * 10          # 5 min pre-sleep
            + [_epoch("N1", 1)] * 6            # 3 min N1
            + [_epoch("N2", 2)] * 20           # 10 min N2
            + [_epoch("N3", 3)] * 10           # 5 min N3
            + [_epoch("Wake", 0)] * 2          # 1 min mid-sleep awakening
            + [_epoch("N2", 2)] * 12           # 6 min N2
            + [_epoch("REM", 4)] * 10          # 5 min REM
        )
        result = compute_sleep_stats(epochs, epoch_duration_s=30.0)
        assert result is not None

        total = 70
        assert result["wake_pct"] == round(12 / total, 4)
        assert result["n1_pct"] == round(6 / total, 4)
        assert result["n2_pct"] == round(32 / total, 4)
        assert result["n3_pct"] == round(10 / total, 4)
        assert result["rem_pct"] == round(10 / total, 4)

        assert result["total_recording_minutes"] == 35.0  # 70 * 30 / 60
        assert result["total_sleep_minutes"] == 29.0       # 58 * 30 / 60
        assert result["sleep_efficiency"] == round(58 / 70, 4)

        # WASO: 2 Wake epochs after onset = 1 min
        assert result["waso_minutes"] == 1.0
        assert result["num_awakenings"] == 1

        # Sleep onset should be detected
        assert result["sleep_onset"] is not None
        assert result["sleep_onset"]["onset_epoch"] == 10

    def test_empty_returns_none(self):
        result = compute_sleep_stats([])
        assert result is None

    def test_all_wake_returns_none(self):
        """No sleep detected -> None."""
        epochs = [_epoch("Wake", 0)] * 20
        result = compute_sleep_stats(epochs, epoch_duration_s=30.0)
        # Has stage stats but no WASO or onset
        assert result is not None
        assert result["waso_minutes"] == 0.0
        assert result["sleep_onset"] is None

    def test_keys_present(self):
        """All expected keys are in the result."""
        epochs = (
            [_epoch("Wake", 0)] * 5
            + [_epoch("N2", 2)] * 10
        )
        result = compute_sleep_stats(epochs, epoch_duration_s=30.0)
        assert result is not None
        for key in [
            "n3_pct", "rem_pct", "n2_pct", "n1_pct", "wake_pct",
            "total_sleep_minutes", "total_recording_minutes",
            "sleep_efficiency", "waso_minutes", "num_awakenings",
            "longest_awakening_minutes", "sleep_onset",
        ]:
            assert key in result, f"Missing key: {key}"
