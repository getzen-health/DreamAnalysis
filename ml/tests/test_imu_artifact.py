"""Tests for IMUArtifactDetector.

Covers:
- Still accelerometer (gravity vector only) → no motion
- Sudden spike → motion detected
- Output keys present and types correct
- artifact_probability in [0, 1]
- calibrate_resting changes baseline
- gyroscope contribution
- singleton getter returns same instance
"""

import numpy as np
import pytest
from models.imu_artifact_detector import IMUArtifactDetector, get_imu_detector

# Muse 2 IMU sampling rate
_FS = 52.0
# 1-second window at 52 Hz
_N = 52


def _gravity_acc(n: int = _N) -> np.ndarray:
    """Return still accelerometer: gravity pointing straight down (0, 0, 9.81 m/s²)."""
    acc = np.zeros((3, n))
    acc[2, :] = 9.81  # z-axis = gravity
    return acc


def _spike_acc(n: int = _N, spike_amplitude: float = 5.0) -> np.ndarray:
    """Return accelerometer with a clear motion spike above resting baseline."""
    acc = _gravity_acc(n)
    # Inject a strong burst in the middle of the window
    mid = n // 2
    width = max(1, n // 8)
    acc[0, mid:mid + width] += spike_amplitude   # x-axis burst
    acc[1, mid:mid + width] += spike_amplitude   # y-axis burst
    return acc


# ── Basic output shape & types ─────────────────────────────────────────────


def test_output_keys_present():
    detector = IMUArtifactDetector()
    result = detector.detect(_gravity_acc(), fs=_FS)
    required_keys = {
        "motion_detected",
        "artifact_probability",
        "motion_rms_g",
        "gyro_rms_dps",
        "contaminated_fraction",
        "recommendation",
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_artifact_probability_in_range():
    detector = IMUArtifactDetector()
    for acc in [_gravity_acc(), _spike_acc()]:
        result = detector.detect(acc, fs=_FS)
        prob = result["artifact_probability"]
        assert 0.0 <= prob <= 1.0, f"artifact_probability {prob} out of [0, 1]"


def test_motion_detected_is_bool():
    detector = IMUArtifactDetector()
    result = detector.detect(_gravity_acc(), fs=_FS)
    assert isinstance(result["motion_detected"], bool)


def test_recommendation_valid_values():
    detector = IMUArtifactDetector()
    valid = {"clean", "mild_motion", "severe_motion"}
    for acc in [_gravity_acc(), _spike_acc()]:
        result = detector.detect(acc, fs=_FS)
        assert result["recommendation"] in valid, (
            f"Unexpected recommendation: {result['recommendation']}"
        )


# ── Core detection logic ───────────────────────────────────────────────────


def test_still_acc_no_motion():
    """Pure gravity vector should not trigger motion detection."""
    detector = IMUArtifactDetector(threshold_g=0.15)
    # Calibrate so resting baseline = exactly 1 g (gravity only)
    detector.calibrate_resting(_gravity_acc(n=200), fs=_FS)
    result = detector.detect(_gravity_acc(), fs=_FS)
    assert result["motion_detected"] is False, (
        "Still accelerometer (gravity only) should not be flagged as motion. "
        f"Got motion_rms_g={result['motion_rms_g']}, recommendation={result['recommendation']}"
    )
    assert result["recommendation"] == "clean"


def test_spike_acc_motion_detected():
    """Strong acceleration spike well above baseline should be flagged."""
    detector = IMUArtifactDetector(threshold_g=0.15)
    detector.calibrate_resting(_gravity_acc(n=200), fs=_FS)
    # 5 m/s² ≈ 0.51 g above gravity — well above 0.15 g threshold
    result = detector.detect(_spike_acc(spike_amplitude=5.0), fs=_FS)
    assert result["motion_detected"] is True, (
        "Spike accelerometer should be flagged as motion. "
        f"Got motion_rms_g={result['motion_rms_g']}, recommendation={result['recommendation']}"
    )
    assert result["recommendation"] in {"mild_motion", "severe_motion"}


def test_artifact_probability_higher_for_spike():
    """Spike should produce higher artifact_probability than still."""
    detector = IMUArtifactDetector(threshold_g=0.15)
    detector.calibrate_resting(_gravity_acc(n=200), fs=_FS)
    still = detector.detect(_gravity_acc(), fs=_FS)
    spike = detector.detect(_spike_acc(spike_amplitude=5.0), fs=_FS)
    assert spike["artifact_probability"] > still["artifact_probability"], (
        "Spike should have higher artifact_probability than still. "
        f"still={still['artifact_probability']}, spike={spike['artifact_probability']}"
    )


# ── Calibration ────────────────────────────────────────────────────────────


def test_calibrate_resting_sets_baseline():
    detector = IMUArtifactDetector()
    default_baseline = detector._resting_baseline
    detector.calibrate_resting(_gravity_acc(n=300), fs=_FS)
    # After calibration with pure gravity, baseline should be close to 1 g
    assert abs(detector._resting_baseline - 1.0) < 0.05, (
        f"Expected baseline ~1.0 g after calibrating on gravity, "
        f"got {detector._resting_baseline}"
    )
    assert detector._baseline_ready is True


def test_update_baseline_ema():
    detector = IMUArtifactDetector()
    detector._resting_baseline = 1.0
    # After one EMA step with alpha=1.0 (instant update), baseline = new_value
    detector.update_baseline(1.5, alpha=1.0)
    assert detector._resting_baseline == pytest.approx(1.5)


def test_update_baseline_slow_alpha():
    detector = IMUArtifactDetector()
    detector._resting_baseline = 1.0
    # Very small alpha — baseline should barely change
    detector.update_baseline(10.0, alpha=0.001)
    assert detector._resting_baseline < 1.01


# ── Gyroscope contribution ─────────────────────────────────────────────────


def test_gyro_rms_zero_when_not_provided():
    detector = IMUArtifactDetector()
    result = detector.detect(_gravity_acc(), gyro_data=None, fs=_FS)
    assert result["gyro_rms_dps"] == 0.0


def test_gyro_data_accepted():
    """Providing gyro data should not raise and gyro_rms_dps should be non-negative."""
    detector = IMUArtifactDetector()
    gyro = np.random.randn(3, _N) * 2.0  # small random rotation
    result = detector.detect(_gravity_acc(), gyro_data=gyro, fs=_FS)
    assert result["gyro_rms_dps"] >= 0.0


def test_large_gyro_increases_probability():
    """Very large gyro signal should push artifact_probability higher."""
    detector = IMUArtifactDetector(threshold_g=0.15)
    detector.calibrate_resting(_gravity_acc(n=200), fs=_FS)
    no_gyro = detector.detect(_gravity_acc(), gyro_data=None, fs=_FS)
    big_gyro = np.ones((3, _N)) * 50.0  # 50 deg/s on all axes — clear rotation
    with_gyro = detector.detect(_gravity_acc(), gyro_data=big_gyro, fs=_FS)
    assert with_gyro["artifact_probability"] >= no_gyro["artifact_probability"]


# ── Input format flexibility ───────────────────────────────────────────────


def test_transposed_input_accepted():
    """(n_samples, 3) shape should work the same as (3, n_samples)."""
    detector = IMUArtifactDetector()
    acc_standard = _gravity_acc()           # (3, 52)
    acc_transposed = acc_standard.T         # (52, 3)
    res_std = detector.detect(acc_standard, fs=_FS)
    res_t = detector.detect(acc_transposed, fs=_FS)
    # Both should agree on motion_detected
    assert res_std["motion_detected"] == res_t["motion_detected"]


# ── Singleton ──────────────────────────────────────────────────────────────


def test_singleton_same_instance():
    d1 = get_imu_detector()
    d2 = get_imu_detector()
    assert d1 is d2


def test_singleton_returns_imu_detector():
    assert isinstance(get_imu_detector(), IMUArtifactDetector)
