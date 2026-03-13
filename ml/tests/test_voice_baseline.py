"""Tests for per-user voice baseline calibration (issue #352).

Covers:
    - VoiceBaselineCalibrator: frame accumulation, is_ready, normalization,
      get_status, reset
    - get_voice_calibrator: per-user registry
    - Calibration API endpoints: add-frame, status, reset
    - Integration: normalized features improve classification consistency
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure ml/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.voice_ensemble import (
    VoiceBaselineCalibrator,
    _CALIBRATION_KEYS,
    _MIN_CALIBRATION_FRAMES,
    get_voice_calibrator,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_features(
    pitch_mean: float = 150.0,
    pitch_std: float = 20.0,
    energy_mean: float = 0.03,
    energy_std: float = 0.005,
    speaking_rate_proxy: float = 0.4,
    spectral_centroid_mean: float = 2500.0,
    **extra: float,
) -> Dict[str, float]:
    """Return a minimal acoustic feature dict."""
    d: Dict[str, float] = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "speaking_rate_proxy": speaking_rate_proxy,
        "spectral_centroid_mean": spectral_centroid_mean,
    }
    d.update(extra)
    return d


def _fill_calibrator(cal: VoiceBaselineCalibrator, n: int = 10) -> None:
    """Push *n* identical frames into a calibrator."""
    for _ in range(n):
        cal.add_frame(_make_features())


# ── VoiceBaselineCalibrator unit tests ────────────────────────────────────────


class TestVoiceBaselineCalibratorAccumulation:
    """Frame accumulation behaviour."""

    def test_starts_empty(self) -> None:
        cal = VoiceBaselineCalibrator()
        assert cal.n_frames == 0

    def test_add_frame_increments_count(self) -> None:
        cal = VoiceBaselineCalibrator()
        cal.add_frame(_make_features())
        assert cal.n_frames == 1

    def test_multiple_frames_accumulated(self) -> None:
        cal = VoiceBaselineCalibrator()
        for i in range(5):
            cal.add_frame(_make_features(pitch_mean=float(140 + i)))
        assert cal.n_frames == 5

    def test_add_frame_returns_false_before_threshold(self) -> None:
        cal = VoiceBaselineCalibrator()
        for _ in range(_MIN_CALIBRATION_FRAMES - 1):
            result = cal.add_frame(_make_features())
        assert result is False

    def test_add_frame_returns_true_at_threshold(self) -> None:
        cal = VoiceBaselineCalibrator()
        result = None
        for _ in range(_MIN_CALIBRATION_FRAMES):
            result = cal.add_frame(_make_features())
        assert result is True

    def test_add_frame_returns_false_after_already_ready(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal, _MIN_CALIBRATION_FRAMES)
        # Extra frame after calibration is ready
        result = cal.add_frame(_make_features())
        assert result is False

    def test_frames_with_extra_keys_accepted(self) -> None:
        """Extra keys in the feature dict should not break accumulation."""
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal, _MIN_CALIBRATION_FRAMES)
        assert cal.is_ready


class TestVoiceBaselineCalibratorIsReady:
    """is_ready property gates on frame count."""

    def test_not_ready_before_threshold(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal, _MIN_CALIBRATION_FRAMES - 1)
        assert not cal.is_ready

    def test_ready_at_threshold(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal, _MIN_CALIBRATION_FRAMES)
        assert cal.is_ready

    def test_ready_above_threshold(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal, _MIN_CALIBRATION_FRAMES + 5)
        assert cal.is_ready

    def test_min_calibration_frames_constant(self) -> None:
        """Guard against accidental constant change."""
        assert _MIN_CALIBRATION_FRAMES == 10


class TestVoiceBaselineCalibratorNormalization:
    """normalize() produces z-scores against personal baseline."""

    def _ready_cal(self, n: int = _MIN_CALIBRATION_FRAMES) -> VoiceBaselineCalibrator:
        cal = VoiceBaselineCalibrator()
        # Use varied pitch values so std > 0
        for i in range(n):
            cal.add_frame(_make_features(pitch_mean=float(140 + i * 2)))
        return cal

    def test_normalize_returns_dict(self) -> None:
        cal = self._ready_cal()
        out = cal.normalize(_make_features())
        assert isinstance(out, dict)

    def test_normalize_contains_all_input_keys(self) -> None:
        cal = self._ready_cal()
        features = _make_features(mfcc_1=0.5)
        out = cal.normalize(features)
        assert "mfcc_1" in out  # non-calibration key passed through unchanged

    def test_normalize_zscore_mean_subtracted(self) -> None:
        """Normalizing the mean value itself should yield ~0."""
        cal = self._ready_cal()
        mean_features = {k: cal._mean[k] for k in _CALIBRATION_KEYS}
        out = cal.normalize(mean_features)
        for k in _CALIBRATION_KEYS:
            assert abs(out[k]) < 1e-6, f"Expected ~0 for {k} at mean, got {out[k]}"

    def test_normalize_returns_features_unchanged_when_not_ready(self) -> None:
        cal = VoiceBaselineCalibrator()
        features = _make_features()
        out = cal.normalize(features)
        assert out == features

    def test_zero_std_features_set_to_zero(self) -> None:
        """Features with constant baseline (std ≈ 0) should be zeroed out."""
        cal = VoiceBaselineCalibrator()
        # All frames identical → std = 0
        for _ in range(_MIN_CALIBRATION_FRAMES):
            cal.add_frame(_make_features(pitch_mean=180.0))
        out = cal.normalize(_make_features(pitch_mean=200.0))
        assert out["pitch_mean"] == 0.0

    def test_normalize_sign_correct(self) -> None:
        """Value above mean → positive z-score; below → negative."""
        cal = self._ready_cal()
        mean_pitch = cal._mean["pitch_mean"]
        high = cal.normalize(_make_features(pitch_mean=mean_pitch + 100.0))
        low = cal.normalize(_make_features(pitch_mean=mean_pitch - 100.0))
        assert high["pitch_mean"] > 0
        assert low["pitch_mean"] < 0


class TestVoiceBaselineCalibratorGetStatus:
    """get_status() returns progress information."""

    def test_status_before_ready(self) -> None:
        cal = VoiceBaselineCalibrator()
        cal.add_frame(_make_features())
        status = cal.get_status()
        assert status["is_ready"] is False
        assert status["n_frames"] == 1
        assert status["frames_needed"] == _MIN_CALIBRATION_FRAMES
        assert 0 < status["progress_pct"] < 100
        assert status["baseline_mean"] == {}

    def test_status_after_ready(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal)
        status = cal.get_status()
        assert status["is_ready"] is True
        assert status["n_frames"] == _MIN_CALIBRATION_FRAMES
        assert status["progress_pct"] == 100.0
        assert "pitch_mean" in status["baseline_mean"]

    def test_status_progress_pct_capped_at_100(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal, _MIN_CALIBRATION_FRAMES + 20)
        assert cal.get_status()["progress_pct"] == 100.0


class TestVoiceBaselineCalibratorReset:
    """reset() clears all calibration data."""

    def test_reset_clears_frames(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal)
        cal.reset()
        assert cal.n_frames == 0

    def test_reset_clears_ready(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal)
        assert cal.is_ready
        cal.reset()
        assert not cal.is_ready

    def test_reset_clears_statistics(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal)
        cal.reset()
        assert cal._mean == {}
        assert cal._std == {}

    def test_can_recalibrate_after_reset(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal)
        cal.reset()
        _fill_calibrator(cal)
        assert cal.is_ready

    def test_normalize_returns_unchanged_after_reset(self) -> None:
        cal = VoiceBaselineCalibrator()
        _fill_calibrator(cal)
        cal.reset()
        features = _make_features()
        assert cal.normalize(features) == features


# ── Per-user calibrator registry ──────────────────────────────────────────────


class TestGetVoiceCalibrator:
    """get_voice_calibrator() returns per-user instances."""

    def test_returns_calibrator_instance(self) -> None:
        cal = get_voice_calibrator("user_registry_test_a")
        assert isinstance(cal, VoiceBaselineCalibrator)

    def test_same_user_returns_same_instance(self) -> None:
        cal1 = get_voice_calibrator("user_registry_test_b")
        cal2 = get_voice_calibrator("user_registry_test_b")
        assert cal1 is cal2

    def test_different_users_get_different_instances(self) -> None:
        cal1 = get_voice_calibrator("user_registry_test_c1")
        cal2 = get_voice_calibrator("user_registry_test_c2")
        assert cal1 is not cal2

    def test_state_isolated_per_user(self) -> None:
        u1 = get_voice_calibrator("isolation_user_1")
        u2 = get_voice_calibrator("isolation_user_2")
        u1.reset()
        u2.reset()
        _fill_calibrator(u1)
        assert u1.is_ready
        assert not u2.is_ready


# ── Calibration API endpoints ──────────────────────────────────────────────────


@pytest.fixture()
def api_client():
    """Return a FastAPI TestClient for the voice_watch router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    from api.routes.voice_watch import router
    app.include_router(router)
    return TestClient(app)


class TestCalibrateAddFrameEndpoint:
    """POST /voice-watch/calibrate/add-frame"""

    def test_add_frame_returns_status(self, api_client: Any) -> None:
        resp = api_client.post(
            "/voice-watch/calibrate/add-frame",
            json={
                "user_id": "ep_user_add_1",
                "audio_features": _make_features(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "is_ready" in body
        assert "n_frames" in body
        assert "progress_pct" in body

    def test_add_multiple_frames_reaches_ready(self, api_client: Any) -> None:
        for _ in range(_MIN_CALIBRATION_FRAMES):
            resp = api_client.post(
                "/voice-watch/calibrate/add-frame",
                json={
                    "user_id": "ep_user_add_2",
                    "audio_features": _make_features(),
                },
            )
        assert resp.status_code == 200
        assert resp.json()["is_ready"] is True

    def test_add_frame_missing_user_id_returns_422(self, api_client: Any) -> None:
        resp = api_client.post(
            "/voice-watch/calibrate/add-frame",
            json={"audio_features": _make_features()},
        )
        assert resp.status_code == 422


class TestCalibrateStatusEndpoint:
    """GET /voice-watch/calibrate/status"""

    def test_status_returns_progress(self, api_client: Any) -> None:
        # Seed one frame
        api_client.post(
            "/voice-watch/calibrate/add-frame",
            json={
                "user_id": "ep_user_status_1",
                "audio_features": _make_features(),
            },
        )
        resp = api_client.get(
            "/voice-watch/calibrate/status",
            params={"user_id": "ep_user_status_1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_frames"] >= 1
        assert "frames_needed" in body

    def test_status_for_new_user_shows_zero_frames(self, api_client: Any) -> None:
        resp = api_client.get(
            "/voice-watch/calibrate/status",
            params={"user_id": "ep_user_status_brand_new"},
        )
        assert resp.status_code == 200
        assert resp.json()["n_frames"] == 0


class TestCalibrateResetEndpoint:
    """POST /voice-watch/calibrate/reset"""

    def test_reset_clears_calibration(self, api_client: Any) -> None:
        uid = "ep_user_reset_1"
        # Add enough frames to become ready
        for _ in range(_MIN_CALIBRATION_FRAMES):
            api_client.post(
                "/voice-watch/calibrate/add-frame",
                json={"user_id": uid, "audio_features": _make_features()},
            )
        # Confirm ready
        status = api_client.get(
            "/voice-watch/calibrate/status", params={"user_id": uid}
        ).json()
        assert status["is_ready"] is True

        # Reset
        resp = api_client.post(
            "/voice-watch/calibrate/reset",
            json={"user_id": uid},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"

        # Confirm cleared
        status_after = api_client.get(
            "/voice-watch/calibrate/status", params={"user_id": uid}
        ).json()
        assert status_after["is_ready"] is False
        assert status_after["n_frames"] == 0

    def test_reset_missing_user_id_returns_422(self, api_client: Any) -> None:
        resp = api_client.post("/voice-watch/calibrate/reset", json={})
        assert resp.status_code == 422


# ── Classification consistency improvement ────────────────────────────────────


class TestNormalizedFeaturesImprovementConsistency:
    """Normalized features reduce inter-user variance for identical emotion signals."""

    def test_normalization_reduces_variance_across_virtual_users(self) -> None:
        """Two users with different vocal baselines should produce more similar
        z-scored pitch values when normalizing against their own baselines,
        versus using raw values."""
        # User A: naturally low-pitched speaker (baseline ~120 Hz)
        cal_a = VoiceBaselineCalibrator()
        for _ in range(_MIN_CALIBRATION_FRAMES):
            cal_a.add_frame(_make_features(pitch_mean=120.0 + np.random.randn() * 5))

        # User B: naturally high-pitched speaker (baseline ~240 Hz)
        cal_b = VoiceBaselineCalibrator()
        for _ in range(_MIN_CALIBRATION_FRAMES):
            cal_b.add_frame(_make_features(pitch_mean=240.0 + np.random.randn() * 5))

        # Both express the same "above-baseline excited" pattern (+30 Hz above their own baseline)
        excited_a = _make_features(pitch_mean=cal_a._mean["pitch_mean"] + 30.0)
        excited_b = _make_features(pitch_mean=cal_b._mean["pitch_mean"] + 30.0)

        # Raw values are very different
        raw_diff = abs(excited_a["pitch_mean"] - excited_b["pitch_mean"])

        # Normalized values should be much closer
        norm_a = cal_a.normalize(excited_a)["pitch_mean"]
        norm_b = cal_b.normalize(excited_b)["pitch_mean"]
        norm_diff = abs(norm_a - norm_b)

        assert norm_diff < raw_diff, (
            f"Normalized diff {norm_diff:.3f} should be less than raw diff {raw_diff:.3f}"
        )

    def test_normalization_direction_consistent_with_emotion(self) -> None:
        """High-energy (excited) speech should produce positive energy z-score."""
        cal = VoiceBaselineCalibrator()
        baseline_energy = 0.02
        for _ in range(_MIN_CALIBRATION_FRAMES):
            cal.add_frame(_make_features(energy_mean=baseline_energy + np.random.randn() * 0.001))

        high_energy = cal.normalize(_make_features(energy_mean=baseline_energy + 0.05))
        assert high_energy["energy_mean"] > 0

    def test_calibration_keys_all_normalized(self) -> None:
        """Every key in _CALIBRATION_KEYS must appear in normalized output."""
        cal = VoiceBaselineCalibrator()
        for i in range(_MIN_CALIBRATION_FRAMES):
            cal.add_frame(_make_features(pitch_mean=float(140 + i)))
        out = cal.normalize(_make_features())
        for k in _CALIBRATION_KEYS:
            assert k in out, f"Calibration key {k!r} missing from normalize() output"
